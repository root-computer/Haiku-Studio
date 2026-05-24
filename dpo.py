#!/usr/bin/env python
"""
dpo.py — Direct Preference Optimization for h2 Haiku/Tanka checkpoints.

Preference data formats supported:
  1) JSONL, one object per line:
     {"prompt":"user: ...\nbot:", "chosen":"...", "rejected":"..."}
     Aliases: user/preferred/good/accepted and bad/dispreferred.

  2) Plain text blocks separated by blank lines:
     user: Explain X.
     chosen: Good answer.
     rejected: Bad answer.

DPO uses a frozen reference model and trains only the policy model:
  loss = -logsigmoid(beta * [(logp_c - logp_r) - (ref_logp_c - ref_logp_r)])

Example:
  python dpo.py --config config/gpt_config.json
  python dpo.py --config config/gpt_config.json --resume
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
import shutil
from dataclasses import asdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pretrain import (
    GPT,
    IGNORE_INDEX,
    autocast_ctx,
    configure_torch,
    deep_get,
    ensure_dir,
    file_sha256,
    get_device,
    gpt_config_from_dict,
    load_json,
    make_grad_scaler,
    optimizer_from_config,
    set_seed,
    tokenizer_vocab_size,
    tok_encode,
)


def stable_hash_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_score(seed: int, key: str) -> float:
    payload = f"{seed}:{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(1 << 64)


def iter_preference_records(path_or_dir: str) -> Iterator[Tuple[str, Dict[str, str]]]:
    paths = sorted(glob.glob(os.path.join(path_or_dir, "*"))) if os.path.isdir(path_or_dir) else [path_or_dir]
    paths = [p for p in paths if os.path.isfile(p) and os.path.basename(p) != ".gitkeep"]
    if not paths:
        raise FileNotFoundError(f"No DPO preference files found at {path_or_dir!r}")

    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    rec = normalize_record(obj)
                    if rec:
                        yield f"{os.path.basename(path)}:{idx}", rec
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                buf: List[str] = []
                doc_idx = 0
                for line in f:
                    if line.strip() == "":
                        if buf:
                            rec = parse_text_block("".join(buf))
                            if rec:
                                yield f"{os.path.basename(path)}:{doc_idx}", rec
                            doc_idx += 1
                            buf = []
                    else:
                        buf.append(line)
                if buf:
                    rec = parse_text_block("".join(buf))
                    if rec:
                        yield f"{os.path.basename(path)}:{doc_idx}", rec


def first_string(obj: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def normalize_record(obj: Dict[str, Any]) -> Optional[Dict[str, str]]:
    prompt = first_string(obj, ["prompt", "user", "input", "instruction", "context"])
    chosen = first_string(obj, ["chosen", "preferred", "good", "accepted", "winner", "response_chosen"])
    rejected = first_string(obj, ["rejected", "dispreferred", "bad", "declined", "loser", "response_rejected"])

    # Common nested shape: {"prompt":..., "responses":{"chosen":...,"rejected":...}}
    responses = obj.get("responses")
    if isinstance(responses, dict):
        chosen = chosen or first_string(responses, ["chosen", "preferred", "good", "accepted"])
        rejected = rejected or first_string(responses, ["rejected", "dispreferred", "bad", "declined"])

    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def parse_text_block(text: str) -> Optional[Dict[str, str]]:
    fields: Dict[str, List[str]] = {"prompt": [], "chosen": [], "rejected": []}
    current: Optional[str] = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        low = line.lower().strip()
        matched = None
        for role, aliases in {
            "prompt": ["prompt:", "user:", "input:", "instruction:"],
            "chosen": ["chosen:", "preferred:", "good:", "accepted:", "winner:"],
            "rejected": ["rejected:", "dispreferred:", "bad:", "declined:", "loser:"],
        }.items():
            for alias in aliases:
                if low.startswith(alias):
                    matched = role
                    content = line[len(alias):].lstrip()
                    current = role
                    if content:
                        fields[role].append(content)
                    break
            if matched:
                break
        if not matched and current:
            fields[current].append(line)

    rec = {k: "\n".join(v).strip() for k, v in fields.items()}
    if rec["prompt"] and rec["chosen"] and rec["rejected"] and rec["chosen"] != rec["rejected"]:
        return rec
    return None


def ensure_prompt_prefix(prompt: str) -> str:
    text = prompt.strip()
    low = text.lower()
    if "bot:" in low or "assistant:" in low:
        return text.rstrip() + "\n"
    if low.startswith("user:"):
        return text.rstrip() + "\nbot: "
    return f"user: {text}\nbot: "


def encode_preference(tokenizer, prompt: str, answer: str, block_size: int) -> Tuple[List[int], List[int]]:
    prompt_text = ensure_prompt_prefix(prompt)
    answer_text = answer.strip() + "\n"
    prompt_ids = tok_encode(tokenizer, prompt_text)
    answer_ids = tok_encode(tokenizer, answer_text)
    ids = prompt_ids + answer_ids
    mask = [0] * len(prompt_ids) + [1] * len(answer_ids)

    # Keep the end of the prompt+answer if it exceeds context, while preserving
    # the answer-token supervision mask alignment.
    if len(ids) > block_size + 1:
        ids = ids[-(block_size + 1):]
        mask = mask[-(block_size + 1):]
    if len(ids) < 2 or sum(mask[1:]) == 0:
        return [], []
    x = ids[:-1]
    y = [tok if m else IGNORE_INDEX for tok, m in zip(ids[1:], mask[1:])]
    return x, y


class PreferenceStore:
    def __init__(self, config: Dict[str, Any], tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.cache_path = deep_get(config, "paths.dpo_cache", "data/dpo_cache.pt")

    def dataset_signature(self, dpo_path: str, tokenizer_sha: str) -> str:
        paths = sorted(glob.glob(os.path.join(dpo_path, "*"))) if os.path.isdir(dpo_path) else [dpo_path]
        fps = []
        for p in paths:
            if not os.path.isfile(p) or os.path.basename(p) == ".gitkeep":
                continue
            st = os.stat(p)
            fps.append({"path": os.path.abspath(p), "bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
        obj = {
            "files": fps,
            "tokenizer_sha256": tokenizer_sha,
            "block_size": int(deep_get(self.config, "model.block_size", 1024)),
            "val_ratio": float(deep_get(self.config, "dpo.val_ratio", 0.05)),
            "seed": int(deep_get(self.config, "runtime.seed", 1337)),
        }
        return stable_hash_json(obj)

    def load_or_build(self, dpo_path: str, tokenizer_sha: str, block_size: int, force: bool = False) -> Dict[str, Any]:
        sig = self.dataset_signature(dpo_path, tokenizer_sha)
        if not force and os.path.exists(self.cache_path):
            cache = torch.load(self.cache_path, map_location="cpu", weights_only=False)
            if cache.get("signature") == sig:
                print(f"Using existing DPO cache: {self.cache_path}")
                return cache

        print(f"Building DPO cache from {dpo_path}")
        ensure_dir(os.path.dirname(self.cache_path) or ".")
        seed = int(deep_get(self.config, "runtime.seed", 1337))
        val_ratio = float(deep_get(self.config, "dpo.val_ratio", 0.05))
        train: List[Tuple[List[int], List[int], List[int], List[int]]] = []
        val: List[Tuple[List[int], List[int], List[int], List[int]]] = []
        skipped = 0
        total = 0
        for key, rec in iter_preference_records(dpo_path):
            total += 1
            cx, cy = encode_preference(self.tokenizer, rec["prompt"], rec["chosen"], block_size)
            rx, ry = encode_preference(self.tokenizer, rec["prompt"], rec["rejected"], block_size)
            if not cx or not rx:
                skipped += 1
                continue
            target = val if split_score(seed, key) < val_ratio else train
            target.append((cx, cy, rx, ry))
        if not train:
            raise RuntimeError("No trainable DPO preference pairs found. Add JSONL/text pairs to the configured DPO dataset path.")
        if not val:
            print("WARNING: no validation DPO pairs found; using a small train subset for eval.")
            val = train[: min(32, len(train))]
        cache = {"signature": sig, "train": train, "val": val, "created_time": time.time(), "skipped": skipped, "total_records": total}
        tmp = self.cache_path + ".tmp"
        torch.save(cache, tmp)
        os.replace(tmp, self.cache_path)
        print(f"DPO cache ready: train={len(train):,} val={len(val):,} skipped={skipped:,} raw_records={total:,}")
        return cache


class PreferenceDataset:
    def __init__(self, examples: List[Tuple[List[int], List[int], List[int], List[int]]], block_size: int):
        self.examples = examples
        self.block_size = int(block_size)

    def __len__(self):
        return len(self.examples)

    def order_for_epoch(self, seed: int, epoch_idx: int, shuffle: bool = True) -> List[int]:
        order = list(range(len(self.examples)))
        if shuffle:
            rng = random.Random(seed + epoch_idx * 1_000_003)
            rng.shuffle(order)
        return order

    def get_batch(self, order: List[int], cursor: int, batch_size: int, device: torch.device):
        if cursor + batch_size > len(order):
            batch = self._make_batch([self.examples[order[i % len(order)]] for i in range(batch_size)], device)
            return (*batch, batch_size, True)
        selected = [self.examples[order[i]] for i in range(cursor, cursor + batch_size)]
        batch = self._make_batch(selected, device)
        return (*batch, cursor + batch_size, False)

    def _make_batch(self, selected, device):
        c_max = min(self.block_size, max(len(x[0]) for x in selected))
        r_max = min(self.block_size, max(len(x[2]) for x in selected))
        bs = len(selected)
        cx = np.zeros((bs, c_max), dtype=np.int64)
        cy = np.full((bs, c_max), IGNORE_INDEX, dtype=np.int64)
        rx = np.zeros((bs, r_max), dtype=np.int64)
        ry = np.full((bs, r_max), IGNORE_INDEX, dtype=np.int64)
        for row, (cxi, cyi, rxi, ryi) in enumerate(selected):
            cn = min(c_max, len(cxi))
            rn = min(r_max, len(rxi))
            cx[row, :cn] = np.asarray(cxi[:cn], dtype=np.int64)
            cy[row, :cn] = np.asarray(cyi[:cn], dtype=np.int64)
            rx[row, :rn] = np.asarray(rxi[:rn], dtype=np.int64)
            ry[row, :rn] = np.asarray(ryi[:rn], dtype=np.int64)
        non_blocking = device.type == "cuda"
        return (
            torch.from_numpy(cx).to(device, non_blocking=non_blocking),
            torch.from_numpy(cy).to(device, non_blocking=non_blocking),
            torch.from_numpy(rx).to(device, non_blocking=non_blocking),
            torch.from_numpy(ry).to(device, non_blocking=non_blocking),
        )


class CosineSchedule:
    def __init__(self, optimizer, lr_max: float, lr_min: float, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))

    def step(self, current_step: int) -> float:
        if current_step < self.warmup_steps:
            lr = self.lr_max * float(current_step + 1) / float(self.warmup_steps)
        else:
            progress = (current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


class JsonlLogger:
    def __init__(self, path: str):
        self.path = path
        ensure_dir(os.path.dirname(path) or ".")

    def log(self, **kwargs):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kwargs, sort_keys=True) + "\n")


def load_checkpoint_state(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def make_model_from_checkpoint(config: Dict[str, Any], vocab_size: int, ckpt_path: str, device: torch.device) -> GPT:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    cfg_dict = dict(config.get("model", {}))
    cfg_dict["vocab_size"] = vocab_size
    ckpt = load_checkpoint_state(ckpt_path, device)
    ckpt_cfg = ckpt.get("config") or {}
    cfg = gpt_config_from_dict({**cfg_dict, **ckpt_cfg})
    model = GPT(cfg).to(device)
    model.logit_chunk_size = int(deep_get(config, "dpo.logit_chunk_size", deep_get(config, "train.logit_chunk_size", 64)))
    model.load_state_dict(ckpt["state"], strict=True)
    return model


def resolve_policy_checkpoint(config: Dict[str, Any]) -> str:
    candidates = [
        deep_get(config, "dpo.policy_checkpoint", None),
        deep_get(config, "paths.sft_best_checkpoint", None),
        deep_get(config, "paths.sft_checkpoint", None),
        deep_get(config, "paths.pretrain_best_checkpoint", None),
        deep_get(config, "paths.pretrain_checkpoint", None),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("No policy checkpoint found for DPO. Run SFT first or set dpo.policy_checkpoint.")


def ensure_reference_checkpoint(config: Dict[str, Any], policy_ckpt: str) -> str:
    ref = deep_get(config, "dpo.reference_checkpoint", "data/model.sft.reference.pt")
    if os.path.exists(ref):
        return ref
    if not bool(deep_get(config, "dpo.create_reference_if_missing", True)):
        raise FileNotFoundError(f"DPO reference checkpoint missing: {ref}")
    ensure_dir(os.path.dirname(ref) or ".")
    shutil.copy2(policy_ckpt, ref)
    print(f"Created frozen DPO reference checkpoint from policy base: {ref}")
    return ref


def sequence_logps(model: GPT, x: torch.Tensor, y: torch.Tensor, use_amp: bool, device: torch.device) -> torch.Tensor:
    with autocast_ctx(device, use_amp and device.type == "cuda"):
        logits, _ = model(x, None)
        logp = F.log_softmax(logits.float(), dim=-1)
    mask = y.ne(IGNORE_INDEX)
    safe_y = y.masked_fill(~mask, 0)
    tok_logp = logp.gather(-1, safe_y.unsqueeze(-1)).squeeze(-1)
    # Sum over supervised answer tokens; DPO literature generally uses summed sequence logprobs.
    return (tok_logp * mask.float()).sum(dim=-1)


def dpo_batch_loss(policy: GPT, reference: GPT, batch, beta: float, use_amp: bool, device: torch.device) -> Dict[str, torch.Tensor]:
    cx, cy, rx, ry = batch
    pol_c = sequence_logps(policy, cx, cy, use_amp, device)
    pol_r = sequence_logps(policy, rx, ry, use_amp, device)
    with torch.no_grad():
        ref_c = sequence_logps(reference, cx, cy, use_amp, device)
        ref_r = sequence_logps(reference, rx, ry, use_amp, device)
    pol_delta = pol_c - pol_r
    ref_delta = ref_c - ref_r
    logits = beta * (pol_delta - ref_delta)
    loss = -F.logsigmoid(logits).mean()
    return {
        "loss": loss,
        "accuracy": (logits > 0).float().mean(),
        "reward_chosen": (beta * (pol_c - ref_c)).mean().detach(),
        "reward_rejected": (beta * (pol_r - ref_r)).mean().detach(),
        "margin": (pol_delta - ref_delta).mean().detach(),
    }


@torch.no_grad()
def evaluate(policy: GPT, reference: GPT, dataset: PreferenceDataset, batch_size: int, max_batches: int, beta: float, device: torch.device, use_amp: bool) -> Dict[str, float]:
    policy.eval()
    reference.eval()
    order = list(range(len(dataset)))
    cursor = 0
    totals = {"loss": 0.0, "accuracy": 0.0, "reward_chosen": 0.0, "reward_rejected": 0.0, "margin": 0.0}
    count = 0
    for _ in range(min(max_batches, max(1, len(order) // max(1, batch_size)))):
        cx, cy, rx, ry, cursor, wrapped = dataset.get_batch(order, cursor, batch_size, device)
        out = dpo_batch_loss(policy, reference, (cx, cy, rx, ry), beta, use_amp, device)
        if torch.isfinite(out["loss"]):
            for k in totals:
                totals[k] += float(out[k].item())
            count += 1
        if wrapped:
            break
    policy.train()
    return {k: v / max(1, count) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/gpt_config.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    device = get_device(config)
    configure_torch(device, config)
    set_seed(int(deep_get(config, "runtime.seed", 1337)), device)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tokenizer_bpe import BPETokenizer

    tokenizer_path = deep_get(config, "paths.tokenizer_path", "data/tokenizer.json")
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_sha = file_sha256(tokenizer_path)
    vocab_size = tokenizer_vocab_size(tokenizer)

    dpo_ckpt = deep_get(config, "paths.dpo_checkpoint", "data/model.dpo.pt")
    dpo_best = deep_get(config, "paths.dpo_best_checkpoint", "data/model.dpo.best.pt")
    dpo_log = deep_get(config, "paths.dpo_log", "data/dpo_loss.jsonl")

    policy_base = resolve_policy_checkpoint(config)
    reference_ckpt = ensure_reference_checkpoint(config, policy_base)
    policy_load = dpo_ckpt if args.resume and os.path.exists(dpo_ckpt) else policy_base

    policy = make_model_from_checkpoint(config, vocab_size, policy_load, device)
    reference = make_model_from_checkpoint(config, vocab_size, reference_ckpt, device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    optimizer_config = {
        **config,
        "optimizer": {
            **config.get("optimizer", {}),
            "lr": float(deep_get(config, "dpo.lr", 3e-6)),
            "lr_min": float(deep_get(config, "dpo.lr_min", 3e-7)),
            "weight_decay": float(deep_get(config, "dpo.weight_decay", 0.0)),
        },
    }
    optimizer = optimizer_from_config(policy, optimizer_config)
    use_amp = bool(deep_get(config, "runtime.use_amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(device, use_amp)

    block_size = int(policy.cfg.block_size)
    dpo_path = deep_get(config, "paths.dpo_dataset", "dpo")
    store = PreferenceStore(config, tokenizer)
    cache = store.load_or_build(dpo_path, tokenizer_sha, block_size, force=args.force_rebuild_cache or bool(deep_get(config, "dpo.force_rebuild_cache", False)))
    train_ds = PreferenceDataset(cache["train"], block_size)
    val_ds = PreferenceDataset(cache["val"], block_size)

    batch_size = int(deep_get(config, "dpo.batch_size", deep_get(config, "train.batch_size", 1)))
    grad_accum = int(deep_get(config, "dpo.grad_accum", deep_get(config, "train.grad_accum", 8)))
    epochs = int(deep_get(config, "dpo.epochs", 1))
    beta = float(deep_get(config, "dpo.beta", 0.1))
    total_steps_cfg = deep_get(config, "dpo.max_steps", None)
    total_steps = int(total_steps_cfg) if total_steps_cfg is not None else max(1, (len(train_ds) * epochs) // max(1, batch_size * grad_accum))
    warmup_steps = int(total_steps * float(deep_get(config, "dpo.warmup_pct", 0.03)))
    schedule = CosineSchedule(optimizer, float(deep_get(config, "dpo.lr", 3e-6)), float(deep_get(config, "dpo.lr_min", 3e-7)), warmup_steps, total_steps)

    global_step = 0
    epoch_idx = 0
    batch_cursor = 0
    best_val = float("inf")

    if args.resume:
        if not os.path.exists(dpo_ckpt):
            raise FileNotFoundError(f"--resume requested but no DPO checkpoint exists: {dpo_ckpt}")
        ckpt = load_checkpoint_state(dpo_ckpt, device)
        if ckpt.get("tokenizer_sha256") and ckpt["tokenizer_sha256"] != tokenizer_sha:
            raise RuntimeError("Tokenizer mismatch. Refusing to resume DPO.")
        if ckpt.get("dataset_signature") and ckpt["dataset_signature"] != cache["signature"]:
            raise RuntimeError("DPO dataset signature mismatch. Refusing to resume.")
        policy.load_state_dict(ckpt["state"], strict=True)
        if ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as e:
                print(f"WARNING: could not restore scaler: {e}")
        ts = ckpt.get("train_state", {})
        global_step = int(ts.get("global_step", 0))
        epoch_idx = int(ts.get("epoch_idx", 0))
        batch_cursor = int(ts.get("batch_cursor", 0))
        best_val = float(ts.get("best_val", float("inf")))
        print(f"Resumed DPO from {dpo_ckpt}: step={global_step:,} epoch={epoch_idx + 1} cursor={batch_cursor:,}")

    logger = JsonlLogger(dpo_log)
    log_every = int(deep_get(config, "dpo.log_every", deep_get(config, "logging.log_every", 25)))
    eval_every = int(deep_get(config, "dpo.eval_every", deep_get(config, "logging.eval_every", 100)))
    save_every = int(deep_get(config, "dpo.save_every", deep_get(config, "logging.save_every", 250)))
    val_batches = int(deep_get(config, "dpo.val_batches", 100))
    max_grad_norm = float(deep_get(config, "optimizer.max_grad_norm", 1.0))
    shuffle = bool(deep_get(config, "dpo.shuffle", True))
    order = train_ds.order_for_epoch(int(deep_get(config, "runtime.seed", 1337)), epoch_idx, shuffle=shuffle)

    session_start_step = global_step
    session_start_time = time.time()
    running = {"loss": 0.0, "accuracy": 0.0, "reward_chosen": 0.0, "reward_rejected": 0.0, "margin": 0.0}
    running_count = 0
    interrupted = False

    def state() -> Dict[str, Any]:
        return {
            "global_step": int(global_step),
            "epoch_idx": int(epoch_idx),
            "batch_cursor": int(batch_cursor),
            "best_val": float(best_val),
            "total_steps": int(total_steps),
            "batch_size": int(batch_size),
            "grad_accum": int(grad_accum),
            "beta": float(beta),
            "reference_checkpoint": reference_ckpt,
        }

    def save(path: str):
        ensure_dir(os.path.dirname(path) or ".")
        payload = {
            "state": policy.state_dict(),
            "config": asdict(policy.cfg),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "train_state": state(),
            "tokenizer_sha256": tokenizer_sha,
            "dataset_signature": cache["signature"],
            "reference_checkpoint": reference_ckpt,
            "saved_time": time.time(),
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def handle_interrupt(signum, frame):
        nonlocal interrupted
        print("\nInterrupt requested. Will save after current optimizer step.")
        interrupted = True

    try:
        signal.signal(signal.SIGINT, handle_interrupt)
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("DPO START")
    print("=" * 60)
    print(f"policy_base={policy_load}")
    print(f"reference={reference_ckpt}")
    print(f"train_pairs={len(train_ds):,} val_pairs={len(val_ds):,}")
    print(f"total_steps={total_steps:,} warmup_steps={warmup_steps:,} beta={beta} effective_batch={batch_size * grad_accum}")

    policy.train()
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps and epoch_idx < epochs:
        for _ in range(grad_accum):
            if interrupted:
                break
            cx, cy, rx, ry, next_cursor, wrapped = train_ds.get_batch(order, batch_cursor, batch_size, device)
            if wrapped:
                epoch_idx += 1
                if epoch_idx >= epochs:
                    break
                order = train_ds.order_for_epoch(int(deep_get(config, "runtime.seed", 1337)), epoch_idx, shuffle=shuffle)
                batch_cursor = 0
                cx, cy, rx, ry, next_cursor, wrapped = train_ds.get_batch(order, batch_cursor, batch_size, device)
            batch_cursor = next_cursor
            out = dpo_batch_loss(policy, reference, (cx, cy, rx, ry), beta, use_amp, device)
            loss = out["loss"]
            if not torch.isfinite(loss):
                print(f"WARNING: NaN/Inf DPO loss at step {global_step}; skipping update")
                optimizer.zero_grad(set_to_none=True)
                running = {k: 0.0 for k in running}
                running_count = 0
                break
            scaler.scale(loss / grad_accum).backward()
            for k in running:
                running[k] += float(out[k].item())
            running_count += 1

        if epoch_idx >= epochs:
            break

        lr = schedule.step(global_step)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % log_every == 0:
            elapsed = time.time() - session_start_time
            session_steps = max(0, global_step - session_start_step)
            sps = session_steps / max(1.0, elapsed)
            eta = (total_steps - global_step) / max(0.001, sps) / 3600.0
            avg = {k: v / max(1, running_count) for k, v in running.items()}
            print(f"  step {global_step:>7} | loss {avg['loss']:.4f} | acc {avg['accuracy']:.3f} | margin {avg['margin']:.3f} | lr {lr:.2e} | ep {epoch_idx + 1} | cursor {batch_cursor:,}/{len(order):,} | {sps:.3f} steps/s | ETA {eta:.1f}h")

        if global_step % eval_every == 0:
            vals = evaluate(policy, reference, val_ds, batch_size, val_batches, beta, device, use_amp)
            train_avg = {k: v / max(1, running_count) for k, v in running.items()}
            improved = vals["loss"] < best_val
            if improved:
                best_val = vals["loss"]
            print(f"\n  ── DPO EVAL step {global_step} ──")
            print(f"     train={train_avg['loss']:.4f} val={vals['loss']:.4f} acc={vals['accuracy']:.3f} margin={vals['margin']:.3f} lr={lr:.2e}{' ★ NEW BEST' if improved else ''}\n")
            logger.log(kind="eval", step=global_step, train_loss=round(train_avg["loss"], 6), val_loss=round(vals["loss"], 6), accuracy=round(vals["accuracy"], 6), margin=round(vals["margin"], 6), lr=lr, epoch_idx=epoch_idx, batch_cursor=batch_cursor, time=time.time())
            if improved:
                save(dpo_best)
            running = {k: 0.0 for k in running}
            running_count = 0

        if global_step % save_every == 0 or interrupted:
            save(dpo_ckpt)
            print(f"  saved checkpoint: {dpo_ckpt} @ step {global_step:,}")
            if interrupted:
                break

    save(dpo_ckpt)
    vals = evaluate(policy, reference, val_ds, batch_size, val_batches, beta, device, use_amp)
    if vals["loss"] < best_val:
        best_val = vals["loss"]
        save(dpo_best)

    print("\n" + "=" * 60)
    print("DPO COMPLETE" if not interrupted else "DPO PAUSED")
    print("=" * 60)
    print(f"Steps:    {global_step:,}")
    print(f"Best val: {best_val:.4f}")
    print(f"Saved:    {dpo_ckpt}")
    print(f"Best:     {dpo_best}")
    print("=" * 60)


if __name__ == "__main__":
    main()
