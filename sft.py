#!/usr/bin/env python
"""
sft.py — Supervised fine-tuning for Haiku/Tanka-style GPT checkpoints.

Input format:
  - Plain UTF-8 .txt files.
  - Dialog documents are separated by one or more blank lines.
  - Turns use literal user: and bot: prefixes.
  - Loss is applied only to bot reply content tokens, not user prompt tokens.

Example:
  python sft.py --config config/gpt_config.json --resume
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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the same architecture/checkpoint/runtime utilities as pretrain.py.
from pretrain import (
    GPT,
    GPTConfig,
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
    save_checkpoint,
    set_seed,
    tokenizer_vocab_size,
    tok_encode,
)


def stable_hash_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def iter_dialog_documents(path_or_dir: str) -> Iterator[Tuple[str, str]]:
    paths: List[str]
    if os.path.isdir(path_or_dir):
        paths = sorted(glob.glob(os.path.join(path_or_dir, "*.txt")))
    else:
        paths = [path_or_dir]
    if not paths:
        raise FileNotFoundError(f"No SFT .txt files found at {path_or_dir!r}")

    for path in paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            buf: List[str] = []
            doc_idx = 0
            for line in f:
                if line.strip() == "":
                    if buf:
                        text = "".join(buf).strip()
                        if text:
                            yield f"{os.path.basename(path)}:{doc_idx}", text
                            doc_idx += 1
                        buf = []
                else:
                    buf.append(line)
            if buf:
                text = "".join(buf).strip()
                if text:
                    yield f"{os.path.basename(path)}:{doc_idx}", text


def split_role_line(line: str) -> Tuple[Optional[str], str]:
    low = line.lower()
    if low.startswith("user:"):
        return "user", line[5:].lstrip()
    if low.startswith("bot:"):
        return "bot", line[4:].lstrip()
    return None, line


def encode_dialog(tokenizer, text: str, eot_marker: str = "<eot>") -> Tuple[List[int], List[int]]:
    """Return ids and a same-length supervision mask: 1 only for bot reply content."""
    ids: List[int] = []
    mask: List[int] = []
    current_role: Optional[str] = None

    def add_text(piece: str, supervised: bool) -> None:
        part = tok_encode(tokenizer, piece)
        ids.extend(part)
        mask.extend([1 if supervised else 0] * len(part))

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        role, content = split_role_line(line)
        if role == "user":
            current_role = "user"
            add_text("user:", False)
            if content:
                add_text(" " + content, False)
            add_text("\n", False)
        elif role == "bot":
            current_role = "bot"
            add_text("bot:", False)
            if content:
                # Completion-only loss: supervise the actual bot reply, not the tag.
                add_text(" " + content, True)
            add_text("\n", True)
        else:
            # Continuation line inherits the previous role.
            supervised = current_role == "bot"
            add_text(line + "\n", supervised)

    # Boundary token is context, not a supervised reply.
    if eot_marker:
        add_text(eot_marker + "\n\n", False)

    return ids, mask


def split_score(seed: int, key: str) -> float:
    payload = f"{seed}:{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(1 << 64)


class SFTExampleStore:
    def __init__(self, config: Dict[str, Any], tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.cache_path = deep_get(config, "paths.sft_cache", "data/sft_cache.pt")
        self.eot_marker = str(deep_get(config, "data.eot_marker", "<eot>"))

    def dataset_signature(self, sft_path: str, tokenizer_sha: str) -> str:
        paths = sorted(glob.glob(os.path.join(sft_path, "*.txt"))) if os.path.isdir(sft_path) else [sft_path]
        fps = []
        for p in paths:
            st = os.stat(p)
            fps.append({
                "path": os.path.abspath(p),
                "bytes": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
            })
        obj = {
            "files": fps,
            "tokenizer_sha256": tokenizer_sha,
            "eot_marker": self.eot_marker,
            "block_size": int(deep_get(self.config, "model.block_size", 1024)),
            "val_ratio": float(deep_get(self.config, "sft.val_ratio", 0.05)),
            "seed": int(deep_get(self.config, "runtime.seed", 1337)),
        }
        return stable_hash_json(obj)

    def load_or_build(self, sft_path: str, tokenizer_sha: str, force: bool = False) -> Dict[str, Any]:
        sig = self.dataset_signature(sft_path, tokenizer_sha)
        if not force and os.path.exists(self.cache_path):
            cache = torch.load(self.cache_path, map_location="cpu", weights_only=False)
            if cache.get("signature") == sig:
                print(f"Using existing SFT cache: {self.cache_path}")
                return cache

        print(f"Building SFT cache from {sft_path}")
        ensure_dir(os.path.dirname(self.cache_path) or ".")
        seed = int(deep_get(self.config, "runtime.seed", 1337))
        val_ratio = float(deep_get(self.config, "sft.val_ratio", 0.05))
        block_size = int(deep_get(self.config, "model.block_size", 1024))
        min_supervised = int(deep_get(self.config, "sft.min_supervised_tokens", 1))

        train: List[Tuple[List[int], List[int]]] = []
        val: List[Tuple[List[int], List[int]]] = []
        skipped = 0

        for key, doc in iter_dialog_documents(sft_path):
            ids, mask = encode_dialog(self.tokenizer, doc, self.eot_marker)
            if len(ids) < 2 or sum(mask) < min_supervised:
                skipped += 1
                continue
            windows = self.make_windows(ids, mask, block_size)
            if not windows:
                skipped += 1
                continue
            target = val if split_score(seed, key) < val_ratio else train
            target.extend(windows)

        if not train:
            raise RuntimeError("No trainable SFT examples found.")
        if not val:
            print("WARNING: no validation SFT examples found; using a small train subset for eval.")
            val = train[: min(64, len(train))]

        cache = {
            "signature": sig,
            "train": train,
            "val": val,
            "created_time": time.time(),
            "skipped_documents": skipped,
            "eot_marker": self.eot_marker,
        }
        tmp = self.cache_path + ".tmp"
        torch.save(cache, tmp)
        os.replace(tmp, self.cache_path)
        print(f"SFT cache ready: train={len(train):,} windows val={len(val):,} windows skipped_docs={skipped:,}")
        return cache

    @staticmethod
    def make_windows(ids: List[int], mask: List[int], block_size: int) -> List[Tuple[List[int], List[int]]]:
        """Create x/label windows. labels supervise next tokens only where mask[next_token] is true."""
        out: List[Tuple[List[int], List[int]]] = []
        max_x_len = block_size
        if len(ids) <= 1:
            return out

        # Use non-overlapping windows by default. Very long dialogs are split safely.
        start = 0
        while start < len(ids) - 1:
            end = min(start + max_x_len + 1, len(ids))
            chunk_ids = ids[start:end]
            chunk_mask = mask[start:end]
            if len(chunk_ids) >= 2:
                x = chunk_ids[:-1]
                y = [tok if m else IGNORE_INDEX for tok, m in zip(chunk_ids[1:], chunk_mask[1:])]
                if any(v != IGNORE_INDEX for v in y):
                    out.append((x, y))
            if end == len(ids):
                break
            start += max_x_len
        return out


class SFTWindowDataset:
    def __init__(self, examples: List[Tuple[List[int], List[int]]], block_size: int):
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

    def get_batch(self, order: List[int], cursor: int, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        if cursor + batch_size > len(order):
            xb, yb, next_cursor, _ = self.get_batch(order, 0, batch_size, device)
            return xb, yb, next_cursor, True

        selected = [self.examples[order[i]] for i in range(cursor, cursor + batch_size)]
        max_len = min(self.block_size, max(len(x) for x, _ in selected))
        x_arr = np.zeros((batch_size, max_len), dtype=np.int64)
        y_arr = np.full((batch_size, max_len), IGNORE_INDEX, dtype=np.int64)

        for row, (x, y) in enumerate(selected):
            n = min(max_len, len(x))
            x_arr[row, :n] = np.asarray(x[:n], dtype=np.int64)
            y_arr[row, :n] = np.asarray(y[:n], dtype=np.int64)

        xb = torch.from_numpy(x_arr).to(device, non_blocking=(device.type == "cuda"))
        yb = torch.from_numpy(y_arr).to(device, non_blocking=(device.type == "cuda"))
        return xb, yb, cursor + batch_size, False


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


@torch.no_grad()
def evaluate(model: GPT, dataset: SFTWindowDataset, batch_size: int, max_batches: int, device: torch.device, use_amp: bool) -> float:
    model.eval()
    order = list(range(len(dataset)))
    cursor = 0
    losses = []
    for _ in range(min(max_batches, max(1, len(order) // max(1, batch_size)))):
        xb, yb, cursor, wrapped = dataset.get_batch(order, cursor, batch_size, device)
        with autocast_ctx(device, use_amp and device.type == "cuda"):
            _, loss = model(xb, yb)
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
        if wrapped:
            break
    model.train()
    return sum(losses) / max(1, len(losses))


class JsonlLogger:
    def __init__(self, path: str):
        self.path = path
        ensure_dir(os.path.dirname(path) or ".")

    def log(self, **kwargs):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kwargs, sort_keys=True) + "\n")


def load_base_model(config: Dict[str, Any], tokenizer_vocab: int, device: torch.device) -> GPT:
    cfg_dict = dict(config.get("model", {}))
    cfg_dict["vocab_size"] = tokenizer_vocab
    cfg = gpt_config_from_dict(cfg_dict)
    model = GPT(cfg).to(device)
    model.logit_chunk_size = int(deep_get(config, "train.logit_chunk_size", 128))

    base_ckpt = deep_get(config, "sft.base_checkpoint", None) or deep_get(config, "paths.pretrain_best_checkpoint", "data/model.best.pt")
    if not os.path.exists(base_ckpt):
        base_ckpt = deep_get(config, "paths.pretrain_checkpoint", "data/model.pt")
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"No base checkpoint found for SFT: {base_ckpt}")

    ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config")
    if ckpt_cfg:
        loaded_cfg = gpt_config_from_dict({**cfg_dict, **ckpt_cfg})
        if asdict(loaded_cfg) != asdict(cfg):
            cfg = loaded_cfg
            model = GPT(cfg).to(device)
            model.logit_chunk_size = int(deep_get(config, "train.logit_chunk_size", 128))
    model.load_state_dict(ckpt["state"], strict=True)
    print(f"Loaded base checkpoint: {base_ckpt}")
    return model


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

    model = load_base_model(config, vocab_size, device)
    optimizer = optimizer_from_config(model, {
        **config,
        "optimizer": {
            **config.get("optimizer", {}),
            "lr": float(deep_get(config, "sft.lr", 5e-5)),
            "lr_min": float(deep_get(config, "sft.lr_min", 5e-6)),
            "weight_decay": float(deep_get(config, "sft.weight_decay", 0.0)),
        },
    })
    use_amp = bool(deep_get(config, "runtime.use_amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(device, use_amp)

    sft_path = deep_get(config, "paths.sft_dataset", "sft")
    store = SFTExampleStore(config, tokenizer)
    cache = store.load_or_build(sft_path, tokenizer_sha, force=args.force_rebuild_cache or bool(deep_get(config, "sft.force_rebuild_cache", False)))

    block_size = int(model.cfg.block_size)
    train_ds = SFTWindowDataset(cache["train"], block_size)
    val_ds = SFTWindowDataset(cache["val"], block_size)

    batch_size = int(deep_get(config, "sft.batch_size", deep_get(config, "train.batch_size", 1)))
    grad_accum = int(deep_get(config, "sft.grad_accum", deep_get(config, "train.grad_accum", 8)))
    epochs = int(deep_get(config, "sft.epochs", 2))
    total_steps_cfg = deep_get(config, "sft.max_steps", None)
    if total_steps_cfg is None:
        total_steps = max(1, (len(train_ds) * epochs) // max(1, batch_size * grad_accum))
    else:
        total_steps = int(total_steps_cfg)

    warmup_steps = int(total_steps * float(deep_get(config, "sft.warmup_pct", 0.03)))
    schedule = CosineSchedule(
        optimizer,
        lr_max=float(deep_get(config, "sft.lr", 5e-5)),
        lr_min=float(deep_get(config, "sft.lr_min", 5e-6)),
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    sft_ckpt = deep_get(config, "paths.sft_checkpoint", "data/model.sft.pt")
    sft_best = deep_get(config, "paths.sft_best_checkpoint", "data/model.sft.best.pt")
    sft_log = deep_get(config, "paths.sft_log", "data/sft_loss.jsonl")

    global_step = 0
    epoch_idx = 0
    batch_cursor = 0
    best_val = float("inf")

    if args.resume:
        if not os.path.exists(sft_ckpt):
            raise FileNotFoundError(f"--resume requested but no SFT checkpoint exists: {sft_ckpt}")
        ckpt = torch.load(sft_ckpt, map_location=device, weights_only=False)
        if ckpt.get("tokenizer_sha256") and ckpt["tokenizer_sha256"] != tokenizer_sha:
            raise RuntimeError("Tokenizer mismatch. Refusing to resume SFT.")
        if ckpt.get("dataset_signature") and ckpt["dataset_signature"] != cache["signature"]:
            raise RuntimeError("SFT dataset signature mismatch. Refusing to resume.")
        model.load_state_dict(ckpt["state"], strict=True)
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
        print(f"Resumed SFT from {sft_ckpt}: step={global_step:,} epoch={epoch_idx + 1} cursor={batch_cursor:,}")

    logger = JsonlLogger(sft_log)
    log_every = int(deep_get(config, "sft.log_every", deep_get(config, "logging.log_every", 50)))
    eval_every = int(deep_get(config, "sft.eval_every", deep_get(config, "logging.eval_every", 100)))
    save_every = int(deep_get(config, "sft.save_every", deep_get(config, "logging.save_every", 500)))
    val_batches = int(deep_get(config, "sft.val_batches", 100))
    max_grad_norm = float(deep_get(config, "optimizer.max_grad_norm", 1.0))
    shuffle = bool(deep_get(config, "sft.shuffle", True))

    order = train_ds.order_for_epoch(int(deep_get(config, "runtime.seed", 1337)), epoch_idx, shuffle=shuffle)

    session_start_step = global_step
    session_start_time = time.time()
    running_loss = 0.0
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
        }

    def save(path: str):
        payload = {
            "state": model.state_dict(),
            "config": asdict(model.cfg),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "train_state": state(),
            "tokenizer_sha256": tokenizer_sha,
            "dataset_signature": cache["signature"],
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
    print("SFT START")
    print("=" * 60)
    print(f"train_windows={len(train_ds):,} val_windows={len(val_ds):,}")
    print(f"total_steps={total_steps:,} warmup_steps={warmup_steps:,} effective_batch={batch_size * grad_accum}")
    print(f"completion_only_loss=True role_tags=user:/bot:\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps and epoch_idx < epochs:
        for _ in range(grad_accum):
            if interrupted:
                break
            xb, yb, next_cursor, wrapped = train_ds.get_batch(order, batch_cursor, batch_size, device)
            if wrapped:
                epoch_idx += 1
                if epoch_idx >= epochs:
                    break
                order = train_ds.order_for_epoch(int(deep_get(config, "runtime.seed", 1337)), epoch_idx, shuffle=shuffle)
                batch_cursor = 0
                xb, yb, next_cursor, wrapped = train_ds.get_batch(order, batch_cursor, batch_size, device)
            batch_cursor = next_cursor

            with autocast_ctx(device, use_amp):
                _, loss = model(xb, yb)
            if not torch.isfinite(loss):
                print(f"WARNING: NaN/Inf SFT loss at step {global_step}; skipping update")
                optimizer.zero_grad(set_to_none=True)
                running_loss = 0.0
                running_count = 0
                break
            scaler.scale(loss / grad_accum).backward()
            running_loss += float(loss.item())
            running_count += 1

        if epoch_idx >= epochs:
            break

        lr = schedule.step(global_step)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % log_every == 0:
            elapsed = time.time() - session_start_time
            session_steps = max(0, global_step - session_start_step)
            sps = session_steps / max(1.0, elapsed)
            eta = (total_steps - global_step) / max(0.001, sps) / 3600.0
            avg = running_loss / max(1, running_count)
            print(f"  step {global_step:>7} | loss {avg:.4f} | lr {lr:.2e} | ep {epoch_idx + 1} | cursor {batch_cursor:,}/{len(order):,} | {sps:.3f} steps/s | ETA {eta:.1f}h")

        if global_step % eval_every == 0:
            val_loss = evaluate(model, val_ds, batch_size, val_batches, device, use_amp)
            train_avg = running_loss / max(1, running_count)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            print(f"\n  ── SFT EVAL step {global_step} ──")
            print(f"     train={train_avg:.4f}  val={val_loss:.4f}  ppl={math.exp(min(val_loss, 20)):.1f}  lr={lr:.2e}{' ★ NEW BEST' if improved else ''}\n")
            logger.log(kind="eval", step=global_step, train_loss=round(train_avg, 6), val_loss=round(val_loss, 6), lr=lr, epoch_idx=epoch_idx, batch_cursor=batch_cursor, time=time.time())
            if improved:
                save(sft_best)
            running_loss = 0.0
            running_count = 0

        if global_step % save_every == 0 or interrupted:
            save(sft_ckpt)
            print(f"  saved checkpoint: {sft_ckpt} @ step {global_step:,}")
            if interrupted:
                break

    save(sft_ckpt)
    val_loss = evaluate(model, val_ds, batch_size, val_batches, device, use_amp)
    if val_loss < best_val:
        best_val = val_loss
        save(sft_best)

    print("\n" + "=" * 60)
    print("SFT COMPLETE" if not interrupted else "SFT PAUSED")
    print("=" * 60)
    print(f"Steps:    {global_step:,}")
    print(f"Best val: {best_val:.4f} (ppl {math.exp(min(best_val, 20)):.1f})")
    print(f"Saved:    {sft_ckpt}")
    print(f"Best:     {sft_best}")
    print("=" * 60)


if __name__ == "__main__":
    main()
