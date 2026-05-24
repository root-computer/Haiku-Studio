#!/usr/bin/env python
"""Single-shot h2 checkpoint inference for Haiku Studio.

The studio calls this as a short-lived subprocess so the desktop backend does
not need a Flask/FastAPI Python service. For heavier chat use, this can later be
replaced with a persistent worker without changing the UI contract.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from pretrain import GPT, gpt_config_from_dict, load_json, deep_get, get_device, configure_torch
from tokenizer_bpe import BPETokenizer


def resolve_path(raw: str, fallback: str) -> Path:
    p = Path(raw or fallback)
    return p if p.is_absolute() else ROOT / p


def pick_checkpoint(config: Dict[str, Any]) -> Path | None:
    candidates = [
        deep_get(config, "paths.dpo_best_checkpoint", "data/model.dpo.best.pt"),
        deep_get(config, "paths.dpo_checkpoint", "data/model.dpo.pt"),
        deep_get(config, "paths.sft_best_checkpoint", "data/model.sft.best.pt"),
        deep_get(config, "paths.sft_checkpoint", "data/model.sft.pt"),
        deep_get(config, "paths.pretrain_best_checkpoint", "data/model.best.pt"),
        deep_get(config, "paths.pretrain_checkpoint", "data/model.pt"),
    ]
    for raw in candidates:
        p = resolve_path(str(raw), "")
        if p.exists():
            return p
    return None


def build_prompt(history: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for msg in history[-25:]:
        role = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        tag = "bot" if role in ("bot", "assistant") else "user"
        lines.append(f"{tag}: {content}")
    lines.append("bot:")
    return "\n".join(lines) + " "


def apply_repetition_penalty(logits: torch.Tensor, ids: List[int], penalty: float) -> torch.Tensor:
    if penalty is None or penalty <= 1.0 or not ids:
        return logits
    unique_ids = set(ids)
    for tok in unique_ids:
        if logits[tok] < 0:
            logits[tok] *= penalty
        else:
            logits[tok] /= penalty
    return logits


def sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    temperature = max(0.05, float(temperature or 0.9))
    logits = logits / temperature
    probs = F.softmax(logits.float(), dim=-1)
    top_p = float(top_p or 0.92)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        remove = cdf > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_probs[remove] = 0
        sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
        return int(sorted_idx[torch.multinomial(sorted_probs, num_samples=1)].item())
    return int(torch.multinomial(probs, num_samples=1).item())


def generate(model: GPT, tokenizer: BPETokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float, repetition_penalty: float, block_size: int) -> str:
    ids = list(tokenizer.encode(prompt, add_special=False))
    original_len = len(ids)
    eos_ids = {x for x in [getattr(tokenizer, "eos_id", None), tokenizer.tokenizer.token_to_id("<eot>")] if x is not None}
    stop_user = tokenizer.encode("\nuser:", add_special=False)
    stop_bot = tokenizer.encode("\nbot:", add_special=False)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = ids[-block_size:]
            x = torch.tensor([ctx], dtype=torch.long, device=next(model.parameters()).device)
            logits, _ = model(x)
            next_logits = logits[0, -1, :].detach().clone()
            next_logits = apply_repetition_penalty(next_logits, ids[-256:], repetition_penalty)
            nxt = sample_next(next_logits, temperature, top_p)
            ids.append(nxt)
            if nxt in eos_ids:
                break
            tail = ids[-max(len(stop_user), len(stop_bot), 1):]
            if stop_user and tail[-len(stop_user):] == stop_user:
                ids = ids[:-len(stop_user)]
                break
            if stop_bot and tail[-len(stop_bot):] == stop_bot:
                ids = ids[:-len(stop_bot)]
                break

    text = tokenizer.decode(ids[original_len:]).strip()
    return text or "[empty response]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/gpt_config.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    payload = json.loads(sys.stdin.read() or "{}")
    config_path = resolve_path(args.config, "config/gpt_config.json")
    config = load_json(str(config_path))
    tokenizer_path = resolve_path(str(deep_get(config, "paths.tokenizer_path", "data/tokenizer.json")), "data/tokenizer.json")
    checkpoint_path = pick_checkpoint(config)
    prompt = build_prompt(payload.get("history", []))

    if checkpoint_path is None:
        print(json.dumps({
            "reply": "No h2 checkpoint found in the active project yet. Train a model or place a checkpoint in the active project's checkpoints folder.",
            "prompt": prompt,
        }))
        return
    if not tokenizer_path.exists():
        print(json.dumps({"reply": f"Tokenizer not found at {tokenizer_path}.", "prompt": prompt}))
        return

    device = get_device(config)
    configure_torch(device, config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = BPETokenizer.load(str(tokenizer_path))
    cfg_payload = {**deep_get(config, "model", {}), **ckpt.get("config", {})}
    cfg_payload.setdefault("vocab_size", len(tokenizer.tokenizer.get_vocab()))
    model_cfg = gpt_config_from_dict(cfg_payload)
    model_cfg.grad_checkpoint = False
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["state"], strict=True)

    reply = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=int(payload.get("max_new_tokens", 128)),
        temperature=float(payload.get("temperature", 0.9)),
        top_p=float(payload.get("top_p", 0.92)),
        repetition_penalty=float(payload.get("repetition_penalty", 1.15 + float(payload.get("presence_penalty", 0.0)))),
        block_size=int(getattr(model_cfg, "block_size", 1024)),
    )
    print(json.dumps({"reply": reply, "prompt": prompt, "checkpoint": str(checkpoint_path)}))


if __name__ == "__main__":
    main()
