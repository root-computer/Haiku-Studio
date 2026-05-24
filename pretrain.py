#!/usr/bin/env python
"""
pretrain.py — Fault-tolerant Haiku/Tanka-style GPT pretraining.

Key properties:
  - Loads all parameters from config/gpt_config.json.
  - Optional deterministic pretokenized .bin cache.
  - Atomic cache manifest and tokenizer/corpus/config signature checks.
  - Exact training resume from the last checkpoint created by this script:
      model + optimizer + AMP scaler + global step + epoch + token-window cursor.
  - Correct resumed ETA / steps-per-second accounting.
  - Optional tied or untied token embeddings.
  - Causal LM objective over pretraining text.

The first run should usually be:
  python pretrain.py --config config/gpt_config.json --build-cache

Then train/resume:
  python pretrain.py --config config/gpt_config.json --resume
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import shutil
import signal
import sys
import time
import gc
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100


# =============================================================================
# CONFIG / UTILITIES
# =============================================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(obj: Dict[str, Any], path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def deep_get(d: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def quick_file_fingerprint(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "name": os.path.basename(path),
        "bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def stable_hash_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def get_device(config: Dict[str, Any]) -> torch.device:
    requested = deep_get(config, "runtime.device", "auto")
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch(device: torch.device, config: Dict[str, Any]) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(deep_get(config, "runtime.allow_tf32", False))
        torch.backends.cudnn.allow_tf32 = bool(deep_get(config, "runtime.allow_tf32", False))
        torch.backends.cudnn.benchmark = bool(deep_get(config, "runtime.cudnn_benchmark", True))
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(bool(deep_get(config, "runtime.enable_flash_sdp", True)))
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(bool(deep_get(config, "runtime.enable_mem_efficient_sdp", True)))
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(bool(deep_get(config, "runtime.enable_math_sdp", True)))


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


try:
    from torch.amp import autocast as amp_autocast
    from torch.amp import GradScaler as TorchGradScaler
    AMP_NEW = True
except Exception:
    from torch.cuda.amp import autocast as amp_autocast
    from torch.cuda.amp import GradScaler as TorchGradScaler
    AMP_NEW = False


class NullContext:
    def __enter__(self): return None
    def __exit__(self, *args): return False


def autocast_ctx(device: torch.device, enabled: bool):
    if not enabled:
        return NullContext()
    if AMP_NEW:
        return amp_autocast(device_type=device.type, enabled=True)
    return amp_autocast(enabled=True)


def make_grad_scaler(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda")
    try:
        return TorchGradScaler(device.type, enabled=enabled)
    except TypeError:
        return TorchGradScaler(enabled=enabled)


def tokenizer_vocab_size(tokenizer) -> int:
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "get_vocab_size"):
        return int(tokenizer.tokenizer.get_vocab_size())
    if hasattr(tokenizer, "get_vocab_size"):
        return int(tokenizer.get_vocab_size())
    raise AttributeError("Tokenizer object has no get_vocab_size method.")


def tok_encode(tokenizer, text: str) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_special=False))
    except TypeError:
        return list(tokenizer.encode(text))


# =============================================================================
# MODEL
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt() * self.weight


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k, positions):
        freqs = torch.einsum("t,d->td", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 26
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 1024
    dropout: float = 0.0
    rope_base: float = 50000.0
    grad_checkpoint: bool = True
    hidden_mult: float = 2.67
    tied_embeddings: bool = False
    residual_init_scale: bool = True


def gpt_config_from_dict(d: Dict[str, Any]) -> GPTConfig:
    valid = {f.name for f in fields(GPTConfig)}
    return GPTConfig(**{k: v for k, v in d.items() if k in valid})


class GQAAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        if self.n_head % self.n_kv != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.rope = RoPE(self.head_dim, base=cfg.rope_base)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, torch.arange(seqlen, device=x.device))

        if self.n_kv != self.n_head:
            try:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, enable_gqa=True)
            except (TypeError, RuntimeError):
                repeat = self.n_head // self.n_kv
                y = F.scaled_dot_product_attention(
                    q,
                    k.repeat_interleave(repeat, dim=1),
                    v.repeat_interleave(repeat, dim=1),
                    dropout_p=0.0,
                    is_causal=True,
                )
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_head * self.head_dim)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 2.67):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = GQAAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg.n_embd, hidden_mult=cfg.hidden_mult)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tied_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self._use_checkpoint = bool(cfg.grad_checkpoint)
        self.logit_chunk_size = 128
        self.apply(self._init_weights)
        # Apply residual init scaling as a separate explicit pass so traversal
        # order in apply() cannot overwrite the scaled values.
        if bool(getattr(self.cfg, "residual_init_scale", True)):
            self._apply_residual_init_scale()

    def _init_weights(self, module):
        # Basic init only. Residual-scaled overrides happen in a separate pass.
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _apply_residual_init_scale(self):
        # GPT-NeoX / LLaMA-style scaled init for residual projections so the
        # residual stream variance does not blow up with depth.
        scale = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.mlp.w3.weight, mean=0.0, std=scale)

    def _chunked_lm_loss(self, hidden, targets):
        bsz, seqlen, dim = hidden.shape
        chunk = max(1, int(getattr(self, "logit_chunk_size", 128)))
        valid_tokens = targets.ne(IGNORE_INDEX).sum().clamp_min(1)
        loss_sum = hidden.new_zeros(())

        def loss_for_chunk(h_flat, y_flat):
            logits = self.lm_head(h_flat)
            return F.cross_entropy(
                logits.float(),
                y_flat,
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )

        use_loss_checkpoint = self.training and self._use_checkpoint and torch.is_grad_enabled()
        if use_loss_checkpoint:
            from torch.utils.checkpoint import checkpoint

        for start in range(0, seqlen, chunk):
            end = min(start + chunk, seqlen)
            h = hidden[:, start:end, :].contiguous().view(-1, dim)
            y = targets[:, start:end].contiguous().view(-1)
            if use_loss_checkpoint:
                loss_sum = loss_sum + checkpoint(loss_for_chunk, h, y, use_reentrant=False)
            else:
                loss_sum = loss_sum + loss_for_chunk(h, y)
        return loss_sum / valid_tokens

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        if self.training and self._use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            for blk in self.blocks:
                x = checkpoint(blk, x, use_reentrant=False)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.norm_f(x)
        if targets is not None:
            return None, self._chunked_lm_loss(x, targets)
        return self.lm_head(x), None


# =============================================================================
# DOCUMENTS / PRETOKENIZATION CACHE
# =============================================================================

def iter_documents(path: str, max_doc_chars: int) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        current: List[str] = []
        current_chars = 0
        for line in f:
            if line.strip() == "":
                if current:
                    text = "".join(current).strip()
                    if text:
                        yield text
                    current = []
                    current_chars = 0
            else:
                current.append(line)
                current_chars += len(line)
                if current_chars >= max_doc_chars:
                    text = "".join(current).strip()
                    if text:
                        yield text
                    current = []
                    current_chars = 0
        if current:
            text = "".join(current).strip()
            if text:
                yield text


def count_documents(path: str) -> int:
    count = 0
    in_doc = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if in_doc:
                    count += 1
                    in_doc = False
            else:
                in_doc = True
        if in_doc:
            count += 1
    return count


def build_corpus_index(corpus_dir: str, hash_files: bool = False) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt files found in corpus_dir={corpus_dir!r}")
    out: List[Dict[str, Any]] = []
    for p in files:
        fp = quick_file_fingerprint(p)
        fp["docs"] = count_documents(p)
        if hash_files:
            fp["sha256"] = file_sha256(p)
        out.append(fp)
    return out


def split_score(seed: int, file_idx: int, doc_idx: int) -> float:
    payload = f"{seed}:{file_idx}:{doc_idx}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") / float(1 << 64)


def is_val_doc(seed: int, file_idx: int, doc_idx: int, val_ratio: float) -> bool:
    return split_score(seed, file_idx, doc_idx) < val_ratio


def tokenize_document(tokenizer, text: str, eot_marker: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    return tok_encode(tokenizer, text + f"\n{eot_marker}\n\n")


class MixedDocumentStreamer:
    def __init__(
        self,
        corpus_index: List[Dict[str, Any]],
        tokenizer,
        seed: int,
        val_ratio: float,
        max_doc_chars: int,
        shuffle_buffer_docs: int,
        shuffle_buffer_tokens: int,
        eot_marker: str,
        want_val: bool,
        max_tokens: Optional[int] = None,
    ):
        self.corpus_index = corpus_index
        self.tokenizer = tokenizer
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.max_doc_chars = int(max_doc_chars)
        self.shuffle_buffer_docs = int(shuffle_buffer_docs)
        self.shuffle_buffer_tokens = int(shuffle_buffer_tokens)
        self.eot_marker = eot_marker
        self.want_val = bool(want_val)
        self.max_tokens = max_tokens
        self.rng = random.Random(self.seed + (9_999_991 if want_val else 0))
        self.streams = []
        for file_idx, entry in enumerate(corpus_index):
            self.streams.append({
                "file_idx": file_idx,
                "name": entry["name"],
                "iter": iter_documents(entry["path"], max_doc_chars=max_doc_chars),
                "doc_idx": 0,
                "weight": max(1, int(entry.get("docs", 1))),
                "active": True,
            })

    def _pick_stream(self):
        active = [s for s in self.streams if s["active"]]
        if not active:
            return None
        total = sum(s["weight"] for s in active)
        r = self.rng.random() * total
        acc = 0
        for s in active:
            acc += s["weight"]
            if r <= acc:
                return s
        return active[-1]

    def _next_doc_from_stream(self, stream):
        while True:
            try:
                doc = next(stream["iter"])
            except StopIteration:
                stream["active"] = False
                return None
            doc_idx = stream["doc_idx"]
            stream["doc_idx"] += 1
            if is_val_doc(self.seed, stream["file_idx"], doc_idx, self.val_ratio) == self.want_val:
                return doc

    def iter_tokenized_docs(self) -> Iterator[List[int]]:
        yielded_tokens = 0
        while True:
            doc_buffer: List[List[int]] = []
            buffer_tokens = 0
            while len(doc_buffer) < self.shuffle_buffer_docs and buffer_tokens < self.shuffle_buffer_tokens:
                stream = self._pick_stream()
                if stream is None:
                    break
                doc = self._next_doc_from_stream(stream)
                if doc is None:
                    continue
                ids = tokenize_document(self.tokenizer, doc, self.eot_marker)
                if ids:
                    doc_buffer.append(ids)
                    buffer_tokens += len(ids)

            if not doc_buffer:
                break

            if len(doc_buffer) > 1:
                self.rng.shuffle(doc_buffer)

            for ids in doc_buffer:
                if self.max_tokens is not None:
                    remaining = int(self.max_tokens) - yielded_tokens
                    if remaining <= 0:
                        return
                    if len(ids) > remaining:
                        ids = ids[:remaining]
                yielded_tokens += len(ids)
                yield ids
                if self.max_tokens is not None and yielded_tokens >= int(self.max_tokens):
                    return


class PretokenizedCache:
    def __init__(self, config: Dict[str, Any], tokenizer, vocab_size: int):
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = int(vocab_size)
        self.cache_dir = deep_get(config, "paths.pretokenized_cache_dir", "data/pretokenized")
        ensure_dir(self.cache_dir)
        self.manifest_path = os.path.join(self.cache_dir, "manifest.json")
        self.train_bin = os.path.join(self.cache_dir, "train.bin")
        self.val_bin = os.path.join(self.cache_dir, "val.bin")
        self.dtype = np.uint16 if self.vocab_size <= 65535 else np.int32

    def expected_manifest(self, corpus_index: List[Dict[str, Any]], tokenizer_sha: str) -> Dict[str, Any]:
        relevant = {
            "tokenizer_sha256": tokenizer_sha,
            "vocab_size": self.vocab_size,
            "dtype": "uint16" if self.dtype == np.uint16 else "int32",
            "corpus_index": corpus_index,
            "split_seed": int(deep_get(self.config, "runtime.seed", 1337)),
            "val_ratio": float(deep_get(self.config, "data.val_ratio", 0.02)),
            "max_doc_chars": int(deep_get(self.config, "data.max_doc_chars", 1_000_000)),
            "shuffle_buffer_docs": int(deep_get(self.config, "data.shuffle_buffer_docs", 256)),
            "shuffle_buffer_tokens": int(deep_get(self.config, "data.shuffle_buffer_tokens", 1_000_000)),
            "eot_marker": str(deep_get(self.config, "data.eot_marker", "<eot>")),
            "max_val_tokens": int(deep_get(self.config, "data.max_val_tokens", 200_000)),
        }
        relevant["signature"] = stable_hash_json(relevant)
        return relevant

    def is_valid(self, expected: Dict[str, Any]) -> bool:
        if not (os.path.exists(self.manifest_path) and os.path.exists(self.train_bin) and os.path.exists(self.val_bin)):
            return False
        try:
            actual = load_json(self.manifest_path)
        except Exception:
            return False
        if actual.get("signature") != expected.get("signature"):
            return False
        train_tokens = int(actual.get("train_tokens", -1))
        val_tokens = int(actual.get("val_tokens", -1))
        itemsize = np.dtype(self.dtype).itemsize
        return (
            os.path.getsize(self.train_bin) == train_tokens * itemsize and
            os.path.getsize(self.val_bin) == val_tokens * itemsize and
            train_tokens > 0 and
            val_tokens > 0
        )

    def build(self, corpus_index: List[Dict[str, Any]], expected: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        if force and os.path.isdir(self.cache_dir):
            for name in ("train.bin", "val.bin", "manifest.json", "train.bin.tmp", "val.bin.tmp"):
                p = os.path.join(self.cache_dir, name)
                if os.path.exists(p):
                    os.remove(p)

        if self.is_valid(expected):
            manifest = load_json(self.manifest_path)
            print(f"Using existing pretokenized cache: {self.cache_dir}")
            print(f"  train tokens: {manifest['train_tokens']:,}")
            print(f"  val tokens:   {manifest['val_tokens']:,}")
            return manifest

        # Do not trust partial previous builds.
        for name in ("train.bin.tmp", "val.bin.tmp", "manifest.json.tmp"):
            p = os.path.join(self.cache_dir, name)
            if os.path.exists(p):
                os.remove(p)

        print(f"Building deterministic pretokenized cache: {self.cache_dir}")
        train_tmp = self.train_bin + ".tmp"
        val_tmp = self.val_bin + ".tmp"

        train_tokens = self._write_split(train_tmp, corpus_index, want_val=False, max_tokens=None)
        val_tokens = self._write_split(
            val_tmp,
            corpus_index,
            want_val=True,
            max_tokens=int(deep_get(self.config, "data.max_val_tokens", 200_000)),
        )

        if train_tokens <= int(deep_get(self.config, "model.block_size", 1024)) + 1:
            raise RuntimeError("Pretokenized train cache is too small.")
        if val_tokens <= int(deep_get(self.config, "model.block_size", 1024)) + 1:
            raise RuntimeError("Pretokenized validation cache is too small.")

        os.replace(train_tmp, self.train_bin)
        os.replace(val_tmp, self.val_bin)

        manifest = dict(expected)
        manifest.update({
            "train_tokens": int(train_tokens),
            "val_tokens": int(val_tokens),
            "created_time": time.time(),
        })
        save_json_atomic(manifest, self.manifest_path)
        print(f"Finished cache: train={train_tokens:,} tokens, val={val_tokens:,} tokens")
        return manifest

    def _write_split(self, out_path: str, corpus_index: List[Dict[str, Any]], want_val: bool, max_tokens: Optional[int]) -> int:
        seed = int(deep_get(self.config, "runtime.seed", 1337))
        streamer = MixedDocumentStreamer(
            corpus_index=corpus_index,
            tokenizer=self.tokenizer,
            seed=seed,
            val_ratio=float(deep_get(self.config, "data.val_ratio", 0.02)),
            max_doc_chars=int(deep_get(self.config, "data.max_doc_chars", 1_000_000)),
            shuffle_buffer_docs=int(deep_get(self.config, "data.shuffle_buffer_docs", 256)),
            shuffle_buffer_tokens=int(deep_get(self.config, "data.shuffle_buffer_tokens", 1_000_000)),
            eot_marker=str(deep_get(self.config, "data.eot_marker", "<eot>")),
            want_val=want_val,
            max_tokens=max_tokens,
        )
        total = 0
        label = "val" if want_val else "train"
        with open(out_path, "wb") as f:
            for i, ids in enumerate(streamer.iter_tokenized_docs(), start=1):
                arr = np.asarray(ids, dtype=self.dtype)
                arr.tofile(f)
                total += int(arr.size)
                if i % 10000 == 0:
                    print(f"  {label}: docs={i:,} tokens={total:,}")
        return total

    def open_memmaps(self) -> Tuple[np.memmap, np.memmap, Dict[str, Any]]:
        manifest = load_json(self.manifest_path)
        dtype = np.uint16 if manifest.get("dtype") == "uint16" else np.int32
        train = np.memmap(self.train_bin, mode="r", dtype=dtype)
        val = np.memmap(self.val_bin, mode="r", dtype=dtype)
        if len(train) != int(manifest["train_tokens"]):
            raise RuntimeError("train.bin size does not match manifest.")
        if len(val) != int(manifest["val_tokens"]):
            raise RuntimeError("val.bin size does not match manifest.")
        return train, val, manifest


class BinLMDataset:
    def __init__(self, ids: np.memmap, block_size: int, stride: Optional[int] = None):
        self.ids = ids
        self.block_size = int(block_size)
        self.stride = int(stride or block_size)
        if len(self.ids) < self.block_size + 1:
            raise RuntimeError(f"Need at least {self.block_size + 1} tokens, got {len(self.ids)}.")
        self.num_samples = ((len(self.ids) - self.block_size - 1) // self.stride) + 1
        # Identity order by default; training callers can install a permutation
        # per epoch via set_epoch_order so that multi-epoch runs do not see
        # windows in the exact same sequence every pass.
        self._order: Optional[np.ndarray] = None

    def set_epoch_order(self, seed: int, epoch_idx: int, shuffle: bool = True) -> None:
        """Install a window-index permutation for this epoch.

        Deterministic given (seed, epoch_idx). Set shuffle=False to revert to
        sequential order (used for val, where order does not matter).
        """
        if not shuffle:
            self._order = None
            return
        rng = np.random.default_rng(int(seed) ^ (int(epoch_idx) * 1_000_003))
        self._order = rng.permutation(self.num_samples).astype(np.int64)

    def _sample_index(self, cursor: int) -> int:
        if self._order is None:
            return cursor
        return int(self._order[cursor])

    def get_batch(self, cursor: int, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        if batch_size > self.num_samples:
            raise RuntimeError(
                f"batch_size={batch_size} exceeds dataset size num_samples={self.num_samples}. "
                "Reduce batch_size, increase the corpus, or lower max_val_tokens."
            )
        if cursor + batch_size > self.num_samples:
            xb, yb, next_cursor, _ = self.get_batch(0, batch_size, device)
            return xb, yb, next_cursor, True

        x = np.empty((batch_size, self.block_size), dtype=np.int64)
        y = np.empty((batch_size, self.block_size), dtype=np.int64)
        for row, pos in enumerate(range(cursor, cursor + batch_size)):
            sample_idx = self._sample_index(pos)
            start = sample_idx * self.stride
            x[row, :] = self.ids[start : start + self.block_size]
            y[row, :] = self.ids[start + 1 : start + 1 + self.block_size]
        xb = torch.from_numpy(x).to(device, non_blocking=(device.type == "cuda"))
        yb = torch.from_numpy(y).to(device, non_blocking=(device.type == "cuda"))
        return xb, yb, cursor + batch_size, False


# =============================================================================
# TRAINING
# =============================================================================

class CosineSchedule:
    def __init__(self, optimizer, lr_max: float, lr_min: float, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.lr_max = float(lr_max)
        self.lr_min = float(lr_min)
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))

    def step(self, current_step: int) -> float:
        # current_step is the number of optimizer updates already completed.
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
def evaluate(model: GPT, dataset: BinLMDataset, batch_size: int, max_batches: int, device: torch.device, use_amp: bool) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    cursor = 0
    batches = min(max_batches, max(1, dataset.num_samples // max(1, batch_size)))
    for _ in range(batches):
        xb, yb, cursor, wrapped = dataset.get_batch(cursor, batch_size, device)
        with autocast_ctx(device, use_amp and device.type == "cuda"):
            _, loss = model(xb, yb)
        if torch.isfinite(loss):
            total_loss += float(loss.item())
            total_batches += 1
        if wrapped:
            break
    model.train()
    return total_loss / max(1, total_batches)


class LossLogger:
    def __init__(self, path: str):
        self.path = path
        ensure_dir(os.path.dirname(path) or ".")

    def log(self, **kwargs):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kwargs, sort_keys=True) + "\n")


def optimizer_from_config(model: GPT, config: Dict[str, Any]):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(deep_get(config, "optimizer.lr", 3e-4)),
        betas=(
            float(deep_get(config, "optimizer.beta1", 0.9)),
            float(deep_get(config, "optimizer.beta2", 0.95)),
        ),
        eps=float(deep_get(config, "optimizer.eps", 1e-8)),
        weight_decay=float(deep_get(config, "optimizer.weight_decay", 0.01)),
        foreach=bool(deep_get(config, "optimizer.foreach", False)),
    )


def save_checkpoint(
    path: str,
    model: GPT,
    optimizer,
    scaler,
    config: Dict[str, Any],
    train_state: Dict[str, Any],
    tokenizer_sha256: str,
    cache_signature: str,
) -> None:
    payload = {
        "state": model.state_dict(),
        "config": asdict(model.cfg),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "train_state": train_state,
        "tokenizer_sha256": tokenizer_sha256,
        "cache_signature": cache_signature,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "saved_time": time.time(),
    }
    atomic_torch_save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def restore_rng_state(payload: Dict[str, Any]) -> None:
    rng = payload.get("rng_state") or {}
    try:
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception as e:
        print(f"WARNING: could not restore RNG state exactly: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/gpt_config.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--build-cache", action="store_true", help="Deprecated/no-op: the cache is always built on demand. Kept for script compatibility.")
    parser.add_argument("--cache-only", action="store_true", help="Build/validate cache and exit.")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    device = get_device(config)
    configure_torch(device, config)
    set_seed(int(deep_get(config, "runtime.seed", 1337)), device)

    paths = config.get("paths", {})
    data_dir = paths.get("data_dir", "data")
    ensure_dir(data_dir)
    tokenizer_path = paths.get("tokenizer_path", os.path.join(data_dir, "tokenizer.json"))
    model_path = paths.get("pretrain_checkpoint", os.path.join(data_dir, "model.pt"))
    best_model_path = paths.get("pretrain_best_checkpoint", os.path.join(data_dir, "model.best.pt"))
    loss_log_path = paths.get("pretrain_log", os.path.join(data_dir, "pretrain_loss.jsonl"))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tokenizer_bpe import BPETokenizer

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = BPETokenizer.load(tokenizer_path)
    tokenizer_sha = file_sha256(tokenizer_path)
    vocab_size = tokenizer_vocab_size(tokenizer)
    model_cfg_dict = dict(config.get("model", {}))
    model_cfg_dict["vocab_size"] = vocab_size
    cfg = gpt_config_from_dict(model_cfg_dict)

    print(f"Device: {device}")
    print(f"Tokenizer vocab: {vocab_size}")
    print(f"Tokenizer sha256: {tokenizer_sha[:16]}...")

    corpus_dir = paths.get("corpus_dir", "corpus")
    corpus_index = build_corpus_index(corpus_dir, hash_files=bool(deep_get(config, "data.hash_corpus_files", False)))
    total_bytes = sum(int(e["bytes"]) for e in corpus_index)
    total_docs = sum(int(e["docs"]) for e in corpus_index)
    print("\nCorpus index:")
    for e in corpus_index:
        print(f"  {e['name']:<40} {e['bytes']/1024/1024:>8.1f} MB {e['docs']:>10,} docs")
    print(f"  {'TOTAL':<40} {total_bytes/1024/1024:>8.1f} MB {total_docs:>10,} docs")

    cache = PretokenizedCache(config, tokenizer, vocab_size)
    expected_manifest = cache.expected_manifest(corpus_index, tokenizer_sha)
    # The pretokenized cache is mandatory: exact resume requires a stable
    # token sequence on disk. We always build/validate it.
    manifest = cache.build(corpus_index, expected_manifest, force=args.force_rebuild_cache or bool(deep_get(config, "data.force_rebuild_cache", False)))

    if args.cache_only:
        return

    train_ids, val_ids, manifest = cache.open_memmaps()
    cache_sig = manifest["signature"]

    model = GPT(cfg).to(device)
    model.logit_chunk_size = int(deep_get(config, "train.logit_chunk_size", 128))
    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")
    print(f"Config: {asdict(cfg)}")

    stride = int(deep_get(config, "data.stride", cfg.block_size))
    train_ds = BinLMDataset(train_ids, cfg.block_size, stride=stride)
    val_ds = BinLMDataset(val_ids, cfg.block_size, stride=stride)
    print(f"Train tokens: {len(train_ids):,} -> {train_ds.num_samples:,} windows")
    print(f"Val tokens:   {len(val_ids):,} -> {val_ds.num_samples:,} windows")

    optimizer = optimizer_from_config(model, config)
    use_amp = bool(deep_get(config, "runtime.use_amp", True)) and device.type == "cuda"
    scaler = make_grad_scaler(device, use_amp)

    batch_size = int(deep_get(config, "train.batch_size", 1))
    grad_accum = int(deep_get(config, "train.grad_accum", 8))
    epochs = int(deep_get(config, "train.epochs", 1))
    max_steps_cfg = deep_get(config, "train.max_steps", None)
    if max_steps_cfg is not None:
        total_steps = int(max_steps_cfg)
    else:
        total_steps = max(1, (train_ds.num_samples * epochs) // max(1, batch_size * grad_accum))

    warmup_steps = int(total_steps * float(deep_get(config, "scheduler.warmup_pct", 0.02)))
    schedule = CosineSchedule(
        optimizer,
        lr_max=float(deep_get(config, "optimizer.lr", 3e-4)),
        lr_min=float(deep_get(config, "optimizer.lr_min", 1e-5)),
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    global_step = 0
    epoch_idx = 0
    sample_cursor = 0
    best_val = float("inf")

    if args.resume:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"--resume requested but checkpoint does not exist: {model_path}")
        ckpt = load_checkpoint(model_path, device)
        if ckpt.get("tokenizer_sha256") and ckpt["tokenizer_sha256"] != tokenizer_sha:
            raise RuntimeError("Tokenizer hash mismatch. Refusing to resume.")
        if ckpt.get("cache_signature") and ckpt["cache_signature"] != cache_sig:
            raise RuntimeError("Pretokenized cache signature mismatch. Refusing to resume.")
        old_cfg = gpt_config_from_dict({**asdict(cfg), **ckpt.get("config", {})})
        if asdict(old_cfg) != asdict(cfg):
            print("WARNING: checkpoint model config differs from current JSON. Loading checkpoint config.")
            cfg = old_cfg
            model = GPT(cfg).to(device)
            model.logit_chunk_size = int(deep_get(config, "train.logit_chunk_size", 128))
            optimizer = optimizer_from_config(model, config)
            schedule = CosineSchedule(
                optimizer,
                lr_max=float(deep_get(config, "optimizer.lr", 3e-4)),
                lr_min=float(deep_get(config, "optimizer.lr_min", 1e-5)),
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            )
        model.load_state_dict(ckpt["state"], strict=True)
        if ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as e:
                print(f"WARNING: could not restore AMP scaler: {e}")
        restore_rng_state(ckpt)
        ts = ckpt.get("train_state", {})
        global_step = int(ts.get("global_step", 0))
        epoch_idx = int(ts.get("epoch_idx", ts.get("epoch", 0)))
        sample_cursor = int(ts.get("sample_cursor", 0))
        best_val = float(ts.get("best_val", float("inf")))

        # Compatibility fallback for older checkpoints without exact cursor.
        if "sample_cursor" not in ts:
            approx_seen = global_step * batch_size * grad_accum
            epoch_idx = approx_seen // train_ds.num_samples
            sample_cursor = approx_seen % train_ds.num_samples
            print("WARNING: old checkpoint had no sample_cursor; using approximate cursor.")
        print(f"Resumed from {model_path}")
        print(f"  step={global_step:,} epoch={epoch_idx + 1} sample_cursor={sample_cursor:,} best_val={best_val:.6f}")

    logger = LossLogger(loss_log_path)
    log_every = int(deep_get(config, "logging.log_every", 50))
    eval_every = int(deep_get(config, "logging.eval_every", 500))
    save_every = int(deep_get(config, "logging.save_every", 1000))
    val_batches = int(deep_get(config, "logging.val_batches", 200))
    max_grad_norm = float(deep_get(config, "optimizer.max_grad_norm", 1.0))

    session_start_step = global_step
    session_start_time = time.time()
    running_loss = 0.0
    running_count = 0
    interrupted = False

    def current_train_state() -> Dict[str, Any]:
        return {
            "global_step": int(global_step),
            "epoch_idx": int(epoch_idx),
            "sample_cursor": int(sample_cursor),
            "best_val": float(best_val),
            "total_steps": int(total_steps),
            "batch_size": int(batch_size),
            "grad_accum": int(grad_accum),
            "stride": int(stride),
            "train_num_samples": int(train_ds.num_samples),
        }

    def handle_interrupt(signum, frame):
        nonlocal interrupted
        print("\nInterrupt requested. Will save after current optimizer step.")
        interrupted = True

    try:
        signal.signal(signal.SIGINT, handle_interrupt)
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("PRETRAINING START")
    print("=" * 60)
    print(f"total_steps={total_steps:,} warmup_steps={warmup_steps:,} effective_batch={batch_size * grad_accum}")
    print(f"start step={global_step:,} epoch={epoch_idx + 1} sample_cursor={sample_cursor:,}\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Install window order for the current (possibly resumed) epoch. Val
    # dataset stays in sequential order — order does not affect eval.
    seed = int(deep_get(config, "runtime.seed", 1337))
    train_ds.set_epoch_order(seed, epoch_idx, shuffle=True)

    while global_step < total_steps and epoch_idx < epochs:
        for micro in range(grad_accum):
            # Fix #4: check interrupt inside micro loop so Ctrl-C exits fast
            if interrupted:
                break
            xb, yb, next_cursor, wrapped = train_ds.get_batch(sample_cursor, batch_size, device)
            if wrapped:
                epoch_idx += 1
                sample_cursor = 0
                if epoch_idx >= epochs:
                    break
                # New epoch: reshuffle window order so we don't re-walk the
                # same sequence. Deterministic given (seed, epoch_idx).
                train_ds.set_epoch_order(seed, epoch_idx, shuffle=True)
                xb, yb, next_cursor, wrapped = train_ds.get_batch(sample_cursor, batch_size, device)
            sample_cursor = next_cursor

            with autocast_ctx(device, use_amp):
                _, loss = model(xb, yb)

            if loss is None or not torch.isfinite(loss):
                print(f"WARNING: NaN/Inf loss at step {global_step}, skipping optimizer update")
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
            steps_per_sec = session_steps / max(1.0, elapsed)
            remaining = max(0, total_steps - global_step)
            eta_hrs = remaining / max(0.001, steps_per_sec) / 3600.0
            avg = running_loss / max(1, running_count)
            print(
                f"  step {global_step:>8} | loss {avg:.4f} | lr {lr:.2e} | "
                f"ep {epoch_idx + 1} | cursor {sample_cursor:,}/{train_ds.num_samples:,} | "
                f"{steps_per_sec:.3f} steps/s | ETA {eta_hrs:.1f}h"
            )

        if global_step % eval_every == 0:
            val_loss = evaluate(model, val_ds, batch_size, val_batches, device, use_amp)
            train_avg = running_loss / max(1, running_count)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            print(f"\n  ── EVAL step {global_step} ──")
            print(
                f"     train={train_avg:.4f}  val={val_loss:.4f}  "
                f"ppl={math.exp(min(val_loss, 20)):.1f}  lr={lr:.2e}"
                f"{' ★ NEW BEST' if improved else ''}\n"
            )
            logger.log(
                kind="eval",
                step=global_step,
                train_loss=round(train_avg, 6),
                val_loss=round(val_loss, 6),
                ppl=round(math.exp(min(val_loss, 20)), 6),
                lr=lr,
                epoch_idx=epoch_idx,
                sample_cursor=sample_cursor,
                time=time.time(),
            )
            if improved:
                save_checkpoint(best_model_path, model, optimizer, scaler, config, current_train_state(), tokenizer_sha, cache_sig)
            running_loss = 0.0
            running_count = 0
            model.train()

        if global_step % save_every == 0 or interrupted:
            save_checkpoint(model_path, model, optimizer, scaler, config, current_train_state(), tokenizer_sha, cache_sig)
            print(f"  saved checkpoint: {model_path} @ step {global_step:,}")
            if interrupted:
                break

    save_checkpoint(model_path, model, optimizer, scaler, config, current_train_state(), tokenizer_sha, cache_sig)
    val_loss = evaluate(model, val_ds, batch_size, val_batches, device, use_amp)
    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(best_model_path, model, optimizer, scaler, config, current_train_state(), tokenizer_sha, cache_sig)

    elapsed = time.time() - session_start_time
    print("\n" + "=" * 60)
    print("PRETRAINING COMPLETE" if not interrupted else "PRETRAINING PAUSED")
    print("=" * 60)
    print(f"Steps:       {global_step:,}")
    print(f"Epoch:       {epoch_idx + 1}")
    print(f"Cursor:      {sample_cursor:,}/{train_ds.num_samples:,}")
    print(f"Best val:    {best_val:.4f} (ppl {math.exp(min(best_val, 20)):.1f})")
    print(f"Session:     {elapsed/3600:.2f} hours")
    print(f"Saved:       {model_path}")
    print(f"Best:        {best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
