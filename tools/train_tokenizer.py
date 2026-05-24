#!/usr/bin/env python
"""Train the h2 BPETokenizer from one text file or a corpus directory.

This script is designed for Haiku Studio's built-in tokenizer tool:
- input must be explicit; no silent fallback corpus is used here
- large corpora are sampled under a RAM guard by default
- tokenizer.json is written atomically
- a project-local copy can be written after the main tokenizer is saved
"""

import argparse
import ctypes
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizer_bpe import BPETokenizer


MiB = 1024 * 1024


def total_ram_gb() -> float:
    """Best-effort total RAM detection without third-party dependencies."""
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(stat.ullTotalPhys) / (1024 ** 3)

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024 ** 3)
    except Exception:
        return 0.0


def default_ram_guard_mb(ram_gb: float) -> int:
    """Conservative tokenizer input cap. BPE stats can expand beyond raw text size."""
    if ram_gb and ram_gb <= 4:
        return 128
    if ram_gb and ram_gb <= 8:
        return 256
    if ram_gb and ram_gb <= 16:
        return 512
    if ram_gb and ram_gb <= 32:
        return 1024
    return 1536


def resolve_input(raw: str, source_kind: str) -> Path:
    if not raw or not str(raw).strip():
        raise ValueError("No tokenizer source selected. Choose a .txt file or corpus folder first.")

    path = Path(raw.strip().strip('"'))
    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Tokenizer source does not exist: {path}")

    if source_kind == "file" and not path.is_file():
        raise ValueError(f"Tokenizer source mode is Single Text File, but this is not a file: {path}")

    if source_kind == "corpus" and not path.is_dir():
        raise ValueError(f"Tokenizer source mode is Corpus Folder, but this is not a folder: {path}")

    if source_kind not in {"file", "corpus", "auto"}:
        raise ValueError(f"Unknown source kind: {source_kind}")

    return path


def discover_text_files(path: Path, source_kind: str) -> List[Path]:
    if source_kind == "file" or path.is_file():
        return [path]

    files = sorted(p for p in path.rglob("*.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .txt files found in corpus folder: {path}")
    return files


def plan_file_budgets(files: List[Path], max_bytes: int) -> List[Tuple[Path, int]]:
    sizes = [(p, max(0, p.stat().st_size)) for p in files]
    total = sum(size for _, size in sizes)

    if max_bytes <= 0 or total <= max_bytes:
        return sizes

    # Sample across the full corpus rather than consuming only the first huge file.
    # Every non-empty file gets at least a small head sample when possible.
    nonempty = [(p, s) for p, s in sizes if s > 0]
    if not nonempty:
        return sizes

    min_slice = min(256 * 1024, max(4096, max_bytes // max(1, len(nonempty))))
    planned: List[Tuple[Path, int]] = []
    remaining = max_bytes

    # First pass: assign a small representative slice to every file.
    for p, s in nonempty:
        if remaining <= 0:
            break
        take = min(s, min_slice, remaining)
        planned.append((p, take))
        remaining -= take

    if remaining <= 0:
        return planned

    # Second pass: distribute remaining budget proportionally by file size.
    total_remaining_size = sum(max(0, s - min_slice) for _, s in nonempty)
    if total_remaining_size <= 0:
        return planned

    planned_map = {p: b for p, b in planned}
    for p, s in nonempty:
        already = planned_map.get(p, 0)
        extra_capacity = max(0, s - already)
        if extra_capacity <= 0:
            continue
        extra = min(extra_capacity, max(0, math.floor(remaining * (extra_capacity / total_remaining_size))))
        planned_map[p] = already + extra

    return [(p, min(s, planned_map.get(p, 0))) for p, s in nonempty if planned_map.get(p, 0) > 0]


def iter_text(plan: List[Tuple[Path, int]], chunk_bytes: int) -> Iterator[str]:
    for file_index, (file_path, byte_budget) in enumerate(plan, start=1):
        print(
            f"[tokenizer] reading {file_index}/{len(plan)}: {file_path} "
            f"(budget={byte_budget / MiB:.2f} MiB)",
            flush=True,
        )
        consumed = 0
        with file_path.open("rb") as f:
            while consumed < byte_budget:
                need = min(chunk_bytes, byte_budget - consumed)
                blob = f.read(need)
                if not blob:
                    break
                consumed += len(blob)
                text = blob.decode("utf-8", errors="ignore")
                if text:
                    yield text
        print(f"[tokenizer] finished {file_path.name}; consumed={consumed / MiB:.2f} MiB", flush=True)


def save_tokenizer(tokenizer: BPETokenizer, output_path: Path, project_copy: Path | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tokenizer.save(str(tmp))
    os.replace(tmp, output_path)
    print(f"[tokenizer] saved primary tokenizer: {output_path}", flush=True)

    if project_copy:
        project_copy.parent.mkdir(parents=True, exist_ok=True)
        tmp_copy = project_copy.with_suffix(project_copy.suffix + ".tmp")
        shutil.copy2(output_path, tmp_copy)
        os.replace(tmp_copy, project_copy)
        print(f"[tokenizer] saved project tokenizer copy: {project_copy}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Explicit .txt file or directory containing .txt files.")
    ap.add_argument("--source-kind", choices=["file", "corpus", "auto"], default="auto")
    ap.add_argument("--output", default="data/tokenizer.json")
    ap.add_argument("--project-copy", default="", help="Optional second tokenizer.json path for the active project.")
    ap.add_argument("--vocab-size", type=int, default=50000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-input-mb", type=int, default=0, help="0 = automatic RAM-safe guard.")
    ap.add_argument("--chunk-mb", type=int, default=8)
    ap.add_argument("--no-ram-guard", action="store_true", help="Use the full selected input, regardless of size.")
    args = ap.parse_args()

    if args.vocab_size < 256:
        raise ValueError("vocab-size must be at least 256")
    if args.min_freq < 1:
        raise ValueError("min-freq must be at least 1")

    input_path = resolve_input(args.input, args.source_kind)
    source_kind = "corpus" if input_path.is_dir() else "file"
    files = discover_text_files(input_path, source_kind)
    total_bytes = sum(p.stat().st_size for p in files)

    ram_gb = total_ram_gb()
    auto_guard_mb = default_ram_guard_mb(ram_gb)
    max_input_mb = args.max_input_mb if args.max_input_mb > 0 else auto_guard_mb
    max_bytes = 0 if args.no_ram_guard else max_input_mb * MiB
    chunk_bytes = max(1, min(max(args.chunk_mb, 1), 64)) * MiB

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    project_copy = None
    if args.project_copy and str(args.project_copy).strip():
        project_copy = Path(str(args.project_copy).strip().strip('"'))
        if not project_copy.is_absolute():
            project_copy = ROOT / project_copy

    plan = plan_file_budgets(files, max_bytes)
    planned_bytes = sum(b for _, b in plan)

    if planned_bytes <= 0:
        raise ValueError("Tokenizer source contains no readable text bytes.")

    print("=" * 72, flush=True)
    print("H2 TOKENIZER TRAINING", flush=True)
    print("=" * 72, flush=True)
    print(f"source_kind:     {source_kind}", flush=True)
    print(f"input:           {input_path}", flush=True)
    print(f"files:           {len(files):,}", flush=True)
    print(f"raw_input_size:  {total_bytes / MiB:.2f} MiB", flush=True)
    print(f"ram_detected:    {ram_gb:.2f} GiB" if ram_gb else "ram_detected:    unknown", flush=True)
    if args.no_ram_guard:
        print("ram_guard:       disabled by user", flush=True)
    else:
        print(f"ram_guard:       enabled; tokenizer input capped at {max_input_mb:,} MiB", flush=True)
    print(f"planned_sample:  {planned_bytes / MiB:.2f} MiB", flush=True)
    print(f"chunk_size:      {chunk_bytes / MiB:.2f} MiB", flush=True)
    print(f"output:          {output_path}", flush=True)
    print(f"project_copy:    {project_copy if project_copy else '(none)'}", flush=True)
    print(f"vocab_size:      {args.vocab_size:,}", flush=True)
    print(f"min_freq:        {args.min_freq}", flush=True)
    print("=" * 72, flush=True)

    tokenizer = BPETokenizer.train_from_iterator(
        iter_text(plan, chunk_bytes=chunk_bytes),
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        special_tokens=("<unk>", "<bos>", "<eos>", "<eot>"),
    )
    save_tokenizer(tokenizer, output_path, project_copy)
    print("[tokenizer] training complete", flush=True)


if __name__ == "__main__":
    main()
