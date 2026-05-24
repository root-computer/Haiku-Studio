#!/usr/bin/env python
"""Train the h2 BPETokenizer from a file or directory of .txt files."""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizer_bpe import BPETokenizer


def iter_text(path: Path, chunk_chars: int = 1_000_000) -> Iterator[str]:
    if path.is_dir():
        files = sorted(path.rglob("*.txt"))
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"No .txt files found at {path}")
    for file_path in files:
        print(f"[tokenizer] reading {file_path}", flush=True)
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(chunk_chars)
                if not chunk:
                    break
                yield chunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Corpus .txt file or directory containing .txt files.")
    ap.add_argument("--output", default="data/tokenizer.json")
    ap.add_argument("--vocab-size", type=int, default=50000)
    ap.add_argument("--min-freq", type=int, default=2)
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("H2 TOKENIZER TRAINING", flush=True)
    print("=" * 60, flush=True)
    print(f"input:      {input_path}", flush=True)
    print(f"output:     {output_path}", flush=True)
    print(f"vocab_size: {args.vocab_size:,}", flush=True)
    print(f"min_freq:   {args.min_freq}", flush=True)

    tokenizer = BPETokenizer.train_from_iterator(
        iter_text(input_path),
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        special_tokens=("<unk>", "<bos>", "<eos>", "<eot>"),
    )
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tokenizer.save(str(tmp))
    os.replace(tmp, output_path)
    print(f"[tokenizer] saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
