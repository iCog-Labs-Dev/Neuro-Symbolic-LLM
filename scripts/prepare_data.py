#!/usr/bin/env python
"""Prepare the local ``data/`` folder used by the real-text demo.

Creates:

    data/
    ├── shakespeare.txt      full tinyshakespeare corpus (~1.1 MB, 40k lines)
    ├── wiki.txt             real Wikipedia text (wikitext-2 train sample)
    ├── tokenizer_gpt2/      GPT-2 tokenizer saved with save_pretrained
    └── tokenizer_pythia/    Pythia/GPT-NeoX tokenizer saved with save_pretrained

Everything is regenerable; run this once while online, then the demo works
fully offline:

    python scripts/prepare_data.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

WIKI_DATASET = "Salesforce/wikitext"
WIKI_CONFIG = "wikitext-2-raw-v1"
WIKI_CHARS = 400_000  # how much Wikipedia text to keep


def download_shakespeare() -> None:
    out = DATA_DIR / "shakespeare.txt"
    if out.is_file():
        print(f"[skip] {out.name} already exists ({out.stat().st_size} bytes)")
        return
    print("[get ] downloading tinyshakespeare ...")
    with urllib.request.urlopen(SHAKESPEARE_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    out.write_text(text, encoding="utf-8")
    print(f"[ok  ] {out} ({len(text)} chars, {text.count(chr(10))} lines)")


def extract_wikipedia() -> None:
    out = DATA_DIR / "wiki.txt"
    if out.is_file():
        print(f"[skip] {out.name} already exists ({out.stat().st_size} bytes)")
        return
    from datasets import load_dataset

    print(f"[get ] loading {WIKI_DATASET}/{WIKI_CONFIG} train split ...")
    ds = load_dataset(WIKI_DATASET, WIKI_CONFIG, split="train")
    chunks: list[str] = []
    total = 0
    for row in ds:
        line = row["text"]
        if not line or not line.strip():
            continue
        chunks.append(line.strip())
        total += len(line) + 1
        if total >= WIKI_CHARS:
            break
    text = "\n".join(chunks)
    out.write_text(text, encoding="utf-8")
    print(f"[ok  ] {out} ({len(text)} chars from {len(chunks)} paragraphs)")


def save_tokenizer(model_id: str, folder: str) -> None:
    from transformers import AutoTokenizer

    out = DATA_DIR / folder
    if out.is_dir() and any(out.iterdir()):
        print(f"[skip] {folder}/ already exists")
        return
    print(f"[get ] tokenizer for {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.save_pretrained(out)
    n_files = len(list(out.iterdir()))
    print(f"[ok  ] {out} ({n_files} files, vocab={tok.vocab_size})")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data folder : {DATA_DIR}\n")
    download_shakespeare()
    extract_wikipedia()
    save_tokenizer("gpt2", "tokenizer_gpt2")
    save_tokenizer("EleutherAI/pythia-70m", "tokenizer_pythia")
    print("\nAll data ready. Try:")
    print("  python scripts/run_real_text_demo.py")
    print("  python scripts/run_real_text_demo.py --model EleutherAI/pythia-70m")
    print("  python scripts/run_real_text_demo.py --text data/wiki.txt --steer 2.0")


if __name__ == "__main__":
    main()
