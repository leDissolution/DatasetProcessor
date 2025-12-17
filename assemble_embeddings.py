"""
Assemble embedding prompts from plain text files into a single JSONL.

- Scans a source directory for *.train.txt files (recursively).
- Wraps each non-empty line as {"prompt": "..."}.
- Writes a combined embeddings.train.jsonl to the target directory.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Dict, Tuple

from assembler import ensure_directories, write_jsonl_file


def _find_train_txt_files(base_path: str) -> List[str]:
    """Recursively list all *.train.txt files under base_path, sorted for determinism."""
    files: List[str] = []
    for root, _, filenames in os.walk(base_path):
        for name in filenames:
            if name.lower().endswith(".train.txt"):
                files.append(os.path.join(root, name))
    files.sort()
    return files


def _collect_prompts(file_paths: List[str]) -> Tuple[List[Dict[str, str]], int]:
    """Load prompt lines from files; skip empty lines and keep ordering within each file."""
    entries: List[Dict[str, str]] = []
    skipped = 0
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                prompt = raw.rstrip("\r\n")
                if not prompt:
                    skipped += 1
                    continue
                entries.append({"prompt": prompt})
    return entries, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble embedding prompts into JSONL")
    parser.add_argument(
        "--base_path",
        required=True,
        help="Folder containing *.train.txt files (recursively searched)",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Where to write the combined file (default: base_path)",
    )
    parser.add_argument(
        "--output_file",
        default="embeddings.train.jsonl",
        help="Output filename (default: embeddings.train.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_path = os.path.normpath(args.base_path)
    if not os.path.isdir(base_path):
        raise SystemExit(f"Source directory not found or not a directory: {base_path}")

    output_dir = os.path.normpath(args.output_dir or base_path)
    ensure_directories(output_dir)
    output_path = os.path.join(output_dir, args.output_file)

    txt_files = _find_train_txt_files(base_path)
    if not txt_files:
        raise SystemExit(f"No *.train.txt files found in {base_path}")

    entries, skipped = _collect_prompts(txt_files)
    if not entries:
        raise SystemExit("No prompts detected (all lines were empty)")

    write_jsonl_file(output_path, entries, ensure_ascii=False)
    print(f"Wrote {len(entries)} prompt(s) to {output_path}")
    if skipped:
        print(f"Skipped {skipped} empty line(s)")


if __name__ == "__main__":
    main()
