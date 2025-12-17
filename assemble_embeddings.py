"""
Assemble embedding prompts from stat lines into train/eval JSONL files.

- Recursively scans for both *.train.txt and *.eval.txt under a source directory.
- Keeps only lines that start with "<stats" (after stripping whitespace).
- Wraps every attribute value (quoted segment) as **/text/**.
- Writes separate embeddings.train.jsonl and embeddings.eval.jsonl files.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Dict, Tuple

from assembler import ensure_directories, write_jsonl_file


def _find_txt_files(base_path: str, suffix: str) -> List[str]:
    """Recursively list all files with the given suffix under base_path, sorted for determinism."""
    files: List[str] = []
    for root, _, filenames in os.walk(base_path):
        for name in filenames:
            if name.lower().endswith(suffix.lower()):
                files.append(os.path.join(root, name))
    files.sort()
    return files


_ATTRIBUTE_VALUE_RE = re.compile(r'"([^"]*)"')


def _wrap_attribute_values(line: str) -> str:
    """
    Mask entire line except attribute values (skipping the first).
    
    - Wraps the whole line with **/ ... /**  (mask start/end).
    - Punches holes for attribute values (except first) using /** ... **/ inside quotes.
    """

    seen_first = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal seen_first
        value = match.group(1)
        if not seen_first:
            seen_first = True
            return match.group(0)
        if value.startswith("/**") and value.endswith("**/"):
            return match.group(0)
        return f'"/**{value}**/"'

    replaced = _ATTRIBUTE_VALUE_RE.sub(_replace, line)
    return f"**/{replaced}/**"


def _collect_stat_prompts(file_paths: List[str]) -> Tuple[List[Dict[str, str]], int]:
    """Collect {"prompt": line} entries for lines starting with <stats."""
    entries: List[Dict[str, str]] = []
    skipped_non_stats = 0
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                stripped = raw.strip()
                if not stripped:
                    continue
                if not stripped.startswith("<stats"):
                    skipped_non_stats += 1
                    continue
                wrapped = _wrap_attribute_values(stripped)
                entries.append({"prompt": wrapped})
    return entries, skipped_non_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble embedding prompts into JSONL files")
    parser.add_argument(
        "--base_path",
        "--source_dir",
        dest="base_path",
        required=True,
        help="Folder containing *.train.txt and *.eval.txt files (recursively searched)",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Where to write the combined files (default: base_path)",
    )
    parser.add_argument(
        "--train_output_file",
        default="embeddings.train.jsonl",
        help="Train output filename (default: embeddings.train.jsonl)",
    )
    parser.add_argument(
        "--eval_output_file",
        default="embeddings.eval.jsonl",
        help="Eval output filename (default: embeddings.eval.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_path = os.path.normpath(args.base_path)
    if not os.path.isdir(base_path):
        raise SystemExit(f"Source directory not found or not a directory: {base_path}")

    output_dir = os.path.normpath(args.output_dir or base_path)
    ensure_directories(output_dir)
    train_output_path = os.path.join(output_dir, args.train_output_file)
    eval_output_path = os.path.join(output_dir, args.eval_output_file)

    train_files = _find_txt_files(base_path, ".train.txt")
    eval_files = _find_txt_files(base_path, ".eval.txt")

    if not train_files:
        raise SystemExit(f"No *.train.txt files found in {base_path}")
    if not eval_files:
        raise SystemExit(f"No *.eval.txt files found in {base_path}")

    train_entries, train_skipped = _collect_stat_prompts(train_files)
    eval_entries, eval_skipped = _collect_stat_prompts(eval_files)

    if not train_entries:
        raise SystemExit("No train prompts detected (no <stats lines found)")
    if not eval_entries:
        raise SystemExit("No eval prompts detected (no <stats lines found)")

    write_jsonl_file(train_output_path, train_entries, ensure_ascii=False)
    write_jsonl_file(eval_output_path, eval_entries, ensure_ascii=False)

    print(f"Wrote {len(train_entries)} train prompt(s) to {train_output_path}")
    print(f"Wrote {len(eval_entries)} eval prompt(s) to {eval_output_path}")
    if train_skipped or eval_skipped:
        print(f"Skipped {train_skipped + eval_skipped} non-<stats line(s)")


if __name__ == "__main__":
    main()
