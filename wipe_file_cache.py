#!/usr/bin/env python3
"""
Wipe file-level cache entries for a specific filename.

This removes rows from the file_cache table for the given file path
(across all file hashes and versions). It intentionally does not modify
`dp_cache`, which is content-based and reused across files.

Usage:
  python wipe_file_cache.py <path-to-source-file> [--db <sqlite-path>] [--yes]

Examples:
    python wipe_file_cache.py ./data/myfile.txt
    python wipe_file_cache.py "C:\\datasets\\input.txt" --yes
    python wipe_file_cache.py ./data/myfile.txt --db ./.cache/processing.sqlite
    python wipe_file_cache.py ./data/myfile.txt --also-dp  # also drop matching dp_cache rows
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from typing import Optional


def default_db_path() -> str:
    base_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(base_dir, ".cache")
    return os.path.join(cache_dir, "processing.sqlite")


def human_path(p: str) -> str:
    try:
        return os.path.abspath(p)
    except Exception:
        return p


def ensure_db_exists(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.isfile(path)
    except Exception:
        return False


def wipe_file_cache(
    file_path: str,
    db_path: Optional[str] = None,
    assume_yes: bool = False,
    also_dp: bool = False,
) -> int:
    db = db_path or default_db_path()
    if not ensure_db_exists(db):
        print(f"Cache database not found at: {db}")
        return 2

    file_key = os.path.abspath(file_path)
    try:
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception as e:
        print(f"Failed to open database '{db}': {e}")
        return 3

    try:
        cur = conn.cursor()
        # Count rows before delete (exact match)
        cur.execute("SELECT COUNT(*) FROM file_cache WHERE file_key=?", (file_key,))
        row = cur.fetchone()
        count = int(row[0]) if row else 0
        # If no exact match (case differences?), try resolving canonical stored key using LOWER()
        if count == 0:
            cur.execute("SELECT file_key FROM file_cache WHERE LOWER(file_key)=LOWER(?) LIMIT 1", (file_key,))
            row2 = cur.fetchone()
            if row2 and isinstance(row2[0], str):
                file_key = row2[0]
                cur.execute("SELECT COUNT(*) FROM file_cache WHERE file_key=?", (file_key,))
                row = cur.fetchone()
                count = int(row[0]) if row else 0

        if count == 0 and not also_dp:
            print(f"No file_cache entries found for: {file_key}")
            return 0

        to_delete_msgs = []
        if count > 0:
            to_delete_msgs.append(f"{count} from file_cache")

        dp_delete_count = 0
        dp_keys: list[str] = []
        if also_dp:
            # Best-effort: recompute datapoint keys for current file content
            try:
                from entity_parser import parse_text  # type: ignore
            except Exception as e:
                print(f"Warning: could not import parser to compute dp keys: {e}")
                parse_text = None  # type: ignore
            try:
                from processing_cache import compute_datapoint_key  # type: ignore
            except Exception as e:
                print(f"Warning: could not import compute_datapoint_key: {e}")
                compute_datapoint_key = None  # type: ignore

            if parse_text is not None:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                except Exception as e:
                    print(f"Warning: failed to read file to compute dp keys: {e}")
                    text = None  # type: ignore

                if text and parse_text is not None:
                    warnings: list[str] = []
                    def _w(msg: str) -> None:
                        warnings.append(msg)
                    try:
                        dps = parse_text(text, on_warning=_w, source_name=file_path)
                        if 'compute_datapoint_key' in globals() and compute_datapoint_key is not None:  # type: ignore[name-defined]
                            dp_keys = [compute_datapoint_key(dp) for dp in dps]  # type: ignore[misc]
                        else:
                            dp_keys = []
                    except Exception as e:
                        print(f"Warning: failed to parse datapoints: {e}")
                        dp_keys = []

            if dp_keys:
                placeholders = ",".join(["?"] * len(dp_keys))
                cur.execute(f"SELECT COUNT(*) FROM dp_cache WHERE dp_key IN ({placeholders})", dp_keys)
                row2 = cur.fetchone()
                dp_delete_count = int(row2[0]) if row2 else 0
                if dp_delete_count > 0:
                    to_delete_msgs.append(f"{dp_delete_count} from dp_cache")

        if not to_delete_msgs:
            print(f"No matching cache entries found for: {file_key}")
            return 0

        if not assume_yes:
            print("About to delete: " + ", ".join(to_delete_msgs) + f" for\n  {file_key}")
            if also_dp and dp_keys:
                print("Note: dp_cache is content-based and shared; removing these may affect reuse in other files with identical datapoints.")
            confirm = input("Proceed? [y/N]: ").strip().lower()
            if confirm not in {"y", "yes"}:
                print("Aborted.")
                return 1

        if count > 0:
            cur.execute("DELETE FROM file_cache WHERE file_key=?", (file_key,))
        if also_dp and dp_keys:
            placeholders = ",".join(["?"] * len(dp_keys))
            cur.execute(f"DELETE FROM dp_cache WHERE dp_key IN ({placeholders})", dp_keys)
        conn.commit()

        print(f"Deleted: " + ", ".join(to_delete_msgs) + f" for {file_key}")
        return 0
    except sqlite3.OperationalError as e:
        print(f"SQLite error: {e}")
        return 4
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 5
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Wipe file-level cache entries for a specific filename.")
    parser.add_argument("file", help="Path to the source file whose cache should be cleared.")
    parser.add_argument("--db", dest="db", default=None, help="Path to processing.sqlite (optional). Defaults to ./.cache/processing.sqlite next to this script.")
    parser.add_argument("-y", "--yes", action="store_true", help="Do not prompt for confirmation.")
    parser.add_argument("--also-dp", action="store_true", help="Also delete dp_cache rows derived from this file's current content (caution: affects shared cache).")

    args = parser.parse_args(argv)

    file_arg = args.file
    db = args.db

    print(f"Target file: {human_path(file_arg)}")
    print(f"Database:    {human_path(db) if db else human_path(default_db_path())}")

    return wipe_file_cache(file_arg, db_path=db, assume_yes=args.yes, also_dp=args.also_dp)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
