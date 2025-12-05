"""
File I/O utilities for dataset assembly.
"""

import os
import json
from typing import List, Dict, Any


def ensure_directories(*paths: str) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_files_by_suffix(directory: str, suffix: str) -> List[str]:
    """Get all files in directory with given suffix."""
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(suffix)
    ]
    files.sort()
    return files


def load_jsonl_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Load and parse JSONL files into a list of dictionaries."""
    entries = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            entries.extend([json.loads(line) for line in f])
    return entries


def write_jsonl_file(file_path: str, entries: List[Dict[str, Any]], ensure_ascii: bool = True) -> None:
    """Write entries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=ensure_ascii) + '\n')
