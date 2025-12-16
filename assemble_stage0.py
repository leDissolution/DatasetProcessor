"""
Stage0 dataset assembler that balances attributes across small batches.

- Consumes synthetic Stage0 JSONL exports (no loss metrics).
- Spreads rare attributes across batches to avoid domination when batch_size is small.
- Smooths batch prompt lengths to reduce padding waste.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Dict, Any, Optional, Callable, Tuple, Set

from collections import defaultdict

from assembler import (
    ensure_directories,
    get_files_by_suffix,
    load_jsonl_files,
    write_jsonl_file,
    filter_entries_by_attr,
    engineer_attr_balanced_batches,
    get_target_attr,
    has_no_change_flag,
)

DEFAULT_STAGE0_BASE = r".\Dataset\Prepared\\"
DEFAULT_BATCH_SIZE = 10
DEFAULT_SEED = 1337


def _compute_epoch_uniqueness(
    epoch_batches_list: List[List[List[Dict[str, Any]]]],
) -> Dict[str, Any]:
    """Compute metrics measuring how different each epoch's batch arrangement is.
    
    Metrics:
    - position_overlap: % of entries in same batch index across epoch pairs (lower = more unique)
    - neighbor_overlap: % of entry pairs that share a batch in both epochs (lower = more unique)
    """
    if len(epoch_batches_list) < 2:
        return {"position_overlap_mean": 0.0, "neighbor_overlap_mean": 0.0}
    
    def get_entry_id(entry: Dict[str, Any]) -> str:
        return entry.get("id", "") or str(id(entry))
    
    def build_batch_index_map(batches: List[List[Dict[str, Any]]]) -> Dict[str, int]:
        """Map entry ID -> batch index."""
        mapping: Dict[str, int] = {}
        for batch_idx, batch in enumerate(batches):
            for entry in batch:
                mapping[get_entry_id(entry)] = batch_idx
        return mapping
    
    def build_neighbor_set(batches: List[List[Dict[str, Any]]]) -> Set[Tuple[str, str]]:
        """Build set of (id1, id2) pairs that share a batch (ordered to avoid duplicates)."""
        neighbors: Set[Tuple[str, str]] = set()
        for batch in batches:
            ids = sorted(get_entry_id(e) for e in batch)
            for i, id1 in enumerate(ids):
                for id2 in ids[i + 1:]:
                    neighbors.add((id1, id2))
        return neighbors
    
    position_overlaps: List[float] = []
    neighbor_overlaps: List[float] = []
    
    # Compare all epoch pairs
    for i in range(len(epoch_batches_list)):
        for j in range(i + 1, len(epoch_batches_list)):
            batches_i = epoch_batches_list[i]
            batches_j = epoch_batches_list[j]
            
            # Position overlap: how many entries are in the same batch index
            map_i = build_batch_index_map(batches_i)
            map_j = build_batch_index_map(batches_j)
            common_ids = set(map_i.keys()) & set(map_j.keys())
            if common_ids:
                same_position = sum(1 for eid in common_ids if map_i[eid] == map_j[eid])
                position_overlaps.append(same_position / len(common_ids) * 100)
            
            # Neighbor overlap: how many entry pairs share a batch in both epochs
            neighbors_i = build_neighbor_set(batches_i)
            neighbors_j = build_neighbor_set(batches_j)
            if neighbors_i:
                shared_neighbors = len(neighbors_i & neighbors_j)
                neighbor_overlaps.append(shared_neighbors / len(neighbors_i) * 100)
    
    return {
        "position_overlap_mean": sum(position_overlaps) / len(position_overlaps) if position_overlaps else 0.0,
        "neighbor_overlap_mean": sum(neighbor_overlaps) / len(neighbor_overlaps) if neighbor_overlaps else 0.0,
        "position_overlap_values": position_overlaps,
        "neighbor_overlap_values": neighbor_overlaps,
    }


def build_training_epochs(
    entries: List[Dict[str, Any]],
    batch_size: int,
    num_epochs: int,
    rng_seed: int,
    max_per_attr: int,
    length_fn: Optional[Callable[[Dict[str, Any]], int]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build training data across multiple epochs with varied batch arrangements.
    
    Each epoch uses the same entries but with a different seed for shuffling,
    resulting in different batch compositions while maintaining the same
    distribution characteristics.
    
    Returns:
        Tuple of (all_entries_across_epochs, combined_metrics)
    """
    import math
    
    all_epochs_entries: List[Dict[str, Any]] = []
    epoch_batches_list: List[List[List[Dict[str, Any]]]] = []
    combined_metrics: Dict[str, Any] = {}
    
    print(f"Building {num_epochs} epoch(s) with {len(entries)} entries each")
    
    for epoch_idx in range(num_epochs):
        epoch_seed = rng_seed + epoch_idx
        
        reordered, metrics = engineer_attr_balanced_batches(
            entries,
            batch_size=batch_size,
            rng_seed=epoch_seed,
            max_per_attr=max_per_attr,
            length_fn=length_fn,
        )
        
        # Reconstruct batches for uniqueness analysis
        total_batches = math.ceil(len(reordered) / batch_size)
        batches = [
            reordered[i * batch_size : (i + 1) * batch_size]
            for i in range(total_batches)
        ]
        epoch_batches_list.append(batches)
        
        all_epochs_entries.extend(reordered)
        
        # Store metrics from first epoch as representative, note epoch count
        if epoch_idx == 0:
            combined_metrics = metrics.copy()
            combined_metrics["num_epochs"] = num_epochs
            combined_metrics["entries_per_epoch"] = len(reordered)
        
        print(f"Epoch {epoch_idx + 1}: {len(reordered)} entries, seed={epoch_seed}")
    
    combined_metrics["total_entries"] = len(all_epochs_entries)
    
    # Compute epoch uniqueness metrics
    if num_epochs > 1:
        uniqueness = _compute_epoch_uniqueness(epoch_batches_list)
        combined_metrics["epoch_position_overlap_pct"] = uniqueness["position_overlap_mean"]
        combined_metrics["epoch_neighbor_overlap_pct"] = uniqueness["neighbor_overlap_mean"]
    
    return all_epochs_entries, combined_metrics


def _build_length_fn(tokenizer_path: Optional[str]) -> Optional[Callable[[Dict[str, Any]], int]]:
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        def _len(entry: Dict[str, Any]) -> int:
            prompt = entry.get("prompt", "") or ""
            try:
                return len(tokenizer.encode(prompt, add_special_tokens=False))
            except Exception:
                return len(prompt)

        return _len
    except Exception as e:
        print(f"[WARN] Falling back to character length; tokenizer unavailable: {e}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble Stage0 dataset with balanced batches")
    parser.add_argument(
        "--base_path",
        type=str,
        default=DEFAULT_STAGE0_BASE,
        help="Base path containing Prepared synthetic outputs (default: ./Dataset/Prepared)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for Stage0 (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed for shuffling",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Optional tokenizer path to measure token lengths instead of chars",
    )
    parser.add_argument(
        "--max_per_attr",
        type=int,
        default=3,
        help="Per-batch cap for any single attribute (default: 3)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Override output directory (default: <base_path>/Assembled_stage0)",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default="train.jsonl",
        help="Output file name (default: train.jsonl)",
    )
    parser.add_argument(
        "--eval_output_file_name",
        type=str,
        default="eval.jsonl",
        help="Output file name for eval set (default: eval.jsonl)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs for training data (default: 1). Each epoch reuses entries in different order with different batch combinations.",
    )
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Print metrics without writing output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_path = os.path.normpath(args.base_path or DEFAULT_STAGE0_BASE)
    synthetic_path = os.path.join(base_path, "Synthetic")
    output_dir = os.path.normpath(
        args.output_dir if args.output_dir else os.path.join(base_path, "Assembled_stage0")
    )
    ensure_directories(output_dir)

    synth_files = get_files_by_suffix(synthetic_path, "train.jsonl")
    if not synth_files:
        raise FileNotFoundError(f"No synthetic Stage0 files found in {synthetic_path}")
    
    eval_files = get_files_by_suffix(synthetic_path, "eval.jsonl")
    if not eval_files:
        raise FileNotFoundError(f"No eval Stage0 files found in {synthetic_path}")

    entries = load_jsonl_files(synth_files)
    entries = filter_entries_by_attr(entries)
    print(f"Loaded {len(entries)} entries from {len(synth_files)} file(s)")

    eval_entries = load_jsonl_files(eval_files)
    eval_entries = filter_entries_by_attr(eval_entries)
    print(f"Loaded {len(eval_entries)} eval entries from {len(eval_files)} file(s)")

    # Count entries per attr
    attr_counts: Dict[str, int] = defaultdict(int)
    attr_no_change_counts: Dict[str, int] = defaultdict(int)
    for entry in entries:
        attr = get_target_attr(entry)
        attr_counts[attr] += 1
        if has_no_change_flag(entry):
            attr_no_change_counts[attr] += 1

    attr_summary = {
        attr: f"{count} ({attr_no_change_counts.get(attr, 0)})"
        for attr, count in sorted(attr_counts.items(), key=lambda kv: -kv[1])
    }
    print("Attr counts:", attr_summary)

    length_fn = _build_length_fn(args.tokenizer_path.strip()) if args.tokenizer_path else None

    reordered, metrics = build_training_epochs(
        entries,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        rng_seed=args.seed,
        max_per_attr=args.max_per_attr,
        length_fn=length_fn,
    )

    def _length_key(entry: Dict[str, Any]) -> int:
        prompt = entry.get("prompt", "") or ""
        if length_fn:
            try:
                return length_fn(entry)
            except Exception:
                pass
        return len(prompt)

    eval_sorted = sorted(eval_entries, key=_length_key)

    print(
        "Epochs: {ep}, Batches/epoch: {b}, Total entries: {tot}".format(
            ep=metrics.get("num_epochs", 1),
            b=metrics["total_batches"],
            tot=metrics.get("total_entries", len(reordered)),
        )
    )
    print(
        "Batch len: mean={mean:.1f} std={std:.1f} min={mn} max={mx}".format(
            mean=metrics["mean_batch_len"],
            std=metrics["std_batch_len"],
            mn=metrics["min_batch_len"],
            mx=metrics["max_batch_len"],
        )
    )
    print("Attr per-batch min: ", metrics["attr_batch_min"])
    print("Attr per-batch max: ", metrics["attr_batch_max"])
    print("Batches missing attr: ", metrics["attr_missing_batches"])
    print(
        "no_change total={tot}, batch min={mn}, max={mx}, batches missing={miss}".format(
            tot=metrics.get("no_change_total", 0),
            mn=metrics.get("no_change_batch_min", 0),
            mx=metrics.get("no_change_batch_max", 0),
            miss=metrics.get("no_change_batches_missing", 0),
        )
    )
    
    if args.epochs > 1:
        print(
            "Epoch uniqueness: position_overlap={pos:.1f}%, neighbor_overlap={neigh:.1f}% (lower = more unique)".format(
                pos=metrics.get("epoch_position_overlap_pct", 0),
                neigh=metrics.get("epoch_neighbor_overlap_pct", 0),
            )
        )

    if args.stats_only:
        return

    output_path = os.path.join(output_dir, args.output_file_name)
    write_jsonl_file(output_path, reordered)
    print(f"Wrote {len(reordered)} entries to {output_path}")

    eval_output_path = os.path.join(output_dir, args.eval_output_file_name)
    write_jsonl_file(eval_output_path, eval_sorted)
    print(f"Wrote {len(eval_sorted)} eval entries to {eval_output_path}")


if __name__ == "__main__":
    main()
