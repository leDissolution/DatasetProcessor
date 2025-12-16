"""
Batch engineering utilities for balanced dataset construction.
"""

import random
import math
import statistics
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple, Optional, Callable

from .entry_utils import (
    get_completion_difficulty,
    get_target_attr,
    get_difficulty_bucket,
    has_no_change_flag,
)


def _pick_diverse_entry_idx(
    pool: List[Dict[str, Any]],
    batch_attr_counts: Dict[str, int],
) -> int:
    """Return the index of an entry in pool that maximizes attr diversity.

    Prefers entries whose target.attr is least represented in the current batch.
    Ties are broken by picking the last element (for efficient pop).
    """
    if not pool:
        return 0
    if len(pool) == 1:
        return 0

    best_idx = len(pool) - 1  # Default to last for efficient pop()
    best_count = batch_attr_counts.get(get_target_attr(pool[best_idx]), 0)

    # Only scan a limited window to avoid O(n²) blowup on large pools
    scan_limit = min(50, len(pool))
    for i in range(len(pool) - 1, max(-1, len(pool) - 1 - scan_limit), -1):
        attr = get_target_attr(pool[i])
        count = batch_attr_counts.get(attr, 0)
        if count < best_count:
            best_count = count
            best_idx = i
            if best_count == 0:
                # Can't do better than an attr not yet in the batch
                break

    return best_idx


def rebalance_batches_for_requirement(
    batches: List[List[Dict[str, Any]]],
    predicate: Callable[[Dict[str, Any]], bool],
) -> int:
    """Ensure each batch satisfies predicate when surplus entries exist.

    Returns the number of batches still lacking entries that satisfy the predicate
    after attempting to rebalance. Swaps preserve batch sizes and aim to keep
    difficulty composition stable when possible.
    """

    def batch_has_requirement(batch: List[Dict[str, Any]]) -> bool:
        return any(predicate(entry) for entry in batch)

    lacking = [idx for idx, batch in enumerate(batches) if not batch_has_requirement(batch)]
    if not lacking:
        return 0

    made_progress = True
    while lacking and made_progress:
        made_progress = False
        for target_idx in list(lacking):
            donor_idx = next(
                (
                    idx
                    for idx, donor_batch in enumerate(batches)
                    if idx != target_idx and sum(1 for entry in donor_batch if predicate(entry)) > 1
                ),
                None,
            )
            if donor_idx is None:
                continue

            donor_batch = batches[donor_idx]
            target_batch = batches[target_idx]

            donor_entry_idx = next(
                (i for i, entry in enumerate(donor_batch) if predicate(entry)),
                None,
            )
            if donor_entry_idx is None:
                continue

            donor_entry = donor_batch[donor_entry_idx]
            donor_bucket = get_difficulty_bucket(donor_entry)

            replacement_idx = next(
                (
                    i
                    for i, entry in enumerate(target_batch)
                    if not predicate(entry) and get_difficulty_bucket(entry) == donor_bucket
                ),
                None,
            )
            if replacement_idx is None:
                replacement_idx = next(
                    (i for i, entry in enumerate(target_batch) if not predicate(entry)),
                    None,
                )
            if replacement_idx is None:
                continue

            replacement_entry = target_batch[replacement_idx]
            donor_batch[donor_entry_idx] = replacement_entry
            target_batch[replacement_idx] = donor_entry
            made_progress = True

        lacking = [idx for idx, batch in enumerate(batches) if not batch_has_requirement(batch)]

    return len(lacking)


def engineer_balanced_batches(
    entries: List[Dict[str, Any]],
    batch_size: int,
    synthetic_ids: Set[int],
    rng_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Arrange entries into batches with balanced difficulty and required coverage.

    Returns the reordered entries along with batch metrics for diagnostics.
    """

    if not entries:
        return [], {
            "total_batches": 0,
            "synthetic_available_batches": 0,
            "synthetic_batches_covered": 0,
            "synthetic_batches_missing": 0,
            "no_change_available_batches": 0,
            "no_change_batches_covered": 0,
            "no_change_batches_missing": 0,
            "mean_of_batch_means": 0.0,
            "std_of_batch_means": 0.0,
            "mean_of_batch_medians": 0.0,
            "std_of_batch_medians": 0.0,
            "mean_unique_attrs": 0.0,
            "min_unique_attrs": 0,
            "max_unique_attrs": 0,
        }

    rng = random.Random(rng_seed)
    pools: Dict[str, List[Dict[str, Any]]] = {"high": [], "medium": [], "low": [], "other": []}
    for entry in entries:
        pools[get_difficulty_bucket(entry)].append(entry)

    for pool in pools.values():
        rng.shuffle(pool)

    total_batches = math.ceil(len(entries) / batch_size)
    batches: List[List[Dict[str, Any]]] = []
    ordered_categories = ("high", "medium", "low", "other")

    # Count total entries per attr across all pools for target diversity
    global_attr_counts: Dict[str, int] = defaultdict(int)
    for cat in ordered_categories:
        for entry in pools[cat]:
            global_attr_counts[get_target_attr(entry)] += 1

    for batch_idx in range(total_batches):
        batch: List[Dict[str, Any]] = []
        batch_attr_counts: Dict[str, int] = defaultdict(int)
        remaining_capacity = batch_size

        # Determine actual batch size (last batch may be smaller)
        remaining_total = sum(len(pools[cat]) for cat in ordered_categories)
        actual_batch_size = min(batch_size, remaining_total)

        # Calculate proportional targets dynamically based on CURRENT pool sizes
        targets: Dict[str, int] = {}
        allocated = 0
        for cat in ordered_categories:
            if remaining_total > 0:
                # Proportional to current pool size
                raw_target = len(pools[cat]) / remaining_total * actual_batch_size
            else:
                raw_target = 0
            if cat == ordered_categories[-1]:
                # Last category gets the remainder to avoid rounding errors
                targets[cat] = max(0, actual_batch_size - allocated)
            else:
                targets[cat] = round(raw_target)
                allocated += targets[cat]

        # Fill batch according to proportional targets, preferring attr diversity
        for category in ordered_categories:
            pool = pools[category]
            target = min(targets[category], len(pool), remaining_capacity)

            for _ in range(target):
                if not pool or remaining_capacity <= 0:
                    break
                # Pick entry that maximizes attr diversity in this batch
                best_idx = _pick_diverse_entry_idx(pool, batch_attr_counts)
                entry = pool.pop(best_idx)
                batch.append(entry)
                batch_attr_counts[get_target_attr(entry)] += 1
                remaining_capacity -= 1

        # Fill any remaining capacity from whatever is available, still preferring diversity
        # Prioritize higher difficulty categories first
        while remaining_capacity > 0:
            picked: Optional[Dict[str, Any]] = None
            best_pool_idx: Optional[int] = None
            best_category: Optional[str] = None
            for category in ordered_categories:
                pool = pools[category]
                if pool:
                    idx = _pick_diverse_entry_idx(pool, batch_attr_counts)
                    picked = pool[idx]
                    best_pool_idx = idx
                    best_category = category
                    break
            if picked is None or best_pool_idx is None or best_category is None:
                break
            entry = pools[best_category].pop(best_pool_idx)
            batch.append(entry)
            batch_attr_counts[get_target_attr(entry)] += 1
            remaining_capacity -= 1

        batches.append(batch)

    synthetic_predicate = lambda entry: id(entry) in synthetic_ids
    no_change_predicate = has_no_change_flag

    synthetic_total = sum(1 for entry in entries if synthetic_predicate(entry))
    no_change_total = sum(1 for entry in entries if no_change_predicate(entry))

    synthetic_available_batches = min(total_batches, synthetic_total)
    no_change_available_batches = min(total_batches, no_change_total)

    if synthetic_total:
        synthetic_missing = rebalance_batches_for_requirement(batches, synthetic_predicate)
    else:
        synthetic_missing = total_batches

    if no_change_total:
        no_change_missing = rebalance_batches_for_requirement(batches, no_change_predicate)
    else:
        no_change_missing = total_batches

    synthetic_batches_covered = total_batches - synthetic_missing
    no_change_batches_covered = total_batches - no_change_missing

    ordered_entries = [entry for batch in batches for entry in batch]

    # Calculate per-batch difficulty statistics
    batch_mean_difficulties: List[float] = []
    batch_median_difficulties: List[float] = []
    batch_unique_attrs: List[int] = []
    for batch in batches:
        if batch:
            difficulties = [get_completion_difficulty(e) for e in batch]
            batch_mean_difficulties.append(statistics.mean(difficulties))
            batch_median_difficulties.append(statistics.median(difficulties))
            batch_unique_attrs.append(len(set(get_target_attr(e) for e in batch)))

    # Compute std of mean and median across batches
    mean_of_batch_means = statistics.mean(batch_mean_difficulties) if batch_mean_difficulties else 0.0
    std_of_batch_means = statistics.stdev(batch_mean_difficulties) if len(batch_mean_difficulties) > 1 else 0.0
    min_batch_mean = min(batch_mean_difficulties) if batch_mean_difficulties else 0.0
    max_batch_mean = max(batch_mean_difficulties) if batch_mean_difficulties else 0.0
    mean_of_batch_medians = statistics.mean(batch_median_difficulties) if batch_median_difficulties else 0.0
    std_of_batch_medians = statistics.stdev(batch_median_difficulties) if len(batch_median_difficulties) > 1 else 0.0
    mean_unique_attrs = statistics.mean(batch_unique_attrs) if batch_unique_attrs else 0.0
    min_unique_attrs = min(batch_unique_attrs) if batch_unique_attrs else 0
    max_unique_attrs = max(batch_unique_attrs) if batch_unique_attrs else 0

    metrics = {
        "total_batches": total_batches,
        "synthetic_available_batches": synthetic_available_batches,
        "synthetic_batches_covered": synthetic_batches_covered,
        "synthetic_batches_missing": synthetic_missing,
        "no_change_available_batches": no_change_available_batches,
        "no_change_batches_covered": no_change_batches_covered,
        "no_change_batches_missing": no_change_missing,
        "mean_of_batch_means": mean_of_batch_means,
        "std_of_batch_means": std_of_batch_means,
        "mean_of_batch_medians": mean_of_batch_medians,
        "std_of_batch_medians": std_of_batch_medians,
        "mean_unique_attrs": mean_unique_attrs,
        "min_unique_attrs": min_unique_attrs,
        "max_unique_attrs": max_unique_attrs,
        "min_batch_mean": min_batch_mean,
        "max_batch_mean": max_batch_mean,
    }

    return ordered_entries, metrics


def engineer_attr_balanced_batches(
    entries: List[Dict[str, Any]],
    batch_size: int,
    rng_seed: int,
    *,
    max_per_attr: Optional[int] = None,
    length_fn: Optional[Callable[[Dict[str, Any]], int]] = None,
    spread_no_change: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Arrange entries to spread attrs across batches and smooth lengths.

    This variant is for Stage0 where no difficulty buckets exist. It works in two
    phases:
    1) Spread rare attributes first with a per-batch cap to avoid domination.
    2) Fill remaining slots by placing the longest prompts into the currently
       shortest batches to reduce padding waste.
    """

    if not entries:
        return [], {
            "total_batches": 0,
            "mean_batch_len": 0.0,
            "std_batch_len": 0.0,
            "min_batch_len": 0,
            "max_batch_len": 0,
            "attr_batch_min": {},
            "attr_batch_max": {},
            "attr_missing_batches": {},
        }

    rng = random.Random(rng_seed)
    pool = list(entries)
    rng.shuffle(pool)

    def _length(entry: Dict[str, Any]) -> int:
        if length_fn is None:
            return len(entry.get("prompt", "") or "")
        try:
            return int(length_fn(entry))
        except Exception:
            return len(entry.get("prompt", "") or "")

    total_batches = math.ceil(len(pool) / batch_size)
    batches: List[List[Dict[str, Any]]] = [[] for _ in range(total_batches)]

    # Build attr pools from ALL entries (including no_change)
    attr_pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in pool:
        attr_pools[get_target_attr(entry)].append(entry)
    for entries_by_attr in attr_pools.values():
        rng.shuffle(entries_by_attr)

    def _attr_count(batch: List[Dict[str, Any]], attr: str) -> int:
        return sum(1 for e in batch if get_target_attr(e) == attr)

    def _no_change_count(batch: List[Dict[str, Any]]) -> int:
        return sum(1 for e in batch if has_no_change_flag(e))

    def _batch_mean_length(batch: List[Dict[str, Any]]) -> float:
        if not batch:
            return 0.0
        return sum(_length(e) for e in batch) / len(batch)

    def _length_deviation(batch: List[Dict[str, Any]], entry_len: int) -> float:
        """How far entry_len is from the batch's current mean length."""
        if not batch:
            return 0.0
        return abs(entry_len - _batch_mean_length(batch))

    # Interleaved attribute spreading: instead of processing attr-by-attr (which causes
    # rare attrs to fill batches before abundant ones get a chance), we process one entry
    # at a time, always picking the (attr, batch) combo that most needs balancing.
    #
    # For each attr, calculate ideal count per batch = total_for_attr / total_batches
    # Then greedily place entries where the deficit (ideal - actual) is largest.

    attr_totals = {attr: len(items) for attr, items in attr_pools.items()}
    attr_ideal_per_batch = {attr: count / total_batches for attr, count in attr_totals.items()}

    def _deficit(batch: List[Dict[str, Any]], attr: str) -> float:
        """How many more of this attr does this batch need vs ideal?"""
        actual = _attr_count(batch, attr)
        return attr_ideal_per_batch[attr] - actual

    # Flatten all entries with their attrs
    all_entries_with_attr: List[tuple] = []
    for attr, items in attr_pools.items():
        for entry in items:
            all_entries_with_attr.append((attr, entry))
    rng.shuffle(all_entries_with_attr)

    for attr, entry in all_entries_with_attr:
        is_no_change = has_no_change_flag(entry)
        entry_len = _length(entry)

        # Only consider batches that have space
        available_batches = [b for b in batches if len(b) < batch_size]
        if not available_batches:
            break

        # Choose batch with highest deficit for this attr (needs it most)
        # Tie-break: no_change balance, length deviation, then size
        if spread_no_change and is_no_change:
            target_batch = max(
                available_batches,
                key=lambda b: (
                    _deficit(b, attr),
                    -_no_change_count(b),
                    -_length_deviation(b, entry_len),
                    -len(b),
                ),
            )
        else:
            target_batch = max(
                available_batches,
                key=lambda b: (
                    _deficit(b, attr),
                    -_length_deviation(b, entry_len),
                    -len(b),
                ),
            )

        target_batch.append(entry)

    ordered_entries = [e for batch in batches for e in batch]

    batch_lengths = [sum(_length(e) for e in batch) for batch in batches if batch]

    def _no_change_count_batch(batch: List[Dict[str, Any]]) -> int:
        return sum(1 for e in batch if has_no_change_flag(e))

    attr_keys = list(attr_pools.keys())
    attr_batch_counts: Dict[str, List[int]] = {a: [] for a in attr_keys}
    for batch in batches:
        counts: Dict[str, int] = defaultdict(int)
        for e in batch:
            counts[get_target_attr(e)] += 1
        for attr in attr_keys:
            attr_batch_counts[attr].append(counts.get(attr, 0))

    no_change_counts = [_no_change_count_batch(batch) for batch in batches]

    metrics = {
        "total_batches": total_batches,
        "mean_batch_len": statistics.mean(batch_lengths) if batch_lengths else 0.0,
        "std_batch_len": statistics.stdev(batch_lengths) if len(batch_lengths) > 1 else 0.0,
        "min_batch_len": min(batch_lengths) if batch_lengths else 0,
        "max_batch_len": max(batch_lengths) if batch_lengths else 0,
        "attr_batch_min": {a: min(v) if v else 0 for a, v in attr_batch_counts.items()},
        "attr_batch_max": {a: max(v) if v else 0 for a, v in attr_batch_counts.items()},
        "attr_missing_batches": {a: sum(1 for c in counts if c == 0) for a, counts in attr_batch_counts.items()},
        "no_change_total": sum(1 for e in entries if has_no_change_flag(e)),
        "no_change_batches_missing": sum(1 for c in no_change_counts if c == 0),
        "no_change_batch_min": min(no_change_counts) if no_change_counts else 0,
        "no_change_batch_max": max(no_change_counts) if no_change_counts else 0,
    }

    return ordered_entries, metrics
