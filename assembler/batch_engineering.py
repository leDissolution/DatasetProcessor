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
