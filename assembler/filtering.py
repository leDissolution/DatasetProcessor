"""
Filtering and selection utilities for dataset entries.
"""

import random
import math
from typing import List, Dict, Any, Set, Tuple, Optional, Callable

from .constants import HIGH_LOSS_THRESHOLD, MEDIUM_LOSS_MIN, LOW_LOSS_MIN
from .entry_utils import (
    get_completion_difficulty,
    get_target_attr,
    need_attr,
    is_flip,
)


def filter_entries_by_attr(
    entries: List[Dict[str, Any]], 
    predicate: Callable[[str], bool] = need_attr
) -> List[Dict[str, Any]]:
    """Filter entries by target.attr using a predicate (str) -> bool."""
    out: List[Dict[str, Any]] = []
    for e in entries:
        if predicate(get_target_attr(e)):
            out.append(e)
    return out


def exclude_top_difficulty_percentile(
    entries: List[Dict[str, Any]], 
    percentile: float, 
    min_difficulty: float = 0.0
) -> List[Dict[str, Any]]:
    """Remove the highest-difficulty entries corresponding to the given percentile.

    Only entries with completion_difficulty >= min_difficulty are considered eligible for removal.
    """
    if not entries or percentile <= 0:
        return entries

    clamped = max(0.0, min(100.0, percentile))
    if clamped == 0.0:
        return entries

    eligible = [entry for entry in entries if get_completion_difficulty(entry) >= min_difficulty]
    if not eligible:
        return entries

    remove_count = math.ceil(len(eligible) * clamped / 100.0)
    if remove_count <= 0:
        return entries

    sorted_eligible = sorted(eligible, key=get_completion_difficulty, reverse=True)
    ids_to_remove = {id(entry) for entry in sorted_eligible[:remove_count]}
    return [entry for entry in entries if id(entry) not in ids_to_remove]


def create_loss_buckets(
    entries: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create loss buckets based on completion difficulty (nested in loss_metrics).
    
    Returns: (high_loss, medium_loss, low_loss, zero_loss) tuples, each shuffled.
    """
    high_loss = [entry for entry in entries if get_completion_difficulty(entry) > HIGH_LOSS_THRESHOLD]
    medium_loss = [entry for entry in entries if MEDIUM_LOSS_MIN < get_completion_difficulty(entry) <= HIGH_LOSS_THRESHOLD]
    low_loss = [entry for entry in entries if LOW_LOSS_MIN < get_completion_difficulty(entry) <= MEDIUM_LOSS_MIN]
    zero_loss = [entry for entry in entries if get_completion_difficulty(entry) == 0]

    random.shuffle(high_loss)
    random.shuffle(medium_loss)
    random.shuffle(low_loss)

    return high_loss, medium_loss, low_loss, zero_loss


def get_eval_ids(eval_entries: List[Dict[str, Any]]) -> Set[str]:
    """Extract unique ids from evaluation entries."""
    return {entry['id'] for entry in eval_entries if 'id' in entry}


def select_unique_source_entries(
    entries: List[Dict[str, Any]], 
    count: int, 
    exclude_ids: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """Select entries ensuring unique ids, up to the specified count."""
    if exclude_ids is None:
        exclude_ids = set()

    seen_ids = exclude_ids.copy()
    selected = []

    for entry in entries:
        entry_id = entry.get('id')
        if entry_id and entry_id not in seen_ids and need_attr(get_target_attr(entry)):
            seen_ids.add(entry_id)
            selected.append(entry)
            if len(selected) >= count:
                break

    return selected


def remove_eval_contamination(
    train_entries: List[Dict[str, Any]],
    eval_ids: Set[str]
) -> List[Dict[str, Any]]:
    """Remove training entries that share ids with eval entries."""
    return [entry for entry in train_entries if entry.get('id') not in eval_ids]


def sort_by_difficulty_and_length(
    entries: List[Dict[str, Any]], 
    reverse: bool = True
) -> List[Dict[str, Any]]:
    """Sort entries by completion difficulty and prompt length."""
    return sorted(
        entries, 
        key=lambda x: (get_completion_difficulty(x), len(x.get('prompt', ''))), 
        reverse=reverse
    )


def sort_by_flip_priority(
    entries: List[Dict[str, Any]], 
    flips_first: bool = True
) -> List[Dict[str, Any]]:
    """Sort entries prioritizing flips first or non-flips first."""
    return sorted(
        entries,
        key=lambda x: (
            not is_flip(x) if flips_first else is_flip(x),
            get_completion_difficulty(x),
            len(x.get('prompt', '')),
        ),
        reverse=True,
    )


def dedupe_entries_by_id(
    entries: List[Dict[str, Any]], 
    seen_ids: Optional[Set[str]] = None
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Return entries with unique ids only, tracking and updating seen_ids.

    If an entry lacks 'id', keep it once using its object id for de-dup within this call.
    """
    if seen_ids is None:
        seen_ids = set()
    local_noid_seen: Set[int] = set()
    out: List[Dict[str, Any]] = []
    for e in entries:
        eid = e.get('id')
        if isinstance(eid, str) and eid:
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            out.append(e)
        else:
            oid = id(e)
            if oid in local_noid_seen:
                continue
            local_noid_seen.add(oid)
            out.append(e)
    return out, seen_ids


def dedupe_entries_by_identity(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return entries with unique object identity, preserving order."""
    seen: Set[int] = set()
    out: List[Dict[str, Any]] = []
    for e in entries:
        oid = id(e)
        if oid in seen:
            continue
        seen.add(oid)
        out.append(e)
    return out


def extract_synth_flips(synth_train: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract synthetic entries where prediction doesn't match actual value."""
    return [entry for entry in synth_train if is_flip(entry)]
