"""
Utilities for accessing and manipulating dataset entry properties.
"""

from typing import Dict, Any

from .constants import HIGH_LOSS_THRESHOLD, MEDIUM_LOSS_MIN, LOW_LOSS_MIN


def get_loss_metrics(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return nested loss_metrics dict or empty."""
    return entry.get('loss_metrics') or {}


def get_completion_difficulty(entry: Dict[str, Any]) -> float:
    """Get completion difficulty from entry's loss_metrics."""
    lm = get_loss_metrics(entry)
    try:
        return float(lm.get('completion_difficulty') or 0.0)
    except Exception:
        return 0.0


def get_worst_loss(entry: Dict[str, Any]) -> float:
    """Get worst loss from entry's loss_metrics."""
    lm = get_loss_metrics(entry)
    try:
        return float(lm.get('worst_loss') or 0.0)
    except Exception:
        return 0.0


def get_critical_token(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Get critical token info from entry's loss_metrics."""
    lm = get_loss_metrics(entry)
    return lm.get('critical_token') or {}


def get_target_attr(entry: Dict[str, Any]) -> str:
    """Get the target attribute from an entry."""
    tgt = entry.get('target') or {}
    return str(tgt.get('attr') or 'unknown')


def need_attr(attr: str) -> bool:
    """Return True if this attribute should be included in datasets.
    
    Attribute selection helper: customize this predicate to restrict which 
    target attrs are included. By default, accept all attributes except blacklisted ones.
    """
    blacklist = {"updated"}
    return attr not in blacklist


def is_flip(entry: Dict[str, Any]) -> bool:
    """Check if an entry represents a flip (prediction doesn't match actual value)."""
    ct = get_critical_token(entry)
    pred = ct.get('pred_decoded_value')
    actual = ct.get('decoded_value')
    return pred is not None and actual is not None and str(pred) != str(actual)


def has_no_change_flag(entry: Dict[str, Any]) -> bool:
    """Return True when entry is explicitly marked as no_change."""
    sentinel = "!!no_change!!\" />"
    prompt = entry.get('prompt')
    return isinstance(prompt, str) and prompt.endswith(sentinel)


def count_flips(entries: list) -> int:
    """Count the number of flips in a list of entries."""
    return sum(1 for entry in entries if is_flip(entry))


def count_no_change(entries: list) -> int:
    """Count entries explicitly marked to keep the original value."""
    sentinel = "!!no_change!!\" />"
    return sum(
        1
        for entry in entries
        if isinstance(entry.get('prompt'), str) and str(entry.get('prompt')).endswith(sentinel)
    )


def get_difficulty_bucket(entry: Dict[str, Any]) -> str:
    """Return the difficulty bucket label for a given entry."""
    difficulty = get_completion_difficulty(entry)
    if difficulty > HIGH_LOSS_THRESHOLD:
        return "high"
    if difficulty > MEDIUM_LOSS_MIN:
        return "medium"
    if difficulty > LOW_LOSS_MIN:
        return "low"
    return "other"
