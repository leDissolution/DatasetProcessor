"""
Statistics and reporting utilities for dataset assembly.
"""

import os
import math
import statistics
from collections import defaultdict
from typing import List, Dict, Any, Set, Optional

from .entry_utils import (
    get_completion_difficulty,
    get_worst_loss,
    get_target_attr,
    is_flip,
    count_flips,
    count_no_change,
)


def print_bucket_stats(
    high_loss: List, 
    medium_loss: List, 
    low_loss: List,
    zero_loss: List, 
    synth_train: List, 
    train: List
) -> None:
    """Print statistics for each loss bucket."""
    high_flips = count_flips(high_loss)
    medium_flips = count_flips(medium_loss)
    low_flips = count_flips(low_loss)
    zero_flips = count_flips(zero_loss)
    synth_flips = count_flips(synth_train)
    high_no_change = count_no_change(high_loss)
    medium_no_change = count_no_change(medium_loss)
    low_no_change = count_no_change(low_loss)
    zero_no_change = count_no_change(zero_loss)
    synth_no_change = count_no_change(synth_train)
    train_no_change = count_no_change(train)

    print(f"High loss: {len(high_loss)} (flips: {high_flips}, no_change: {high_no_change})")
    print(f"Medium loss: {len(medium_loss)} (flips: {medium_flips}, no_change: {medium_no_change})")
    print(f"Low loss: {len(low_loss)} (flips: {low_flips}, no_change: {low_no_change})")
    print(f"Zero loss: {len(zero_loss)} (flips: {zero_flips}, no_change: {zero_no_change})")
    print(f"Synth train: {len(synth_train)} (flips: {synth_flips}, no_change: {synth_no_change})")
    print(f"Total: {len(train)} (no_change: {train_no_change})")


def print_top_unique_sources(entries: List[Dict[str, Any]], limit: int = 50) -> None:
    """Print top unique source IDs by worst_loss."""
    print(f"Top {limit} unique source IDs by worst_loss:")
    seen_ids: Set[str] = set()
    unique_entries = []

    for entry in sorted(entries, key=lambda x: get_worst_loss(x), reverse=True):
        entry_id = entry.get('id', 'unknown')
        if entry_id not in seen_ids:
            seen_ids.add(entry_id)
            unique_entries.append(entry)
            if len(unique_entries) >= limit:
                break

    for entry in unique_entries:
        entry_id = entry.get('id', 'unknown')
        print(f"Loss: {get_worst_loss(entry):.4f}, Source: {entry_id}, Prompt: {entry.get('prompt','')[:50]}...")


def print_zero_loss_sources(zero_loss: List[Dict[str, Any]]) -> None:
    """Print zero-loss counts aggregated per source file for easier scanning."""
    if not zero_loss:
        print("Zero loss entries: 0")
        return

    zero_loss_total = len(zero_loss)
    per_file: Dict[str, Set[str]] = defaultdict(set)
    unknown_ids = 0

    for entry in zero_loss:
        eid = entry.get('id')
        if not isinstance(eid, str) or not eid:
            unknown_ids += 1
            continue
        source = eid.split(':', 1)[0]
        per_file[source].add(eid)

    unique_total = sum(len(ids) for ids in per_file.values())
    print(f"Zero loss entries: {zero_loss_total} (unique ids: {unique_total})")

    for source in sorted(per_file.keys()):
        count = len(per_file[source])
        print(f"{source}: {count}")

    if unknown_ids:
        print(f"<unknown>: {unknown_ids}")


def print_file_contributions(
    selected_entries: List[Dict[str, Any]],
    total_pool: List[Dict[str, Any]],
    eval_entries: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Print per-file contributions for selected entries relative to totals in that file.

    File name is taken as the first part of the 'id' when split by ':'.
        Format:
        filename: X entries (y%); flips: f (p%); avg: a; eval: e (z% of file)
        where
            - y = X / total_in_file * 100
            - p = f / X * 100
        - a = mean completion_difficulty across selected entries in this file
            - z = e / total_in_file * 100
    """
    totals: Dict[str, int] = defaultdict(int)
    selected: Dict[str, int] = defaultdict(int)
    flips_in_selected: Dict[str, int] = defaultdict(int)
    # Sum of completion_difficulty to compute average per file
    sum_loss_in_selected: Dict[str, float] = defaultdict(float)
    # Track unique IDs per file for denominator of eval%
    unique_ids_by_file: Dict[str, Set[str]] = defaultdict(set)
    # Track unique eval IDs per file for numerator of eval%
    eval_ids_by_file: Dict[str, Set[str]] = defaultdict(set)

    def file_of(e: Dict[str, Any]) -> Optional[str]:
        eid = e.get('id')
        if not isinstance(eid, str) or not eid:
            return None
        return eid.split(':', 1)[0]

    for e in total_pool:
        fn = file_of(e)
        if fn:
            totals[fn] += 1
            eid = e.get('id')
            if isinstance(eid, str) and eid:
                unique_ids_by_file[fn].add(eid)

    for e in selected_entries:
        fn = file_of(e)
        if fn:
            selected[fn] += 1
            if is_flip(e):
                flips_in_selected[fn] += 1
            # accumulate difficulty for average
            try:
                sum_loss_in_selected[fn] += float(get_completion_difficulty(e))
            except Exception:
                pass

    if eval_entries:
        for e in eval_entries:
            fn = file_of(e)
            if fn:
                eid = e.get('id')
                if isinstance(eid, str) and eid:
                    eval_ids_by_file[fn].add(eid)

    if not selected:
        print("No per-file contributions to report.")
        return

    print("Per-file contribution:")

    # Build rows for pretty printing
    rows = []
    for fn in sorted(selected.keys()):
        sel = selected[fn]
        tot = totals.get(fn, 0)
        y_pct = (sel / tot * 100.0) if tot > 0 else 0.0
        f = flips_in_selected.get(fn, 0)
        p_pct = (f / sel * 100.0) if sel > 0 else 0.0
        # average loss across selected entries from this file
        avg_loss = (sum_loss_in_selected.get(fn, 0.0) / sel) if sel > 0 else 0.0
        # eval stats computed on unique IDs
        e_unique = len(eval_ids_by_file.get(fn) or set())
        uniq_total = len(unique_ids_by_file.get(fn) or set())
        z_pct = (e_unique / uniq_total * 100.0) if uniq_total > 0 else 0.0
        rows.append((fn, sel, tot, y_pct, f, p_pct, avg_loss, e_unique, z_pct))

    # Determine column widths
    fn_w = max(4, min(60, max((len(r[0]) for r in rows), default=4)))
    num_w = 7
    pct_w = 7
    avg_w = 7

    # Print header
    header = (
        f"{'file':<{fn_w}}  {'sel':>{num_w}}  {'tot':>{num_w}}  {'sel%':>{pct_w}}  "
        f"{'flips':>{num_w}}  {'flip%':>{pct_w}}  {'avg':>{avg_w}}  {'eval':>{num_w}}  {'eval%':>{pct_w}}"
    )
    print(header)
    print('-' * len(header))

    # Print rows
    for fn, sel, tot, y_pct, f, p_pct, a, e, z_pct in rows:
        print(
            f"{fn:<{fn_w}}  {sel:>{num_w}d}  {tot:>{num_w}d}  {y_pct:>{pct_w}.1f}  "
            f"{f:>{num_w}d}  {p_pct:>{pct_w}.1f}  {a:>{avg_w}.2f}  {e:>{num_w}d}  {z_pct:>{pct_w}.1f}"
        )

    # Totals row
    total_sel = sum(r[1] for r in rows)
    total_tot = sum(r[2] for r in rows)
    total_flips = sum(r[4] for r in rows)
    total_eval = sum(r[7] for r in rows)
    total_sum_loss = sum(sum_loss_in_selected.values())
    total_sel_pct = (total_sel / total_tot * 100.0) if total_tot > 0 else 0.0
    total_flip_pct = (total_flips / total_sel * 100.0) if total_sel > 0 else 0.0
    total_eval_pct = (total_eval / total_tot * 100.0) if total_tot > 0 else 0.0
    total_avg_loss = (total_sum_loss / total_sel) if total_sel > 0 else 0.0
    print('-' * len(header))
    print(
        f"{'TOTAL':<{fn_w}}  {total_sel:>{num_w}d}  {total_tot:>{num_w}d}  {total_sel_pct:>{pct_w}.1f}  "
        f"{total_flips:>{num_w}d}  {total_flip_pct:>{pct_w}.1f}  {total_avg_loss:>{avg_w}.2f}  {total_eval:>{num_w}d}  {total_eval_pct:>{pct_w}.1f}"
    )


def calculate_median_loss(entries: List[Dict[str, Any]]) -> float:
    """Calculate the median completion_difficulty for a list of entries."""
    threshold = 0.05  # Minimum difficulty to consider
    losses = [get_completion_difficulty(entry) for entry in entries if get_completion_difficulty(entry) >= threshold]
    if not losses:
        return 0.0
    return statistics.median(losses)


def print_eval_stat_counts(eval_entries: List[Dict[str, Any]]) -> None:
    """Print the count of each target stat (target.attr) present in the eval split."""
    if not eval_entries:
        print("Eval stat distribution: (empty)")
        return

    by_stat: Dict[str, int] = defaultdict(int)
    for e in eval_entries:
        by_stat[get_target_attr(e)] += 1

    total = len(eval_entries)
    print("Eval stat distribution:")
    # Sort by count desc, then name asc for stable output
    for stat, cnt in sorted(by_stat.items(), key=lambda kv: (-kv[1], kv[0])):
        pct = (cnt / total * 100.0) if total else 0.0
        print(f" - {stat}: {cnt} ({pct:.1f}%)")
    print(f"Total eval entries: {total}")


def plot_difficulty_histogram(
    difficulties: List[float], 
    output_path: Optional[str] = None
) -> None:
    """Plot or save a histogram of completion_difficulty values."""
    filtered: List[float] = []
    for d in difficulties:
        try:
            val = float(d)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            filtered.append(val)
    if not filtered:
        print("No completion_difficulty values to plot.")
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("[ERROR] matplotlib is required for --plot_difficulty_hist. Install it with `pip install matplotlib`.")
        return

    plt.figure(figsize=(10, 6))
    # Bin count derived from data volume with sensible defaults
    bin_count = min(100, max(10, int(math.sqrt(len(filtered))) or 10))
    plt.hist(filtered, bins=bin_count, color="steelblue", edgecolor="black", alpha=0.85)
    plt.title("Completion Difficulty Distribution")
    plt.xlabel("completion_difficulty")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if output_path:
        directory = os.path.dirname(os.path.abspath(output_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(output_path)
        print(f"Wrote histogram to {os.path.abspath(output_path)}")
    else:
        plt.show()

    plt.close()
