"""
Dataset Assembler for ML training data preparation.
Processes synthetic and chat data into training/evaluation splits with difficulty-based sampling.

CLI options:
    --base_path PATH          Override the default dataset base path.
    --with_chats / --no-with_chats
                                                     Include Chat data (default: True).
    --with_synthetic / --no-with_synthetic
                                                     Include Synthetic data (default: True).
    --with_stage3 / --no-with_stage3
                                                     Write Stage3 outputs (default: False).
    --exclude_top_percentile X Exclude the top X percent most difficult entries (default: 0).
    --exclude_top_percentile_min_difficulty V
                              Only consider entries with difficulty >= V when applying percentile filtering (default: 0).
    --existing_eval_file PATH Reuse an existing eval JSONL and filter its IDs from training (default: none).
    --big_eval_size N         If > 0, also assemble big_eval.jsonl of size N that includes
                              the regular eval, all synthetic eval entries, and fills the rest
                              50/50 from medium and low difficulty buckets. Big eval does not
                              exclude entries from training.
    --train_file_name NAME    Output filename for the Stage 2 train split (default: train_high.jsonl).
    --eval_file_name NAME     Output filename for the Stage 2 eval split (default: eval.jsonl).
"""

from collections import defaultdict
import os
import json
import random
import statistics
import argparse
from typing import List, Dict, Any, Set, Tuple, Optional, Callable
import math

BASE_PATH = r".\Dataset\Prepared_st2\\"
# The concrete paths are derived in main() to allow --base_path override.

BATCH_SIZE = 24
REGULAR_BATCHES = 1300
REGULARIZATION_BATCHES = 700
REGULAR_COUNT = REGULAR_BATCHES * BATCH_SIZE
REGULARIZATION_COUNT = REGULARIZATION_BATCHES * BATCH_SIZE
TOTAL_STEPS = REGULAR_BATCHES + REGULARIZATION_BATCHES

EVAL_BATCH_SIZE = 18
EVAL_BATCHES = 30
REPLACE_EVAL_THRESHOLD = 0.05

HIGH_LOSS_THRESHOLD = 1
MEDIUM_LOSS_MIN = 0.25
LOW_LOSS_MIN = 0.0

# Deterministic RNG seed for stable shuffles across runs
RNG_SEED = 1337


def get_loss_metrics(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return nested loss_metrics dict or empty."""
    return entry.get('loss_metrics') or {}


def get_completion_difficulty(entry: Dict[str, Any]) -> float:
    lm = get_loss_metrics(entry)
    try:
        return float(lm.get('completion_difficulty') or 0.0)
    except Exception:
        return 0.0


def get_worst_loss(entry: Dict[str, Any]) -> float:
    lm = get_loss_metrics(entry)
    try:
        return float(lm.get('worst_loss') or 0.0)
    except Exception:
        return 0.0


def get_critical_token(entry: Dict[str, Any]) -> Dict[str, Any]:
    lm = get_loss_metrics(entry)
    return lm.get('critical_token') or {}


def get_target_attr(entry: Dict[str, Any]) -> str:
    tgt = entry.get('target') or {}
    return str(tgt.get('attr') or 'unknown')


# Attribute selection helper: customize this predicate to restrict which target attrs are included.
# By default, accept all attributes. You can modify this function or have it read from env/config.
def need_attr(attr: str) -> bool:
    """Return True if this attribute should be included in datasets."""
    blacklist = {"updated"}
    return attr not in blacklist


def filter_entries_by_attr(entries: List[Dict[str, Any]], predicate: Callable[[str], bool] = need_attr) -> List[Dict[str, Any]]:
    """Filter entries by target.attr using a predicate (str) -> bool."""
    out: List[Dict[str, Any]] = []
    for e in entries:
        if predicate(get_target_attr(e)):
            out.append(e)
    return out


def exclude_top_difficulty_percentile(
    entries: List[Dict[str, Any]], percentile: float, min_difficulty: float = 0.0
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


def create_loss_buckets(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], ...]:
    """Create loss buckets based on completion difficulty (nested in loss_metrics)."""
    high_loss = [entry for entry in entries if get_completion_difficulty(entry) > HIGH_LOSS_THRESHOLD]
    medium_loss = [entry for entry in entries if MEDIUM_LOSS_MIN < get_completion_difficulty(entry) <= HIGH_LOSS_THRESHOLD]
    low_loss = [entry for entry in entries if LOW_LOSS_MIN < get_completion_difficulty(entry) <= MEDIUM_LOSS_MIN]
    zero_loss = [entry for entry in entries if get_completion_difficulty(entry) == 0]

    random.shuffle(high_loss)
    random.shuffle(medium_loss)
    random.shuffle(low_loss)

    return high_loss, medium_loss, low_loss, zero_loss


def print_bucket_stats(high_loss: List, medium_loss: List, low_loss: List,
                      zero_loss: List, synth_train: List, train: List) -> None:
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


def get_eval_ids(eval_entries: List[Dict[str, Any]]) -> Set[str]:
    """Extract unique ids from evaluation entries."""
    return {entry['id'] for entry in eval_entries if 'id' in entry}


def select_unique_source_entries(entries: List[Dict[str, Any]], count: int, exclude_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """Select entries ensuring unique ids, up to the specified count."""
    if exclude_ids is None:
        exclude_ids = set()

    seen_ids = exclude_ids.copy()
    selected = []

    for entry in entries:
        id = entry.get('id')
        if id and id not in seen_ids and need_attr(get_target_attr(entry)):
            seen_ids.add(id)
            selected.append(entry)
            if len(selected) >= count:
                break

    return selected


def remove_eval_contamination(train_entries: List[Dict[str, Any]],
                            eval_ids: Set[str]) -> List[Dict[str, Any]]:
    """Remove training entries that share ids with eval entries."""
    return [entry for entry in train_entries if entry.get('id') not in eval_ids]


def sort_by_difficulty_and_length(entries: List[Dict[str, Any]], reverse: bool = True) -> List[Dict[str, Any]]:
    """Sort entries by completion difficulty and prompt length."""
    return sorted(entries, key=lambda x: (get_completion_difficulty(x), len(x.get('prompt', ''))), reverse=reverse)


def print_top_unique_sources(entries: List[Dict[str, Any]], limit: int = 50) -> None:
    """Print top unique source IDs by worst_loss."""
    print(f"Top {limit} unique source IDs by worst_loss:")
    seen_ids = set()
    unique_entries = []

    for entry in sorted(entries, key=lambda x: get_worst_loss(x), reverse=True):
        id = entry.get('id', 'unknown')
        if id not in seen_ids:
            seen_ids.add(id)
            unique_entries.append(entry)
            if len(unique_entries) >= limit:
                break

    for entry in unique_entries:
        id = entry.get('id', 'unknown')
        print(f"Loss: {get_worst_loss(entry):.4f}, Source: {id}, Prompt: {entry.get('prompt','')[:50]}...")


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


def dedupe_entries_by_id(entries: List[Dict[str, Any]], seen_ids: Optional[Set[str]] = None) -> Tuple[List[Dict[str, Any]], Set[str]]:
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


def is_flip(entry: Dict[str, Any]) -> bool:
    """Check if an entry represents a flip (prediction doesn't match actual value)."""
    ct = get_critical_token(entry)
    pred = ct.get('pred_decoded_value')
    actual = ct.get('decoded_value')
    return pred is not None and actual is not None and str(pred) != str(actual)


def count_flips(entries: List[Dict[str, Any]]) -> int:
    """Count the number of flips in a list of entries."""
    return sum(1 for entry in entries if is_flip(entry))


def count_no_change(entries: List[Dict[str, Any]]) -> int:
    """Count entries explicitly marked to keep the original value."""
    sentinel = "!!no_change!!\" />"
    return sum(
        1
        for entry in entries
        if isinstance(entry.get('prompt'), str) and str(entry.get('prompt')).endswith(sentinel)
    )


def has_no_change_flag(entry: Dict[str, Any]) -> bool:
    """Return True when entry is explicitly marked as no_change."""
    sentinel = "!!no_change!!\" />"
    prompt = entry.get('prompt')
    return isinstance(prompt, str) and prompt.endswith(sentinel)


def sort_by_flip_priority(entries: List[Dict[str, Any]], flips_first: bool = True) -> List[Dict[str, Any]]:
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


def extract_synth_flips(synth_train: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract synthetic entries where prediction doesn't match actual value."""
    return [entry for entry in synth_train if is_flip(entry)]


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

    # Calculate the ideal proportion of each category per batch
    total_entries = len(entries)
    category_ratios = {cat: len(pools[cat]) / total_entries if total_entries else 0 for cat in ordered_categories}

    for batch_idx in range(total_batches):
        batch: List[Dict[str, Any]] = []
        remaining_capacity = batch_size

        # Determine actual batch size (last batch may be smaller)
        actual_batch_size = min(batch_size, sum(len(pools[cat]) for cat in ordered_categories))

        # Calculate proportional targets for this batch based on original ratios
        targets: Dict[str, int] = {}
        allocated = 0
        for cat in ordered_categories:
            if cat == ordered_categories[-1]:
                # Last category gets the remainder to avoid rounding errors
                targets[cat] = actual_batch_size - allocated
            else:
                targets[cat] = round(category_ratios[cat] * actual_batch_size)
                allocated += targets[cat]

        # Fill batch according to proportional targets
        for category in ordered_categories:
            pool = pools[category]
            target = min(targets[category], len(pool), remaining_capacity)

            for _ in range(target):
                if not pool or remaining_capacity <= 0:
                    break
                batch.append(pool.pop())
                remaining_capacity -= 1

        # Fill any remaining capacity from whatever is available
        while remaining_capacity > 0:
            picked: Optional[Dict[str, Any]] = None
            for category in ordered_categories:
                pool = pools[category]
                if pool:
                    picked = pool.pop()
                    break
            if picked is None:
                break
            batch.append(picked)
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
    for batch in batches:
        if batch:
            difficulties = [get_completion_difficulty(e) for e in batch]
            batch_mean_difficulties.append(statistics.mean(difficulties))
            batch_median_difficulties.append(statistics.median(difficulties))

    # Compute std of mean and median across batches
    mean_of_batch_means = statistics.mean(batch_mean_difficulties) if batch_mean_difficulties else 0.0
    std_of_batch_means = statistics.stdev(batch_mean_difficulties) if len(batch_mean_difficulties) > 1 else 0.0
    mean_of_batch_medians = statistics.mean(batch_median_difficulties) if batch_median_difficulties else 0.0
    std_of_batch_medians = statistics.stdev(batch_median_difficulties) if len(batch_median_difficulties) > 1 else 0.0

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
    }

    return ordered_entries, metrics


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


def plot_difficulty_histogram(difficulties: List[float], output_path: Optional[str] = None) -> None:
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


def main():
    """Main dataset assembly process."""
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Assemble training/eval datasets from chat and synthetic sources.")
    # Keep option names as requested (underscored);
    # BooleanOptionalAction auto-creates --no-with_* variants.
    parser.add_argument("--base_path", type=str, default=BASE_PATH, help="Base path containing Synthetic, Chat exports, Assembled, Stage3")
    parser.add_argument("--with_chats", action=argparse.BooleanOptionalAction, default=True, help="Include Chat data inputs")
    parser.add_argument("--with_synthetic", action=argparse.BooleanOptionalAction, default=True, help="Include Synthetic data inputs")
    parser.add_argument("--with_stage3", action=argparse.BooleanOptionalAction, default=False, help="Write Stage3 outputs (train.jsonl, eval.jsonl)")
    parser.add_argument("--big_eval_size", type=int, default=0, help="If > 0, also assemble big_eval.jsonl of this size")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=r".\Models\g2b-stage1",
        help="Tokenizer path or model id for AutoTokenizer.from_pretrained",
    )
    parser.add_argument(
        "--exclude_top_percentile",
        type=float,
        default=0.0,
        help="Exclude the top X percent of highest-difficulty entries from each training pool",
    )
    parser.add_argument(
        "--exclude_top_percentile_min_difficulty",
        type=float,
        default=0.0,
        help="Minimum completion_difficulty an entry must have to be eligible for percentile exclusion",
    )
    parser.add_argument(
        "--existing_eval_file",
        type=str,
        default="",
        help="Path (absolute or relative to Assembled/) of an eval JSONL to reuse",
    )
    parser.add_argument(
        "--plot_difficulty_hist",
        action="store_true",
        help="Plot completion_difficulty histogram from loaded training pools and exit",
    )
    parser.add_argument(
        "--difficulty_hist_output",
        type=str,
        default="",
        help="Optional path to save the histogram when --plot_difficulty_hist is used",
    )
    parser.add_argument(
        "--train_file_name",
        type=str,
        default="train_high.jsonl",
        help="File name for the assembled high-priority training split",
    )
    parser.add_argument(
        "--eval_file_name",
        type=str,
        default="eval.jsonl",
        help="File name for the assembled evaluation split",
    )
    args = parser.parse_args()

    # Resolve paths from base_path (normalize to avoid trailing separators issues)
    base_path = os.path.normpath(args.base_path or BASE_PATH)
    CHATS_PATH = os.path.join(base_path, "Chat exports")
    SYNTHETIC_PATH = os.path.join(base_path, "Synthetic")
    ASSEMBLED_PATH = os.path.join(base_path, "Assembled")
    STAGE3_PATH = os.path.join(base_path, "Stage3")

    # Ensure output directories exist (Stage3 optional)
    ensure_directories(ASSEMBLED_PATH)
    if args.with_stage3:
        ensure_directories(STAGE3_PATH)
    random.seed(RNG_SEED)

    existing_eval_path: Optional[str] = None
    if args.existing_eval_file:
        candidate = args.existing_eval_file.strip()
        if candidate:
            if not os.path.isabs(candidate):
                candidate = os.path.join(ASSEMBLED_PATH, candidate)
            candidate = os.path.normpath(candidate)
            if not os.path.isfile(candidate):
                raise FileNotFoundError(f"Existing eval file not found: {candidate}")
            existing_eval_path = candidate

    # Load data files
    train_files = get_files_by_suffix(CHATS_PATH, '.train.jsonl') if args.with_chats else []
    synth_train_files = get_files_by_suffix(SYNTHETIC_PATH, '.train.jsonl') if args.with_synthetic else []
    eval_files = get_files_by_suffix(SYNTHETIC_PATH, '.eval.jsonl') if args.with_synthetic else []
    train = load_jsonl_files(train_files) if train_files else []
    synth_train = load_jsonl_files(synth_train_files) if synth_train_files else []
    eval_data = load_jsonl_files(eval_files) if eval_files else []
    # Preserve the full synthetic eval (unfiltered) for big eval construction
    synth_eval_all_unfiltered = list(eval_data)

    # Write out the entirety of the synthetic eval data (unfiltered)
    if args.with_synthetic:
        write_jsonl_file(os.path.join(ASSEMBLED_PATH, "synthetic_eval_unfiltered.jsonl"), eval_data)

    chat_eval_files = get_files_by_suffix(CHATS_PATH, '.eval.jsonl') if args.with_chats else []
    chat_eval_data = load_jsonl_files(chat_eval_files) if chat_eval_files else []
    if args.with_chats:
        write_jsonl_file(
            os.path.join(ASSEMBLED_PATH, "chat_eval_unfiltered.jsonl"),
            chat_eval_data)

    # Filter entries by attribute after writing unfiltered evals
    train = filter_entries_by_attr(train)
    synth_train = filter_entries_by_attr(synth_train)
    eval_data = filter_entries_by_attr(eval_data)

    if args.plot_difficulty_hist:
        histogram_entries = list(train)
        histogram_entries.extend(synth_train)
        histogram_entries.extend(eval_data)
        difficulties = [get_completion_difficulty(entry) for entry in histogram_entries]
        output_path = args.difficulty_hist_output.strip() or None
        plot_difficulty_histogram(difficulties, output_path)
        return

    if args.exclude_top_percentile > 0:
        pct = max(0.0, min(100.0, args.exclude_top_percentile))
        pct_floor = args.exclude_top_percentile_min_difficulty
        eligible_train = sum(1 for entry in train if get_completion_difficulty(entry) >= pct_floor)
        before_train = len(train)
        train = exclude_top_difficulty_percentile(train, pct, pct_floor)
        print(
            f"Excluded {before_train - len(train)} entries from chat training pool (top {pct}% difficulty over {eligible_train} eligible entries with difficulty >= {pct_floor})"
        )

    use_existing_eval = existing_eval_path is not None
    if use_existing_eval:
        existing_eval_entries = load_jsonl_files([existing_eval_path])
        filtered_existing_eval = filter_entries_by_attr(existing_eval_entries)
        removed_from_existing = len(existing_eval_entries) - len(filtered_existing_eval)
        if removed_from_existing > 0:
            print(
                f"[INFO] Excluded {removed_from_existing} entries from existing eval due to attribute predicate"
            )
        eval_data = filtered_existing_eval
        print(f"Reusing {len(eval_data)} eval entries from {existing_eval_path}")

    high_loss, medium_loss, low_loss, zero_loss = create_loss_buckets(train)

    if not use_existing_eval:
        eval_data = [
            entry for entry in eval_data if get_completion_difficulty(entry) >= REPLACE_EVAL_THRESHOLD
        ]
        existing_eval_ids = get_eval_ids(eval_data)

        #eval_count = (len(eval_data) // EVAL_BATCH_SIZE + EXTRA_EVAL_BATCHES) * EVAL_BATCH_SIZE - len(eval_data)
        eval_count = max(0, EVAL_BATCHES * EVAL_BATCH_SIZE - len(eval_data))
        print(f"Adding {eval_count} entries to eval set")

        # Select unique id entries from medium_loss
        unique_medium_entries = select_unique_source_entries(
            medium_loss, eval_count, existing_eval_ids
        )
        eval_data += unique_medium_entries

        print(
            f"Successfully added {len(unique_medium_entries)} unique entries from medium_loss to eval set"
        )

    eval_ids = get_eval_ids(eval_data)
    if use_existing_eval:
        print(f"Found {len(eval_ids)} unique ids in reused eval set")
    else:
        print(f"Found {len(eval_ids)} unique ids in eval set")

    original_train_count = len(train)
    train = remove_eval_contamination(train, eval_ids)
    print(f"Removed {original_train_count - len(train)} training entries that share ids with eval entries")

    high_loss, medium_loss, low_loss, zero_loss = create_loss_buckets(train)

    # Build optional Big Eval set (does not affect training contamination)
    big_eval: List[Dict[str, Any]] = []
    if args.big_eval_size and args.big_eval_size > 0:
        # Start with the regular eval (already filtered and topped up)
        big_eval.extend(eval_data)
        # Add entire synthetic eval (unfiltered)
        big_eval_ids: Set[str] = set()
        for _be in big_eval:
            _eid = _be.get('id')
            if isinstance(_eid, str):
                big_eval_ids.add(_eid)
        synth_to_add = [e for e in synth_eval_all_unfiltered if not (isinstance(e.get('id'), str) and e.get('id') in big_eval_ids)]
        big_eval.extend(synth_to_add)

        # Fill the remainder 50/50 from medium and low buckets
        remaining = args.big_eval_size - len(big_eval)
        if remaining < 0:
            print(f"[WARN] big_eval_size ({args.big_eval_size}) is smaller than the mandatory portion (regular eval + all synthetic eval = {len(big_eval)}). Writing {len(big_eval)} entries.")
            remaining = 0

        if remaining > 0:
            # Ensure deterministic order from already shuffled buckets
            half_medium = math.ceil(remaining // 4)
            half_low = remaining - half_medium

            # Helper to pick unique-by-id entries not already in big_eval
            def pick_from(bucket: List[Dict[str, Any]], k: int, seen_ids: Set[str]) -> List[Dict[str, Any]]:
                picked: List[Dict[str, Any]] = []
                for e in bucket:
                    eid = e.get('id')
                    if isinstance(eid, str) and eid in seen_ids:
                        continue
                    # respect attr filter for buckets (they already are filtered)
                    picked.append(e)
                    if isinstance(eid, str):
                        seen_ids.add(eid)
                    if len(picked) >= k:
                        break
                return picked

            big_eval_ids = set(big_eval_ids)  # copy for mutation
            add_medium = pick_from(medium_loss, half_medium, big_eval_ids)
            add_low = pick_from(low_loss, half_low, big_eval_ids)

            # If one bucket underfilled, top up from the other
            short = remaining - (len(add_medium) + len(add_low))
            if short > 0:
                # Try medium first, then low
                add_more_m = pick_from([e for e in medium_loss if e not in add_medium], short, big_eval_ids)
                short -= len(add_more_m)
                add_more_l: List[Dict[str, Any]] = []
                if short > 0:
                    add_more_l = pick_from([e for e in low_loss if e not in add_low], short, big_eval_ids)
                big_eval.extend(add_medium + add_low + add_more_m + add_more_l)
            else:
                big_eval.extend(add_medium + add_low)

    eval_data = sorted(eval_data, key=lambda x: len(x.get('prompt', '')))
    print(f"No-change entries in eval: {count_no_change(eval_data)}")

    synth_train = synth_train * 2

    if args.with_stage3:
        write_jsonl_file(os.path.join(STAGE3_PATH, "train.jsonl"), train + synth_train)
        write_jsonl_file(os.path.join(STAGE3_PATH, "eval.jsonl"), eval_data)
        if args.big_eval_size and args.big_eval_size > 0:
            write_jsonl_file(os.path.join(STAGE3_PATH, "big_eval.jsonl"), big_eval)

    # Print distribution of target stats inside the eval set
    print_eval_stat_counts(eval_data)

    print_bucket_stats(high_loss, medium_loss, low_loss, zero_loss, synth_train, train)

    # Prioritize high and medium difficulty entries without replicating identical objects
    high_loss_oids = {id(entry) for entry in high_loss}
    medium_loss_oids = {id(entry) for entry in medium_loss}
    unique_train = dedupe_entries_by_identity(train)

    def priority_key(entry: Dict[str, Any]) -> Tuple[int, float, int]:
        oid = id(entry)
        priority = 2 if oid in high_loss_oids else 1 if oid in medium_loss_oids else 0
        return (
            priority,
            get_completion_difficulty(entry),
            len(entry.get('prompt', '')),
        )

    prioritized_pool = sorted(unique_train, key=priority_key, reverse=True)

    # Create Stage 2 dataset (high-quality subset)
    count = max(0, min(REGULAR_COUNT - len(synth_train), len(prioritized_pool)))
    train_high = prioritized_pool[:count]

    if len(train_high) > 0:
        print(f"Minimum loss: {min(get_completion_difficulty(entry) for entry in train_high)}")

    train_high += synth_train

    # Create low-priority training data, prioritizing flips
    train_remaining = prioritized_pool[count:]
    train_remaining_flips_first = sort_by_flip_priority(train_remaining, flips_first=True)

    # For regularization, take from the end (non-flips prioritized)
    train_remaining_nonflips_first = sort_by_flip_priority(train_remaining, flips_first=True)
    regularization_entries = train_remaining_nonflips_first[:REGULARIZATION_COUNT]

    train_high += regularization_entries

    # Update train_low to be the remaining entries after regularization selection
    used_for_regularization = set(id(entry) for entry in regularization_entries)
    train_low = [entry for entry in train_remaining if id(entry) not in used_for_regularization]
    print(f"Selected {len(regularization_entries)} regularization entries (flips: {count_flips(regularization_entries)})")
    print(f"Remaining in train_low: {len(train_low)} (flips: {count_flips(train_low)})")

    # Create stat buckets for analysis
    buckets = defaultdict(list)
    for entry in train_high:
        buckets[get_target_attr(entry)].append(entry)
        if (get_target_attr(entry) == "unknown"):
            print(f"Entry with unknown attr: ID={entry.get('id','unknown')}, prompt={entry.get('prompt','')[:50]}...")

    # Totals per stat across available pool (train + synth_train) for y%
    totals_by_stat: Dict[str, int] = defaultdict(int)
    for e in (train + synth_train):
        totals_by_stat[get_target_attr(e)] += 1

    # Total selected entries count for share-of-selected percentage
    total_selected = len(train_high)

    for stat, entries in buckets.items():
        flip_count = count_flips(entries)
        tot = totals_by_stat.get(stat, 0)
        pct = ((len(entries) / tot) * 100.0) if tot > 0 else 0.0
        pct_of_selected = ((len(entries) / total_selected) * 100.0) if total_selected > 0 else 0.0
        print(f"{stat}: {len(entries)} (flips: {flip_count}, {pct:.1f}% pool, {pct_of_selected:.1f}% selected)")

    synthetic_entry_ids = {id(entry) for entry in synth_train}
    train_high, batch_metrics = engineer_balanced_batches(
        train_high,
        BATCH_SIZE,
        synthetic_entry_ids,
        RNG_SEED,
    )

    total_batches = batch_metrics["total_batches"]
    if total_batches:
        synthetic_missing = batch_metrics["synthetic_batches_missing"]
        synthetic_available = batch_metrics["synthetic_available_batches"]
        synthetic_covered = batch_metrics["synthetic_batches_covered"]
        if synthetic_available < total_batches:
            print(
                f"[WARN] Synthetic supply covers only {synthetic_available} of {total_batches} batches; "
                f"{total_batches - synthetic_available} batches lack synthetic entries."
            )
        elif synthetic_missing:
            print(
                f"[WARN] Could not rebalance synthetic entries for {synthetic_missing} batches despite available supply."
            )

        no_change_missing = batch_metrics["no_change_batches_missing"]
        no_change_available = batch_metrics["no_change_available_batches"]
        no_change_covered = batch_metrics["no_change_batches_covered"]
        if no_change_available < total_batches:
            print(
                f"[WARN] Only {no_change_available} of {total_batches} batches can include no_change entries; "
                f"{total_batches - no_change_available} batches lack them."
            )
        elif no_change_missing:
            print(
                f"[WARN] Could not rebalance no_change entries for {no_change_missing} batches despite available supply."
            )

        print(
            f"Balanced batches: synthetic coverage {synthetic_covered}/{total_batches}, no_change coverage {no_change_covered}/{total_batches}."
        )

        # Print batch difficulty distribution stats
        print(
            f"Batch difficulty stats: mean={batch_metrics['mean_of_batch_means']:.3f} (std={batch_metrics['std_of_batch_means']:.3f}), "
            f"median={batch_metrics['mean_of_batch_medians']:.3f} (std={batch_metrics['std_of_batch_medians']:.3f})"
        )

        # Print overall dataset difficulty stats
        all_difficulties = [get_completion_difficulty(e) for e in train_high]
        if all_difficulties:
            overall_mean = statistics.mean(all_difficulties)
            overall_median = statistics.median(all_difficulties)
            overall_std = statistics.stdev(all_difficulties) if len(all_difficulties) > 1 else 0.0
            print(f"Overall dataset difficulty: mean={overall_mean:.3f}, median={overall_median:.3f}, std={overall_std:.3f}")

    print(f"No-change entries in train_high: {count_no_change(train_high)}")

    # Write assembled datasets
    train_output_path = os.path.join(ASSEMBLED_PATH, args.train_file_name)
    eval_output_path = os.path.join(ASSEMBLED_PATH, args.eval_file_name)

    write_jsonl_file(eval_output_path, eval_data)
    if args.big_eval_size and args.big_eval_size > 0:
        write_jsonl_file(os.path.join(ASSEMBLED_PATH, "big_eval.jsonl"), big_eval)
    write_jsonl_file(train_output_path, train_high)
    write_jsonl_file(os.path.join(ASSEMBLED_PATH, "train_low.jsonl"), train_low)

    # Analysis and reporting
    print_top_unique_sources(train_high)
    print_zero_loss_sources(zero_loss)

    # Median loss calculation for the entire train set
    median_loss = calculate_median_loss(train_high + train_low)
    print(f"Median loss over the entire train set: {median_loss:.4f}")

    # Per-file contribution of selected training entries relative to total available per file
    print_file_contributions(train_high, train + synth_train, eval_entries=eval_data)

    # Extract and write synthetic flips
    synth_flips = extract_synth_flips(synth_train)
    write_jsonl_file(os.path.join(ASSEMBLED_PATH, "synth_flips.jsonl"), synth_flips, ensure_ascii=False)
    print(f"Wrote {len(synth_flips)} flips")

    # Print top-10 longest tokenized prompts in train_high, unique by source_id (pick longest per source_id)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        # Track total completion token usage for the finalized training split
        completion_token_lengths: List[int] = []
        missing_completion_entries = 0
        completion_texts: List[str] = []
        for entry in train_high:
            completion_text = entry.get("expected_value")
            if not isinstance(completion_text, str) or not completion_text:
                completion_text = entry.get("completion") or entry.get("response") or ""
            if not completion_text:
                missing_completion_entries += 1
                continue
            completion_texts.append(completion_text)

        if completion_texts:
            try:
                batch = tokenizer(
                    completion_texts,
                    add_special_tokens=False,
                    return_length=True,
                    padding=False,
                    truncation=False,
                )
                lengths = batch.get("length")
                if isinstance(lengths, list):
                    completion_token_lengths.extend(int(v) for v in lengths)
                elif lengths is not None:
                    completion_token_lengths.extend(int(v) for v in list(lengths))
                else:
                    input_ids = batch.get("input_ids")
                    if isinstance(input_ids, list):
                        completion_token_lengths.extend(len(ids) for ids in input_ids if isinstance(ids, list))
            except Exception:
                # Fallback to per-entry tokenization if batching fails
                for text in completion_texts:
                    try:
                        completion_token_lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
                    except Exception:
                        pass

        total_completion_tokens = sum(completion_token_lengths)
        if completion_token_lengths:
            avg_completion_tokens = statistics.mean(completion_token_lengths)
            max_completion_tokens = max(completion_token_lengths)
            print(
                f"Completion tokens (train_high): total={total_completion_tokens}, mean={avg_completion_tokens:.1f}, max={max_completion_tokens}"
            )
        else:
            print("Completion tokens (train_high): total=0")

        if missing_completion_entries:
            print(f"[INFO] Missing completion text in {missing_completion_entries} train_high entries")
        # Map from source_id to (prompt_length, entry)
        source_id_to_entry = {}
        for entry in train_high:
            source_id = entry.get('source_id', '')
            prompt = entry.get("prompt", "")
            if not prompt:
                continue
            # Only keep the entry with the longest prompt (string length) for each source_id
            if source_id not in source_id_to_entry or len(prompt) > len(source_id_to_entry[source_id][1].get("prompt", "")):
                source_id_to_entry[source_id] = (prompt, entry)
        # Now tokenize and sort by tokenized length
        tokenized_lengths: List[Tuple[int, Dict[str, Any]]] = []
        prompt_entries = list(source_id_to_entry.values())
        if prompt_entries:
            prompts = [p for p, _ in prompt_entries]
            entries_for_prompts = [e for _, e in prompt_entries]
            try:
                batch = tokenizer(
                    prompts,
                    add_special_tokens=True,
                    return_length=True,
                    padding=False,
                    truncation=False,
                )
                lengths = batch.get("length")
                prompt_lengths: List[int]
                if isinstance(lengths, list):
                    prompt_lengths = [int(v) for v in lengths]
                elif lengths is not None:
                    prompt_lengths = [int(v) for v in list(lengths)]
                else:
                    input_ids = batch.get("input_ids")
                    if isinstance(input_ids, list):
                        prompt_lengths = [len(ids) for ids in input_ids if isinstance(ids, list)]
                    else:
                        prompt_lengths = []
                if len(prompt_lengths) != len(entries_for_prompts):
                    raise ValueError("Tokenized length count mismatch for prompts")
                tokenized_lengths = [(prompt_lengths[i], entries_for_prompts[i]) for i in range(len(entries_for_prompts))]
            except Exception:
                # Fallback to per-entry tokenization if batching fails
                fallback_lengths: List[Tuple[int, Dict[str, Any]]] = []
                for prompt, entry in prompt_entries:
                    try:
                        fallback_lengths.append((len(tokenizer.encode(prompt)), entry))
                    except Exception:
                        pass
                tokenized_lengths = fallback_lengths
        tokenized_lengths.sort(reverse=True, key=lambda x: x[0])
        # Length stats
        lengths = [t for t, _ in tokenized_lengths]
        if lengths:
            try:
                mean_len = statistics.mean(lengths)
            except Exception:
                mean_len = 0.0
            # Median (p50)
            p50 = statistics.median(lengths)
            # Percentiles helper (linear interpolation between closest ranks)
            def percentile(data, p):
                if not data:
                    return 0
                xs = sorted(data)
                k = (len(xs)-1) * (p/100.0)
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return xs[int(k)]
                d0 = xs[f] * (c - k)
                d1 = xs[c] * (k - f)
                return d0 + d1
            p90 = percentile(lengths, 90)
            p95 = percentile(lengths, 95)
            print(f"Tokenized length stats (unique by source_id in train_high): mean={mean_len:.1f}, p50={p50:.1f}, p90={p90:.1f}, p95={p95:.1f}")
        print("Top 10 longest tokenized prompts in train_high (unique by source_id):")
        for i, (tok_len, entry) in enumerate(tokenized_lengths[:10], 1):
            print(f"{i:2d}. Tokens: {tok_len:4d} | ID: {entry.get('id','unknown')} | Prompt: {entry.get('prompt','')[:80]}...")
    except Exception as e:
        print(f"[WARN] Could not print top-10 longest tokenized prompts: {e}")


if __name__ == "__main__":
    main()