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
    --big_eval_size N         If > 0, also assemble big_eval.jsonl of size N that includes
                              the regular eval, all synthetic eval entries, and fills the rest
                              50/50 from medium and low difficulty buckets. Big eval does not
                              exclude entries from training.
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
REQUIRED_COUNT = 1400 * BATCH_SIZE
REGULARIZATION_COUNT = 600 * BATCH_SIZE

EVAL_BATCH_SIZE = 18
EVAL_BATCHES = 30
REPLACE_EVAL_THRESHOLD = 0.05

HIGH_LOSS_THRESHOLD = 3.0
MEDIUM_LOSS_MIN = 0.4
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

    print(f"High loss: {len(high_loss)} (flips: {high_flips})")
    print(f"Medium loss: {len(medium_loss)} (flips: {medium_flips})")
    print(f"Low loss: {len(low_loss)} (flips: {low_flips})")
    print(f"Zero loss: {len(zero_loss)} (flips: {zero_flips})")
    print(f"Synth train: {len(synth_train)} (flips: {synth_flips})")
    print(f"Total: {len(train)}")


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
    """Print unique zero loss source IDs."""
    if len(zero_loss) > 0:
        print(f"Zero loss entries: {len(zero_loss)}")
        zero_loss_ids = {entry['id'] for entry in zero_loss if 'id' in entry}
        for id in zero_loss_ids:
            print(f"{id}")


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


def is_flip(entry: Dict[str, Any]) -> bool:
    """Check if an entry represents a flip (prediction doesn't match actual value)."""
    ct = get_critical_token(entry)
    pred = ct.get('pred_decoded_value')
    actual = ct.get('decoded_value')
    return pred is not None and actual is not None and str(pred) != str(actual)


def count_flips(entries: List[Dict[str, Any]]) -> int:
    """Count the number of flips in a list of entries."""
    return sum(1 for entry in entries if is_flip(entry))


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

    high_loss, medium_loss, low_loss, zero_loss = create_loss_buckets(train)

    eval_data = [entry for entry in eval_data if get_completion_difficulty(entry) >= REPLACE_EVAL_THRESHOLD]
    existing_eval_ids = get_eval_ids(eval_data)

    #eval_count = (len(eval_data) // EVAL_BATCH_SIZE + EXTRA_EVAL_BATCHES) * EVAL_BATCH_SIZE - len(eval_data)
    eval_count = max(0, EVAL_BATCHES * EVAL_BATCH_SIZE - len(eval_data))
    print(f"Adding {eval_count} entries to eval set")

    # Select unique id entries from medium_loss
    unique_medium_entries = select_unique_source_entries(medium_loss, eval_count, existing_eval_ids)
    eval_data += unique_medium_entries

    print(f"Successfully added {len(unique_medium_entries)} unique entries from medium_loss to eval set")

    eval_ids = get_eval_ids(eval_data)
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

    synth_train = synth_train * 2

    if args.with_stage3:
        write_jsonl_file(os.path.join(STAGE3_PATH, "train.jsonl"), train + synth_train)
        write_jsonl_file(os.path.join(STAGE3_PATH, "eval.jsonl"), eval_data)
        if args.big_eval_size and args.big_eval_size > 0:
            write_jsonl_file(os.path.join(STAGE3_PATH, "big_eval.jsonl"), big_eval)

    # Print distribution of target stats inside the eval set
    print_eval_stat_counts(eval_data)

    print_bucket_stats(high_loss, medium_loss, low_loss, zero_loss, synth_train, train)

    train += high_loss * 2 + medium_loss

    # Create Stage 2 dataset (high-quality subset)
    count = REQUIRED_COUNT - len(synth_train)
    train_high = sort_by_difficulty_and_length(train)[:count]

    if len(train_high) > 0:
        print(f"Minimum loss: {min(get_completion_difficulty(entry) for entry in train_high)}")

    train_high += synth_train

    # Create low-priority training data, prioritizing flips
    train_remaining = sort_by_difficulty_and_length(train)[count:]
    train_remaining_flips_first = sort_by_flip_priority(train_remaining, flips_first=True)

    # For regularization, take from the end (non-flips prioritized)
    train_remaining_nonflips_first = sort_by_flip_priority(train_remaining, flips_first=False)
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

    random.shuffle(train_high)

    # Write assembled datasets
    write_jsonl_file(os.path.join(ASSEMBLED_PATH, "eval.jsonl"), eval_data)
    if args.big_eval_size and args.big_eval_size > 0:
        write_jsonl_file(os.path.join(ASSEMBLED_PATH, "big_eval.jsonl"), big_eval)
    write_jsonl_file(os.path.join(ASSEMBLED_PATH, "train_high.jsonl"), train_high)
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
        tokenizer = AutoTokenizer.from_pretrained(r".\Models\g2b-stage1")
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
        tokenized_lengths = []
        for prompt, entry in source_id_to_entry.values():
            tokens = tokenizer.encode(prompt)
            tokenized_lengths.append((len(tokens), entry))
        tokenized_lengths.sort(reverse=True, key=lambda x: x[0])
        print("Top 10 longest tokenized prompts in train_high (unique by source_id):")
        for i, (tok_len, entry) in enumerate(tokenized_lengths[:10], 1):
            print(f"{i:2d}. Tokens: {tok_len:4d} | ID: {entry.get('id','unknown')} | Prompt: {entry.get('prompt','')[:80]}...")
    except Exception as e:
        print(f"[WARN] Could not print top-10 longest tokenized prompts: {e}")


if __name__ == "__main__":
    main()