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
    --batch_size N            Batch size for training data assembly (default: 16).
    --epochs N                Number of epochs for training data (default: 1). Each epoch
                              reuses high/medium entries in different order but samples a
                              new low set.
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
    --eval_correct_pct PCT    Target percent correct within no_change and non-no_change groups (default: 25).
    --eval_no_change_pct PCT  Target percent of per-attr eval entries with no_change (default: 50).
"""

from collections import defaultdict
import os
import random
import statistics
import argparse
import math
from typing import List, Dict, Any, Set, Tuple, Optional

from assembler import (
    # Constants
    BASE_PATH,
    BATCH_SIZE,
    REGULAR_BATCHES,
    REGULARIZATION_BATCHES,
    EVAL_BATCH_SIZE,
    EVAL_BATCHES,
    REPLACE_EVAL_THRESHOLD,
    RNG_SEED,
    # Entry utilities
    get_completion_difficulty,
    get_target_attr,
    count_flips,
    count_no_change,
    need_attr,
    # IO utilities
    ensure_directories,
    get_files_by_suffix,
    load_jsonl_files,
    write_jsonl_file,
    # Filtering
    filter_entries_by_attr,
    exclude_top_difficulty_percentile,
    remove_eval_contamination,
    dedupe_entries_by_id,
    dedupe_entries_by_identity,
    create_loss_buckets,
    sort_by_flip_priority,
    extract_synth_flips,
    get_eval_ids,
    # Batch engineering
    engineer_balanced_batches,
    # Reporting
    print_bucket_stats,
    print_top_unique_sources,
    print_zero_loss_sources,
    print_file_contributions,
    print_eval_stat_counts,
    plot_difficulty_histogram,
    calculate_median_loss,
)
from assembler.entry_utils import has_no_change_flag, is_flip


def build_big_eval(
    eval_data: List[Dict[str, Any]],
    synth_eval_all_unfiltered: List[Dict[str, Any]],
    medium_loss: List[Dict[str, Any]],
    low_loss: List[Dict[str, Any]],
    big_eval_size: int,
) -> List[Dict[str, Any]]:
    """Build optional Big Eval set (does not affect training contamination)."""
    big_eval: List[Dict[str, Any]] = []
    
    # Start with the regular eval (already filtered and topped up)
    big_eval.extend(eval_data)
    
    # Add entire synthetic eval (unfiltered)
    big_eval_ids: Set[str] = set()
    for _be in big_eval:
        _eid = _be.get('id')
        if isinstance(_eid, str):
            big_eval_ids.add(_eid)
    synth_to_add = [
        e for e in synth_eval_all_unfiltered 
        if not (isinstance(e.get('id'), str) and e.get('id') in big_eval_ids)
    ]
    big_eval.extend(synth_to_add)

    # Fill the remainder 50/50 from medium and low buckets
    remaining = big_eval_size - len(big_eval)
    if remaining < 0:
        print(
            f"[WARN] big_eval_size ({big_eval_size}) is smaller than the mandatory portion "
            f"(regular eval + all synthetic eval = {len(big_eval)}). Writing {len(big_eval)} entries."
        )
        remaining = 0

    if remaining > 0:
        half_medium = math.ceil(remaining // 4)
        half_low = remaining - half_medium

        def pick_from(bucket: List[Dict[str, Any]], k: int, seen_ids: Set[str]) -> List[Dict[str, Any]]:
            picked: List[Dict[str, Any]] = []
            for e in bucket:
                eid = e.get('id')
                if isinstance(eid, str) and eid in seen_ids:
                    continue
                picked.append(e)
                if isinstance(eid, str):
                    seen_ids.add(eid)
                if len(picked) >= k:
                    break
            return picked

        big_eval_ids = set(big_eval_ids)
        add_medium = pick_from(medium_loss, half_medium, big_eval_ids)
        add_low = pick_from(low_loss, half_low, big_eval_ids)

        # If one bucket underfilled, top up from the other
        short = remaining - (len(add_medium) + len(add_low))
        if short > 0:
            add_more_m = pick_from([e for e in medium_loss if e not in add_medium], short, big_eval_ids)
            short -= len(add_more_m)
            add_more_l: List[Dict[str, Any]] = []
            if short > 0:
                add_more_l = pick_from([e for e in low_loss if e not in add_low], short, big_eval_ids)
            big_eval.extend(add_medium + add_low + add_more_m + add_more_l)
        else:
            big_eval.extend(add_medium + add_low)

    return big_eval


def _allocate_attr_targets(
    attr_buckets: Dict[str, List[Dict[str, Any]]],
    target_total: int,
) -> Dict[str, int]:
    """Allocate target counts per attr proportional to bucket availability."""
    if target_total <= 0 or not attr_buckets:
        return {}

    total_available = sum(len(v) for v in attr_buckets.values())
    if total_available == 0:
        return {}

    target_total = min(target_total, total_available)
    allocations: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []

    for attr, entries in attr_buckets.items():
        share = len(entries) / total_available
        raw_target = share * target_total
        base = int(math.floor(raw_target))
        allocations[attr] = base
        remainders.append((raw_target - base, attr))

    # Distribute leftover slots by largest remainder first
    remaining = target_total - sum(allocations.values())
    for _, attr in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocations[attr] < len(attr_buckets[attr]):
            allocations[attr] += 1
            remaining -= 1

    # If still short due to caps, top up from attrs with supply left
    if remaining > 0:
        attrs_by_slack = sorted(
            attr_buckets.keys(),
            key=lambda a: len(attr_buckets[a]) - allocations.get(a, 0),
            reverse=True,
        )
        for attr in attrs_by_slack:
            if remaining <= 0:
                break
            slack = len(attr_buckets[attr]) - allocations.get(attr, 0)
            if slack <= 0:
                continue
            add = min(slack, remaining)
            allocations[attr] = allocations.get(attr, 0) + add
            remaining -= add

    return allocations


def _sample_attr_entries(
    entries: List[Dict[str, Any]],
    target_count: int,
    correct_ratio: float,
    no_change_ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample entries for a single attr with no_change and correct ratios."""
    if target_count <= 0 or not entries:
        return [], list(entries)

    pool = list(entries)
    rng.shuffle(pool)

    no_change_entries = [e for e in pool if has_no_change_flag(e)]
    non_no_change_entries = [e for e in pool if not has_no_change_flag(e)]

    target_no_change = min(len(no_change_entries), round(target_count * no_change_ratio))
    target_non_no_change = target_count - target_no_change
    if target_non_no_change > len(non_no_change_entries):
        short = target_non_no_change - len(non_no_change_entries)
        target_non_no_change = len(non_no_change_entries)
        target_no_change = min(len(no_change_entries), target_no_change + short)

    def _sample_subset(
        subset: List[Dict[str, Any]],
        subset_target: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if subset_target <= 0 or not subset:
            return [], list(subset)

        subset_pool = list(subset)
        rng.shuffle(subset_pool)
        correct_entries = [e for e in subset_pool if not is_flip(e)]
        flip_entries = [e for e in subset_pool if is_flip(e)]

        target_correct = min(len(correct_entries), round(subset_target * correct_ratio))
        selected: List[Dict[str, Any]] = []

        selected_correct = correct_entries[:target_correct]
        selected.extend(selected_correct)

        remaining_slots = subset_target - len(selected)
        take_flips = min(remaining_slots, len(flip_entries))
        selected.extend(flip_entries[:take_flips])
        remaining_slots -= take_flips

        take_more_correct = 0
        if remaining_slots > 0:
            take_more_correct = min(remaining_slots, len(correct_entries) - len(selected_correct))
            if take_more_correct > 0:
                start = len(selected_correct)
                selected.extend(correct_entries[start:start + take_more_correct])
                remaining_slots -= take_more_correct

        leftover_pool = (
            correct_entries[len(selected_correct) + take_more_correct:]
            + flip_entries[take_flips:]
        )
        return selected, leftover_pool

    selected_no_change, leftover_nc = _sample_subset(no_change_entries, target_no_change)
    selected_non_no_change, leftover_non = _sample_subset(non_no_change_entries, target_non_no_change)

    selected = selected_no_change + selected_non_no_change

    remaining_slots = target_count - len(selected)
    if remaining_slots > 0:
        if len(selected_no_change) < target_no_change and leftover_nc:
            take_nc = min(remaining_slots, target_no_change - len(selected_no_change), len(leftover_nc))
            selected.extend(leftover_nc[:take_nc])
            leftover_nc = leftover_nc[take_nc:]
            remaining_slots -= take_nc

        if remaining_slots > 0 and leftover_non:
            take_non = min(remaining_slots, len(leftover_non))
            selected.extend(leftover_non[:take_non])
            leftover_non = leftover_non[take_non:]
            remaining_slots -= take_non

        if remaining_slots > 0 and leftover_nc:
            take_nc = min(remaining_slots, len(leftover_nc))
            selected.extend(leftover_nc[:take_nc])
            leftover_nc = leftover_nc[take_nc:]
            remaining_slots -= take_nc

    leftover_pool = leftover_nc + leftover_non
    return selected, leftover_pool


def build_attr_balanced_eval(
    medium_entries: List[Dict[str, Any]],
    target_total: int,
    correct_pct: float,
    no_change_pct: float,
    seed_entries: Optional[List[Dict[str, Any]]] = None,
    rng_seed: int = RNG_SEED,
) -> List[Dict[str, Any]]:
    """Build an eval set from the medium bucket with attr-level controls."""
    seed_entries = seed_entries or []
    seed_deduped, seed_ids = dedupe_entries_by_id(seed_entries)

    rng = random.Random(rng_seed)

    correct_ratio = max(0.0, min(1.0, correct_pct / 100.0))
    no_change_ratio = max(0.0, min(1.0, no_change_pct / 100.0))

    # Remove seed ids from the selection pool to avoid duplicates
    medium_pool = [e for e in medium_entries if e.get('id') not in seed_ids]
    if not medium_pool or target_total <= len(seed_deduped):
        return seed_deduped[:target_total]

    selection_target = max(0, min(target_total - len(seed_deduped), len(medium_pool)))

    attr_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in medium_pool:
        attr_buckets[get_target_attr(entry)].append(entry)

    # Shuffle each bucket deterministically
    for bucket_entries in attr_buckets.values():
        rng.shuffle(bucket_entries)

    allocations = _allocate_attr_targets(attr_buckets, selection_target)

    selected: List[Dict[str, Any]] = []
    leftover: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set(seed_ids)

    for attr, target_count in allocations.items():
        chosen, remainder = _sample_attr_entries(
            attr_buckets[attr], target_count, correct_ratio, no_change_ratio, rng
        )
        for entry in chosen:
            eid = entry.get('id')
            if eid and eid in seen_ids:
                continue
            if eid:
                seen_ids.add(eid)
            selected.append(entry)
        leftover.extend(remainder)

    remaining_needed = selection_target - len(selected)
    if remaining_needed > 0 and leftover:
        rng.shuffle(leftover)
        for entry in leftover:
            if remaining_needed <= 0:
                break
            eid = entry.get('id')
            if eid and eid in seen_ids:
                continue
            if eid:
                seen_ids.add(eid)
            selected.append(entry)
            remaining_needed -= 1

    final_eval, _ = dedupe_entries_by_id(seed_deduped + selected)

    if len(final_eval) < target_total:
        extra_needed = target_total - len(final_eval)
        filler_pool = [e for e in leftover if e.get('id') not in seen_ids]
        rng.shuffle(filler_pool)
        final_eval.extend(filler_pool[:extra_needed])

    return final_eval[:target_total]


def print_eval_attr_stats(
    eval_entries: List[Dict[str, Any]],
    correct_pct_target: float,
    no_change_pct_target: float,
) -> None:
    """Print per-attr stats for the eval set."""
    total = len(eval_entries)
    if total == 0:
        print("[INFO] Eval set is empty")
        return

    attr_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "no_change": 0,
            "nc_total": 0,
            "nc_correct": 0,
            "non_nc_total": 0,
            "non_nc_correct": 0,
        }
    )
    for entry in eval_entries:
        attr = get_target_attr(entry)
        stats = attr_stats[attr]
        stats["total"] += 1
        is_correct = not is_flip(entry)
        is_no_change = has_no_change_flag(entry)
        if is_correct:
            stats["correct"] += 1
        if is_no_change:
            stats["no_change"] += 1
            stats["nc_total"] += 1
            if is_correct:
                stats["nc_correct"] += 1
        else:
            stats["non_nc_total"] += 1
            if is_correct:
                stats["non_nc_correct"] += 1

    print(
        f"Eval attr targets: correct~{correct_pct_target:.1f}% (within no_change and non_no_change), "
        f"no_change~{no_change_pct_target:.1f}%"
    )
    for attr, stats in sorted(attr_stats.items(), key=lambda kv: kv[1]["total"], reverse=True):
        tot = stats["total"]
        correct_pct = (stats["correct"] / tot) * 100.0 if tot else 0.0
        no_change_pct = (stats["no_change"] / tot) * 100.0 if tot else 0.0
        nc_correct_pct = (stats["nc_correct"] / stats["nc_total"]) * 100.0 if stats["nc_total"] else 0.0
        non_nc_correct_pct = (
            (stats["non_nc_correct"] / stats["non_nc_total"]) * 100.0 if stats["non_nc_total"] else 0.0
        )
        share_pct = (tot / total) * 100.0
        print(
            f"  {attr}: n={tot}, correct={correct_pct:.1f}%, no_change={no_change_pct:.1f}%, "
            f"correct_nc={nc_correct_pct:.1f}%, correct_non_nc={non_nc_correct_pct:.1f}%, "
            f"share={share_pct:.1f}%"
        )


def build_training_epochs(
    high_medium_entries: List[Dict[str, Any]],
    low_pool_for_epochs: List[Dict[str, Any]],
    synth_train: List[Dict[str, Any]],
    low_count_per_epoch: int,
    num_epochs: int,
) -> List[Dict[str, Any]]:
    """Build training data across multiple epochs with varied sampling."""
    train_high_all_epochs: List[Dict[str, Any]] = []
    
    print(
        f"Building {num_epochs} epoch(s): {len(high_medium_entries)} high+medium, "
        f"{len(synth_train)} synthetic, {low_count_per_epoch} low per epoch"
    )
    
    for epoch_idx in range(num_epochs):
        epoch_rng = random.Random(RNG_SEED + epoch_idx)
        
        # Shuffle high+medium entries for this epoch
        epoch_high_medium = list(high_medium_entries)
        epoch_rng.shuffle(epoch_high_medium)
        
        # Shuffle and sample low entries for this epoch
        epoch_low_pool = list(low_pool_for_epochs)
        epoch_rng.shuffle(epoch_low_pool)
        epoch_low = epoch_low_pool[:low_count_per_epoch]
        
        # Shuffle synthetic for this epoch
        epoch_synth = list(synth_train)
        epoch_rng.shuffle(epoch_synth)
        
        # Combine for this epoch
        epoch_entries = epoch_high_medium + epoch_low + epoch_synth
        train_high_all_epochs.extend(epoch_entries)
        
        if epoch_idx == 0 and (len(epoch_high_medium) + len(epoch_low) > 0):
            min_loss = min(get_completion_difficulty(e) for e in (epoch_high_medium + epoch_low))
            print(f"Epoch {epoch_idx + 1}: Minimum loss: {min_loss}")
    
    return train_high_all_epochs


def print_stat_buckets(
    train_high: List[Dict[str, Any]],
    train: List[Dict[str, Any]],
    synth_train: List[Dict[str, Any]],
) -> None:
    """Print stat distribution for training entries."""
    buckets = defaultdict(list)
    for entry in train_high:
        buckets[get_target_attr(entry)].append(entry)
        if get_target_attr(entry) == "unknown":
            print(
                f"Entry with unknown attr: ID={entry.get('id','unknown')}, "
                f"prompt={entry.get('prompt','')[:50]}..."
            )

    # Totals per stat across available pool
    totals_by_stat: Dict[str, int] = defaultdict(int)
    for e in (train + synth_train):
        totals_by_stat[get_target_attr(e)] += 1

    total_selected = len(train_high)

    for stat, entries in buckets.items():
        flip_count = count_flips(entries)
        tot = totals_by_stat.get(stat, 0)
        pct = ((len(entries) / tot) * 100.0) if tot > 0 else 0.0
        pct_of_selected = ((len(entries) / total_selected) * 100.0) if total_selected > 0 else 0.0
        print(f"{stat}: {len(entries)} (flips: {flip_count}, {pct:.1f}% pool, {pct_of_selected:.1f}% selected)")


def print_batch_metrics(batch_metrics: Dict[str, Any], train_high: List[Dict[str, Any]]) -> None:
    """Print batch engineering metrics and warnings."""
    total_batches = batch_metrics["total_batches"]
    if not total_batches:
        return

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
        f"Balanced batches: synthetic coverage {synthetic_covered}/{total_batches}, "
        f"no_change coverage {no_change_covered}/{total_batches}."
    )

    # Print batch difficulty distribution stats
    print(
        f"Batch difficulty stats: mean={batch_metrics['mean_of_batch_means']:.3f} "
        f"(std={batch_metrics['std_of_batch_means']:.3f}), "
        f"median={batch_metrics['mean_of_batch_medians']:.3f} "
        f"(std={batch_metrics['std_of_batch_medians']:.3f})"
    )

    # Print overall dataset difficulty stats
    all_difficulties = [get_completion_difficulty(e) for e in train_high]
    if all_difficulties:
        overall_mean = statistics.mean(all_difficulties)
        overall_median = statistics.median(all_difficulties)
        overall_std = statistics.stdev(all_difficulties) if len(all_difficulties) > 1 else 0.0
        print(
            f"Overall dataset difficulty: mean={overall_mean:.3f}, "
            f"median={overall_median:.3f}, std={overall_std:.3f}"
        )
        print(
            f"Min batch mean difficulty: {batch_metrics['min_batch_mean']:.3f}, "
            f"Max batch mean difficulty: {batch_metrics['max_batch_mean']:.3f}"
        )

    # Print attr diversity stats
    print(
        f"Batch attr diversity: mean={batch_metrics['mean_unique_attrs']:.1f} unique attrs/batch, "
        f"min={batch_metrics['min_unique_attrs']}, max={batch_metrics['max_unique_attrs']}"
    )


def print_tokenization_stats(train_high: List[Dict[str, Any]], tokenizer_path: str) -> None:
    """Print tokenization statistics for training entries."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Track completion token usage
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
                f"Completion tokens (train_high): total={total_completion_tokens}, "
                f"mean={avg_completion_tokens:.1f}, max={max_completion_tokens}"
            )
        else:
            print("Completion tokens (train_high): total=0")

        if missing_completion_entries:
            print(f"[INFO] Missing completion text in {missing_completion_entries} train_high entries")
        
        # Print top-10 longest tokenized prompts
        _print_longest_prompts(train_high, tokenizer)
        
    except Exception as e:
        print(f"[WARN] Could not print tokenization stats: {e}")


def _print_longest_prompts(train_high: List[Dict[str, Any]], tokenizer) -> None:
    """Print top 10 longest tokenized prompts (unique by source_id)."""
    source_id_to_entry = {}
    for entry in train_high:
        source_id = entry.get('source_id', '')
        prompt = entry.get("prompt", "")
        if not prompt:
            continue
        if source_id not in source_id_to_entry or len(prompt) > len(source_id_to_entry[source_id][1].get("prompt", "")):
            source_id_to_entry[source_id] = (prompt, entry)
    
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
            for prompt, entry in prompt_entries:
                try:
                    tokenized_lengths.append((len(tokenizer.encode(prompt)), entry))
                except Exception:
                    pass
    
    tokenized_lengths.sort(reverse=True, key=lambda x: x[0])
    
    # Length stats
    lengths = [t for t, _ in tokenized_lengths]
    if lengths:
        mean_len = statistics.mean(lengths)
        p50 = statistics.median(lengths)
        
        def percentile(data, p):
            if not data:
                return 0
            xs = sorted(data)
            k = (len(xs)-1) * (p/100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return xs[int(k)]
            return xs[f] * (c - k) + xs[c] * (k - f)
        
        p90 = percentile(lengths, 90)
        p95 = percentile(lengths, 95)
        print(
            f"Tokenized length stats (unique by source_id in train_high): "
            f"mean={mean_len:.1f}, p50={p50:.1f}, p90={p90:.1f}, p95={p95:.1f}"
        )
    
    print("Top 10 longest tokenized prompts in train_high (unique by source_id):")
    for i, (tok_len, entry) in enumerate(tokenized_lengths[:10], 1):
        print(f"{i:2d}. Tokens: {tok_len:4d} | ID: {entry.get('id','unknown')} | Prompt: {entry.get('prompt','')[:80]}...")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Assemble training/eval datasets from chat and synthetic sources."
    )
    parser.add_argument(
        "--base_path", type=str, default=BASE_PATH, 
        help="Base path containing Synthetic, Chat exports, Assembled, Stage3"
    )
    parser.add_argument(
        "--with_chats", action=argparse.BooleanOptionalAction, default=True, 
        help="Include Chat data inputs"
    )
    parser.add_argument(
        "--with_synthetic", action=argparse.BooleanOptionalAction, default=True, 
        help="Include Synthetic data inputs"
    )
    parser.add_argument(
        "--with_stage3", action=argparse.BooleanOptionalAction, default=False, 
        help="Write Stage3 outputs (train.jsonl, eval.jsonl)"
    )
    parser.add_argument(
        "--big_eval_size", type=int, default=0, 
        help="If > 0, also assemble big_eval.jsonl of this size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, 
        help="Batch size for training data assembly"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, 
        help="Number of epochs for training data"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=r".\Models\g2b-stage1",
        help="Tokenizer path or model id for AutoTokenizer.from_pretrained",
    )
    parser.add_argument(
        "--exclude_top_percentile", type=float, default=0.0,
        help="Exclude the top X percent of highest-difficulty entries from each training pool",
    )
    parser.add_argument(
        "--exclude_top_percentile_min_difficulty", type=float, default=0.0,
        help="Minimum completion_difficulty an entry must have to be eligible for percentile exclusion",
    )
    parser.add_argument(
        "--existing_eval_file", type=str, default="",
        help="Path (absolute or relative to Assembled/) of an eval JSONL to reuse",
    )
    parser.add_argument(
        "--plot_difficulty_hist", action="store_true",
        help="Plot completion_difficulty histogram from loaded training pools and exit",
    )
    parser.add_argument(
        "--difficulty_hist_output", type=str, default="",
        help="Optional path to save the histogram when --plot_difficulty_hist is used",
    )
    parser.add_argument(
        "--train_file_name", type=str, default="train_high.jsonl",
        help="File name for the assembled high-priority training split",
    )
    parser.add_argument(
        "--eval_file_name", type=str, default="eval.jsonl",
        help="File name for the assembled evaluation split",
    )
    parser.add_argument(
        "--eval_correct_pct", type=float, default=25.0,
        help="Target percent correct within no_change and non-no_change groups",
    )
    parser.add_argument(
        "--eval_no_change_pct", type=float, default=50.0,
        help="Target percent of per-attr eval entries that are explicit no_change",
    )
    return parser.parse_args()


def main():
    """Main dataset assembly process."""
    args = parse_args()

    # Resolve paths from base_path
    base_path = os.path.normpath(args.base_path or BASE_PATH)
    CHATS_PATH = os.path.join(base_path, "Chat exports")
    SYNTHETIC_PATH = os.path.join(base_path, "Synthetic")
    ASSEMBLED_PATH = os.path.join(base_path, "Assembled")
    STAGE3_PATH = os.path.join(base_path, "Stage3")

    # Ensure output directories exist
    ensure_directories(ASSEMBLED_PATH)
    if args.with_stage3:
        ensure_directories(STAGE3_PATH)
    random.seed(RNG_SEED)

    # Resolve existing eval file path if provided
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
    synth_eval_all_unfiltered = list(eval_data)

    # Write unfiltered synthetic eval
    if args.with_synthetic:
        write_jsonl_file(os.path.join(ASSEMBLED_PATH, "synthetic_eval_unfiltered.jsonl"), eval_data)

    # Load and write chat eval
    chat_eval_files = get_files_by_suffix(CHATS_PATH, '.eval.jsonl') if args.with_chats else []
    chat_eval_data = load_jsonl_files(chat_eval_files) if chat_eval_files else []
    if args.with_chats:
        write_jsonl_file(os.path.join(ASSEMBLED_PATH, "chat_eval_unfiltered.jsonl"), chat_eval_data)

    # Filter entries by attribute
    train = filter_entries_by_attr(train)
    synth_train = filter_entries_by_attr(synth_train)
    eval_data = filter_entries_by_attr(eval_data)

    # Handle histogram plotting mode
    if args.plot_difficulty_hist:
        histogram_entries = list(train) + synth_train + eval_data
        difficulties = [get_completion_difficulty(entry) for entry in histogram_entries]
        output_path = args.difficulty_hist_output.strip() or None
        plot_difficulty_histogram(difficulties, output_path)
        return

    # Apply top percentile exclusion
    if args.exclude_top_percentile > 0:
        pct = max(0.0, min(100.0, args.exclude_top_percentile))
        pct_floor = args.exclude_top_percentile_min_difficulty
        eligible_train = sum(1 for entry in train if get_completion_difficulty(entry) >= pct_floor)
        before_train = len(train)
        train = exclude_top_difficulty_percentile(train, pct, pct_floor)
        print(
            f"Excluded {before_train - len(train)} entries from chat training pool "
            f"(top {pct}% difficulty over {eligible_train} eligible entries with difficulty >= {pct_floor})"
        )

    # Handle existing eval file
    use_existing_eval = existing_eval_path is not None
    if use_existing_eval:
        existing_eval_entries = load_jsonl_files([existing_eval_path])
        filtered_existing_eval = filter_entries_by_attr(existing_eval_entries)
        removed_from_existing = len(existing_eval_entries) - len(filtered_existing_eval)
        if removed_from_existing > 0:
            print(f"[INFO] Excluded {removed_from_existing} entries from existing eval due to attribute predicate")
        eval_data = filtered_existing_eval
        print(f"Reusing {len(eval_data)} eval entries from {existing_eval_path}")

    high_loss, medium_loss, low_loss, zero_loss = create_loss_buckets(train)

    eval_correct_pct = max(0.0, min(100.0, args.eval_correct_pct))
    eval_no_change_pct = max(0.0, min(100.0, args.eval_no_change_pct))

    # Build eval set if not using existing
    if not use_existing_eval:
        eval_data = [entry for entry in eval_data if get_completion_difficulty(entry) >= REPLACE_EVAL_THRESHOLD]
        total_eval_target = EVAL_BATCHES * EVAL_BATCH_SIZE

        eval_data = build_attr_balanced_eval(
            medium_loss,
            total_eval_target,
            eval_correct_pct,
            eval_no_change_pct,
            seed_entries=eval_data,
            rng_seed=RNG_SEED,
        )
        print(
            f"Eval set built from medium bucket: size={len(eval_data)} target={total_eval_target} "
            f"correct~{eval_correct_pct:.1f}% no_change~{eval_no_change_pct:.1f}%"
        )
    else:
        total_eval_target = EVAL_BATCHES * EVAL_BATCH_SIZE
        if len(eval_data) > total_eval_target:
            eval_data = eval_data[:total_eval_target]
            print(f"[INFO] Truncated reused eval to {total_eval_target} entries to match eval target")

    eval_ids = get_eval_ids(eval_data)
    print(f"Found {len(eval_ids)} unique ids in {'reused' if use_existing_eval else ''} eval set")

    # Remove eval contamination from training
    original_train_count = len(train)
    train = remove_eval_contamination(train, eval_ids)
    print(f"Removed {original_train_count - len(train)} training entries that share ids with eval entries")

    high_loss, medium_loss, low_loss, zero_loss = create_loss_buckets(train)

    # Build big eval if requested
    big_eval: List[Dict[str, Any]] = []
    if args.big_eval_size and args.big_eval_size > 0:
        big_eval = build_big_eval(eval_data, synth_eval_all_unfiltered, medium_loss, low_loss, args.big_eval_size)

    eval_data = sorted(eval_data, key=lambda x: len(x.get('prompt', '')))
    print(f"No-change entries in eval: {count_no_change(eval_data)}")
    print_eval_attr_stats(eval_data, eval_correct_pct, eval_no_change_pct)

    synth_train = synth_train

    # Write Stage3 outputs if requested
    if args.with_stage3:
        write_jsonl_file(os.path.join(STAGE3_PATH, "train.jsonl"), train + synth_train)
        write_jsonl_file(os.path.join(STAGE3_PATH, "eval.jsonl"), eval_data)
        if args.big_eval_size and args.big_eval_size > 0:
            write_jsonl_file(os.path.join(STAGE3_PATH, "big_eval.jsonl"), big_eval)

    print_eval_stat_counts(eval_data)
    print_bucket_stats(high_loss, medium_loss, low_loss, zero_loss, synth_train, train)

    # Prioritize high and medium difficulty entries
    high_loss_oids = {id(entry) for entry in high_loss}
    medium_loss_oids = {id(entry) for entry in medium_loss}
    unique_train = dedupe_entries_by_identity(train)

    def priority_key(entry: Dict[str, Any]) -> Tuple[int, float, int]:
        oid = id(entry)
        priority = 2 if oid in high_loss_oids else 1 if oid in medium_loss_oids else 0
        return (priority, get_completion_difficulty(entry), len(entry.get('prompt', '')))

    prioritized_pool = sorted(unique_train, key=priority_key, reverse=True)

    # Create Stage 2 dataset
    regular_count = REGULAR_BATCHES * args.batch_size
    count = max(0, min(regular_count - len(synth_train), len(prioritized_pool)))
    
    # Separate high+medium from low entries for epoch handling
    high_medium_entries = [e for e in prioritized_pool[:count] if id(e) in high_loss_oids or id(e) in medium_loss_oids]
    low_entries_in_regular = [e for e in prioritized_pool[:count] if id(e) not in high_loss_oids and id(e) not in medium_loss_oids]
    
    # Create low-priority training data pool
    train_remaining = prioritized_pool[count:]
    train_remaining_nonflips_first = sort_by_flip_priority(train_remaining, flips_first=True)
    regularization_count = REGULARIZATION_BATCHES * args.batch_size
    
    low_pool_for_epochs = low_entries_in_regular + train_remaining_nonflips_first
    
    # Build epochs
    num_epochs = max(1, args.epochs)
    low_count_per_epoch = len(low_entries_in_regular) + regularization_count
    
    train_high = build_training_epochs(
        high_medium_entries, low_pool_for_epochs, synth_train, low_count_per_epoch, num_epochs
    )
    
    # For regularization tracking
    first_epoch_rng = random.Random(RNG_SEED)
    first_epoch_low_pool = list(low_pool_for_epochs)
    first_epoch_rng.shuffle(first_epoch_low_pool)
    regularization_entries = first_epoch_low_pool[:regularization_count]

    used_for_regularization = set(id(entry) for entry in regularization_entries)
    train_low = [entry for entry in train_remaining if id(entry) not in used_for_regularization]
    
    print(f"Selected {len(regularization_entries)} regularization entries (flips: {count_flips(regularization_entries)})")
    print(f"Remaining in train_low: {len(train_low)} (flips: {count_flips(train_low)})") 
    print(f"Total train_high entries across {num_epochs} epoch(s): {len(train_high)}")

    print_stat_buckets(train_high, train, synth_train)

    # Engineer balanced batches
    synthetic_entry_ids = {id(entry) for entry in synth_train}
    train_high, batch_metrics = engineer_balanced_batches(
        train_high, args.batch_size, synthetic_entry_ids, RNG_SEED
    )

    print_batch_metrics(batch_metrics, train_high)
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

    median_loss = calculate_median_loss(train_high + train_low)
    print(f"Median loss over the entire train set: {median_loss:.4f}")

    print_file_contributions(train_high, train + synth_train, eval_entries=eval_data)

    # Extract and write synthetic flips
    synth_flips = extract_synth_flips(synth_train)
    write_jsonl_file(os.path.join(ASSEMBLED_PATH, "synth_flips.jsonl"), synth_flips, ensure_ascii=False)
    print(f"Wrote {len(synth_flips)} flips")

    # Print tokenization stats
    print_tokenization_stats(train_high, args.tokenizer_path)


if __name__ == "__main__":
    main()
