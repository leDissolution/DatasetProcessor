"""
Dataset Assembler package for ML training data preparation.
"""

from .entry_utils import (
    get_loss_metrics,
    get_completion_difficulty,
    get_worst_loss,
    get_critical_token,
    get_target_attr,
    is_flip,
    has_no_change_flag,
    count_flips,
    count_no_change,
    get_difficulty_bucket,
    need_attr,
)

from .io_utils import (
    ensure_directories,
    get_files_by_suffix,
    load_jsonl_files,
    write_jsonl_file,
)

from .filtering import (
    filter_entries_by_attr,
    exclude_top_difficulty_percentile,
    select_unique_source_entries,
    remove_eval_contamination,
    dedupe_entries_by_id,
    dedupe_entries_by_identity,
    create_loss_buckets,
    sort_by_difficulty_and_length,
    sort_by_flip_priority,
    extract_synth_flips,
    get_eval_ids,
)

from .batch_engineering import (
    engineer_balanced_batches,
    rebalance_batches_for_requirement,
    engineer_attr_balanced_batches,
)

from .reporting import (
    print_bucket_stats,
    print_top_unique_sources,
    print_zero_loss_sources,
    print_file_contributions,
    print_eval_stat_counts,
    plot_difficulty_histogram,
    calculate_median_loss,
)

from .constants import (
    BASE_PATH,
    BATCH_SIZE,
    REGULAR_BATCHES,
    REGULARIZATION_BATCHES,
    REGULAR_COUNT,
    REGULARIZATION_COUNT,
    TOTAL_STEPS,
    EVAL_BATCH_SIZE,
    EVAL_BATCHES,
    REPLACE_EVAL_THRESHOLD,
    HIGH_LOSS_THRESHOLD,
    MEDIUM_LOSS_MIN,
    LOW_LOSS_MIN,
    RNG_SEED,
)

__all__ = [
    # Entry utilities
    "get_loss_metrics",
    "get_completion_difficulty",
    "get_worst_loss",
    "get_critical_token",
    "get_target_attr",
    "is_flip",
    "has_no_change_flag",
    "count_flips",
    "count_no_change",
    "get_difficulty_bucket",
    "need_attr",
    # IO utilities
    "ensure_directories",
    "get_files_by_suffix",
    "load_jsonl_files",
    "write_jsonl_file",
    # Filtering
    "filter_entries_by_attr",
    "exclude_top_difficulty_percentile",
    "select_unique_source_entries",
    "remove_eval_contamination",
    "dedupe_entries_by_id",
    "dedupe_entries_by_identity",
    "create_loss_buckets",
    "sort_by_difficulty_and_length",
    "sort_by_flip_priority",
    "extract_synth_flips",
    "get_eval_ids",
    # Batch engineering
    "engineer_balanced_batches",
    "rebalance_batches_for_requirement",
    "engineer_attr_balanced_batches",
    # Reporting
    "print_bucket_stats",
    "print_top_unique_sources",
    "print_zero_loss_sources",
    "print_file_contributions",
    "print_eval_stat_counts",
    "plot_difficulty_histogram",
    "calculate_median_loss",
    # Constants
    "BASE_PATH",
    "BATCH_SIZE",
    "REGULAR_BATCHES",
    "REGULARIZATION_BATCHES",
    "REGULAR_COUNT",
    "REGULARIZATION_COUNT",
    "TOTAL_STEPS",
    "EVAL_BATCH_SIZE",
    "EVAL_BATCHES",
    "REPLACE_EVAL_THRESHOLD",
    "HIGH_LOSS_THRESHOLD",
    "MEDIUM_LOSS_MIN",
    "LOW_LOSS_MIN",
    "RNG_SEED",
]
