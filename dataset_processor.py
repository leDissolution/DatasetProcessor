import orjson
import os
import random
from typing import Optional, List, Dict, Any, Tuple
from entity_parser import parse_text
from datapoint import Datapoint, LossMetrics
from loss_engine import compute_losses_for_datapoints
try:
    from pipeline import run_pipeline, PipelineContext
except Exception:
    run_pipeline = None
    PipelineContext = object
from processing_cache import (
    ProcessingCache,
    compute_file_hash,
    compute_run_version,
    compute_datapoint_key,
    remap_cached_entries_source,
)
from Entities.registry import default_registry
from types import SimpleNamespace

def ingest_entities(
    path: str,
    new_path: str,
    with_loss: bool = True,
    with_json: bool = False,
    batch_size: int = 35,
    tok_per_batch: int = 26624,
    pipeline_passes: Optional[list] = None,
    pipeline_ctx: Optional[object] = None,
    # Cache behavior controls:
    cache_read: Optional[bool] = None,
    file_cache_read: Optional[bool] = None,
    cache_write: Optional[bool] = None,
    debug: bool = False,
) -> None:
    """Parallel ingestion using the new entity parser with optional pipeline processing.

    - Parses entities into Datapoints
    - Optionally runs a pipeline of passes over the datapoints
    - Optionally computes loss metrics (GPU parallel) and attaches to each Datapoint
    - Writes JSONL with prompt, source_id, target and loss metrics.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file {path} not found.")
        return

    # Determine cache behavior (defaults + env overrides + explicit args)
    env_force = str(os.environ.get("STATSUITE_FORCE_REPROCESS", "")).strip().lower() in {"1", "true", "yes", "y"}
    env_skip_read = str(os.environ.get("STATSUITE_SKIP_CACHE_READ", "")).strip().lower() in {"1", "true", "yes", "y"}
    env_disable_write = str(os.environ.get("STATSUITE_DISABLE_CACHE_WRITE", "")).strip().lower() in {"1", "true", "yes", "y"}
    env_skip_file_read = str(os.environ.get("STATSUITE_SKIP_FILE_CACHE_READ", "")).strip().lower() in {"1", "true", "yes", "y"}

    if cache_read is None:
        # If forcing reprocess or skip-read is set, don't read from cache; otherwise read.
        cache_read = not (env_force or env_skip_read)
    if cache_write is None:
        # Default to writing; allow disabling via env.
        cache_write = not env_disable_write
    # Allow overriding file-level cache read independently; default to same as cache_read
    if file_cache_read is None:
        # Respect dedicated file-cache skip if set; otherwise mirror dp-level read flag
        file_cache_read = (cache_read is True) and (not env_skip_file_read) or (cache_read is False and False) or (cache_read is None and (not env_skip_file_read))

    cache = ProcessingCache()
    file_key = os.path.abspath(path)
    file_hash = compute_file_hash(text)
    version = compute_run_version(pipeline_passes, with_loss, with_json)

    cached_file = cache.get_file(file_key, file_hash, version) if file_cache_read else None
    if cached_file is not None:
        try:
            print(f"[CACHE] Full-file cache hit: {path} entries={len(cached_file)} version={version[:8]}")
        except Exception:
            pass
        if len(cached_file) > 0:
            try:
                with open(new_path, "w", encoding='utf-8') as out:
                    lines = [orjson.dumps(entry) for entry in cached_file]
                    out.write('\n'.join([line.decode('utf-8') for line in lines]))
            except Exception as e:
                print(f"Error writing output file {new_path} from cache: {e}")
            finally:
                cache.close()
            return
        else:
            # Avoid sticky state where an empty file-level cache prevents reprocessing
            try:
                print(f"[CACHE] Ignoring empty full-file cache for '{path}'; will reprocess.")
            except Exception:
                pass

    warnings: list[str] = []
    def _warn(msg: str) -> None:
        warnings.append(msg)

    dps: list[Datapoint] = parse_text(text, on_warning=_warn, source_name=path)

    all_entries: List[Dict[str, Any]] = []
    dp_keys: List[str] = []
    cached_slices: Dict[int, List[Dict[str, Any]]] = {}
    misses: List[Tuple[int, Datapoint, str]] = []  # (position, dp, dp_key)

    for idx, dp in enumerate(dps):
        dp_key = compute_datapoint_key(dp)
        dp_keys.append(dp_key)
        cached = cache.get_dp(dp_key, version) if cache_read else None
        if cached is not None:
            cached_slices[idx] = remap_cached_entries_source(cached, dp.source_id)
        else:
            misses.append((idx, dp, dp_key))

    try:
        print(
            f"[CACHE] DP summary: file='{path}' total={len(dps)} from_cache={len(cached_slices)} reprocess={len(misses)} read_dp={cache_read} read_file={file_cache_read} write={cache_write}"
        )
    except Exception:
        pass

    new_slices: Dict[int, List[Dict[str, Any]]] = {}
    # Track success/failure to avoid writing bad empty caches
    total_outs = 0
    total_success = 0
    if misses:
        processed_pairs: List[Tuple[int, str, List[Datapoint]]] = []
        if pipeline_passes and run_pipeline is not None:
            for idx, dp, dp_key in misses:
                rng_seed = int(dp_key[:8], 16)
                base_registry = getattr(pipeline_ctx, "registry", default_registry()) if pipeline_ctx else default_registry()
                split_val = getattr(pipeline_ctx, "split", None) if pipeline_ctx else None
                ctx_local = SimpleNamespace(rng=random.Random(rng_seed), registry=base_registry, split=split_val)
                try:
                    outs = run_pipeline([dp], pipeline_passes, ctx=ctx_local)  # type: ignore[arg-type]
                except Exception as e:
                    print(f"Error running pipeline for {path} (dp idx {idx}): {e}")
                    outs = [dp]
                processed_pairs.append((idx, dp_key, outs))
        else:
            for idx, dp, dp_key in misses:
                processed_pairs.append((idx, dp_key, [dp]))

        flat_dps: List[Tuple[int, str, Datapoint]] = []
        for idx, dp_key, outs in processed_pairs:
            for out_dp in outs:
                flat_dps.append((idx, dp_key, out_dp))

        # Fault-tolerant loss computation: isolate failing datapoints and skip caching them
        if with_loss and flat_dps:
            def compute_metrics_safe(dps: List[Datapoint], tpb: int) -> List[Optional[Dict[str, Any]]]:
                # Returns list of metrics dict or None on failure per dp
                if not dps:
                    return []
                try:
                    res = compute_losses_for_datapoints(dps, batch_size=batch_size, debug_print=debug, debug_samples=batch_size, max_tokens_per_batch=tpb)
                    return list(res)
                except Exception as e:
                    if (tpb // 2) < 500:
                        return [None]
                    res = compute_losses_for_datapoints(dps, batch_size=batch_size, debug_print=debug, debug_samples=batch_size, max_tokens_per_batch=tpb // 2)
                    return list(res)

            only_dps = [t[2] for t in flat_dps]
            metrics_list_optional = compute_metrics_safe(only_dps, tpb=tok_per_batch)
            # Assign metrics where available
            for (pos, _key, out_dp), m in zip(flat_dps, metrics_list_optional):
                if m is not None:
                    out_dp.loss_metrics = LossMetrics(
                        completion_difficulty=m.get("completion_difficulty"),
                        mean_loss=m.get("mean_loss"),
                        worst_loss=m.get("worst_loss"),
                        critical_token=m.get("critical_token"),
                    )
                else:
                    out_dp.loss_metrics = None

        total_entries = 0
        total_failed = 0
        for idx, dp_key, outs in processed_pairs:
            if with_loss:
                successful_outs = [od for od in outs if od.loss_metrics is not None]
                # Aggregate totals for cache write decision
                total_outs += len(outs)
                success_count = len(successful_outs)
                total_success += success_count
                failed = success_count < len(outs)
                if failed:
                    total_failed += len(outs) - success_count
                    total_entries += len(outs)
                entries_for_dp = []
                for od in successful_outs:
                    out = od.to_json()
                    if with_json:
                        try:
                            out["datapoint"] = od.to_raw_json()
                        except Exception:
                            pass
                    entries_for_dp.append(out)
                new_slices[idx] = entries_for_dp
                if cache_write and not failed:
                    try:
                        cache.set_dp(dp_key, version, entries_for_dp)
                    except Exception:
                        pass
            else:
                entries_for_dp = []
                for od in outs:
                    out = od.to_json()
                    if with_json:
                        try:
                            out["datapoint"] = od.to_raw_json()
                        except Exception:
                            pass
                    entries_for_dp.append(out)
                new_slices[idx] = entries_for_dp
                if cache_write:
                    try:
                        cache.set_dp(dp_key, version, entries_for_dp)
                    except Exception:
                        pass

        print(f"[LOSS] Failed {total_failed} out of {total_entries}")

    for i in range(len(dps)):
        if i in cached_slices:
            all_entries.extend(cached_slices[i])
        elif i in new_slices:
            all_entries.extend(new_slices[i])
        else:
            base = dps[i].to_json()
            if with_json:
                try:
                    base["datapoint"] = dps[i].to_raw_json()
                except Exception:
                    pass
            all_entries.append(base)

    # Entry-level reuse/regeneration summary (post-pipeline)
    try:
        entries_from_cache = sum(len(v) for v in cached_slices.values())
        entries_new = sum(len(v) for v in new_slices.values())
        print(
            f"[CACHE] Entries summary: file='{path}' reused={entries_from_cache} regenerated={entries_new} total={len(all_entries)}"
        )
    except Exception:
        pass
    try:
        with open(new_path, "w", encoding='utf-8') as out:
            lines = [orjson.dumps(entry) for entry in all_entries]
            out.write('\n'.join([line.decode('utf-8') for line in lines]))
            if cache_write:
                # Don't persist an empty file cache (prevents sticky empty-cache bug)
                # Also skip caching when all loss computations failed (no successful outputs)
                all_loss_failed = bool(with_loss and total_outs > 0 and total_success == 0)
                should_write_file_cache = len(all_entries) > 0 and not all_loss_failed
                if not should_write_file_cache:
                    try:
                        reason = "all loss failed" if all_loss_failed else "empty entries"
                        print(f"[CACHE] Skipping file cache write for '{path}' due to {reason}.")
                    except Exception:
                        pass
                else:
                    try:
                        cache.set_file(file_key, file_hash, version, all_entries)
                    except Exception:
                        pass
    except Exception as e:
        print(f"Error writing output file {new_path}: {e}")
        return
    finally:
        cache.close()

    if warnings:
        print(f"ingest_entities completed with {len(warnings)} warnings (showing up to 5):")
        for w in warnings[:5]:
            print(f" - {w}")
