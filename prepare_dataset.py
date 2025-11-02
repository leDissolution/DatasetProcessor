from __future__ import annotations

import os
import argparse
from typing import List, Optional
import random

from Entities.entity import StatsEntity
from dataset_processor import ingest_entities
from pipeline import MetaUpdatedPass, MirrorEntitiesPass, PipelineContext, ReplaceUnchangedWithNoopPass, StatUnwinderPass, NameRandomizerPass, DuplicateAugmentationPass
from Entities.registry import default_registry
import loss_engine
from loss_engine import validate_devices


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def list_txt_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    txt_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, f))
    return txt_files


def _parse_bool(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def main(argv: Optional[List[str]] = None) -> None:
    default_base_path = rf".\Dataset\\"
    default_model_name = getattr(loss_engine, "MODEL_NAME", rf".\Models\slm2-stage2")

    parser = argparse.ArgumentParser(description="Prepare StatsSuite datasets")
    parser.add_argument("--base_path", type=str, default=default_base_path, help="Base dataset path (default: current script value)")
    parser.add_argument("--out_path", type=str, default=None, help="Output folder; if omitted, uses <base_path>/Prepared_st2")
    parser.add_argument("--with_chats", type=str, default="true", help="Include chat exports (true/false)")
    parser.add_argument("--with_synthetic", type=str, default="true", help="Include synthetic data (true/false)")
    parser.add_argument("--with_loss", type=str, default="true", help="Compute loss metrics (true/false)")
    parser.add_argument("--with_json", type=str, default="false", help="Also include the raw datapoint as nested JSON (true/false)")
    parser.add_argument("--model_name", type=str, default=default_model_name, help="HF model path/name for loss engine (default: current in loss_engine)")
    parser.add_argument("--clean", type=str, default="false", help="Clean output directory before processing (true/false)")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated list of GPU device indices to use for loss computation (e.g. 0,1). If omitted uses defaults in loss_engine. Invalid devices are skipped with a warning.")
    parser.add_argument("--stats", type=str, default=None, help="Comma-separated stat attribute names to include (case-insensitive). Default processes all stats.")

    args = parser.parse_args(argv)

    base_path = args.base_path
    out_root = args.out_path if args.out_path else os.path.join(base_path, "Prepared")
    ensure_dirs(out_root)

    if _parse_bool(args.clean, False):
        print(f"Cleaning output directory: {out_root}")
        for root, dirs, files in os.walk(out_root, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    with_chats = _parse_bool(args.with_chats, True)
    with_synthetic = _parse_bool(args.with_synthetic, True)
    with_loss = _parse_bool(args.with_loss, True)
    with_json = _parse_bool(args.with_json, False)

    stats_filter: Optional[List[str]] = None
    if isinstance(args.stats, str):
        raw_stats = args.stats.strip()
        if raw_stats:
            parts = [p.strip().lower() for p in raw_stats.replace(";", ",").split(",") if p.strip()]
            if parts:
                stats_filter = parts
                print(f"Restricting processing to stats: {', '.join(stats_filter)}")

    user_devices: Optional[List[int]] = None
    if args.devices is not None:
        raw = str(args.devices).strip()
        if raw:
            parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
            parsed: List[int] = []
            for p in parts:
                try:
                    parsed.append(int(p))
                except ValueError:
                    print(f"[DEVICES] Ignoring non-integer device entry '{p}'.")
            user_devices = parsed
        else:
            user_devices = []

    filtered_devices = validate_devices(user_devices) if user_devices is not None else None
    if with_loss and user_devices is not None and not filtered_devices:
        raise SystemExit("No requested GPU devices are available; cannot compute loss metrics.")

    try:
        if isinstance(args.model_name, str) and args.model_name:
            loss_engine.MODEL_NAME = args.model_name
    except Exception:
        pass

    bypass_cache = False
    bypass_file_cache = True

    tok_per_batch = 10000

    total = 0

    if with_chats:
        chat_in = os.path.join(base_path, "Chat exports")
        chat_out = os.path.join(out_root, "Chat exports")
        ensure_dirs(chat_out)

        files = list_txt_files(chat_in)
        print(f"Parsing {len(files)} chat file(s) from: {chat_in}")
        ctx = PipelineContext(rng=random.Random(42), registry=default_registry(), split="train")
        passes = [
            #MetaUpdatedPass(),
            StatUnwinderPass(),
            DuplicateAugmentationPass(),
            NameRandomizerPass(0.05),
            MirrorEntitiesPass(base_cls=StatsEntity),
            ReplaceUnchangedWithNoopPass(),
        ]
        eval_passes = [
            StatUnwinderPass(),
            ReplaceUnchangedWithNoopPass(),
        ]
        for fp in files:
            name, _ = os.path.splitext(os.path.basename(fp))
            try:
                out_train = os.path.join(chat_out, f"{name}.train.jsonl")
                print(f"Processing chat file: {fp} -> {out_train}")
                ingest_entities(
                    fp,
                    out_train,
                    with_loss=with_loss,
                    **({"with_json": with_json}),
                    batch_size=60,
                    tok_per_batch=tok_per_batch,
                    devices=filtered_devices,
                    pipeline_passes=passes,
                    pipeline_ctx=ctx,
                    cache_read=(not bypass_cache),
                    file_cache_read=(not bypass_file_cache),
                    cache_write=True,
                    debug=False,
                    allowed_stats=stats_filter,
                )

                out_eval = os.path.join(chat_out, f"{name}.eval.jsonl")
                print(f"Processing chat file: {fp} -> {out_eval}")
                ingest_entities(
                    fp,
                    out_eval,
                    with_loss=with_loss,
                    **({"with_json": with_json}),
                    batch_size=60,
                    tok_per_batch=tok_per_batch,
                    devices=filtered_devices,
                    pipeline_passes=eval_passes,
                    pipeline_ctx=ctx,
                    cache_read=(not bypass_cache),
                    file_cache_read=(not bypass_file_cache),
                    cache_write=True,
                    allowed_stats=stats_filter,
                )
                total += 1
            except Exception as e:
                print(f"Error processing {fp}: {e}")

    if with_synthetic:
        synth_in = os.path.join(base_path, "Synthetic")
        synth_out = os.path.join(out_root, "Synthetic")
        ensure_dirs(synth_out)

        files = list_txt_files(synth_in)
        print(f"Parsing {len(files)} synthetic file(s) from: {synth_in}")
        if files:
            ctx = PipelineContext(rng=random.Random(1337), registry=default_registry(), split="train")
            passes = [
                DuplicateAugmentationPass(),
                StatUnwinderPass(),
                MirrorEntitiesPass(base_cls=StatsEntity),
                ReplaceUnchangedWithNoopPass(),
            ]
            for fp in files:
                name, _ = os.path.splitext(os.path.basename(fp))
                out_train = os.path.join(synth_out, f"{name}.jsonl")
                print(f"Processing synthetic file: {fp} -> {out_train}")
                try:
                    ingest_entities(
                        fp,
                        out_train,
                        with_loss=with_loss,
                        **({"with_json": with_json}),
                        batch_size=60,
                        tok_per_batch=tok_per_batch,
                        devices=filtered_devices,
                        pipeline_passes=passes,
                        pipeline_ctx=ctx,
                        cache_read=(not bypass_cache),
                        file_cache_read=(not bypass_file_cache),
                        cache_write=True,
                        allowed_stats=stats_filter,
                    )
                    total += 1
                except Exception as e:
                    print(f"Error processing {fp}: {e}")

    print(f"Completed. Processed {total} file(s).")


if __name__ == "__main__":
    main()
