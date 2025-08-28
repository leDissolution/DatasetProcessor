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


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def list_txt_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.txt')]


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
    # Defaults matching current behavior
    default_base_path = rf".\Dataset\\"
    default_model_name = getattr(loss_engine, "MODEL_NAME", rf".\Models\slm2-stage2")

    parser = argparse.ArgumentParser(description="Prepare StatsSuite datasets")
    parser.add_argument("--base_path", type=str, default=default_base_path, help="Base dataset path (default: current script value)")
    parser.add_argument("--out_path", type=str, default=None, help="Output folder; if omitted, uses <base_path>/Prepared_st2")
    parser.add_argument("--with_chats", type=str, default="true", help="Include chat exports (true/false)")
    parser.add_argument("--with_synthetic", type=str, default="true", help="Include synthetic data (true/false)")
    parser.add_argument("--with_loss", type=str, default="true", help="Compute loss metrics (true/false)")
    parser.add_argument("--model_name", type=str, default=default_model_name, help="HF model path/name for loss engine (default: current in loss_engine)")

    args = parser.parse_args(argv)

    base_path = args.base_path
    out_root = args.out_path if args.out_path else os.path.join(base_path, "Prepared")
    ensure_dirs(out_root)

    with_chats = _parse_bool(args.with_chats, True)
    with_synthetic = _parse_bool(args.with_synthetic, True)
    with_loss = _parse_bool(args.with_loss, True)

    # Allow overriding the loss model used by the loss engine
    try:
        if isinstance(args.model_name, str) and args.model_name:
            loss_engine.MODEL_NAME = args.model_name
    except Exception:
        pass

    # Existing cache flags and batching behavior
    bypass_cache = False
    bypass_file_cache = True

    tok_per_batch = 18432

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
                    batch_size=60,
                    tok_per_batch=tok_per_batch,
                    pipeline_passes=passes,
                    pipeline_ctx=ctx,
                    cache_read=(not bypass_cache),
                    file_cache_read=(not bypass_file_cache),
                    cache_write=True,
                    debug=False,
                )

                out_eval = os.path.join(chat_out, f"{name}.eval.jsonl")
                print(f"Processing chat file: {fp} -> {out_eval}")
                ingest_entities(
                    fp,
                    out_eval,
                    with_loss=with_loss,
                    batch_size=60,
                    tok_per_batch=tok_per_batch,
                    pipeline_passes=eval_passes,
                    pipeline_ctx=ctx,
                    cache_read=(not bypass_cache),
                    file_cache_read=(not bypass_file_cache),
                    cache_write=True,
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
                        batch_size=60,
                        tok_per_batch=tok_per_batch,
                        pipeline_passes=passes,
                        pipeline_ctx=ctx,
                        cache_read=(not bypass_cache),
                        file_cache_read=(not bypass_file_cache),
                        cache_write=True,
                    )
                    total += 1
                except Exception as e:
                    print(f"Error processing {fp}: {e}")

    print(f"Completed. Processed {total} file(s).")


if __name__ == "__main__":
    main()
