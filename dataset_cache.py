from __future__ import annotations

"""
Simple on-disk cache for prepared dataset outputs.

If an input source file's content and the processing configuration haven't
changed, reuse the previously produced JSONL output to skip parsing, pipeline
passes, and optional loss computation.

Cache layout: <output_dir>/.cache/<sha256 key>.jsonl

The cache key includes:
- normalized input file path
- sha256 of the input text
- a normalized signature of the pipeline passes, context split, and RNG state
- a normalized signature of the loss configuration (model name, dtype, devices,
  batch size) when with_loss is True; otherwise, just with_loss=False
"""

from dataclasses import asdict, is_dataclass
from typing import Any, Optional, Sequence
import hashlib
import json
import os


def _to_jsonable(obj: Any) -> Any:
    """Attempt to convert objects into a JSON-serializable form deterministically."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, set):
        # Deterministic order
        return [_to_jsonable(x) for x in sorted(list(obj), key=lambda v: str(v))]
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    try:
        if is_dataclass(obj) and not isinstance(obj, type):
            return _to_jsonable(asdict(obj))
    except Exception:
        pass
    if hasattr(obj, "__name__") and isinstance(getattr(obj, "__name__"), str):
        # Likely a class/type
        return getattr(obj, "__name__")
    # Fallback to class name (avoids memory addresses from repr)
    return getattr(obj, "__class__", type(obj)).__name__


def _pass_signature(p: Any) -> dict[str, Any]:
    """Produce a normalized signature for a pipeline pass instance.

    We capture the class name and JSONable public attributes when possible.
    """
    sig: dict[str, Any] = {"class": p.__class__.__name__}
    try:
        attrs = {k: v for k, v in vars(p).items() if not k.startswith("_")}
    except Exception:
        attrs = {}
    norm_attrs: dict[str, Any] = {}
    for k, v in attrs.items():
        try:
            norm_attrs[k] = _to_jsonable(v)
        except Exception:
            # Skip non-serializable fields
            continue
    if norm_attrs:
        sig["attrs"] = norm_attrs
    return sig


def pipeline_signature(passes: Optional[Sequence[Any]], ctx: Optional[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"version": 1}
    if passes:
        out["passes"] = [_pass_signature(p) for p in passes]
    else:
        out["passes"] = []
    # Include split if present and RNG fingerprint so different seeds don't reuse cache.
    try:
        split = getattr(ctx, "split", None)
    except Exception:
        split = None
    out["split"] = split

    rng_state = None
    try:
        rng = getattr(ctx, "rng", None)
        if rng is not None and hasattr(rng, "getstate"):
            s = rng.getstate()  # type: ignore[assignment]
            payload = repr(s).encode("utf-8")
            rng_state = hashlib.sha256(payload).hexdigest()
    except Exception:
        rng_state = None
    out["rng_state"] = rng_state
    return out


def loss_signature(with_loss: bool, batch_size: int) -> dict[str, Any]:
    sig: dict[str, Any] = {"with_loss": bool(with_loss), "batch_size": int(batch_size)}
    if with_loss:
        try:
            # Import lazily to avoid heavy initialization costs until needed.
            import loss_engine  # type: ignore
        except Exception:
            # Fallback: no model metadata available
            loss_meta = {"model": None, "dtype": None, "devices": None}
        else:
            model = getattr(loss_engine, "MODEL_NAME", None)
            dtype = getattr(loss_engine, "DTYPE", None)
            # Render dtype as string to make it JSONable and stable
            dtype_s = str(dtype) if dtype is not None else None
            loss_meta = {"model": model, "dtype": dtype_s}
        sig.update(loss_meta)
    return sig


def _normalize_path(p: str) -> str:
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(p)))
    except Exception:
        return p


def compute_cache_key(
    input_path: str,
    input_text: str,
    pipeline_sig: dict[str, Any],
    loss_sig: dict[str, Any],
) -> str:
    payload = {
        "v": 1,
        "input_path": _normalize_path(input_path),
        "text_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        "pipeline": pipeline_sig,
        "loss": loss_sig,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_dir_for_output(new_path: str) -> str:
    parent = os.path.dirname(new_path)
    return os.path.join(parent, ".cache")


def maybe_load_cached_output(new_path: str, key: str) -> Optional[bytes]:
    cache_dir = _cache_dir_for_output(new_path)
    cache_fp = os.path.join(cache_dir, f"{key}.jsonl")
    try:
        with open(cache_fp, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception:
        return None


def store_cached_output(new_path: str, key: str, content: bytes) -> None:
    cache_dir = _cache_dir_for_output(new_path)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_fp = os.path.join(cache_dir, f"{key}.jsonl.tmp")
        final_fp = os.path.join(cache_dir, f"{key}.jsonl")
        with open(tmp_fp, "wb") as f:
            f.write(content)
        os.replace(tmp_fp, final_fp)
    except Exception:
        # Best-effort caching; ignore errors
        return


__all__ = [
    "pipeline_signature",
    "loss_signature",
    "compute_cache_key",
    "maybe_load_cached_output",
    "store_cached_output",
]
