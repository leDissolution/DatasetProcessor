from __future__ import annotations

from datapoint import Datapoint

from typing import List, Dict, Tuple, Optional, Any, cast
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import multiprocessing as mp

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = r".\Models\slm2-stage2"
DTYPE      = torch.bfloat16
DEVICES    = [0, 1]
DTYPE_CPU  = torch.float32  # safer default on CPU

_VRAM_FRACTION_CAP: float = float(os.getenv("LOSS_ENGINE_VRAM_FRACTION", "0.90"))
_VRAM_POLL_SLEEP_S: float = float(os.getenv("LOSS_ENGINE_VRAM_POLL_SLEEP_S", "0.1"))
_VRAM_DEBUG: bool = os.getenv("LOSS_ENGINE_VRAM_DEBUG", "0") not in ("", "0", "false", "False")
_EMPTY_CACHE_EVERY_N: int = max(0, int(os.getenv("LOSS_ENGINE_EMPTY_CACHE_EVERY", "0")))

# Globals that will live INSIDE EVERY WORKER PROCESS (initialized in _worker_init)
_tokenizer: Any = None
_model: Any = None
_device: Optional[str] = None
_comma_id: Optional[int] = None
_cpu_tokenizer: Any = None  # CPU-side tokenizer for planning/scheduling
_BATCH_COUNTER: int = 0


def _get_cpu_tokenizer() -> Any:
    global _cpu_tokenizer
    if _cpu_tokenizer is None:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        _cpu_tokenizer = tok
    return _cpu_tokenizer


def _plan_token_batches(
    prompts: List[str],
    max_tokens_per_batch: int,
    sort_by_len: bool = True,
) -> Tuple[List[List[int]], List[int]]:
    """Plan batches by limiting effective tokens: batch_eff_tokens = max_len * batch_size.
    Returns (batches_of_indices, token_lengths).
    """
    tok = _get_cpu_tokenizer()
    # include EOS to mirror worker inputs
    enc = tok(prompts, add_special_tokens=True, padding=False, truncation=False)
    lengths: List[int] = [len(ids) + (1 if tok.eos_token_id is not None else 0) for ids in enc["input_ids"]]

    if sort_by_len:
        order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    else:
        order = list(range(len(prompts)))

    batches: List[List[int]] = []
    cur: List[int] = []
    cur_max = 0
    for i in order:
        L = lengths[i]
        new_max = max(cur_max, L)
        new_bs = len(cur) + 1
        eff_tokens = new_max * new_bs
        if cur and eff_tokens > max_tokens_per_batch:
            batches.append(cur)
            cur = [i]
            cur_max = L
        else:
            cur.append(i)
            cur_max = new_max
    if cur:
        batches.append(cur)
    return batches, lengths

def _worker_init(device_id: int, model_name: str, dtype: torch.dtype) -> None:
    """Runs once per worker process. Loads tokenizer + model on the given GPU."""
    global _tokenizer, _model, _device, _comma_id
    if device_id >= 0:
        _device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
    else:
        _device = "cpu"

    # Cap this worker's memory usage to a safe fraction of the GPU to avoid system/driver instability.
    if device_id >= 0:
        try:
            # Note: Must be set before large allocations (e.g., model load).
            frac = max(0.1, min(0.95, _VRAM_FRACTION_CAP))
            torch.cuda.set_per_process_memory_fraction(frac, device=device_id)
            if _VRAM_DEBUG:
                free_b, tot_b = torch.cuda.mem_get_info(device_id)
                print(f"[VRAM] per-process cap set to {frac:.2f}; device {device_id} free={free_b/1e6:.0f}MB total={tot_b/1e6:.0f}MB")
        except Exception as e:
            if _VRAM_DEBUG:
                print(f"[VRAM] set_per_process_memory_fraction not available or failed: {e}")

    # Enable performance-friendly backends where available
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    _tokenizer = tok
    _comma_id = _tokenizer.encode(",")[0]

    if device_id >= 0:
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": device_id},
            attn_implementation="flash_attention_2",
            use_cache=False,
        ).eval()
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=DTYPE_CPU,
            device_map={"": "cpu"},
            use_cache=False,
        ).eval()

    if _VRAM_DEBUG and device_id >= 0:
        try:
            reserved = torch.cuda.memory_reserved(device_id) / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
            free_b, tot_b = torch.cuda.mem_get_info(device_id)
            print(f"[VRAM] after model load: reserved={reserved:.0f}MB allocated={allocated:.0f}MB free={free_b/1e6:.0f}MB")
        except Exception:
            pass


def _is_oom(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return any(kw in msg for kw in (
        "out of memory",
        "cuda error: out of memory",
        "cublas workspace",
        "memory allocation",
    ))


def _build_prefix(prompt: str, attr: Optional[str]) -> str:
    """Return the textual prefix to mask (everything up to the target value)."""
    if attr:
        needle = f"{attr}=\""
        idx = prompt.rfind(needle)
        if idx != -1:
            return prompt[: idx + len(needle)]
    
    if "=\"" in prompt:
        return prompt.rsplit("=\"", 1)[0] + "=\""
    return prompt


def _process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute loss metrics for a batch of items."""
    assert _tokenizer is not None and _model is not None and _device is not None
    assert _comma_id is not None

    try:
        # Predeclare for safe cleanup in error/early-return paths
        inputs: Any = None
        labels: Any = None
        logits: Any = None

        # Optional periodic cleanup to reduce fragmentation without doing it every batch
        global _BATCH_COUNTER
        _BATCH_COUNTER += 1
        if _EMPTY_CACHE_EVERY_N > 0 and (_BATCH_COUNTER % _EMPTY_CACHE_EVERY_N == 0):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        prompts: List[str] = [str(it["prompt"]) + _tokenizer.eos_token for it in batch]
        target_attrs: List[Optional[str]] = [it.get("target_attr") for it in batch]

        pad_id = _tokenizer.pad_token_id

        # Prefer pre-tokenized ids if present in payload; otherwise, tokenize here.
        if all("_input_ids" in it for it in batch):
            seqs = [it["_input_ids"] for it in batch]
            max_len = max(1, max(len(s) for s in seqs))
            # Build in pinned CPU memory first, then async transfer to device for smoother overlap
            input_ids_cpu = torch.full((len(seqs), max_len), fill_value=pad_id, dtype=torch.long, pin_memory=True)
            for i, ids in enumerate(seqs):
                n = min(len(ids), max_len)
                if n:
                    input_ids_cpu[i, :n] = torch.as_tensor(ids[:n], dtype=torch.long)
            attention_mask_cpu = (input_ids_cpu != pad_id).to(dtype=torch.long).pin_memory()
            input_ids = input_ids_cpu.to(_device, non_blocking=True)
            attention_mask = attention_mask_cpu.to(_device, non_blocking=True)
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            enc = _tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            # Try to pin host memory for smoother async H2D
            inputs = {}
            for k, v in enc.items():
                try:
                    v = v.pin_memory()
                    inputs[k] = v.to(_device, non_blocking=True)
                except Exception:
                    inputs[k] = v.to(_device)

        labels = inputs["input_ids"].clone()

        # Determine prefix lengths (tokens masked out of loss). Use pre-tokenized when available.
        if all("_prefix_ids" in it for it in batch):
            prefix_lens = [len(it["_prefix_ids"]) for it in batch]
        else:
            prefixes = [_build_prefix(p, a) for p, a in zip(prompts, target_attrs)]
            prefix_tokens = _tokenizer(prefixes, padding=True, return_tensors="pt").input_ids
            prefix_lens = [int((prefix_tokens[i] != pad_id).sum().item()) for i in range(len(prefixes))]

        for i in range(len(prompts)):
            nonpad_len = int((inputs["attention_mask"][i]).sum().item()) if "attention_mask" in inputs else int((inputs["input_ids"][i] != pad_id).sum().item())
            prefix_len = min(prefix_lens[i], nonpad_len)
            labels[i, :prefix_len] = -100
            labels[i, inputs["input_ids"][i] == pad_id] = -100
            if prefix_len >= max(nonpad_len - 1, 0):
                generic_prefix = prompts[i].rsplit("=\"", 1)[0] + "=\"" if "=\"" in prompts[i] else prompts[i]
                gen_tokens = _tokenizer(generic_prefix, return_tensors="pt").input_ids[0]
                gen_len = int((gen_tokens != pad_id).sum().item())
                if gen_len < max(nonpad_len - 1, 0):
                    labels[i, :] = inputs["input_ids"][i]
                    labels[i, :gen_len] = -100
                    labels[i, inputs["input_ids"][i] == pad_id] = -100

        debug_enabled = any(bool(it.get("_debug_print")) for it in batch)
        if debug_enabled:
            dbg_max_val: Optional[Any] = next((it.get("_debug_samples") for it in batch if it.get("_debug_samples") is not None), None)
            dbg_chars_val: Optional[Any] = next((it.get("_debug_chars") for it in batch if it.get("_debug_chars") is not None), None)
            try:
                dbg_max = int(dbg_max_val) if dbg_max_val is not None else 3
            except Exception:
                dbg_max = 3
            try:
                dbg_chars = int(dbg_chars_val) if dbg_chars_val is not None else 200
            except Exception:
                dbg_chars = 200

            shown = 0
            for i in range(len(prompts)):
                if shown >= dbg_max:
                    break

                nonpad_len = int((inputs["attention_mask"][i]).sum().item()) if "attention_mask" in inputs else int((inputs["input_ids"][i] != pad_id).sum().item())
                prefix_len = min(prefix_lens[i], nonpad_len)

                try:
                    prefix_text = _tokenizer.decode(inputs["input_ids"][i, :prefix_len], skip_special_tokens=False)
                    completion_text = _tokenizer.decode(inputs["input_ids"][i, prefix_len:nonpad_len], skip_special_tokens=False)
                except Exception:
                    prefix_text = "<decode_error>"
                    completion_text = "<decode_error>"
                attr = batch[i].get("target_attr")
                print(f"[LOSS-DEBUG] sample={i} attr={attr} prefix_tokens={prefix_len} total_tokens={nonpad_len}")
                print("[LOSS-DEBUG] prefix_tail: ..." + prefix_text[-dbg_chars:])
                print("[LOSS-DEBUG] >>> completion >>> " + completion_text[:dbg_chars])
                shown += 1

        with torch.inference_mode():
            logits = _model(**inputs, use_cache=False).logits

        # Loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        loss_tok = loss_fct(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        mask = (shift_labels != -100).float()
        mean_loss = (loss_tok * mask).sum(1) / mask.sum(1).clamp(min=1)

        is_comma = ((shift_labels == _comma_id) & (mask.bool()))
        first_pos = torch.zeros_like(mask, dtype=torch.bool)
        first_pos[:, 0] = mask[:, 0].bool()
        first_pos[:, 1:] |= is_comma[:, :-1]

        masked_loss = loss_tok.masked_fill(mask == 0, float("-inf"))
        worst_loss = masked_loss.max(1).values

        bf_scores = loss_tok.masked_fill((~first_pos) | (mask == 0), float("-inf"))
        block_first_loss = bf_scores.max(1).values

        critical_token_positions = masked_loss.argmax(dim=1)
        critical_tokens: List[Dict[str, Any]] = []
        valid_token_ids: List[int] = []
        valid_indices: List[int] = []
        pred_token_ids: List[int] = []
        pred_valid_indices: List[int] = []

        for i, pos in enumerate(critical_token_positions):
            if mask[i, pos] > 0:
                token_id = shift_labels[i, pos].item()
                pred_token_id = shift_logits[i, pos].argmax().item()
                valid_token_ids.append(token_id)
                valid_indices.append(i)
                pred_token_ids.append(pred_token_id)
                pred_valid_indices.append(i)
                critical_tokens.append({
                    "position": int(pos.item()),
                    "token_id": int(token_id),
                    "decoded_value": None,
                    "loss": float(loss_tok[i, pos].item()),
                    "pred_token_id": int(pred_token_id),
                    "pred_decoded_value": None,
                })
            else:
                critical_tokens.append({
                    "position": -1,
                    "token_id": -1,
                    "decoded_value": "",
                    "loss": 0.0,
                    "pred_token_id": -1,
                    "pred_decoded_value": "",
                })

        if valid_token_ids:
            try:
                decoded_tokens = [_tokenizer.decode([tid], skip_special_tokens=False) for tid in valid_token_ids]  # type: ignore[arg-type]
            except Exception:
                decoded_tokens = [f"<token_{tid}>" for tid in valid_token_ids]
            for valid_idx, decoded_token in zip(valid_indices, decoded_tokens):
                critical_tokens[valid_idx]["decoded_value"] = decoded_token

        if pred_token_ids:
            try:
                pred_decoded_tokens = [_tokenizer.decode([tid], skip_special_tokens=False) for tid in pred_token_ids]  # type: ignore[arg-type]
            except Exception:
                pred_decoded_tokens = [f"<token_{tid}>" for tid in pred_token_ids]
            for pred_idx, pred_decoded_token in zip(pred_valid_indices, pred_decoded_tokens):
                critical_tokens[pred_idx]["pred_decoded_value"] = pred_decoded_token

        valid_counts = mask.sum(1)
        mean_loss = torch.nan_to_num(mean_loss, nan=0.0, posinf=1e3, neginf=0.0)
        worst_loss = torch.where(valid_counts > 0, torch.nan_to_num(worst_loss, nan=0.0, posinf=1e3, neginf=0.0), torch.zeros_like(worst_loss))
        block_first_loss = torch.where(valid_counts > 0, torch.nan_to_num(block_first_loss, nan=0.0, posinf=1e3, neginf=0.0), torch.zeros_like(block_first_loss))

        difficulty = (0.40 * mean_loss + 0.40 * block_first_loss + 0.20 * worst_loss).tolist()

        out: List[Dict[str, Any]] = []
        for d, m, w, c in zip(difficulty, mean_loss.tolist(), worst_loss.tolist(), critical_tokens):
            out.append({
                "completion_difficulty": float(d),
                "mean_loss": float(m),
                "worst_loss": float(w),
                "critical_token": c,
            })

        # Cleanup
        try:
            if logits is not None:
                del logits
            if inputs is not None:
                del inputs
            if labels is not None:
                del labels
        except Exception:
            pass
        if _EMPTY_CACHE_EVERY_N > 0 and (_BATCH_COUNTER % _EMPTY_CACHE_EVERY_N == 0):
            for _ in range(5):
                try:
                    torch.cuda.empty_cache()
                    time.sleep(_VRAM_POLL_SLEEP_S)
                except Exception:
                    pass

        return out

    except RuntimeError:
        raise


def _batch_worker(args: Tuple[int, List[Dict[str, Any]]]) -> Tuple[int, List[Dict[str, Any]]]:
    idx, batch = args
    return idx, _process_batch(batch)

_EXECUTORS: Dict[int, ProcessPoolExecutor] = {}

def _get_executor(device_id: int) -> ProcessPoolExecutor:
    if device_id in _EXECUTORS:
        return _EXECUTORS[device_id]
    ctx = mp.get_context("spawn")
    ex = ProcessPoolExecutor(
        max_workers=1,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(device_id, MODEL_NAME, DTYPE),
    )
    _EXECUTORS[device_id] = ex
    return ex


def compute_losses_for_prompts(
    prompts: List[str],
    target_attrs: Optional[List[Optional[str]]] = None,
    batch_size: int = 35,
    max_tokens_per_batch: Optional[int] = None,
    devices: List[int] = DEVICES,
    use_cpu: bool = False,
    sort_by_len: bool = True,
    show_progress: bool = True,
    debug_print: bool = False,
    debug_samples: int = 3,
    debug_chars: int = 200,
) -> List[Dict[str, Any]]:
    """Compute loss metrics for a list of prompts."""
    if target_attrs is None:
        target_attrs = [None for _ in range(len(prompts))]

    items: List[Dict[str, Any]] = []
    for p, a in zip(prompts, target_attrs):
        item: Dict[str, Any] = {"prompt": p, "target_attr": a}
        if debug_print:
            item["_debug_print"] = True
            item["_debug_samples"] = int(debug_samples)
            item["_debug_chars"] = int(debug_chars)
        items.append(item)

    # Pre-tokenize prompts and prefixes on CPU to reduce per-batch overhead in workers.
    try:
        tok = _get_cpu_tokenizer()
        eos = tok.eos_token or ""
        for it in items:
            full_prompt = str(it["prompt"]) + eos
            enc = tok(full_prompt, add_special_tokens=True, padding=False, truncation=True)
            it["_input_ids"] = enc.get("input_ids")
            # Pre-tokenize prefix used for loss masking
            pfx_text = _build_prefix(full_prompt, it.get("target_attr"))
            penc = tok(pfx_text, add_special_tokens=True, padding=False, truncation=True)
            it["_prefix_ids"] = penc.get("input_ids")
    except Exception:
        pass

    perm: Optional[List[int]] = None
    sorted_items: List[Dict[str, Any]]
    if max_tokens_per_batch is not None and max_tokens_per_batch > 0:
        # plan token-aware batches on the original prompt list to preserve stable mapping
        planned_batches, _lengths = _plan_token_batches(
            [it["prompt"] for it in items],
            max_tokens_per_batch=max_tokens_per_batch,
            sort_by_len=sort_by_len,
        )
        # Flatten into sorted order of indices
        flat_order: List[int] = [idx for batch in planned_batches for idx in batch]
        if sort_by_len:
            perm = flat_order
            sorted_items = [items[i] for i in perm]
        else:
            # Not sorting: flat_order equals range(n)
            sorted_items = [items[i] for i in flat_order]
        # Build batches as consecutive segments of sorted_items according to planned sizes
        sizes = [len(b) for b in planned_batches]
        offset = 0
        batches: List[Tuple[int, List[Dict[str, Any]]]] = []
        for idx, sz in enumerate(sizes):
            batches.append((idx, sorted_items[offset: offset + sz]))
            offset += sz
    else:
        if sort_by_len:
            perm = sorted(range(len(items)), key=lambda i: len(items[i]["prompt"]))
            sorted_items = [items[i] for i in perm]
        else:
            sorted_items = items
        batches = [
            (idx, sorted_items[i : i + batch_size])
            for idx, i in enumerate(range(0, len(sorted_items), batch_size))
        ]

    # Simple round-robin scheduler: one in-flight batch per device; exceptions propagate
    dev_list: List[int] = list(devices)
    if use_cpu:
        dev_list = dev_list + [-1]

    pbar = tqdm(total=len(batches), desc="GPU-loss batches", unit="batch", leave=False) if show_progress else None
    batch_results: List[Optional[List[Dict[str, Any]]]] = [None] * len(batches)

    futures: set = set()
    future_meta: Dict[Any, Dict[str, Any]] = {}

    def submit_to(dev: int, parent_idx: int, payload: List[Dict[str, Any]]) -> None:
        ex = _get_executor(dev)
        fut = ex.submit(_batch_worker, (parent_idx, payload))
        futures.add(fut)
        future_meta[fut] = {"dev": dev, "parent_idx": parent_idx}

    next_batch = 0
    # Warm start: one per device
    for dev in dev_list:
        if next_batch >= len(batches):
            break
        parent_idx, payload = batches[next_batch]
        submit_to(dev, parent_idx, payload)
        next_batch += 1

    while futures:
        done_set, _ = wait(futures, return_when=FIRST_COMPLETED)
        for fut in list(done_set):
            if fut not in futures:
                continue
            futures.remove(fut)
            meta = future_meta.pop(fut, {})
            idx_ret, processed = fut.result()
            parent_idx = meta.get("parent_idx", idx_ret)
            batch_results[parent_idx] = processed
            if pbar:
                pbar.update(1)
            # Submit the next batch to the same device, if any
            dev = int(meta.get("dev", 0))
            if next_batch < len(batches):
                p_idx, pay = batches[next_batch]
                submit_to(dev, p_idx, pay)
                next_batch += 1

    if pbar:
        pbar.close()

    flat: List[Dict[str, Any]] = []
    for b in batch_results:
        if b is None:
            continue
        flat.extend(b)

    if sort_by_len and perm is not None:
        restored: List[Optional[Dict[str, Any]]] = [None] * len(flat)
        for new_idx, entry in enumerate(flat):
            original_idx = perm[new_idx]
            restored[original_idx] = entry
        flat = [cast(Dict[str, Any], e) for e in restored if e is not None]

    return flat


def compute_losses_for_datapoints(
    dps: List[Datapoint],
    batch_size: int = 35,
    max_tokens_per_batch: Optional[int] = None,
    devices: List[int] = DEVICES,
    use_cpu: bool = False,
    sort_by_len: bool = True,
    show_progress: bool = True,
    debug_print: bool = False,
    debug_samples: int = 3,
    debug_chars: int = 200,
) -> List[Dict[str, Any]]:
    """Convenience wrapper: build prompts/attrs from Datapoints and compute losses."""
    prompts: List[str] = []
    attrs: List[Optional[str]] = []
    for dp in dps:
        prompts.append(dp.rehydrate_prompt())
        attr = None
        tgt = dp.target
        if tgt is not None:
            attr = tgt.attr
        attrs.append(attr)

    return compute_losses_for_prompts(
        prompts,
        target_attrs=attrs,
        batch_size=batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
        devices=devices,
    use_cpu=use_cpu,
        sort_by_len=sort_by_len,
        show_progress=show_progress,
        debug_print=debug_print,
        debug_samples=debug_samples,
        debug_chars=debug_chars,
    )


def benchmark_max_tokens_per_batch(
    prompts: List[str],
    candidate_tokens: Optional[List[int]] = None,
    devices: List[int] = DEVICES,
    use_cpu: bool = False,
    sort_by_len: bool = True,
    warmup: int = 0,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Simple benchmark across candidate max_tokens_per_batch values.
    Returns dict with per-candidate metrics and the best setting by tokens/sec.
    """
    import math
    if candidate_tokens is None:
        # Conservative defaults; will skip values that produce OOM by virtue of inner guardrails
        candidate_tokens = [16384, 32768, 65536, 131072, 262144]

    # Pre-compute planned effective tokens for each candidate without running model
    eff_tokens_map: Dict[int, int] = {}
    for m in candidate_tokens:
        batches, lengths = _plan_token_batches(prompts, m, sort_by_len=sort_by_len)
        eff = 0
        for b in batches:
            if not b:
                continue
            max_len = max(lengths[i] for i in b)
            eff += max_len * len(b)
        eff_tokens_map[m] = eff

    results: Dict[int, Dict[str, Any]] = {}
    # Optional warmup at the smallest setting
    if warmup > 0:
        _ = compute_losses_for_prompts(
            prompts[: max(1, min(len(prompts), warmup))],
            max_tokens_per_batch=min(candidate_tokens),
            devices=devices,
            use_cpu=use_cpu,
            sort_by_len=sort_by_len,
            show_progress=False,
        )

    for m in candidate_tokens:
        start = time.perf_counter()
        status = "ok"
        try:
            _ = compute_losses_for_prompts(
                prompts,
                max_tokens_per_batch=m,
                devices=devices,
                use_cpu=use_cpu,
                sort_by_len=sort_by_len,
                show_progress=show_progress,
            )
        except RuntimeError as err:
            status = "oom" if _is_oom(err) else f"error: {type(err).__name__}"
        except Exception as err:
            raise RuntimeError(f"Benchmark failed: {err}") from err
        elapsed = max(1e-6, time.perf_counter() - start)
        eff = eff_tokens_map.get(m, 0)
        tps = eff / elapsed if status == "ok" and eff > 0 else 0.0
        results[m] = {"status": status, "seconds": elapsed, "effective_tokens": eff, "tokens_per_sec": tps}

    # Choose best among successful runs
    ok = [(
        m,
        results[m]["tokens_per_sec"],
    ) for m in candidate_tokens if results.get(m, {}).get("status") == "ok"]
    best = max(ok, key=lambda x: x[1]) if ok else None
    return {"candidates": results, "best": {"max_tokens_per_batch": best[0], "tokens_per_sec": best[1]} if best else None}


__all__ = [
    "compute_losses_for_prompts",
    "compute_losses_for_datapoints",
    "benchmark_max_tokens_per_batch",
]

# --- Device utilities ---
from typing import Sequence  # placed late to avoid clutter

def validate_devices(devices: Optional[Sequence[int]]) -> List[int]:
    """Return the subset of requested device indices that are actually available.

    If devices is None, returns the module default DEVICES filtered for availability.
    Prints warnings (never raises) if some requested devices are missing or CUDA not available.
    """
    if devices is None:
        devices = DEVICES
    try:
        import torch  # local import; may fail in CPU-only env
        if not torch.cuda.is_available():
            if len(devices) > 0:
                print("[DEVICES] Warning: CUDA not available; using no GPU devices.")
            return []
        count = torch.cuda.device_count()
        avail = [d for d in devices if isinstance(d, int) and 0 <= d < count]
        missing = [d for d in devices if d not in avail]
        if missing:
            print(f"[DEVICES] Warning: requested device(s) {missing} not present. Available: {list(range(count))}. Using subset {avail}.")
        return avail
    except Exception as e:  # pragma: no cover - very rare path
        print(f"[DEVICES] Warning: device validation failed ({e}); using requested list as-is.")
        return list(devices)

__all__.append("validate_devices")
