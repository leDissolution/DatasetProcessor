from __future__ import annotations

import os
import sqlite3
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import time

import orjson

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def sha1_text(s: str) -> str:
    return _sha1_bytes(s.encode("utf-8", errors="ignore"))


class ProcessingCache:
    def __init__(self, db_path: Optional[str] = None) -> None:
        base_dir = os.path.dirname(__file__)
        cache_dir = os.path.join(base_dir, ".cache")
        _ensure_dir(cache_dir)
        self.db_path = db_path or os.path.join(cache_dir, "processing.sqlite")
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS file_cache (
                file_key   TEXT NOT NULL,
                file_hash  TEXT NOT NULL,
                version    TEXT NOT NULL,
                entries    BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                last_used  INTEGER NOT NULL,
                PRIMARY KEY(file_key, file_hash, version)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dp_cache (
                dp_key     TEXT NOT NULL,
                version    TEXT NOT NULL,
                entries    BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                last_used  INTEGER NOT NULL,
                PRIMARY KEY(dp_key, version)
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # File-level API
    def get_file(self, file_key: str, file_hash: str, version: str) -> Optional[List[Dict[str, Any]]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT entries FROM file_cache WHERE file_key=? AND file_hash=? AND version=?",
            (file_key, file_hash, version),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            cur.execute(
                "UPDATE file_cache SET last_used=? WHERE file_key=? AND file_hash=? AND version=?",
                (int(time.time()), file_key, file_hash, version),
            )
            self._conn.commit()
        except Exception:
            pass
        try:
            entries = orjson.loads(row[0])
            if not isinstance(entries, list):
                return None
            return entries  # type: ignore[return-value]
        except Exception:
            return None

    def set_file(self, file_key: str, file_hash: str, version: str, entries: List[Dict[str, Any]]) -> None:
        payload = orjson.dumps(entries)
        ts = int(time.time())
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO file_cache(file_key, file_hash, version, entries, created_at, last_used)
            VALUES(?, ?, ?, ?, COALESCE((SELECT created_at FROM file_cache WHERE file_key=? AND file_hash=? AND version=?), ?), ?)
            """,
            (file_key, file_hash, version, payload, file_key, file_hash, version, ts, ts),
        )
        self._conn.commit()

    # Datapoint-level API
    def get_dp(self, dp_key: str, version: str) -> Optional[List[Dict[str, Any]]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT entries FROM dp_cache WHERE dp_key=? AND version=?",
            (dp_key, version),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            cur.execute(
                "UPDATE dp_cache SET last_used=? WHERE dp_key=? AND version=?",
                (int(time.time()), dp_key, version),
            )
            self._conn.commit()
        except Exception:
            pass
        try:
            entries = orjson.loads(row[0])
            if not isinstance(entries, list):
                return None
            return entries  # type: ignore[return-value]
        except Exception:
            return None

    def set_dp(self, dp_key: str, version: str, entries: List[Dict[str, Any]]) -> None:
        payload = orjson.dumps(entries)
        ts = int(time.time())
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO dp_cache(dp_key, version, entries, created_at, last_used)
            VALUES(?, ?, ?, COALESCE((SELECT created_at FROM dp_cache WHERE dp_key=? AND version=?), ?), ?)
            """,
            (dp_key, version, payload, dp_key, version, ts, ts),
        )
        self._conn.commit()


# -------- Keys and Versions ---------

def compute_file_hash(text: str) -> str:
    return sha1_text(text)


def _norm_attrs(attrs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    try:
        items = sorted(attrs.items(), key=lambda kv: kv[0])
    except Exception:
        items = list(attrs.items())
        items.sort(key=lambda kv: str(kv[0]))
    return tuple((str(k), v) for k, v in items)


def _entity_repr(e: Any) -> Tuple:
    cls_name = e.__class__.__name__
    subj = getattr(e, "subject", None)
    subj_key = getattr(subj, "key", None)
    subj_id = getattr(subj, "id", None)
    attrs = getattr(e, "attrs", {}) or {}
    order = getattr(e, "attr_order", None)
    return (
        cls_name,
        (subj_key, subj_id),
        _norm_attrs(attrs),
        tuple(order) if order else None,
    )


def compute_datapoint_key(dp: Any) -> str:
    # Build a stable content signature independent of file line numbers.
    try:
        prev_msg = (
            "previousMessage",
            getattr(dp.previous_message, "text", None),
            _norm_attrs(getattr(dp.previous_message, "attrs", {}) or {}),
            tuple(getattr(dp.previous_message, "attr_order", []) or []),
        )
        msg = (
            "message",
            getattr(dp.message, "text", None),
            _norm_attrs(getattr(dp.message, "attrs", {}) or {}),
            tuple(getattr(dp.message, "attr_order", []) or []),
        )
        prev_state = tuple(_entity_repr(e) for e in getattr(dp, "previous_state", []) or [])
        state = tuple(_entity_repr(e) for e in getattr(dp, "state", []) or [])
        payload = (
            prev_msg,
            msg,
            prev_state,
            state,
        )
        return _sha1_bytes(orjson.dumps(payload))
    except Exception:
        # In worst case fallback to prompt text hash
        try:
            from datapoint import Datapoint as _DP  # type: ignore
            if isinstance(dp, _DP):
                return sha1_text(dp.rehydrate_prompt())
        except Exception:
            pass
        return sha1_text(repr(dp))


def _read_file_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def compute_pipeline_signature(passes: Optional[Sequence[Any]]) -> str:
    parts: List[bytes] = []

    def _jsonable(x: Any) -> Any:
        # Basic types
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        # Classes / types
        if isinstance(x, type):
            mod = getattr(x, "__module__", "")
            name = getattr(x, "__name__", str(x))
            return {"__type__": f"{mod}.{name}"}
        # Regex patterns (duck-typed)
        pat = getattr(x, "pattern", None)
        flags = getattr(x, "flags", None)
        if isinstance(pat, str) and isinstance(flags, int):
            return {"__regex__": pat, "flags": flags}
        # Sequences
        if isinstance(x, (list, tuple)):
            return [_jsonable(i) for i in x]
        # Mappings
        if isinstance(x, dict):
            try:
                items = sorted(x.items(), key=lambda kv: str(kv[0]))
            except Exception:
                items = list(x.items())
            return {str(k): _jsonable(v) for k, v in items}
        # Fallback to repr
        return {"__repr__": repr(x)}

    # Include pass names and simple configs
    names: List[Dict[str, Any]] = []
    if passes:
        for p in passes:
            raw_cfg = {k: v for k, v in getattr(p, "__dict__", {}).items() if not str(k).startswith("_")}
            cfg = _jsonable(raw_cfg)
            names.append({"name": p.__class__.__name__, "cfg": cfg})
    parts.append(orjson.dumps(names))
    # Include pipeline.py source hash if available
    try:
        pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.py")
        src = _read_file_bytes(pipeline_path)
        if src:
            parts.append(src)
    except Exception:
        pass
    return _sha1_bytes(b"|".join(parts))


def compute_loss_signature() -> str:
    parts: List[bytes] = []
    try:
        from loss_engine import MODEL_NAME, DTYPE  # type: ignore
        parts.append(str(MODEL_NAME).encode("utf-8"))
        parts.append(str(DTYPE).encode("utf-8"))
        # Include model directory/file modified time as a version input
        try:
            mname = str(MODEL_NAME)
            if os.path.isdir(mname):
                max_mtime = 0.0
                file_count = 0
                try:
                    for name in os.listdir(mname):
                        p = os.path.join(mname, name)
                        if os.path.isfile(p):
                            st = os.stat(p)
                            max_mtime = max(max_mtime, st.st_mtime)
                            file_count += 1
                except Exception:
                    pass
                parts.append(f"model_dir_mtime:{int(max_mtime)}".encode("utf-8"))
                parts.append(f"model_dir_files:{file_count}".encode("utf-8"))
            elif os.path.isfile(mname):
                try:
                    st = os.stat(mname)
                    parts.append(f"model_file_mtime:{int(st.st_mtime)}".encode("utf-8"))
                except Exception:
                    pass
        except Exception:
            pass
        loss_path = os.path.join(os.path.dirname(__file__), "loss_engine.py")
        src = _read_file_bytes(loss_path)
        if src:
            parts.append(src)
    except Exception:
        pass
    return _sha1_bytes(b"|".join(parts))


def compute_run_version(passes: Optional[Sequence[Any]], with_loss: bool, with_json: bool = False) -> str:
    pl = compute_pipeline_signature(passes)
    ls = compute_loss_signature() if with_loss else "loss:0"
    return _sha1_bytes((pl + "|" + ls + "|" + str(bool(with_loss)) + "|json=" + ("1" if with_json else "0")).encode("utf-8"))


def remap_cached_entries_source(entries: List[Dict[str, Any]], new_source_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in entries:
        try:
            e2 = dict(e)
            e2["source_id"] = new_source_id
            tgt = e2.get("target") or None
            attr = None
            if isinstance(tgt, dict):
                attr = tgt.get("attr")
            e2["id"] = f"{new_source_id}:{attr or ''}"
            # If nested datapoint exists, update its source_id as well
            try:
                dp_nested = e2.get("datapoint")
                if isinstance(dp_nested, dict):
                    dp_nested = dict(dp_nested)
                    dp_nested["source_id"] = new_source_id
                    e2["datapoint"] = dp_nested
            except Exception:
                pass
            out.append(e2)
        except Exception:
            out.append(e)
    return out


__all__ = [
    "ProcessingCache",
    "compute_file_hash",
    "compute_datapoint_key",
    "compute_run_version",
    "remap_cached_entries_source",
]
