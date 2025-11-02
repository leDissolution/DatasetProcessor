"""
Convert a raw annotated dataset file into a JSONL of chat-style messages.

The script reuses the existing entity parser so that conversions stay aligned
with the rest of the tooling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import orjson

from datapoint import Datapoint
from entity_parser import parse_text
from Entities.entity import CharacterStatsEntity, SceneStatsEntity


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Input file not found: {path}") from exc


def _truthy(val: object) -> bool:
    if not isinstance(val, str):
        return False
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _clean_stat_value(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    cleaned = val.strip()
    if not cleaned or cleaned == "!!no_change!!":
        return None
    # Drop common escape sequences introduced by the annotated format so values stay human-readable.
    cleaned = cleaned.replace("\\\\", "\\").replace('\\"', '"').replace("\\'", "'")
    return cleaned


def _collect_entities(dp: Datapoint, entity_cls: type) -> Iterable[Any]:
    containers: Sequence[Iterable[Any]] = (dp.previous_state, dp.state_context, dp.state)
    for container in containers:
        for entity in container:
            if isinstance(entity, entity_cls):
                yield entity


def _gather_stats(datapoints: Sequence[Datapoint]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    characters: Dict[str, Dict[str, str]] = {}
    scenes: Dict[str, Dict[str, str]] = {}

    for dp in datapoints:
        for entity in _collect_entities(dp, CharacterStatsEntity):
            subject = (entity.subject.id or "").strip()
            if not subject:
                continue
            attrs = characters.setdefault(subject, {})
            for key, value in entity.attrs.items():
                cleaned = _clean_stat_value(value)
                if cleaned is not None:
                    attrs[key] = cleaned
        for entity in _collect_entities(dp, SceneStatsEntity):
            subject = (entity.subject.id or "").strip()
            if not subject:
                continue
            attrs = scenes.setdefault(subject, {})
            for key, value in entity.attrs.items():
                cleaned = _clean_stat_value(value)
                if cleaned is not None:
                    attrs[key] = cleaned

    return characters, scenes


def _build_present(characters: Dict[str, Dict[str, str]]) -> List[str]:
    present: List[str] = []
    for name in characters.keys():
        clean_name = name.strip()
        if not clean_name:
            continue
        candidate = f"{clean_name}.png"
        if candidate not in present:
            present.append(candidate)
    return present


def _extract_send_date(attrs: Dict[str, str]) -> str:
    for key in ("send_date", "timestamp", "time", "date", "datetime", "sent"):
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_name(attrs: Dict[str, str], fallback: Optional[str]) -> str:
    for key in ("from", "name", "speaker", "character", "author"):
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return (fallback or "").strip()


def _infer_roles(attrs: Dict[str, str]) -> Dict[str, bool]:
    result = {"user": False, "system": False}

    role_value = ""
    for key in ("role", "class", "type"):
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            role_value = value.strip().lower()
            break

    if role_value == "system":
        result["system"] = True
    elif role_value == "user":
        result["user"] = True

    sys_flag = attrs.get("system")
    if sys_flag is not None and _truthy(sys_flag):
        result["system"] = True

    usr_flag = attrs.get("user")
    if usr_flag is not None and _truthy(usr_flag):
        result["user"] = True

    return result


def _build_entry(group: Sequence[Datapoint], user_name: Optional[str]) -> Dict[str, Any]:
    primary = group[0]
    msg_attrs = dict(primary.message.attrs)
    name = _extract_name(msg_attrs, user_name)
    inferred = _infer_roles(msg_attrs)
    is_user = inferred["user"]
    if user_name:
        is_user = name.lower() == user_name.strip().lower()
    is_system = inferred["system"]
    send_date = _extract_send_date(msg_attrs)

    characters, scenes = _gather_stats(group)
    stats_payload: List[Dict[str, Dict[str, Dict[str, str]]]] = []
    if characters or scenes:
        bundle: Dict[str, Dict[str, Dict[str, str]]] = {}
        if characters:
            bundle["Characters"] = characters
        if scenes:
            bundle["Scenes"] = scenes
        stats_payload.append(bundle)

    avatar_template = """/thumbnail?type=avatar&file={name}.png"""
    user_avatar = "/thumbnail?type=persona&file=1731080026074-Alex.png"

    entry: Dict[str, Any] = {
        "name": name,
        "is_user": bool(is_user),
        "is_system": bool(is_system),
        "send_date": send_date,
        "mes": primary.message.text.strip(),
        "present": _build_present(characters),
        "stats": stats_payload,
        "force_avatar": user_avatar if is_user else avatar_template.format(name=name),
    }
    return entry


def convert_file(input_path: Path, output_path: Path, user_name: Optional[str]) -> None:
    raw = _read_text(input_path)
    warnings: List[str] = []
    datapoints = parse_text(raw, on_warning=warnings.append, source_name=str(input_path))

    if not datapoints:
        raise SystemExit("No datapoints detected in input. Nothing to write.")

    grouped: Dict[str, List[Datapoint]] = {}
    order: List[str] = []
    for idx, dp in enumerate(datapoints):
        key = dp.source_id or f"line-{idx}"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(dp)

    entries = [_build_entry(grouped[key], user_name) for key in order]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(orjson.dumps(entry).decode("utf-8"))
            handle.write("\n")

    print(f"Wrote {len(entries)} message(s) to {output_path}")
    if warnings:
        print(f"Parser emitted {len(warnings)} warning(s). Showing the first 5:")
        for warning in warnings[:5]:
            print(f" - {warning}")


def _default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_suffix(".jsonl")
    return input_path.parent / f"{input_path.name}.jsonl"


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert dataset file into JSONL chat messages")
    parser.add_argument("-f", "--file", required=True, help="Input dataset file")
    parser.add_argument("-o", "--output", help="Output JSONL file path")
    parser.add_argument("-user", dest="user", help="Name to mark as the user speaker")

    args = parser.parse_args(argv)

    input_path = Path(args.file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(input_path)
    user_name = args.user.strip() if isinstance(args.user, str) else None

    convert_file(input_path, output_path, user_name)


if __name__ == "__main__":
    main()
