"""
Entity-aware parser (parallel to existing code) that converts annotated text
into Datapoint objects using the Entities registry. It does not modify the
current ingestion; intended for side-by-side evaluation and a gradual switch.

Assumptions (per design):
- Global structure: <previousMessage> ... </previousMessage> then zero+ entities,
  then <message> ... </message> then zero+ entities, then next datapoint.
- Entities between anchors are self-closing tags like <stats ... /> or <meta ... />.
- Everything is flat; any tags inside messages remain text (not parsed as entities).
- Unknown entities are filtered out by the registry.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple
import os
import regex as re

from Entities.registry import EntityRegistry, default_registry
from Entities.entity import (
    Entity,
    GenericEntity,
    PreviousMessageEntity,
    MessageEntity,
)
from datapoint import Datapoint, EntityList

_OPEN_ANCHOR_RE = re.compile(r"<(previousMessage|message)([^>]*)>")
_CLOSE_PREV_RE = re.compile(r"</previousMessage>")
_CLOSE_MSG_RE = re.compile(r"</message>")

_SELF_CLOSING_RE = re.compile(r"<([A-Za-z][\w\-]*)\s*([^/>]*?)\s*/>")
_ATTR_RE = re.compile(r"(\w+)\s*=\s*\"((?:[^\"\\]|\\.)*)\"")


def _parse_attrs(attr_str: str) -> Tuple[dict[str, str], list[str]]:
    attrs: dict[str, str] = {}
    order: list[str] = []
    for m in _ATTR_RE.finditer(attr_str):
        key = m.group(1)
        val = m.group(2)
        attrs[key] = val
        order.append(key)
    return attrs, order


def _build_anchor(tag: str, open_match: re.Match, text: str) -> Tuple[Optional[Entity], int]:
    """Given an opening anchor match, capture attributes and body, return entity and index after close tag."""
    attrs_s = open_match.group(2) or ""
    attrs, order = _parse_attrs(attrs_s)

    if tag == "previousMessage":
        close_re = _CLOSE_PREV_RE
        cls = PreviousMessageEntity
    else:
        close_re = _CLOSE_MSG_RE
        cls = MessageEntity

    close_m = close_re.search(text, pos=open_match.end())
    if not close_m:
        return None, open_match.end()

    body = text[open_match.end(): close_m.start()]
    ent = cls(text=body, attrs=attrs, attr_order=order)
    return ent, close_m.end()


def _entities_in_span(s: str, registry: EntityRegistry, on_warning: Optional[Callable[[str], None]]) -> List[Entity]:
    out: List[Entity] = []
    for m in _SELF_CLOSING_RE.finditer(s):
        tag = m.group(1)
        attrs_s = m.group(2) or ""
        attrs, order = _parse_attrs(attrs_s)

        ent = registry.try_build(tag=tag, attrs=attrs, attr_order=order)
        if ent is None or not ent.validate():
            if on_warning:
                on_warning(f"Failed to build entity for <{tag} ... /> with provided attrs.")
            continue
        out.append(ent)

    # Deduplicate by identity within the span (keep first, warn):
    seen: set[tuple] = set()
    deduped: List[Entity] = []
    for e in out:
        key = e.identity_key()
        if key in seen and not e.__class__ is GenericEntity:
            if on_warning:
                on_warning(f"Duplicate entity identity within phase: {e.__class__.__name__} {key} (keeping first)")
            continue
        seen.add(key)
        deduped.append(e)
    return deduped


def parse_text(
    text: str,
    *,
    registry: Optional[EntityRegistry] = None,
    on_warning: Optional[Callable[[str], None]] = None,
    source_name: Optional[str] = None,
) -> List[Datapoint]:
    """Parse entire text into a list of Datapoint objects.

    Unknown/invalid datapoints are skipped. The parser is non-destructive and
    does not attempt recovery inside malformed blocks.
    """
    reg = registry or default_registry()
    datapoints: List[Datapoint] = []

    i = 0
    N = len(text)
    while i < N:
        open_prev = _OPEN_ANCHOR_RE.search(text, pos=i)
        if not open_prev or open_prev.group(1) != "previousMessage":
            break

        prev_ent, after_prev = _build_anchor("previousMessage", open_prev, text)
        if prev_ent is None:
            # malformed; advance to avoid infinite loop
            i = open_prev.end()
            continue

        # Collect entities until <message>
        open_msg = _OPEN_ANCHOR_RE.search(text, pos=after_prev)
        if not open_msg or open_msg.group(1) != "message":
            i = after_prev
            if on_warning:
                on_warning("Skipping block: missing <message> after <previousMessage>.")
            continue

        before_msg_span = text[after_prev: open_msg.start()]
        prev_entities = _entities_in_span(before_msg_span, reg, on_warning)

        msg_ent, after_msg = _build_anchor("message", open_msg, text)
        if msg_ent is None:
            i = open_msg.end()
            if on_warning:
                on_warning("Skipping block: malformed <message> closing tag.")
            continue

        # Next datapoint starts at next <previousMessage>; entities span ends before that
        next_prev = _OPEN_ANCHOR_RE.search(text, pos=after_msg)
        after_msg_to = next_prev.start() if (next_prev and next_prev.group(1) == "previousMessage") else N

        after_entities_span = text[after_msg: after_msg_to]
        next_entities = _entities_in_span(after_entities_span, reg, on_warning)

        if not isinstance(prev_ent, PreviousMessageEntity) or not isinstance(msg_ent, MessageEntity) or len(next_entities) == 0:
            if on_warning:
                on_warning("Skipping block: anchors not recognized as proper entities.")
            i = after_msg
            continue

        # Compute 1-based line number where this datapoint starts (opening <previousMessage>)
        start_idx = open_prev.start()
        line_no = text.count("\n", 0, start_idx) + 1
        file_part = os.path.basename(source_name) if source_name else None
        sid = f"{file_part}:{line_no}" if file_part is not None else ""

        if len(next_entities) == 1:
            subj = getattr(next_entities[0], "subject", None)
            subj_id = getattr(subj, "id", None) if subj else None
            dp = Datapoint(
                previous_message=prev_ent,  # type: ignore[arg-type]
                message=msg_ent,             # type: ignore[arg-type]
                previous_state=EntityList(prev_entities),
                state=EntityList(next_entities),
                source_id=sid,
                original_subject_id=subj_id,
            )
            datapoints.append(dp)
        else:
            for ent in next_entities:
                context_cls = ent.__class__.context_classes
                subj = getattr(ent, "subject", None)
                subj_id = getattr(subj, "id", None) if subj else None

                dp = Datapoint(
                    previous_message=prev_ent,  # type: ignore[arg-type]
                    message=msg_ent,             # type: ignore[arg-type]
                    previous_state=EntityList(prev_entities),
                    state_context=EntityList([ent for ent in next_entities if any(isinstance(ent, cls) for cls in context_cls)]),
                    state=EntityList([ent]),
                    source_id=sid,
                    original_subject_id=subj_id,
                )
                datapoints.append(dp)

        i = after_msg_to

    return datapoints


__all__ = ["parse_text"]
