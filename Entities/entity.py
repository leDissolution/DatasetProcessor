"""
Foundational entity model used by the dataset processor.

Goals this module supports:
- Everything is an entity (including anchors), but messages are also exposed separately on Datapoint.
- Exact rehydration of tags (e.g., <stats character="a" ... /> vs <stats scene="room" ... />)
- Easy diffing via stable identity keys.
- Strings-only attributes; unknown entities can be filtered by a registry.

This module intentionally does not modify existing code paths. It provides
building blocks to be used by a future parser/state machine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, ClassVar, Type, TypedDict


@dataclass(frozen=True)
class Subject:
    """Represents the subject of an entity (e.g., a character or a scene).

    key: which attribute key declared the subject (e.g., "character", "scene").
    id:  the subject identifier (e.g., "a", "room-1").

    Both can be None for subject-less entities (if allowed by the entity type).
    """

    key: Optional[str] = None
    id: Optional[str] = None


class Entity(ABC):
    """Abstract base for all entities.

    Attributes
    ----------
    family: Logical family (e.g., "stats", "meta", "anchor").
    tag:    Concrete tag name used for rehydration (e.g., "stats", "meta",
            "previousMessage", "message").
    subject: Subject of this entity (may be None when not required).
    attrs:   Attributes (strings only). Subject attribute is not duplicated here.
    attr_order: Optional order of attributes as they originally appeared (for
                lossless-ish rehydration/debuggability).
    """

    family: str = "entity"
    tag: str = "entity"

    def __init__(
        self,
        *,
        subject: Optional[Subject] = None,
        attrs: Optional[Dict[str, str]] = None,
        attr_order: Optional[List[str]] = None,
    ) -> None:
        self.subject: Subject = subject or Subject()
        self.attrs: Dict[str, str] = dict(attrs or {})
        # Preserve attribute insertion order if not explicitly provided
        self.attr_order: Optional[List[str]] = (
            list(attrs.keys()) if (attr_order is None and attrs is not None) else attr_order
        )
        self.validate()

    subject_required: bool = False
    context_classes: List[Type[Entity]] = []

    def identity_key(self) -> Tuple[object, Optional[str], Optional[str]]:
        """A stable identity for diffing within a phase.

        By default, uses the concrete class type and subject (key, id).
        Subclasses can override if additional disambiguation is required.
        """

        return (self.__class__, self.subject.key, self.subject.id)

    def validate(self) -> bool:
        """Lightweight validation according to policy flags."""

        if self.subject_required:
            if not (self.subject and self.subject.key and self.subject.id):
                return False

        return True

    @staticmethod
    def _attrs_to_str(attrs: Dict[str, str], order: Optional[List[str]] = None) -> str:
        keys = order or list(attrs.keys())
        parts: List[str] = []
        for k in keys:
            if k in attrs:
                v = attrs[k]
                parts.append(f'{k}="{v}"')
        return " ".join(parts)

    @staticmethod
    def _extract_subject_from_attrs(
        attrs: Optional[Dict[str, str]],
        attr_order: Optional[List[str]],
        subject_key: str,
    ) -> Tuple[Subject, Dict[str, str], Optional[List[str]]]:
        """Pop subject_key from attrs to build a Subject and return cleaned attrs/order.

        Returns (subject, new_attrs, new_attr_order)
        """
        new_attrs: Dict[str, str] = dict(attrs or {})
        subject_id = new_attrs.pop(subject_key, None)
        new_order = None if attr_order is None else [k for k in attr_order if k != subject_key]
        return Subject(subject_key, subject_id), new_attrs, new_order

    @abstractmethod
    def rehydrate(self) -> str:
        """Render this entity back to a tag string.

        Implementations should not add the subject attribute into attrs; they
        are responsible for emitting the subject separately (when applicable).
        """


class AnchorEntity(Entity):
    """Base class for anchored entities (previousMessage, message)."""

    family: str = "anchor"
    subject_required: bool = False

    def __init__(
        self,
        *,
        text: str,
        attrs: Optional[Dict[str, str]] = None,
        attr_order: Optional[List[str]] = None,
    ) -> None:
        self.text: str = text
        super().__init__(subject=None, attrs=attrs, attr_order=attr_order)


class PreviousMessageEntity(AnchorEntity):
    tag: str = "previousMessage"

    def rehydrate(self) -> str:
        attrs_part = self._attrs_to_str(self.attrs, self.attr_order)
        attrs_part = (" " + attrs_part) if attrs_part else ""
        return f"<previousMessage{attrs_part}>{self.text}</previousMessage>"


class MessageEntity(AnchorEntity):
    tag: str = "message"

    def rehydrate(self) -> str:
        attrs_part = self._attrs_to_str(self.attrs, self.attr_order)
        attrs_part = (" " + attrs_part) if attrs_part else ""
        return f"<message{attrs_part}>{self.text}</message>"


class StatsEntity(Entity):
    """Base for <stats ... /> entities (subject required)."""

    family: str = "stats"
    tag: str = "stats"
    subject_required: bool = True

    # --- Typed config surface for descendant stats definitions ---
    class StatFieldConfig(TypedDict):
        dependencies: List[str]
        order: int
        defaultValue: str
        concrete: bool

    StatsConfig = Dict[str, StatFieldConfig]

    STAT_CONFIG: ClassVar[StatsConfig] = {}

    def stats_config(self) -> StatsConfig:
        """Return the typed stats configuration for this entity type."""
        return type(self).STAT_CONFIG

    def rehydrate(self) -> str:
        if not (self.subject and self.subject.key and self.subject.id):
            return ""
        subject_part = f"{self.subject.key}=\"{self.subject.id}\""
        attrs_part = self._attrs_to_str(self.attrs, self.attr_order)
        parts = [subject_part]
        if attrs_part:
            parts.append(attrs_part)
        inside = " ".join(parts)
        return f"<{self.tag} {inside} />"


class CharacterStatsEntity(StatsEntity):
    """<stats character="..." ... />"""

    STAT_CONFIG = {
        'pose':        {'dependencies': [],           'order': 0, 'defaultValue': 'unspecified', 'concrete': True},
        'location':    {'dependencies': ['pose'],     'order': 1, 'defaultValue': 'unspecified', 'concrete': True},
        'outfit':      {'dependencies': [],           'order': 2, 'defaultValue': 'unspecified', 'concrete': True},
        'exposure':    {'dependencies': ['outfit'],   'order': 3, 'defaultValue': 'none', 'concrete': True},
        'accessories': {'dependencies': ['outfit'],   'order': 4, 'defaultValue': 'unspecified', 'concrete': True},
        'mood':        {'dependencies': [],           'order': 5, 'defaultValue': 'neutral', 'concrete': False},
        'bodystate':   {'dependencies': [],           'order': 6, 'defaultValue': 'unspecified', 'concrete': True}
    }

    def __init__(
        self,
        *,
        attrs: Optional[Dict[str, str]] = None,
        attr_order: Optional[List[str]] = None,
    ) -> None:
        subject, attrs, attr_order = self._extract_subject_from_attrs(attrs, attr_order, "character")
        super().__init__(subject=subject, attrs=attrs, attr_order=attr_order)


class SceneStatsEntity(StatsEntity):
    """<stats scene="..." ... />"""

    context_classes: List[Type[Entity]] = [CharacterStatsEntity]

    STAT_CONFIG = {
        'fixtures': {'dependencies': [], 'order': 0, 'defaultValue': 'unspecified', 'concrete': True},
        'items': {'dependencies': ['fixtures'], 'order': 1, 'defaultValue': 'unspecified', 'concrete': True}
    }

    def __init__(
        self,
        *,
        attrs: Optional[Dict[str, str]] = None,
        attr_order: Optional[List[str]] = None,
    ) -> None:
        subject, attrs, attr_order = self._extract_subject_from_attrs(attrs, attr_order, "scene")
        super().__init__(subject=subject, attrs=attrs, attr_order=attr_order)


class MetaEntity(Entity):
    """Base for <meta ... /> entities (subject optional by policy)."""

    family: str = "meta"
    tag: str = "meta"
    subject_required: bool = False

    def rehydrate(self) -> str:
        attrs_dict = dict(self.attrs)
        if self.subject and self.subject.key and self.subject.id:
            ordered: List[str] = [self.subject.key]
            attrs_dict[self.subject.key] = self.subject.id
            if self.attr_order:
                ordered.extend([k for k in self.attr_order if k != self.subject.key])
            attrs_part = self._attrs_to_str(attrs_dict, ordered)
        else:
            attrs_part = self._attrs_to_str(attrs_dict, self.attr_order)
        attrs_part = (" " + attrs_part) if attrs_part else ""

        return f"<{self.tag}{attrs_part} />"

    @staticmethod
    def updated_meta(subject: Subject, updated: bool) -> MetaEntity:
        return MetaEntity(subject=subject, attrs={"updated": "true" if updated else "false"}, attr_order=["updated"])

class AugmentEntity(Entity):
    """<augment ... />"""

    def __init__(
        self,
        *,
        attrs: Optional[Dict[str, str]] = None,
        attr_order: Optional[List[str]] = None,
    ) -> None:
        super().__init__(subject=None, attrs=attrs, attr_order=attr_order)

    def rehydrate(self) -> str:
        attrs_part = self._attrs_to_str(self.attrs, self.attr_order)
        attrs_part = (" " + attrs_part) if attrs_part else ""
        return f"<augment{attrs_part} />"

    @property
    def duplicate(self) -> Optional[float]:
        try:
            val = self.attrs.get("duplicate")
            return int(val) if val is not None and val != "" else None
        except Exception:
            return None


class GenericEntity(Entity):
    """Generic passthrough entity for unknown tags.

    Renders back to a self-closing tag preserving attribute order when provided.
    Subject is not inferred; all attributes are kept as-is.
    """

    def __init__(
        self,
        *,
        tag: str,
        attrs: Optional[Dict[str, str]] = None,
        attr_order: Optional[List[str]] = None,
    ) -> None:
        # Set the concrete tag for rehydration
        self.tag = tag
        super().__init__(subject=None, attrs=attrs, attr_order=attr_order)

    def rehydrate(self) -> str:
        attrs_part = self._attrs_to_str(self.attrs, self.attr_order)
        attrs_part = (" " + attrs_part) if attrs_part else ""
        return f"<{self.tag}{attrs_part} />"


__all__ = [
    "Subject",
    "Entity",
    "AnchorEntity",
    "PreviousMessageEntity",
    "MessageEntity",
    "StatsEntity",
    "CharacterStatsEntity",
    "SceneStatsEntity",
    "MetaEntity",
    "AugmentEntity",
    "GenericEntity",
]
