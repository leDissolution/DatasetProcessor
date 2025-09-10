"""
Datapoint container that exposes messages explicitly while keeping everything as entities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Type, TypeVar, Generic, Optional, overload, cast

from Entities.entity import Entity, PreviousMessageEntity, MessageEntity


T = TypeVar("T", bound=Entity)
U = TypeVar("U", bound=Entity)


class EntityList(List[T], Generic[T]):
    def of_type(self, cls: Type[U]) -> "EntityList[U]":
        filtered = [e for e in self if isinstance(e, cls)]
        return cast("EntityList[U]", EntityList(filtered))

    def where(self, predicate: Callable[[T], bool]) -> "EntityList[T]":
        return EntityList([e for e in self if predicate(e)])

    def count_of(self, predicate: Callable[[T], bool]) -> int:
        return len(self.where(predicate))

    def find_matching(self, ref: U) -> Optional[U]:
        """Find the entity matching `ref`'s concrete class and subject.id.

        - The match must be `isinstance(e, ref.__class__)` and have the same
          `subject.id`.
        - Raises ValueError if `ref` has no subject or subject.id.
        - Raises ValueError if more than one match is found.

        Returns the single match or None if not found.
        """
        subject = ref.subject
        if subject is None or subject.id is None:
            raise ValueError("find_matching requires a reference entity with subject.id")

        matches: list[U] = []
        for e in self:  # type: ignore[assignment]
            if isinstance(e, ref.__class__):
                if e.subject.id == subject.id:
                    matches.append(cast(U, e))
        if len(matches) > 1:
            raise ValueError("find_matching found multiple matches for class and subject.id")
        return matches[0] if matches else None

    def target_value(self, target: "Target", default: Optional[str] = None) -> Optional[str]:
        """Return the attribute value for target, or default if not found.

        Looks for the first entity matching target.entity_cls and target.subject_id
        (if provided), then returns entity.attrs.get(target.attr) when present.
        """
        for e in self:  # type: ignore[assignment]
            if isinstance(e, target.entity_cls):
                subj_ok = True
                sid = getattr(getattr(e, "subject", None), "id", None)
                if target.subject_id is not None and sid != target.subject_id:
                    subj_ok = False
                if subj_ok:
                    val = getattr(e, "attrs", {}).get(target.attr)
                    if val is not None:
                        return val
        return default

    def set_target_value(self, target: "Target", value: str) -> None:
        """Set the attribute value for target."""
        for e in self:  # type: ignore[assignment]
            if isinstance(e, target.entity_cls):
                subj_ok = True
                sid = getattr(getattr(e, "subject", None), "id", None)
                if target.subject_id is not None and sid != target.subject_id:
                    subj_ok = False
                if subj_ok:
                    e.attrs[target.attr] = value
                    return

    def target_entity(self, target: "Target") -> Optional[Entity]:
        """Return the single entity matching the Target or None if not found.

        Matching rules:
        - isinstance(e, target.entity_cls)
        - If target.subject_id is provided, the entity's subject.id must equal it.

        Raises ValueError if more than one matching entity is found.
        """
        matches: list[Entity] = []
        for e in self:  # type: ignore[assignment]
            if isinstance(e, target.entity_cls):
                sid = getattr(getattr(e, "subject", None), "id", None)
                if target.subject_id is not None and sid != target.subject_id:
                    continue
                matches.append(e)
        if len(matches) > 1:
            raise ValueError("target_entity found multiple matches for Target")
        return matches[0] if matches else None

    @overload
    def first(self) -> Optional[T]:
        ...

    @overload
    def first(self, default: T) -> T:
        ...

    def first(self, default: Optional[T] = None) -> Optional[T]:
        for e in self:
            return e
        return default


@dataclass
class Datapoint:
    previous_message: PreviousMessageEntity
    message: MessageEntity
    previous_state: EntityList[Entity] = field(default_factory=EntityList)
    state_context: EntityList[Entity] = field(default_factory=EntityList)  # part of new state that is not trained on
    state: EntityList[Entity] = field(default_factory=EntityList)
    # Metadata: training target for this datapoint (optional)
    target: Optional["Target"] = None
    # Metadata: origin of the datapoint as "{filename}:{line}"
    source_id: str = ""

    loss_metrics: Optional["LossMetrics"] = None

    def __post_init__(self) -> None:
        if not isinstance(self.previous_state, EntityList):
            self.previous_state = EntityList(self.previous_state)
        if not isinstance(self.state, EntityList):
            self.state = EntityList(self.state)

    def all_entities(self) -> EntityList[Entity]:
        """Return all entities including anchors in order: prevMsg, prev_state..., msg, state..."""
        return EntityList([self.previous_message, *self.previous_state, self.message, *self.state])

    def rehydrate_prompt(self) -> str:
        """Render the datapoint back into the textual format used for training prompts."""
        parts: List[str] = []
        parts.append(self.previous_message.rehydrate())
        for ent in self.previous_state:
            parts.append(ent.rehydrate())
        parts.append(self.message.rehydrate())
        for ent in self.state_context:
            parts.append(ent.rehydrate())
        for ent in self.state:
            parts.append(ent.rehydrate())
        return "\n".join(parts)

    def to_json(self) -> Dict[str, Any]:
        """Serialize datapoint to JSON including prompt and metadata."""
        target_json: Optional[Dict[str, Any]]
        if self.target is None:
            target_json = None
        else:
            target_json = {
                "entity": getattr(self.target.entity_cls, "__name__", str(self.target.entity_cls)),
                "subject_id": self.target.subject_id,
                "attr": self.target.attr,
            }

        loss_metrics_json = None if self.loss_metrics is None else {
            "completion_difficulty": self.loss_metrics.completion_difficulty,
            "mean_loss": self.loss_metrics.mean_loss,
            "worst_loss": self.loss_metrics.worst_loss,
            "critical_token": self.loss_metrics.critical_token,
        }

        prompt = self.rehydrate_prompt()
        expected_value = self.state.target_value(self.target) if self.target else prompt.rsplit("=\"", 1)[-1]
        if (self.target and expected_value == "!!no_change!!"):
            expected_value = self.previous_state.target_value(self.target, default=expected_value)

        out: Dict[str, Any] = {
            "prompt": prompt,
            "id": self.source_id + ":" + str(self.target.attr if self.target else ""),
            "source_id": self.source_id,
            "target": target_json,
            "loss_metrics": loss_metrics_json,
            "expected_value": expected_value,
        }
        return out

    def to_raw_json(self) -> Dict[str, Any]:
        """Serialize the raw datapoint structure for debugging/auditing.

        Includes anchors, states, subjects, attributes, and rehydrated text.
        Safe for JSONL embedding.
        """

        def _entity_to_json(e: Entity) -> Dict[str, Any]:
            subj = getattr(e, "subject", None)
            subj_json: Optional[Dict[str, Any]] = None
            if subj is not None:
                subj_json = {"key": getattr(subj, "key", None), "id": getattr(subj, "id", None)}
            out: Dict[str, Any] = {
                "type": e.__class__.__name__,
                "family": getattr(e, "family", None),
                "tag": getattr(e, "tag", None),
                "subject": subj_json,
                "attrs": dict(getattr(e, "attrs", {}) or {}),
                "attr_order": list(getattr(e, "attr_order", []) or []) or None,
                "rehydrated": e.rehydrate(),
            }
            # Anchors (previous/message) carry text content.
            text_val = getattr(e, "text", None)
            if isinstance(text_val, str):
                out["text"] = text_val
            return out

        def _target_json() -> Optional[Dict[str, Any]]:
            if self.target is None:
                return None
            return {
                "entity": getattr(self.target.entity_cls, "__name__", str(self.target.entity_cls)),
                "subject_id": self.target.subject_id,
                "attr": self.target.attr,
            }

        return {
            "previous_message": _entity_to_json(self.previous_message),
            "message": _entity_to_json(self.message),
            "previous_state": [_entity_to_json(e) for e in self.previous_state],
            "state_context": [_entity_to_json(e) for e in self.state_context],
            "state": [_entity_to_json(e) for e in self.state],
            "target": _target_json(),
            "source_id": self.source_id,
        }

    def clone(self, **overrides: Any) -> "Datapoint":
        """Create a shallow clone with optional field overrides.

        By default, list-like fields are shallow-copied into new EntityList instances
        to reduce accidental cross-mutation between datapoints. Pass explicit values
        in overrides to replace any field.
        """
        data: Dict[str, Any] = {
            "previous_message": self.previous_message,
            "message": self.message,
            "previous_state": EntityList(self.previous_state),
            "state_context": EntityList(self.state_context),
            "state": EntityList(self.state),
            "target": self.target,
            "source_id": self.source_id,
            "loss_metrics": (
                None
                if self.loss_metrics is None
                else LossMetrics(
                    completion_difficulty=self.loss_metrics.completion_difficulty,
                    mean_loss=self.loss_metrics.mean_loss,
                    worst_loss=self.loss_metrics.worst_loss,
                    critical_token=(
                        None
                        if self.loss_metrics.critical_token is None
                        else dict(self.loss_metrics.critical_token)
                    ),
                )
            ),
        }
        data.update(overrides)

        if not isinstance(data.get("previous_state"), EntityList):
            data["previous_state"] = EntityList(data.get("previous_state") or [])
        if not isinstance(data.get("state"), EntityList):
            data["state"] = EntityList(data.get("state") or [])

        return Datapoint(**data)


@dataclass
class LossMetrics:
    completion_difficulty: Optional[float] = None
    mean_loss: Optional[float] = None
    worst_loss: Optional[float] = None
    critical_token: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class Target:
    """Composite key identifying the training target for a datapoint.

    entity_cls: the entity type to target (e.g., CharacterStatsEntity)
    subject_id: optional subject ID to disambiguate which entity instance
    attr:       the attribute name within the entity to predict/learn
    """
    entity_cls: Type[Entity]
    subject_id: Optional[str]
    attr: str

__all__ = ["Datapoint", "EntityList", "Target", "LossMetrics"]