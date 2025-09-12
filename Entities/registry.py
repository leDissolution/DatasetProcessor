"""
Allowlist registry and factory for constructing typed entities from parsed tags.

Key is tag only. Subject detection/validation is handled by entity constructors
themselves (they pop their own subject keys from attrs). Unknown tags are
passed through as GenericEntity by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .entity import (CharacterEntity, Entity, CharacterStatsEntity, SceneStatsEntity, AugmentEntity, GenericEntity)


@dataclass(frozen=True)
class EntityRule:
    entity_cls: Type[Entity]


class EntityRegistry:
    """Registry mapping tag -> ordered list of entity classes.

    try_build tries each class for a tag in registration order until one
    successfully constructs. Unknown tags are passed through as GenericEntity.
    """

    def __init__(self) -> None:
        self._rules_by_tag: Dict[str, List[EntityRule]] = {}

    def register(self, tag: str, rule: EntityRule) -> None:
        self._rules_by_tag.setdefault(tag, []).append(rule)

    def is_known(self, tag: str) -> bool:
        return tag in self._rules_by_tag and len(self._rules_by_tag[tag]) > 0

    def try_build(
        self,
        *,
        tag: str,
        attrs: Dict[str, str],
        attr_order: Optional[list[str]] = None,
    ) -> Optional[Entity]:
        rules = self._rules_by_tag.get(tag)
        if not rules:
            return GenericEntity(tag=tag, attrs=attrs, attr_order=attr_order)

        for rule in rules:
            cls = rule.entity_cls
            try:
                attrs_copy = dict(attrs)
                order_copy = list(attr_order) if attr_order is not None else None
                ent = cls(attrs=attrs_copy, attr_order=order_copy)
            except Exception:
                continue

            try:
                if ent.validate():
                    return ent
            except Exception:
                continue

        return GenericEntity(tag=tag, attrs=attrs, attr_order=attr_order)


def default_registry() -> EntityRegistry:
    reg = EntityRegistry()

    reg.register("augment", EntityRule(entity_cls=AugmentEntity))

    reg.register("stats", EntityRule(entity_cls=CharacterStatsEntity))
    reg.register("stats", EntityRule(entity_cls=SceneStatsEntity))

    reg.register("character", EntityRule(entity_cls=CharacterEntity))

    return reg


__all__ = [
    "EntityRegistry",
    "EntityRule",
    "default_registry",
]
