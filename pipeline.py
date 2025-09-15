from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Dict
import random
from functools import lru_cache
import regex as re

from Entities.registry import EntityRegistry, default_registry
from Entities.entity import AugmentEntity, CharacterStatsEntity, Entity, SceneStatsEntity, StatsEntity, PreviousMessageEntity, MessageEntity, MetaEntity, Subject
from datapoint import Datapoint, EntityList, Target


@dataclass
class PipelineContext:
    rng: random.Random
    registry: EntityRegistry
    split: Optional[str] = None

class DatapointPass:
    name: str = "DatapointPass"

    def applies_to(self, dp: Datapoint, ctx: PipelineContext) -> bool:
        return True

    def transform(self, dp: Datapoint, ctx: PipelineContext) -> List[Datapoint]:
        raise NotImplementedError

class AugmentationPass(DatapointPass):
    name: str = "AugmentationPass"

def run_pipeline(
    datapoints: Iterable[Datapoint],
    passes: Sequence[DatapointPass],
    *,
    ctx: Optional[PipelineContext] = None,
    seed: int = 42,
) -> List[Datapoint]:
    if ctx is None:
        ctx = PipelineContext(rng=random.Random(seed), registry=default_registry())

    current: List[Datapoint] = list(datapoints)
    for p in passes:
        next_batch: List[Datapoint] = []
        for dp in current:
            if not p.applies_to(dp, ctx):
                next_batch.append(dp)
                continue
            out = p.transform(dp, ctx)
            if not out:
                continue
            next_batch.extend(out)
        current = next_batch
    return current

class MetaUpdatedPass(DatapointPass):
    name: str = "MetaUpdatedPass"

    def applies_to(self, dp: Datapoint, ctx: PipelineContext) -> bool:
        return len(dp.state.of_type(StatsEntity)) == 1

    def transform(self, dp: Datapoint, ctx: PipelineContext) -> List[Datapoint]:
        new_stats = dp.state.of_type(StatsEntity)[0]
        old_stats = dp.previous_state.find_matching(new_stats)

        if not old_stats or not new_stats.subject or not old_stats.subject:
            return [dp]

        config = new_stats.stats_config()
        concrete_fields = [
            k for k, v in config.items()
            if (v.get("concrete", False) is True)
        ]

        if not concrete_fields:
            return [dp]

        # Both states must contain all concrete attributes
        if not all(k in new_stats.attrs for k in concrete_fields):
            return [dp]
        if not all(k in old_stats.attrs for k in concrete_fields):
            return [dp]

        updated = any(new_stats.attrs[k] != old_stats.attrs[k] for k in concrete_fields)
        meta_entity = MetaEntity.updated_meta(
            subject=new_stats.subject,
            updated=updated
        )

        new_dp = dp.clone(
            previous_state=EntityList(dp.previous_state),
            state=EntityList([meta_entity]),
            target=Target(
                entity_cls=StatsEntity,
                subject_id=new_stats.subject.id,
                attr="updated",
            ),
        )

        return [dp, new_dp]

class ReplaceUnchangedWithNoopPass(DatapointPass):
    name = "ReplaceUnchangedWithNoopPass"

    def applies_to(self, dp: Datapoint, ctx: PipelineContext) -> bool:
        if dp.target is None:
            return False

        new_value = dp.state.target_value(dp.target)
        prev_value = dp.previous_state.target_value(dp.target)

        return new_value is not None and prev_value is not None and new_value == prev_value

    def transform(self, dp: Datapoint, ctx: PipelineContext) -> List[Datapoint]:
        if dp.target is None:
            return [dp]

        dp.state.set_target_value(dp.target, "!!no_change!!")
        return [dp]

class StatUnwinderPass(AugmentationPass):
    """Expand a single new-state stats entity into progressive examples."""

    name = "StatUnwinderPass"

    def applies_to(self, dp: Datapoint, ctx: PipelineContext) -> bool:
        return len(dp.state) == 1 and isinstance(dp.state[0], StatsEntity)

    def transform(self, dp: Datapoint, ctx: PipelineContext) -> List[Datapoint]:
        ent = dp.state[0]
        if not isinstance(ent, StatsEntity):
            return [dp]
        cfg = ent.stats_config()
        attrs = dict(ent.attrs)
        order_map = {k: v.get("order", 9999) for k, v in cfg.items()}

        def collect(stat: str, acc: set[str]) -> None:
            if stat in acc:
                return
            acc.add(stat)
            for dep in cfg.get(stat, {}).get("dependencies", []):
                collect(dep, acc)

        fields_in_order = ent.attr_order or list(attrs.keys())
        requested_stats = [k for k in fields_in_order if k in attrs]

        out: List[Datapoint] = []
        seen_signatures: set[tuple] = set()
        
        subject_key = ent.subject.key if ent.subject and ent.subject.key else None
        subject_id = ent.subject.id if ent.subject and ent.subject.id else None
        if not subject_key or not subject_id:
            return [dp]

        for stat in requested_stats:
            relevant: set[str] = set()
            collect(stat, relevant)
            # Unknown stats will only include themselves in 'relevant'
            relevant_ordered = sorted(relevant, key=lambda k: order_map.get(k, 9999))

            progressive: List[str] = []
            for field in relevant_ordered:
                progressive.append(field)
                step_attrs: dict[str, str] = {}
                for f in progressive:
                    if f in attrs:
                        step_attrs[f] = attrs[f]

                if not step_attrs:
                    continue

                signature = (
                    ent.__class__,
                    subject_key,
                    subject_id,
                    tuple(sorted(step_attrs.items())),
                )
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)

                step_order = [subject_key] + [
                    f for f in sorted(step_attrs.keys(), key=lambda k: order_map.get(k, 9999))
                ]
                ctor_attrs = {subject_key: subject_id, **step_attrs}

                try:
                    new_ent = ent.__class__(attrs=ctor_attrs, attr_order=step_order)
                except Exception:
                    continue

                new_dp = dp.clone(
                    previous_state=EntityList(dp.previous_state),
                    state=EntityList([new_ent]),
                    target=Target(entity_cls=ent.__class__, subject_id=subject_id, attr=field),
                )
                out.append(new_dp)

        return out or [dp]


class MirrorEntitiesPass(AugmentationPass):
    name = "MirrorEntitiesPass"

    def __init__(self, probability: float = 1.0, base_cls: type[Entity] = StatsEntity) -> None:
        self.probability = probability
        self.base_cls = base_cls

    def applies_to(self, dp: Datapoint, ctx: PipelineContext) -> bool:
        prev = dp.previous_state

        counts: dict[type, int] = {}
        for e in prev:
            if isinstance(e, self.base_cls):
                counts[e.__class__] = counts.get(e.__class__, 0) + 1

        return any(c >= 2 for c in counts.values())

    def transform(self, dp: Datapoint, ctx: PipelineContext) -> List[Datapoint]:
        if ctx.rng.random() >= self.probability:
            return [dp]

        prev = EntityList(dp.previous_state)
        n = len(prev)
        if n == 0:
            return [dp]

        by_class: dict[type, list[int]] = {}
        for idx, e in enumerate(prev):
            if isinstance(e, self.base_cls):
                by_class.setdefault(e.__class__, []).append(idx)

        changed = False

        for cls, idxs in by_class.items():
            if len(idxs) < 2:
                continue

            seq = [prev[i] for i in idxs]
            seq.reverse()

            for i, pos in enumerate(idxs):
                if prev[pos] is not seq[i]:
                    changed = True
                prev[pos] = seq[i]

        if not changed:
            return [dp]

        mirrored = dp.clone(
            previous_state=prev,
            state=EntityList(dp.state),
        )

        return [dp, mirrored]

class DuplicateAugmentationPass(AugmentationPass):
    name = "DuplicateAugmentationPass"

    def applies_to(self, dp: Datapoint, ctx: PipelineContext) -> bool:
        return dp.previous_state.of_type(AugmentEntity).count_of(lambda e: e.duplicate is not None) > 0

    def transform(self, dp: Datapoint, ctx: PipelineContext) -> List[Datapoint]:
        first_entity = dp.previous_state.of_type(AugmentEntity).first()
        times = getattr(first_entity, "duplicate", 1) if first_entity is not None else 1

        return [dp] * times

class DuplicateChangeAugmentationPass(AugmentationPass):
    name = "DuplicateChangeAugmentationPass"

    def __init__(self, target_stat: str, multiplier: float) -> None:
        super().__init__()
        self.target_stat = target_stat
        self.multiplier = multiplier

    def applies_to(self, dp: Datapoint, ctx: PipelineContext) -> bool:
        if dp.target is None:
            return False

        if dp.target.attr != self.target_stat:
            return False

        new_value = dp.state.target_value(dp.target)
        prev_value = dp.previous_state.target_value(dp.target)

        return new_value is not None and prev_value is not None and new_value != prev_value

    def transform(self, dp: Datapoint, ctx: PipelineContext) -> List[Datapoint]:
        m = max(1.0, self.multiplier)
        base = int(m)
        frac = m - base
        extra = 1 if ctx.rng.random() < frac else 0
        copies = base + extra
        return [dp] * copies


class NameRandomizerPass(AugmentationPass):
    """Randomize character names consistently across entities and messages.

    Behavior is migrated from NameRandomizerAugmentation:
    - Every original name maps to a unique simple base name within the sample.
    - With probability pcomp, an original may also map to a composite form.
    - Replacement rules:
        attributes: base if not flagged, composite if flagged
        prose:      base for simple names; composite for compounds when flagged
    - Excludes specific names from replacement.
    """

    name = "NameRandomizerPass"

    NAMES_TO_EXCLUDE: Set[str] = {
        "System", "Rat", "Older boy", "Orion (older boy)", "Teen one", "Teen two"
    }

    NAMES = [
        "Tion", "Bailey", "Casey", "Drew", "Eden", "Finn", "Gray", "Harper",
        "Indigo", "Jordan", "Kennedy", "Logan", "Morgan", "Nova", "Parker",
        "Quinn", "Riley", "Sage", "Taylor", "Unity", "Kim", "Winter", "Xen",
        "Yuri", "Zen", "Ash", "Blake", "Charlie", "Dana", "Eli", "Francis",
        "Tay", "Haven", "Ivy", "Jamie", "Kit", "Lee", "Max", "Nico",
        "Ocean", "Peyton", "River", "Sky", "Tate", "Uriel", "Rose", "Wren",
        "Xavier", "Yael", "Zephyr", "Aiden", "Blair", "Cedar", "Dell",
        "Echo", "Fae", "Glen", "Hazel", "Isle", "Jay", "Kai", "Lake",
        "Mare", "Nell", "Oak", "Pine", "Rain", "Storm", "Tide", "Uma",
        "Vex", "Wave", "Xael", "Yang", "Zale", "Atlas", "Brook", "Cloud",
        "Dune", "East", "Frost", "Grove", "Heath", "Inlet", "June", "Kite",
        "Lark", "Mist", "North", "Orion", "Peak", "Quest", "Reed", "Star",
        "Trace", "Umber", "Vale", "West", "Kin'astra", "Yarrow", "Franco"
    ]

    _COMPOSITE_TEMPLATES = [
        "{name}, the {title}",
        "{title} {name}",
        "{name} the {title}",
        "{name}, {title}",
    ]
    _TITLES = [
        "Elder", "Wise", "Brave", "Swift", "Strong", "Gentle", "Kind", "Bold",
        "Quiet", "Fierce", "Calm", "Young", "Old", "Tall", "Small", "Great",
        "Noble", "Fair", "Dark", "Bright", "Silent", "Loud", "Quick", "Slow",
        "Sharp",
    ]

    _compound_re = re.compile(r"[\s,-]")

    def __init__(self, composite_probability: float = 0.30) -> None:
        self.pcomp = composite_probability

    @staticmethod
    def _is_compound(name: str) -> bool:
        return bool(NameRandomizerPass._compound_re.search(name))

    @staticmethod
    def _components(full: str) -> List[str]:
        titles = {
            "mr", "mrs", "ms", "dr", "prof", "father", "mother",
            "sir", "lady", "lord", "the",
        }
        raw = re.split(r"[\s,--]+", full.lower())
        return [c.capitalize() for c in raw if c and c not in titles]

    def _new_base(self, used: Set[str], ctx: PipelineContext) -> str:
        pool = [n for n in self.NAMES if n not in used and n not in self.NAMES_TO_EXCLUDE]
        if not pool:
            used.clear()
            pool = [n for n in self.NAMES if n not in self.NAMES_TO_EXCLUDE]
        name = ctx.rng.choice(pool)
        used.add(name)
        return name

    def _composite(self, base: str, ctx: PipelineContext) -> str:
        tpl = ctx.rng.choice(self._COMPOSITE_TEMPLATES)
        title = ctx.rng.choice(self._TITLES)
        return tpl.format(name=base, title=title)

    @staticmethod
    @lru_cache(maxsize=1024)
    def _compile_name_pattern(original: str) -> re.Pattern:
        o = re.escape(original)
        pattern = rf"\b{o}(?:\'s|\'|\b)"
        return re.compile(pattern)

    def _collect_originals(self, dp: Datapoint) -> Set[str]:
        originals: Set[str] = set()
        def _from_entities(entities: EntityList) -> None:
            for e in entities:
                if not isinstance(e, StatsEntity):
                    continue
                # Only collect subjects from character stats
                if isinstance(e, CharacterStatsEntity) or (e.subject and e.subject.key == "character"):
                    sid = e.subject.id
                    if isinstance(sid, str) and sid:
                        originals.add(sid)
        _from_entities(dp.previous_state)
        _from_entities(dp.state)
        return originals - self.NAMES_TO_EXCLUDE

    def applies_to(self, dp: Datapoint, ctx: PipelineContext) -> bool:
        return bool(self._collect_originals(dp))

    def transform(self, dp: Datapoint, ctx: PipelineContext) -> List[Datapoint]:
        originals = self._collect_originals(dp)
        if not originals:
            return [dp]

        component_parent: Dict[str, str] = {}
        for full in list(originals):
            for c in self._components(full):
                component_parent[c] = full
                originals.add(c)

        used_bases: Set[str] = set()
        base_map: Dict[str, str] = {}
        comp_flag: Dict[str, bool] = {}
        comp_cache: Dict[str, str] = {}

        for orig in originals:
            parent = component_parent.get(orig, orig)
            if parent not in base_map:
                base_map[parent] = self._new_base(used_bases, ctx)
                comp_flag[parent] = ctx.rng.random() < self.pcomp
            base_map[orig] = base_map[parent]
            comp_flag[orig] = comp_flag[parent]

        def get_composite(base: str) -> str:
            if base not in comp_cache:
                comp_cache[base] = self._composite(base, ctx)
            return comp_cache[base]

        def replace_attr_value(val: str) -> str:
            if val in self.NAMES_TO_EXCLUDE:
                return val
            base = base_map.get(val)
            if not base:
                return val
            return get_composite(base) if comp_flag.get(val, False) else base

        def replace_in_attr(value: str) -> str:
            s = value or ""
            for orig in sorted(originals, key=len, reverse=True):
                base = base_map[orig]
                repl = get_composite(base) if (self._is_compound(orig) and comp_flag.get(orig, False)) else base
                pat = self._compile_name_pattern(orig)
                def _sub(m: re.Match) -> str:
                    tok = m.group(0)
                    if tok.endswith("'s"):
                        return f"{repl}'s"
                    if tok.endswith("'"):
                        return f"{repl}'"
                    return repl
                s = pat.sub(_sub, s)
            return s

        def remap_entities(entities: EntityList[Entity]) -> EntityList[Entity]:
            out: List[Entity] = []
            for e in entities:
                new_attrs = dict(e.attrs)
                for k, v in list(new_attrs.items()):
                    if isinstance(v, str):
                        new_attrs[k] = replace_in_attr(v)

                skey = e.subject.key if e.subject else None
                sid = e.subject.id if e.subject else None
                if skey and isinstance(sid, str) and sid:
                    if isinstance(e, SceneStatsEntity):
                        new_attrs[skey] = replace_in_attr(sid)
                    else:
                        new_attrs[skey] = replace_attr_value(sid)
                try:
                    new_e = e.__class__(attrs=new_attrs, attr_order=getattr(e, "attr_order", None))
                    if skey and isinstance(sid, str) and sid and (new_e.subject is None or new_e.subject.key is None):
                        new_e.subject = Subject(key=skey, id=new_attrs[skey])
                        del new_e.attrs[skey]
                except Exception:
                    new_e = e
                out.append(new_e)
            return EntityList(out)

        new_prev = remap_entities(dp.previous_state)
        new_context = remap_entities(dp.state_context)
        new_state = remap_entities(dp.state)

        def replace_in_text(text: str) -> str:
            s = text or ""
            for orig in sorted(originals, key=len, reverse=True):
                base = base_map[orig]
                repl = get_composite(base) if (self._is_compound(orig) and comp_flag.get(orig, False)) else base
                pat = self._compile_name_pattern(orig)
                def _sub(m: re.Match) -> str:
                    tok = m.group(0)
                    if tok.endswith("'s"):
                        return f"{repl}'s"
                    if tok.endswith("'"):
                        return f"{repl}'"
                    return repl
                s = pat.sub(_sub, s)
            return s

        prev_attrs = dict(dp.previous_message.attrs)
        for k, v in list(prev_attrs.items()):
            if isinstance(v, str):
                prev_attrs[k] = replace_in_attr(v)
        new_prev_msg = PreviousMessageEntity(
            text=replace_in_text(dp.previous_message.text),
            attrs=prev_attrs,
            attr_order=list(dp.previous_message.attr_order) if dp.previous_message.attr_order else None,
        )

        msg_attrs = dict(dp.message.attrs)
        # 'from' is a name token: apply direct mapping
        if isinstance(msg_attrs.get("from"), str):
            msg_attrs["from"] = replace_attr_value(msg_attrs["from"])
        for k, v in list(msg_attrs.items()):
            if k == "from":
                continue
            if isinstance(v, str):
                msg_attrs[k] = replace_in_attr(v)
        new_msg = MessageEntity(
            text=replace_in_text(dp.message.text),
            attrs=msg_attrs,
            attr_order=list(dp.message.attr_order) if dp.message.attr_order else None,
        )

        new_target = dp.target
        if dp.target is not None and isinstance(dp.target.subject_id, str):
            sid = dp.target.subject_id
            if sid not in self.NAMES_TO_EXCLUDE and sid in base_map:
                base = base_map[sid]
                new_sid = get_composite(base) if comp_flag.get(sid, False) else base
                if new_sid != sid:
                    new_target = Target(
                        entity_cls=dp.target.entity_cls,
                        subject_id=new_sid,
                        attr=dp.target.attr,
                    )

        new_dp = dp.clone(
            previous_message=new_prev_msg,
            message=new_msg,
            previous_state=new_prev,
            state=new_state,
            target=new_target,
            state_context=new_context,
        )

        return [new_dp]

__all__ = [
    "PipelineContext",
    "DatapointPass",
    "AugmentationPass",
    "run_pipeline",
    "StatUnwinderPass",
    "MirrorEntitiesPass",
    "DuplicateAugmentationPass",
    "DuplicateChangeAugmentationPass",
    "NameRandomizerPass",
]
