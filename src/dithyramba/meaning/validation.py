"""Exact, policy-closed validation and content addressing for P6 proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from dithyramba.contracts import canonical_content_id, canonical_sha256_hex, sha256_hex

from .errors import MeaningBudgetError, MeaningEvidenceError, MeaningProfileError
from .models import (
    ConceptMeaningProposal,
    CoverageState,
    EntityKind,
    EvidencePolarity,
    ExactMentionProposal,
    ExtractionProfile,
    MeaningInputFragment,
    MeaningOutputManifest,
    MeaningProposal,
    ReviewBudget,
    SemanticAlignment,
    StatementKind,
    VoiceKind,
)

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class PreparedVoice:
    voice_id: str
    kind: VoiceKind
    label: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class PreparedEntity:
    entity_id: str
    kind: EntityKind
    label: str
    aliases: tuple[str, ...]
    content_hash: str


@dataclass(frozen=True, slots=True)
class PreparedConcept:
    concept_id: str
    label: str
    aliases: tuple[str, ...]
    content_hash: str


@dataclass(frozen=True, slots=True)
class PreparedMention:
    mention_id: str
    target_id: str
    collection_id: str
    source_fragment_id: str
    quote_start: int
    quote_end: int
    quote_text: str
    quote_sha256: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class PreparedTimeContext:
    time_context_id: str
    event_valid_start: str | None
    event_valid_end: str | None
    recorded_at: str | None
    available_at: str | None
    revealed_at: str | None
    reviewed_at: str | None
    content_hash: str


@dataclass(frozen=True, slots=True)
class PreparedStatement:
    statement_id: str
    collection_id: str
    kind: StatementKind
    statement_text: str
    voice_id: str
    time_context_id: str | None
    content_hash: str


@dataclass(frozen=True, slots=True)
class PreparedEvidenceLink:
    evidence_link_id: str
    statement_id: str
    source_fragment_id: str
    source_version_id: str
    attributed_voice_id: str
    polarity: EvidencePolarity
    alignment: SemanticAlignment
    limits_text: str | None
    quote_start: int
    quote_end: int
    quote_text: str
    quote_sha256: str
    extraction_method: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class PreparedConceptMeaning:
    concept_meaning_id: str
    concept_id: str
    voice_id: str
    time_context_id: str | None
    meaning_text: str
    statement_ids: tuple[str, ...]
    content_hash: str


@dataclass(frozen=True, slots=True)
class PreparedMeaningGraph:
    voices: tuple[PreparedVoice, ...]
    entities: tuple[PreparedEntity, ...]
    entity_mentions: tuple[PreparedMention, ...]
    concepts: tuple[PreparedConcept, ...]
    concept_mentions: tuple[PreparedMention, ...]
    time_contexts: tuple[PreparedTimeContext, ...]
    statements: tuple[PreparedStatement, ...]
    evidence_links: tuple[PreparedEvidenceLink, ...]
    concept_meanings: tuple[PreparedConceptMeaning, ...]
    output: MeaningOutputManifest


def meaning_input_hash(fragments: tuple[MeaningInputFragment, ...]) -> str:
    """Hash the exact permitted provider/import input without persisting text."""

    return canonical_sha256_hex(
        {
            "schema": "dithyramba.meaning_input/2.0",
            "fragments": [fragment.receipt_payload() for fragment in fragments],
        }
    )


def prepare_meaning_graph(
    *,
    library_id: str,
    profile: ExtractionProfile,
    review_budget: ReviewBudget,
    proposal: MeaningProposal,
    fragments: tuple[MeaningInputFragment, ...],
) -> PreparedMeaningGraph:
    """Resolve, ground, content-address, and cap one complete proposal.

    Every fragment in ``fragments`` has already crossed the repository's
    AccessPolicy boundary. A reference not present in that set is rejected
    without persisting or echoing its identifier.
    """

    if proposal.state is not CoverageState.COMPLETE:
        raise MeaningProfileError("only complete proposals can create semantic candidates")
    if not isinstance(profile, ExtractionProfile) or not isinstance(review_budget, ReviewBudget):
        raise MeaningProfileError("profile and ReviewBudget must be typed")
    _validate_profile_outputs(profile, proposal)

    fragment_by_id = {fragment.source_fragment_id: fragment for fragment in fragments}
    if len(fragment_by_id) != len(fragments):
        raise MeaningEvidenceError("permitted semantic input contains duplicate fragments")

    voice_by_key: dict[str, PreparedVoice] = {}
    for voice_item in proposal.voices:
        payload: dict[str, object] = {
            "schema": "dithyramba.voice/1.0",
            "library_id": library_id,
            "kind": voice_item.kind.value,
            "label": voice_item.label,
            "status": "candidate",
        }
        content_hash = canonical_sha256_hex(payload)
        voice_by_key[voice_item.key] = PreparedVoice(
            canonical_content_id("voice", payload),
            voice_item.kind,
            voice_item.label,
            content_hash,
        )

    entity_by_key: dict[str, PreparedEntity] = {}
    for entity_item in proposal.entities:
        payload = {
            "schema": "dithyramba.entity/1.0",
            "library_id": library_id,
            "kind": entity_item.kind.value,
            "label": entity_item.label,
            "status": "candidate",
        }
        content_hash = canonical_sha256_hex(payload)
        entity_by_key[entity_item.key] = PreparedEntity(
            canonical_content_id("entity", payload),
            entity_item.kind,
            entity_item.label,
            entity_item.aliases,
            content_hash,
        )

    concept_by_key: dict[str, PreparedConcept] = {}
    for concept_item in proposal.concepts:
        payload = {
            "schema": "dithyramba.concept/1.0",
            "library_id": library_id,
            "label": concept_item.label,
            "status": "candidate",
        }
        content_hash = canonical_sha256_hex(payload)
        concept_by_key[concept_item.key] = PreparedConcept(
            canonical_content_id("concept", payload),
            concept_item.label,
            concept_item.aliases,
            content_hash,
        )

    time_by_key: dict[str, PreparedTimeContext] = {}
    for time_item in proposal.time_contexts:
        payload = {
            "schema": "dithyramba.time_context/1.0",
            "library_id": library_id,
            "event_valid_start": time_item.event_valid_start,
            "event_valid_end": time_item.event_valid_end,
            "recorded_at": time_item.recorded_at,
            "available_at": time_item.available_at,
            "revealed_at": time_item.revealed_at,
            "reviewed_at": time_item.reviewed_at,
            "status": "candidate",
        }
        content_hash = canonical_sha256_hex(payload)
        time_by_key[time_item.key] = PreparedTimeContext(
            canonical_content_id("time_context", payload),
            time_item.event_valid_start,
            time_item.event_valid_end,
            time_item.recorded_at,
            time_item.available_at,
            time_item.revealed_at,
            time_item.reviewed_at,
            content_hash,
        )

    entity_mentions = tuple(
        _prepare_mention(
            mention=item,
            target=_require_reference(entity_by_key, item.target_key, "Entity"),
            id_prefix="entity_mention",
            fragments=fragment_by_id,
        )
        for item in proposal.entity_mentions
    )
    concept_mentions = tuple(
        _prepare_mention(
            mention=item,
            target=_require_reference(concept_by_key, item.target_key, "Concept"),
            id_prefix="concept_mention",
            fragments=fragment_by_id,
        )
        for item in proposal.concept_mentions
    )

    statements: list[PreparedStatement] = []
    evidence_links: list[PreparedEvidenceLink] = []
    statement_by_key: dict[str, PreparedStatement] = {}
    statement_source_versions: dict[str, set[str]] = {}
    for statement_item in proposal.statements:
        voice = _require_reference(voice_by_key, statement_item.voice_key, "Voice")
        time_context = (
            None
            if statement_item.time_context_key is None
            else _require_reference(
                time_by_key,
                statement_item.time_context_key,
                "TimeContext",
            )
        )
        statement_payload: dict[str, object] = {
            "schema": "dithyramba.statement/1.0",
            "library_id": library_id,
            "collection_id": statement_item.collection_id,
            "kind": statement_item.kind.value,
            "statement_text": statement_item.statement_text,
            "voice_id": voice.voice_id,
            "time_context_id": (None if time_context is None else time_context.time_context_id),
            "status": "candidate",
            "lifecycle": "active",
        }
        statement_hash = canonical_sha256_hex(statement_payload)
        statement = PreparedStatement(
            canonical_content_id("statement", statement_payload),
            statement_item.collection_id,
            statement_item.kind,
            statement_item.statement_text,
            voice.voice_id,
            None if time_context is None else time_context.time_context_id,
            statement_hash,
        )
        versions: set[str] = set()
        for evidence in statement_item.evidence:
            if evidence.attributed_voice_key != statement_item.voice_key:
                raise MeaningEvidenceError(
                    "EvidenceLink attribution must retain the Statement Voice"
                )
            fragment = _require_fragment(fragment_by_id, evidence.source_fragment_id)
            _validate_collection_closure(fragment, statement_item.collection_id)
            _validate_exact_quote(
                fragment,
                evidence.quote_start,
                evidence.quote_end,
                evidence.quote_text,
                evidence.quote_sha256,
            )
            evidence_payload: dict[str, object] = {
                "schema": "dithyramba.evidence_link/1.0",
                "statement_id": statement.statement_id,
                "source_fragment_id": fragment.source_fragment_id,
                "attributed_voice_id": voice.voice_id,
                "polarity": evidence.polarity.value,
                "alignment": evidence.alignment.value,
                "limits_text": evidence.limits_text,
                "quote_start": evidence.quote_start,
                "quote_end": evidence.quote_end,
                "quote_sha256": evidence.quote_sha256,
                "extraction_method": evidence.extraction_method,
                "status": "candidate",
            }
            evidence_hash = canonical_sha256_hex(evidence_payload)
            evidence_links.append(
                PreparedEvidenceLink(
                    canonical_content_id("evidence", evidence_payload),
                    statement.statement_id,
                    fragment.source_fragment_id,
                    fragment.source_version_id,
                    voice.voice_id,
                    evidence.polarity,
                    evidence.alignment,
                    evidence.limits_text,
                    evidence.quote_start,
                    evidence.quote_end,
                    evidence.quote_text,
                    evidence.quote_sha256,
                    evidence.extraction_method,
                    evidence_hash,
                )
            )
            versions.add(fragment.source_version_id)
        if not versions:
            raise MeaningEvidenceError("every Statement requires exact permitted evidence")
        statement_by_key[statement_item.key] = statement
        statement_source_versions[statement_item.key] = versions
        statements.append(statement)

    concept_meanings: list[PreparedConceptMeaning] = []
    for meaning_item in proposal.concept_meanings:
        concept_meanings.append(
            _prepare_concept_meaning(
                meaning_item,
                concept_by_key=concept_by_key,
                voice_by_key=voice_by_key,
                time_by_key=time_by_key,
                statement_by_key=statement_by_key,
            )
        )

    _validate_grounding(
        proposal=proposal,
        voices=voice_by_key,
        entities=entity_by_key,
        concepts=concept_by_key,
        times=time_by_key,
    )

    output = MeaningOutputManifest(
        voice_ids=tuple(item.voice_id for item in voice_by_key.values()),
        entity_ids=tuple(item.entity_id for item in entity_by_key.values()),
        entity_mention_ids=tuple(item.mention_id for item in entity_mentions),
        concept_ids=tuple(item.concept_id for item in concept_by_key.values()),
        concept_mention_ids=tuple(item.mention_id for item in concept_mentions),
        concept_meaning_ids=tuple(item.concept_meaning_id for item in concept_meanings),
        time_context_ids=tuple(item.time_context_id for item in time_by_key.values()),
        statement_ids=tuple(item.statement_id for item in statements),
        evidence_link_ids=tuple(item.evidence_link_id for item in evidence_links),
    )
    if output.candidate_count > review_budget.max_candidates:
        raise MeaningBudgetError("proposal exceeds the explicit run candidate budget")

    _validate_source_version_caps(
        profile=profile,
        entity_mentions=entity_mentions,
        concept_mentions=concept_mentions,
        statements=statement_by_key,
        statement_source_versions=statement_source_versions,
        evidence_links=evidence_links,
        concept_meaning_proposals=proposal.concept_meanings,
        concept_meanings=tuple(concept_meanings),
        entity_by_key=entity_by_key,
        concept_by_key=concept_by_key,
        voice_by_key=voice_by_key,
        time_by_key=time_by_key,
        fragments=fragment_by_id,
    )
    return PreparedMeaningGraph(
        voices=tuple(sorted(voice_by_key.values(), key=lambda item: item.voice_id)),
        entities=tuple(sorted(entity_by_key.values(), key=lambda item: item.entity_id)),
        entity_mentions=tuple(sorted(entity_mentions, key=lambda item: item.mention_id)),
        concepts=tuple(sorted(concept_by_key.values(), key=lambda item: item.concept_id)),
        concept_mentions=tuple(sorted(concept_mentions, key=lambda item: item.mention_id)),
        time_contexts=tuple(sorted(time_by_key.values(), key=lambda item: item.time_context_id)),
        statements=tuple(sorted(statements, key=lambda item: item.statement_id)),
        evidence_links=tuple(sorted(evidence_links, key=lambda item: item.evidence_link_id)),
        concept_meanings=tuple(sorted(concept_meanings, key=lambda item: item.concept_meaning_id)),
        output=output,
    )


def _validate_profile_outputs(profile: ExtractionProfile, proposal: MeaningProposal) -> None:
    if profile is ExtractionProfile.INDEX and (
        proposal.voices
        or proposal.statements
        or proposal.concept_meanings
        or proposal.time_contexts
    ):
        raise MeaningProfileError("index profile permits deterministic mentions only")
    if profile is ExtractionProfile.LIGHT:
        if proposal.concept_meanings or proposal.time_contexts:
            raise MeaningProfileError("light profile cannot emit ConceptMeanings or TimeContexts")
        if any(
            evidence.limits_text is not None
            for statement in proposal.statements
            for evidence in statement.evidence
        ):
            raise MeaningProfileError("light profile cannot emit EvidenceLink limits")


def _prepare_mention(
    *,
    mention: ExactMentionProposal,
    target: PreparedEntity | PreparedConcept,
    id_prefix: str,
    fragments: dict[str, MeaningInputFragment],
) -> PreparedMention:
    source_fragment_id = mention.source_fragment_id
    fragment = _require_fragment(fragments, source_fragment_id)
    collection_id = mention.collection_id
    _validate_collection_closure(fragment, collection_id)
    quote_start = mention.quote_start
    quote_end = mention.quote_end
    quote_text = mention.quote_text
    quote_sha256 = mention.quote_sha256
    _validate_exact_quote(
        fragment,
        quote_start,
        quote_end,
        quote_text,
        quote_sha256,
    )
    target_id = target.entity_id if isinstance(target, PreparedEntity) else target.concept_id
    payload: dict[str, object] = {
        "schema": f"dithyramba.{id_prefix}/1.0",
        "target_id": target_id,
        "collection_id": collection_id,
        "source_fragment_id": source_fragment_id,
        "quote_start": quote_start,
        "quote_end": quote_end,
        "quote_sha256": quote_sha256,
    }
    content_hash = canonical_sha256_hex(payload)
    return PreparedMention(
        canonical_content_id(id_prefix, payload),
        target_id,
        collection_id,
        source_fragment_id,
        quote_start,
        quote_end,
        quote_text,
        quote_sha256,
        content_hash,
    )


def _prepare_concept_meaning(
    item: ConceptMeaningProposal,
    *,
    concept_by_key: dict[str, PreparedConcept],
    voice_by_key: dict[str, PreparedVoice],
    time_by_key: dict[str, PreparedTimeContext],
    statement_by_key: dict[str, PreparedStatement],
) -> PreparedConceptMeaning:
    concept = _require_reference(concept_by_key, item.concept_key, "Concept")
    voice = _require_reference(voice_by_key, item.voice_key, "Voice")
    time_context = (
        None
        if item.time_context_key is None
        else _require_reference(time_by_key, item.time_context_key, "TimeContext")
    )
    supporting = tuple(
        _require_reference(statement_by_key, statement_key, "Statement")
        for statement_key in item.statement_keys
    )
    if any(statement.voice_id != voice.voice_id for statement in supporting):
        raise MeaningEvidenceError("ConceptMeaning must retain its supporting Statement Voice")
    statement_ids = tuple(sorted(statement.statement_id for statement in supporting))
    payload: dict[str, object] = {
        "schema": "dithyramba.concept_meaning/1.0",
        "concept_id": concept.concept_id,
        "voice_id": voice.voice_id,
        "time_context_id": None if time_context is None else time_context.time_context_id,
        "meaning_text": item.meaning_text,
        "statement_ids": list(statement_ids),
        "status": "candidate",
    }
    content_hash = canonical_sha256_hex(payload)
    return PreparedConceptMeaning(
        canonical_content_id("concept_meaning", payload),
        concept.concept_id,
        voice.voice_id,
        None if time_context is None else time_context.time_context_id,
        item.meaning_text,
        statement_ids,
        content_hash,
    )


def _validate_grounding(
    *,
    proposal: MeaningProposal,
    voices: dict[str, PreparedVoice],
    entities: dict[str, PreparedEntity],
    concepts: dict[str, PreparedConcept],
    times: dict[str, PreparedTimeContext],
) -> None:
    used_voice_keys = {statement.voice_key for statement in proposal.statements}
    used_voice_keys.update(meaning.voice_key for meaning in proposal.concept_meanings)
    if set(voices) != used_voice_keys:
        raise MeaningEvidenceError("every Voice must ground a Statement or ConceptMeaning")
    mentioned_entities = {mention.target_key for mention in proposal.entity_mentions}
    if set(entities) != mentioned_entities:
        raise MeaningEvidenceError("every Entity must have an exact separate mention")
    grounded_concepts = {mention.target_key for mention in proposal.concept_mentions}
    grounded_concepts.update(meaning.concept_key for meaning in proposal.concept_meanings)
    if set(concepts) != grounded_concepts:
        raise MeaningEvidenceError("every Concept must have a mention or ConceptMeaning")
    used_time_keys = {
        statement.time_context_key
        for statement in proposal.statements
        if statement.time_context_key is not None
    }
    used_time_keys.update(
        meaning.time_context_key
        for meaning in proposal.concept_meanings
        if meaning.time_context_key is not None
    )
    if set(times) != used_time_keys:
        raise MeaningEvidenceError("every TimeContext must ground semantic output")


def _validate_source_version_caps(
    *,
    profile: ExtractionProfile,
    entity_mentions: tuple[PreparedMention, ...],
    concept_mentions: tuple[PreparedMention, ...],
    statements: dict[str, PreparedStatement],
    statement_source_versions: dict[str, set[str]],
    evidence_links: list[PreparedEvidenceLink],
    concept_meaning_proposals: tuple[ConceptMeaningProposal, ...],
    concept_meanings: tuple[PreparedConceptMeaning, ...],
    entity_by_key: dict[str, PreparedEntity],
    concept_by_key: dict[str, PreparedConcept],
    voice_by_key: dict[str, PreparedVoice],
    time_by_key: dict[str, PreparedTimeContext],
    fragments: dict[str, MeaningInputFragment],
) -> None:
    candidates_by_version: dict[str, set[str]] = {}

    def add(version_id: str, *candidate_ids: str | None) -> None:
        bucket = candidates_by_version.setdefault(version_id, set())
        bucket.update(value for value in candidate_ids if value is not None)

    entity_id_by_mention = {item.entity_id: item for item in entity_by_key.values()}
    for mention in entity_mentions:
        fragment = fragments[mention.source_fragment_id]
        add(
            fragment.source_version_id,
            entity_id_by_mention[mention.target_id].entity_id,
            mention.mention_id,
        )
    concept_id_by_mention = {item.concept_id: item for item in concept_by_key.values()}
    for mention in concept_mentions:
        fragment = fragments[mention.source_fragment_id]
        add(
            fragment.source_version_id,
            concept_id_by_mention[mention.target_id].concept_id,
            mention.mention_id,
        )

    statement_by_id = {item.statement_id: item for item in statements.values()}
    for evidence in evidence_links:
        statement = statement_by_id[evidence.statement_id]
        add(
            evidence.source_version_id,
            evidence.evidence_link_id,
            statement.statement_id,
            statement.voice_id,
            statement.time_context_id,
        )

    meaning_by_key = dict(
        zip(
            (item.key for item in concept_meaning_proposals),
            concept_meanings,
            strict=True,
        )
    )
    for proposal_item in concept_meaning_proposals:
        meaning = meaning_by_key[proposal_item.key]
        for statement_key in proposal_item.statement_keys:
            for version_id in statement_source_versions[statement_key]:
                add(
                    version_id,
                    meaning.concept_meaning_id,
                    meaning.concept_id,
                    meaning.voice_id,
                    meaning.time_context_id,
                )

    cap = profile.source_version_candidate_cap
    if any(len(candidate_ids) > cap for candidate_ids in candidates_by_version.values()):
        raise MeaningBudgetError("proposal exceeds its per-SourceVersion profile cap")


def _require_reference(mapping: dict[str, _T], key: str, label: str) -> _T:
    try:
        return mapping[key]
    except KeyError as exc:
        raise MeaningEvidenceError(f"proposal references an unknown {label}") from exc


def _require_fragment(
    fragments: dict[str, MeaningInputFragment],
    source_fragment_id: str,
) -> MeaningInputFragment:
    try:
        return fragments[source_fragment_id]
    except KeyError as exc:
        raise MeaningEvidenceError(
            "proposal references a fragment outside the permitted read set"
        ) from exc


def _validate_collection_closure(fragment: MeaningInputFragment, collection_id: str) -> None:
    if collection_id not in fragment.collection_ids:
        raise MeaningEvidenceError("semantic evidence is outside its declared Collection")


def _validate_exact_quote(
    fragment: MeaningInputFragment,
    start: int,
    end: int,
    quote_text: str,
    quote_sha256: str,
) -> None:
    if end > len(fragment.text) or fragment.text[start:end] != quote_text:
        raise MeaningEvidenceError("exact quote offsets do not match the permitted SourceFragment")
    if sha256_hex(fragment.text[start:end].encode("utf-8")) != quote_sha256:
        raise MeaningEvidenceError("exact quote SHA-256 does not match the SourceFragment slice")
