"""Optional no-add reranker contract over an already permitted fused pool."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dithyramba.contracts import canonical_content_id, canonical_sha256_hex

from .hybrid_models import HybridContractError

_ID_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


class RerankerContractError(HybridContractError):
    """The optional reranker changed the candidate set or emitted invalid scores."""


class RerankerProfile(BaseModel):
    SCHEMA: ClassVar[str] = "dithyramba.reranker_profile/1.0"
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    provider: str = "sentence_transformers_cross_encoder"
    model_id: str
    revision: str
    license: str
    max_tokens: int = Field(ge=1, le=32_768)
    local_files_only: bool = True
    trust_remote_code: bool = False

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        if value != "sentence_transformers_cross_encoder":
            raise RerankerContractError("unsupported reranker provider")
        return value

    @field_validator("model_id")
    @classmethod
    def _model_id(cls, value: str) -> str:
        if type(value) is not str or _MODEL_ID_PATTERN.fullmatch(value) is None:
            raise RerankerContractError("reranker model_id must be owner/repository")
        return value

    @field_validator("revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        if type(value) is not str or _REVISION_PATTERN.fullmatch(value) is None:
            raise RerankerContractError("reranker revision must be a 40-character commit")
        return value

    @field_validator("license")
    @classmethod
    def _license(cls, value: str) -> str:
        if type(value) is not str or _LABEL_PATTERN.fullmatch(value) is None:
            raise RerankerContractError("reranker license must be a bounded ASCII label")
        return value

    @field_validator("local_files_only")
    @classmethod
    def _offline(cls, value: bool) -> bool:
        if value is not True:
            raise RerankerContractError("reranker must be local_files_only")
        return value

    @field_validator("trust_remote_code")
    @classmethod
    def _remote_code(cls, value: bool) -> bool:
        if value is not False:
            raise RerankerContractError("reranker must disable remote code")
        return value

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "provider": self.provider,
            "model_id": self.model_id,
            "revision": self.revision,
            "license": self.license,
            "max_tokens": self.max_tokens,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }

    @property
    def profile_hash(self) -> str:
        return canonical_sha256_hex(self.semantic_payload())

    @property
    def profile_id(self) -> str:
        return canonical_content_id("reranker_profile", self.semantic_payload())


MMARCO_MINILM_RERANKER_PROFILE = RerankerProfile(
    model_id="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    revision="1427fd652930e4ba29e8149678df786c240d8825",
    license="Apache-2.0",
    max_tokens=512,
)

BGE_M3_RERANKER_PROFILE = RerankerProfile(
    model_id="BAAI/bge-reranker-v2-m3",
    revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    license="Apache-2.0",
    max_tokens=8_192,
)


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    source_fragment_id: str
    source_id: str
    text: str

    def __post_init__(self) -> None:
        _require_id(self.source_fragment_id, "fragment")
        _require_id(self.source_id, "source")
        if type(self.text) is not str or not self.text or len(self.text) > 100_000:
            raise RerankerContractError("reranker candidate text is invalid")
        if "\x00" in self.text or unicodedata.normalize("NFC", self.text) != self.text:
            raise RerankerContractError("reranker candidate text must be NFC without NUL")


@dataclass(frozen=True, slots=True)
class RerankerScore:
    source_fragment_id: str
    score_hex: str

    def __post_init__(self) -> None:
        _require_id(self.source_fragment_id, "fragment")
        if type(self.score_hex) is not str:
            raise RerankerContractError("reranker score must use float.hex text")
        try:
            score = float.fromhex(self.score_hex)
        except ValueError as error:
            raise RerankerContractError("reranker score must use float.hex text") from error
        if not math.isfinite(score):
            raise RerankerContractError("reranker score must be finite")
        if score.hex() != self.score_hex:
            raise RerankerContractError("reranker score must use canonical float.hex text")

    @classmethod
    def from_float(cls, source_fragment_id: str, score: float) -> RerankerScore:
        if type(score) is not float or not math.isfinite(score):
            raise RerankerContractError("reranker score must be a finite Python float")
        return cls(source_fragment_id=source_fragment_id, score_hex=score.hex())

    @property
    def score(self) -> float:
        return float.fromhex(self.score_hex)


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    candidate: RerankCandidate
    rank: int
    score_hex: str


class RerankerProvider(Protocol):
    """Provider receives only the already-authorized fused pool."""

    def score(
        self,
        question: str,
        candidates: tuple[RerankCandidate, ...],
    ) -> tuple[RerankerScore, ...]: ...


def run_reranker(
    provider: RerankerProvider,
    *,
    question: str,
    candidates: tuple[RerankCandidate, ...],
) -> tuple[RerankedCandidate, ...]:
    """Invoke once, then independently enforce the no-add/no-drop contract."""

    if not callable(getattr(provider, "score", None)):
        raise RerankerContractError("reranker provider must expose score()")
    if type(question) is not str:
        raise RerankerContractError("reranker question must be bounded unpadded NFC text")
    normalized = unicodedata.normalize("NFC", question)
    if (
        not question.strip()
        or question != question.strip()
        or normalized != question
        or "\x00" in question
        or len(question) > 2_000
    ):
        raise RerankerContractError("reranker question must be bounded unpadded NFC text")
    _candidate_map(candidates)
    scores = provider.score(question, candidates)
    if type(scores) is not tuple:
        raise RerankerContractError("reranker provider must return a tuple")
    return rerank_candidates(candidates, scores)


def rerank_candidates(
    candidates: tuple[RerankCandidate, ...],
    scores: tuple[RerankerScore, ...],
) -> tuple[RerankedCandidate, ...]:
    """Sort score DESC and binary ID ASC while preserving the exact input set."""

    by_id = _candidate_map(candidates)
    score_by_id: dict[str, RerankerScore] = {}
    for score in scores:
        if not isinstance(score, RerankerScore):
            raise RerankerContractError("reranker output contains an invalid score")
        if score.source_fragment_id in score_by_id:
            raise RerankerContractError("reranker output contains duplicate IDs")
        score_by_id[score.source_fragment_id] = score
    if set(score_by_id) != set(by_id):
        raise RerankerContractError("reranker must return exactly the input candidate IDs")
    ordered = sorted(
        score_by_id.values(),
        key=lambda item: (-item.score, item.source_fragment_id.encode("ascii")),
    )
    return tuple(
        RerankedCandidate(
            candidate=by_id[item.source_fragment_id],
            rank=rank,
            score_hex=item.score_hex,
        )
        for rank, item in enumerate(ordered, start=1)
    )


def _candidate_map(
    candidates: tuple[RerankCandidate, ...],
) -> dict[str, RerankCandidate]:
    if not candidates or len(candidates) > 500:
        raise RerankerContractError("reranker requires 1-500 candidates")
    result: dict[str, RerankCandidate] = {}
    for candidate in candidates:
        if not isinstance(candidate, RerankCandidate):
            raise RerankerContractError("reranker input contains an invalid candidate")
        if candidate.source_fragment_id in result:
            raise RerankerContractError("reranker input candidate IDs must be unique")
        result[candidate.source_fragment_id] = candidate
    return result


def _require_id(value: str, prefix: str) -> str:
    expected = f"{prefix}_"
    if type(value) is not str or not value.startswith(expected):
        raise RerankerContractError(f"identifier must use the {expected} prefix")
    suffix = value[len(expected) :]
    if _ID_SUFFIX_PATTERN.fullmatch(suffix) is None:
        raise RerankerContractError(f"identifier must use a canonical {expected} suffix")
    return value
