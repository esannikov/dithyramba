"""Business boundary for scoped ReviewDecision creation and inspection."""

from __future__ import annotations

from dithyramba.persistence.repository import LibraryRepository
from dithyramba.persistence.review import ReviewDecisionRepository, ReviewTarget

from .models import ReviewDecision, ReviewDecisionRequest, ReviewTargetType


class ReviewService:
    """Thin service that keeps HTTP/CLI code free of persistence logic."""

    def __init__(self, repository: LibraryRepository) -> None:
        self._decisions = ReviewDecisionRepository(repository)

    def target(self, target_type: ReviewTargetType, target_id: str) -> ReviewTarget:
        return self._decisions.get_target(target_type, target_id)

    def decide(self, request: ReviewDecisionRequest) -> ReviewDecision:
        return self._decisions.create(request)

    def packet_item_targets(self, evidence_packet_id: str) -> tuple[ReviewTarget, ...]:
        return self._decisions.list_packet_item_targets(evidence_packet_id)

    def get(self, review_decision_id: str) -> ReviewDecision:
        return self._decisions.get(review_decision_id)

    def list(
        self,
        *,
        target_type: ReviewTargetType | None = None,
        target_id: str | None = None,
        limit: int = 100,
    ) -> tuple[ReviewDecision, ...]:
        return self._decisions.list(
            target_type=target_type,
            target_id=target_id,
            limit=limit,
        )
