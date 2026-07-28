"""Stable errors for the P6 meaning contract."""


class MeaningError(Exception):
    """Base class for P6 meaning failures."""


class MeaningContractError(MeaningError, ValueError):
    """A typed proposal or semantic request violates the frozen contract."""


class MeaningEvidenceError(MeaningContractError):
    """Exact evidence, mention, attribution, or access closure is invalid."""


class MeaningProfileError(MeaningContractError):
    """A proposal contains output not allowed by its extraction profile."""


class MeaningBudgetError(MeaningContractError):
    """A candidate, read, backlog, or review-session budget was exceeded."""


class MeaningPersistenceError(MeaningError):
    """Persisted P6 state is missing, conflicting, or noncanonical."""


class MeaningNotFoundError(MeaningPersistenceError):
    """A requested P6 run, candidate, session, or decision does not exist."""


class MeaningReviewError(MeaningError):
    """A semantic ReviewDecision or queue operation is invalid."""
