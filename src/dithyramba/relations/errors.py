"""Errors for typed, source-grounded Relations and path receipts."""


class RelationError(Exception):
    """Base Relation error."""


class RelationContractError(RelationError, ValueError):
    """A Relation request or result violates its typed contract."""


class RelationAuthorizationError(RelationError):
    """A Relation operation attempts to cross its permitted evidence scope."""


class RelationIntegrityError(RelationError):
    """Persisted Relation lineage or a path receipt is inconsistent."""


class RelationNotFoundError(RelationError):
    """A requested Relation artifact does not exist."""
