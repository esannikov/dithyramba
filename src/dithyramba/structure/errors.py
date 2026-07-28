"""Errors for explicit, provenance-bearing StructureUnits."""


class StructureError(Exception):
    """Base StructureUnit error."""


class StructureContractError(StructureError, ValueError):
    """A StructureUnit contract is malformed."""


class StructureIntegrityError(StructureError):
    """Persisted or source StructureUnit lineage is inconsistent."""


class StructureAuthorizationError(StructureError):
    """A StructureUnit attempts to cross its permitted source boundary."""


class StructureNotFoundError(StructureError):
    """A requested StructureUnit artifact does not exist."""
