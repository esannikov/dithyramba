"""Library identity and private application-data boundary helpers."""

from .errors import (
    ApplicationDataPathError,
    InvalidLibraryIdError,
    InvalidLibraryNameError,
    LibraryAlreadyExistsError,
    LibraryError,
    LibraryLayoutError,
    PathOverlapError,
)
from .models import LibraryConfig, new_library_id, validate_library_id, validate_library_name
from .paths import (
    LibraryPaths,
    application_data_root,
    create_library_layout,
    ensure_paths_disjoint,
    is_path_within,
    library_paths,
    validate_library_layout,
)

__all__ = [
    "ApplicationDataPathError",
    "InvalidLibraryIdError",
    "InvalidLibraryNameError",
    "LibraryAlreadyExistsError",
    "LibraryConfig",
    "LibraryError",
    "LibraryLayoutError",
    "LibraryPaths",
    "PathOverlapError",
    "application_data_root",
    "create_library_layout",
    "ensure_paths_disjoint",
    "is_path_within",
    "library_paths",
    "new_library_id",
    "validate_library_id",
    "validate_library_layout",
    "validate_library_name",
]
