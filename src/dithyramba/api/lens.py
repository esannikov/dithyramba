"""One public dispatcher for Dithyramba's read-only human projections."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from fastapi import FastAPI

from .concept_lens import create_concept_lens_app
from .flow_view import create_flow_view_app
from .reading_room import create_reading_room_app
from .research_atlas import create_research_atlas_app
from .session_lens import create_session_lens_app


class LensMode(StrEnum):
    """The five intentionally distinct human views behind one Lens surface."""

    LIBRARY = "library"
    SESSION = "session"
    ATLAS = "atlas"
    CONCEPTS = "concepts"
    FLOW = "flow"


_FACTORIES: dict[LensMode, Callable[..., FastAPI]] = {
    LensMode.LIBRARY: create_reading_room_app,
    LensMode.SESSION: create_session_lens_app,
    LensMode.ATLAS: create_research_atlas_app,
    LensMode.CONCEPTS: create_concept_lens_app,
    LensMode.FLOW: create_flow_view_app,
}


def create_lens_app(mode: LensMode, /, **configuration: Any) -> FastAPI:
    """Create one Lens mode with that mode's explicit factory arguments.

    The dispatcher unifies the public product surface without pretending that
    a live Library, a durable session, and an immutable Atlas share one data
    contract. Each underlying factory remains independently testable.
    """

    if not isinstance(mode, LensMode):
        raise TypeError("mode must be a LensMode")
    app = _FACTORIES[mode](**configuration)
    app.state.lens_mode = mode
    return app
