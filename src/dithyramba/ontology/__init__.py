"""Scoped candidate ontology: orientation with exact source closure."""

from .engine import OntologyConfig, OntologyFragment, build_candidate_ontology
from .errors import OntologyError
from .models import (
    CandidateOntologyManifest,
    LoadedCandidateOntology,
    OntologyCluster,
    OntologyConcept,
    OntologyEvidence,
    OntologyMetrics,
    OntologyRelation,
    OntologyRelationKind,
    load_candidate_ontology,
)

__all__ = [
    "CandidateOntologyManifest",
    "LoadedCandidateOntology",
    "OntologyCluster",
    "OntologyConcept",
    "OntologyConfig",
    "OntologyError",
    "OntologyEvidence",
    "OntologyFragment",
    "OntologyMetrics",
    "OntologyRelation",
    "OntologyRelationKind",
    "build_candidate_ontology",
    "load_candidate_ontology",
]
