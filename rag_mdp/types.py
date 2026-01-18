from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx


@dataclass
class EvidenceItem:
    """A single evidence fact in the toy world."""

    src: str
    dst: str
    relation: str


@dataclass
class Episode:
    question_id: int
    question_text: str
    location: str
    animal: str
    color: str
    ground_truth_graph: nx.DiGraph
    G_t: nx.DiGraph
    entropy_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class WorldSpec:
    locations: List[str]
    animals: List[str]
    colors: List[str]
    location_to_animal: Dict[str, str]
    animal_to_colors: Dict[str, List[str]]
