from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from rag_mdp.metrics import entropy_from_feasible_count
from rag_mdp.types import EvidenceItem, Episode, WorldSpec


class SyntheticWorld:
    """Toy 2-hop QA world with exact entropy computation."""

    def __init__(
        self,
        seed: Optional[int] = None,
        locations: Optional[List[str]] = None,
        animals: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)

        self.locations = locations or ["forest", "desert", "ocean", "tundra"]
        self.animals = animals or ["fox", "camel", "seal", "hare"]
        self.colors = colors or ["red", "tan", "gray", "white", "black", "gold"]

        self.spec = self._build_world_spec()
        self.entity_to_id, self.id_to_entity = self._build_entity_index()

    def _build_world_spec(self) -> WorldSpec:
        location_to_animal: Dict[str, str] = {}
        animal_to_colors: Dict[str, List[str]] = {}

        # Deterministic pairing of locations->animals for reproducibility.
        for location, animal in zip(self.locations, self.animals):
            location_to_animal[location] = animal

        # Each animal can have multiple plausible colors in the prior.
        shuffled_colors = list(self.colors)
        self.rng.shuffle(shuffled_colors)
        for idx, animal in enumerate(self.animals):
            color_slice = shuffled_colors[idx : idx + 2]
            if len(color_slice) < 2:
                color_slice = shuffled_colors[:2]
            animal_to_colors[animal] = color_slice

        return WorldSpec(
            locations=self.locations,
            animals=self.animals,
            colors=self.colors,
            location_to_animal=location_to_animal,
            animal_to_colors=animal_to_colors,
        )

    def _build_entity_index(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        entities = list(self.locations) + list(self.animals) + list(self.colors)
        entity_to_id = {entity: idx for idx, entity in enumerate(entities, start=1)}
        id_to_entity = {idx: entity for entity, idx in entity_to_id.items()}
        return entity_to_id, id_to_entity

    def generate_episode(self, seed: Optional[int] = None) -> Episode:
        rng = self.rng if seed is None else np.random.default_rng(seed)
        location = rng.choice(self.spec.locations)
        animal = self.spec.location_to_animal[location]
        color = rng.choice(self.spec.animal_to_colors[animal])

        ground_truth_graph = nx.DiGraph()
        ground_truth_graph.add_edge(location, animal, relation="lives_in")
        ground_truth_graph.add_edge(animal, color, relation="has_color")

        question_id = self.spec.locations.index(location)
        question_text = f"What is the color of the animal that lives in {location}?"

        return Episode(
            question_id=question_id,
            question_text=question_text,
            location=location,
            animal=animal,
            color=color,
            ground_truth_graph=ground_truth_graph,
            G_t=nx.DiGraph(),
        )

    def retrieve_evidence(self, episode: Episode, G_t: nx.DiGraph) -> Optional[EvidenceItem]:
        if not self._has_edge(G_t, episode.location, episode.animal, "lives_in"):
            return EvidenceItem(episode.location, episode.animal, "lives_in")
        if not self._has_edge(G_t, episode.animal, episode.color, "has_color"):
            return EvidenceItem(episode.animal, episode.color, "has_color")
        return None

    def consolidate_graph(self, G_t: nx.DiGraph) -> nx.DiGraph:
        # No-op in Phase 1; placeholder for future consolidation logic.
        return G_t

    def add_evidence(self, G_t: nx.DiGraph, evidence: EvidenceItem) -> nx.DiGraph:
        G_t.add_node(evidence.src)
        G_t.add_node(evidence.dst)
        G_t.add_edge(evidence.src, evidence.dst, relation=evidence.relation)
        return G_t

    def compute_true_entropy(self, episode: Episode, G_t: nx.DiGraph) -> float:
        feasible_colors = self._feasible_colors(episode, G_t)
        return entropy_from_feasible_count(len(feasible_colors))

    def _feasible_colors(self, episode: Episode, G_t: nx.DiGraph) -> List[str]:
        # If we have explicit color evidence, entropy collapses.
        color_edges = [
            dst
            for src, dst, data in G_t.edges(data=True)
            if src == episode.animal and data.get("relation") == "has_color"
        ]
        if color_edges:
            return list(dict.fromkeys(color_edges))

        # If we know the animal, limit to that animal's plausible colors.
        if self._has_edge(G_t, episode.location, episode.animal, "lives_in"):
            return list(self.spec.animal_to_colors[episode.animal])

        # Otherwise, all colors are possible.
        return list(self.spec.colors)

    def graph_to_arrays(
        self,
        G_t: nx.DiGraph,
        max_nodes: int,
        max_edges: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        nodes = list(G_t.nodes())[:max_nodes]
        node_ids = [self.entity_to_id[n] for n in nodes]
        node_ids += [0] * (max_nodes - len(node_ids))

        edges = list(G_t.edges())[:max_edges]
        edge_ids = [
            (self.entity_to_id[src], self.entity_to_id[dst]) for src, dst in edges
        ]
        edge_ids += [(0, 0)] * (max_edges - len(edge_ids))

        return np.array(node_ids, dtype=np.int32), np.array(edge_ids, dtype=np.int32)

    def compute_optimal_stopping_time(
        self,
        episode: Episode,
        max_steps: int,
        action_costs: Dict[str, float],
    ) -> int:
        # Brute-force: assume retrieve until t, then return.
        best_t = 0
        best_utility = float("-inf")
        base_entropy = self.compute_true_entropy(episode, nx.DiGraph())

        for t in range(max_steps + 1):
            G_t = nx.DiGraph()
            for _ in range(t):
                evidence = self.retrieve_evidence(episode, G_t)
                if evidence is None:
                    break
                self.add_evidence(G_t, evidence)

            entropy_t = self.compute_true_entropy(episode, G_t)
            retrieve_cost = action_costs.get("retrieve", 0.0) * t
            return_cost = action_costs.get("return", 0.0)
            utility = (base_entropy - entropy_t) - retrieve_cost - return_cost
            if utility > best_utility:
                best_utility = utility
                best_t = t

        return best_t

    @staticmethod
    def _has_edge(G_t: nx.DiGraph, src: str, dst: str, relation: str) -> bool:
        if not G_t.has_edge(src, dst):
            return False
        return G_t.edges[src, dst].get("relation") == relation
