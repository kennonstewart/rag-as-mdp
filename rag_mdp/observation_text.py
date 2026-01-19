from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from rag_mdp.synthetic_world import SyntheticWorld


def query_text_from_id(world: SyntheticWorld, query_id: int) -> str:
    """Reconstruct the question text from the query id."""
    location = world.locations[int(query_id)]
    return f"What is the color of the animal that lives in {location}?"


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _edges_from_arrays(
    world: SyntheticWorld, nodes: np.ndarray, edges: np.ndarray
) -> List[Tuple[str, str]]:
    id_to_entity = world.id_to_entity
    edges_list: List[Tuple[str, str]] = []
    for src_id, dst_id in edges.tolist():
        if src_id == 0 or dst_id == 0:
            continue
        src = id_to_entity.get(int(src_id))
        dst = id_to_entity.get(int(dst_id))
        if src is None or dst is None:
            continue
        edges_list.append((src, dst))
    return edges_list


def evidence_summary_from_graph_arrays(
    world: SyntheticWorld, nodes: np.ndarray, edges: np.ndarray
) -> str:
    """Summarize the evidence graph for LLM inputs."""
    node_names = []
    for node_id in nodes.tolist():
        if node_id == 0:
            continue
        name = world.id_to_entity.get(int(node_id))
        if name is not None:
            node_names.append(name)

    node_names = _dedupe_preserve_order(node_names)
    edge_pairs = _edges_from_arrays(world, nodes, edges)
    edge_strings = _dedupe_preserve_order([f"{src}->{dst}" for src, dst in edge_pairs])

    nodes_text = ", ".join(node_names) if node_names else "none"
    edges_text = ", ".join(edge_strings) if edge_strings else "none"

    return f"Nodes: {nodes_text}. Edges: {edges_text}."
