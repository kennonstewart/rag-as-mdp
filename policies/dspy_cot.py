from __future__ import annotations

from typing import Dict

import dspy

from rag_mdp.observation_text import (
    evidence_summary_from_graph_arrays,
    query_text_from_id,
)
from rag_mdp.synthetic_world import SyntheticWorld


ACTION_MAP: Dict[str, int] = {
    "retrieve": 0,
    "reflect": 1,
    "return": 2,
}


class RAGActionSelection(dspy.Signature):
    """Select an action based on the information state."""

    query_text: str = dspy.InputField(desc="Original question text.")
    evidence_summary: str = dspy.InputField(desc="Summary of the evidence graph.")
    entropy: float = dspy.InputField(desc="Current posterior entropy.")
    step: int = dspy.InputField(desc="Current step index.")
    confidence_threshold: float = dspy.InputField(desc="Target entropy threshold.")

    reasoning: str = dspy.OutputField(desc="Short reasoning for the action.")
    action: str = dspy.OutputField(desc="One of: retrieve, reflect, return.")


class DSPyCoTPolicy:
    """DSPy Chain-of-Thought policy for RAG-as-MDP."""

    def __init__(self, world: SyntheticWorld, confidence_threshold: float = 0.5) -> None:
        self.world = world
        self.confidence_threshold = confidence_threshold
        self.selector = dspy.ChainOfThought(RAGActionSelection)

    def select_action(self, obs) -> int:
        query_text = query_text_from_id(self.world, obs["query_id"])
        evidence_summary = evidence_summary_from_graph_arrays(
            self.world, obs["nodes"], obs["edges"]
        )
        entropy = float(obs["entropy"][0])
        step = int(obs["step"][0])

        result = self.selector(
            query_text=query_text,
            evidence_summary=evidence_summary,
            entropy=entropy,
            step=step,
            confidence_threshold=self.confidence_threshold,
        )

        action = str(getattr(result, "action", "")).strip().lower()
        return ACTION_MAP.get(action, ACTION_MAP["return"])
