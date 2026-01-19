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


class RAGReActSelection(dspy.Signature):
    """ReAct-style action selection for RAG-as-MDP."""

    query_text: str = dspy.InputField(desc="Original question text.")
    evidence_summary: str = dspy.InputField(desc="Summary of the evidence graph.")
    entropy: float = dspy.InputField(desc="Current posterior entropy.")
    step: int = dspy.InputField(desc="Current step index.")
    confidence_threshold: float = dspy.InputField(desc="Target entropy threshold.")

    reasoning: str = dspy.OutputField(desc="Short reasoning for the action.")
    action: str = dspy.OutputField(desc="One of: retrieve, reflect, return.")


class DSPyReActPolicy:
    """DSPy ReAct policy wrapper."""

    def __init__(
        self, world: SyntheticWorld, confidence_threshold: float = 0.5, tools=None
    ) -> None:
        if not hasattr(dspy, "ReAct"):
            raise ImportError("DSPy ReAct module not available in this version.")
        self.world = world
        self.confidence_threshold = confidence_threshold
        if tools is None:
            tools = []
        self.selector = dspy.ReAct(RAGReActSelection, tools=tools)

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
