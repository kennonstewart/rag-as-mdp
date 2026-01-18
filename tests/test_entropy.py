from __future__ import annotations

import math

from rag_mdp.synthetic_world import SyntheticWorld


def test_entropy_progression():
    world = SyntheticWorld(
        seed=0,
        locations=["loc1"],
        animals=["fox"],
        colors=["red", "blue", "green"],
    )
    episode = world.generate_episode(seed=1)

    empty_entropy = world.compute_true_entropy(episode, episode.G_t)
    assert math.isclose(empty_entropy, math.log(len(world.colors)))

    evidence = world.retrieve_evidence(episode, episode.G_t)
    world.add_evidence(episode.G_t, evidence)
    mid_entropy = world.compute_true_entropy(episode, episode.G_t)
    expected_mid = math.log(len(world.spec.animal_to_colors[episode.animal]))
    assert math.isclose(mid_entropy, expected_mid)

    evidence = world.retrieve_evidence(episode, episode.G_t)
    world.add_evidence(episode.G_t, evidence)
    final_entropy = world.compute_true_entropy(episode, episode.G_t)
    assert math.isclose(final_entropy, 0.0)
