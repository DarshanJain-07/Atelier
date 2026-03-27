from collections import OrderedDict

import pytest
import torch

from main import (
    SOCIETY_CACHE,
    SOCIETY_CACHE_LOCK,
    RunProfile,
    aggregate_social_state,
    create_sim_config,
    prepare_society_for_debug,
    prepare_society_sync,
)


@pytest.fixture()
def isolated_society_cache():
    with SOCIETY_CACHE_LOCK:
        snapshot = OrderedDict(SOCIETY_CACHE.items())
        SOCIETY_CACHE.clear()

    try:
        yield
    finally:
        with SOCIETY_CACHE_LOCK:
            SOCIETY_CACHE.clear()
            SOCIETY_CACHE.update(snapshot)


def test_prepare_society_cache_isolates_topology_variants(tmp_path, isolated_society_cache):
    topo_run = RunProfile(
        seed=42,
        temperature=0.7,
        agent_count=128,
        use_power_law=False,
        use_network_topology=True,
        enable_evolution=False,
    )
    flat_run = topo_run.model_copy(update={"use_network_topology": False})

    _, _, _, _, _, _, adjacency_topo, warnings_topo = prepare_society_sync(
        topo_run, str(tmp_path / "topology")
    )
    _, _, _, _, _, _, adjacency_flat, warnings_flat = prepare_society_sync(
        flat_run, str(tmp_path / "flat")
    )

    assert adjacency_topo is not None
    assert adjacency_flat is None
    assert warnings_topo == []
    assert warnings_flat == []

    with SOCIETY_CACHE_LOCK:
        assert len(SOCIETY_CACHE) == 2


def test_prepare_society_cache_isolates_evolution_variants(tmp_path, isolated_society_cache):
    evolved_run = RunProfile(
        seed=7,
        temperature=0.4,
        agent_count=96,
        use_power_law=False,
        use_network_topology=False,
        enable_evolution=True,
        evolution_generations=2,
    )
    baseline_run = evolved_run.model_copy(update={"enable_evolution": False})

    _, evolved_meta, _, _, _, _, _, _ = prepare_society_sync(
        evolved_run, str(tmp_path / "evolved")
    )
    _, baseline_meta, _, _, _, _, _, _ = prepare_society_sync(
        baseline_run, str(tmp_path / "baseline")
    )

    assert baseline_meta["Class"].eq("Agent").all()
    assert not evolved_meta["Class"].eq("Agent").all()

    with SOCIETY_CACHE_LOCK:
        assert len(SOCIETY_CACHE) == 2


def test_prepare_society_returns_fresh_request_memory(tmp_path, isolated_society_cache):
    run = RunProfile(
        seed=99,
        temperature=0.1,
        agent_count=64,
        use_power_law=False,
        use_network_topology=False,
        enable_evolution=False,
        use_agent_memory=True,
    )

    _, _, exposures, _, _, memory_first, _, _ = prepare_society_sync(
        run, str(tmp_path / "first")
    )
    memory_first[0, 0] = 123.0

    _, _, _, _, _, memory_second, _, _ = prepare_society_sync(
        run, str(tmp_path / "second")
    )

    assert memory_first.data_ptr() != memory_second.data_ptr()
    assert memory_second.shape == exposures.shape
    assert memory_second[0, 0].item() == pytest.approx(0.0)


@pytest.mark.parametrize("num_agents", [10, 50, 200])
def test_generated_topology_handles_small_populations(tmp_path, num_agents):
    society = prepare_society_for_debug(
        config=create_sim_config(
            num_agents=num_agents,
            use_network_topology=True,
            enable_evolution=False,
        ),
        output_dir=str(tmp_path / f"society_{num_agents}"),
        evolve=False,
    )

    adjacency = society.adjacency_matrix
    assert adjacency is not None
    assert adjacency.shape == (num_agents, num_agents)

    row_sums = torch.sparse.sum(adjacency, dim=1).to_dense()
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_acting_ratio_uses_scoped_population_size():
    emotions = torch.zeros(10, 8)
    emotions[:, 6] = 1.0
    influence = torch.ones(10)
    engagement = torch.ones(10)

    state = aggregate_social_state(
        create_sim_config(
            num_agents=1000,
            use_granovetter_thresholds=False,
            dominant_emotion_threshold=0.1,
        ),
        emotions,
        influence,
        engagement_scores=engagement,
    )

    assert state["population_size"] == 10
    assert state["acting_count"] == 10
    assert state["acting_ratio"] == pytest.approx(1.0)
