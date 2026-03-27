import tracemalloc
import torch

from main import create_sim_config, prepare_society_for_debug, run_debug_simulation


def test_full_debug_pipeline_stays_within_reasonable_memory(tmp_path):
    config = create_sim_config(
        num_agents=300,
        use_agent_memory=True,
        use_network_topology=True,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "ram"), evolve=False
    )

    world = torch.zeros(1, 12)
    world[0, 0] = -0.6
    world[0, 4] = -0.4

    tracemalloc.start()
    run_debug_simulation(config, world, society=society, urgency=0.5)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak > 0
    assert current <= peak
    assert peak < 256 * 1024 * 1024
