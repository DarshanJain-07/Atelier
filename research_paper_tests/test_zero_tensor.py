import torch

from cognitive_engine import CognitiveEngine
from main import build_debug_society, run_debug_simulation
from physics_engine import SocialPhysicsEngine
from research_paper_tests.config_schema import SimConfig


def test_zero_tensor_produces_neutral_inert_state():
    torch.manual_seed(42)

    config = SimConfig()
    cog_engine = CognitiveEngine(config)
    phys_engine = SocialPhysicsEngine(config)

    world_tensor_raw = torch.zeros(1, 12)
    personalities = torch.rand(3, 5)
    exposures = torch.zeros(3, 12)
    agent_affinities = torch.ones(3, 12)

    context_vector, attention_weights, engagement_scores = cog_engine.run(
        world_tensor_raw=world_tensor_raw,
        urgency=0.0,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=agent_affinities,
    )
    emotions = cog_engine.project_emotions(context_vector)
    result = phys_engine.aggregate_society(
        emotions,
        torch.ones(3),
        engagement_scores,
    )

    assert torch.equal(context_vector, torch.zeros_like(context_vector))
    assert torch.allclose(attention_weights.sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.equal(engagement_scores, torch.zeros_like(engagement_scores))
    assert torch.allclose(emotions.sum(dim=1), torch.ones(3), atol=1e-5)

    assert result["dominant_emotion"] == "Neutral"
    assert result["confidence"] == 0.125
    assert result["sentiment_valence"] == 0.0
    assert result["action_vector"] is None


def test_algorithmic_amplification_keeps_zero_world_neutral():
    torch.manual_seed(42)

    config = SimConfig(
        num_agents=16,
        use_signal_distortion=False,
        use_network_topology=False,
        enable_evolution=False,
        use_algorithmic_amplification=True,
        algo_sample_size=0.25,
        algo_exaggeration_factor=2.0,
    )
    exposures = torch.zeros(16, 12)
    personalities = torch.rand(16, 5)
    agent_affinities = torch.ones(16, 12)
    society = build_debug_society(
        config,
        exposures=exposures,
        personalities=personalities,
        affinities=agent_affinities,
        influence_scores=torch.ones(16),
    )

    result = run_debug_simulation(
        config,
        torch.zeros(1, 12),
        society=society,
        urgency=0.5,
    )

    assert torch.equal(result.final_world_tensor, torch.zeros(1, 12))
    assert torch.equal(
        result.engagement_scores,
        torch.zeros_like(result.engagement_scores),
    )
    assert result.social_state["dominant_emotion"] == "Neutral"
