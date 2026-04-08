import torch

from cognitive_engine import CognitiveEngine
from schema import SimConfig


def test_tensor_magnitudes_stay_finite_and_probabilistic():
    torch.manual_seed(42)

    config = SimConfig()
    cog_engine = CognitiveEngine(config)

    world_tensor_raw = torch.randn(1, 12) * 1.5
    personalities = torch.rand(3, 5)
    exposures = torch.rand(3, 12) * 2 - 1
    agent_affinities = torch.ones(3, 12)

    context_vector, attention_weights, engagement_scores = cog_engine.run(
        world_tensor_raw=world_tensor_raw,
        urgency=0.5,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=agent_affinities,
    )
    emotions = cog_engine.project_emotions(context_vector)

    assert context_vector.shape == (3, 12)
    assert attention_weights.shape == (3, 12)
    assert engagement_scores.shape == (3,)
    assert emotions.shape == (3, 8)

    assert torch.isfinite(context_vector).all()
    assert torch.isfinite(attention_weights).all()
    assert torch.isfinite(engagement_scores).all()
    assert torch.isfinite(emotions).all()

    assert torch.allclose(attention_weights.sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(emotions.sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.all(torch.norm(context_vector, dim=1) > 0)
    assert torch.all(engagement_scores > 0)
