import torch
import sys
import os

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SimConfig
from physics_engine import SocialPhysicsEngine


def test_endogenous_events():
    print("--- Testing Autopoietic Endogenous Event Generation ---")

    config = SimConfig(
        num_agents=2000,
        seed=42,
        elite_divergence_threshold=0.3,  # lower thresholds to force trigger
        polarization_threshold=0.3,
        stewing_ticks=1,
    )

    phys_engine = SocialPhysicsEngine(config)

    N = config.num_agents
    influence = torch.ones(N)

    print("\n[ Scenario 1: Stable Society (No Event Expected) ]")
    emotions_stable = torch.zeros(N, 8)
    emotions_stable[:, 0] = 0.5  # Moderate joy everywhere

    state_stable = phys_engine.aggregate_society(emotions_stable, influence)
    print(f"Polarization: {state_stable['polarization']:.3f}")
    print(f"Elite Divergence: {state_stable['elite_divergence']:.3f}")
    print(f"Action Triggered: {state_stable.get('action_name', 'None')}")

    assert (
        state_stable.get("action_name") is None
    ), "Failed: Action triggered in stable society."

    print("\n[ Scenario 2: Highly Polarized Society (Protest Expected) ]")
    # Group A furious, Group B joyful
    emotions_polarized = torch.zeros(N, 8)
    emotions_polarized[:1000, 4] = 1.0  # Anger
    emotions_polarized[1000:, 0] = 1.0  # Joy

    state_polarized = phys_engine.aggregate_society(emotions_polarized, influence)
    print(f"Polarization: {state_polarized['polarization']:.3f}")
    print(f"Elite Divergence: {state_polarized['elite_divergence']:.3f}")
    print(f"Action Triggered: {state_polarized.get('action_name', 'None')}")

    assert state_polarized.get("action_name") in [
        "Civil Protest",
        "Populist Uprising",
    ], "Failed: No protest triggered."

    print("\n[ Scenario 3: Elite Divergence (Policy Shift Expected) ]")
    # Make the elite 5% completely out of touch with the rest
    emotions_divergent = torch.zeros(N, 8)
    emotions_divergent[:1900, 4] = 0.5  # Population is moderately angry
    emotions_divergent[1900:, 0] = 1.0  # Elite are extremely joyful

    # Need to simulate unequal influence to make the top 5% actual "elites"
    unequal_influence = torch.ones(N)
    unequal_influence[1900:] = 100.0  # Whales

    state_divergent = phys_engine.aggregate_society(
        emotions_divergent, unequal_influence
    )
    print(f"Polarization: {state_divergent['polarization']:.3f}")
    print(f"Elite Divergence: {state_divergent['elite_divergence']:.3f}")
    print(f"Action Triggered: {state_divergent.get('action_name', 'None')}")

    assert state_divergent.get("action_name") in [
        "Elite Policy Shift",
        "Populist Uprising",
    ], "Failed: No divergence event triggered."

    print("\n--- Success! Autopoietic loop thresholds are working. ---")


if __name__ == "__main__":
    test_endogenous_events()
