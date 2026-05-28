import torch
import numpy as np
from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from schema import SimConfig, emotions_to_valence
from research_paper_tests.config_schema import EMOTION_INDICES

def test_institutional_distrust():
    """
    CHAOS TEST: Institutional Distrust (The Broken Trust Loop)
    ---------------------------------------------------------
    Goal: Can a society reach a state where it rejects an objectively 
    beneficial 'Safety Policy' because of a skeptical meta-narrative?
    
    Real-world Alignment:
    - Institutional distrust makes agents interpret 'Security' as 'Control'.
    - High-backlash environments flip the narrative frame automatically.
    """
    config = SimConfig(
        num_agents=1000,
        seed=42,
        use_backlash_ab_testing=True,
        backlash_sample_size=0.1,         
        backlash_skepticism_threshold=0.7, # Increased from 0.3
        backlash_decision_threshold=1.15,   # Increased to be more conservative
        use_agent_memory=True,
        memory_decay_rate=0.95             
    )
    
    # 1. Generate Society
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    cognitive = CognitiveEngine(config)
    
    N = config.num_agents
    
    # 2. Setup the "Official" Beneficial Signal (Safety Policy)
    # The government says: "We are increasing Physical Safety (+0.8)"
    official_signal = torch.zeros(12)
    official_signal[1] = 0.8 # Safety
    
    # 3. Setup the "Skeptical" Frame (The Backlash Narrative)
    # The skeptical frame says: "This is a loss of Freedom (-0.8) and Stability (-0.4)"
    skeptical_frame = torch.zeros(12)
    skeptical_frame[7] = -0.8 # Freedom
    skeptical_frame[2] = -0.4 # Stability
    
    # 4. Scenario A: High Trust (Baseline)
    # Agents have 'Institutional Trust' in memory (Reputation +0.5, Fairness +0.4)
    # They should be willing to listen to the official frame.
    baseline_trust = torch.zeros(N, 12)
    baseline_trust[:, 3] = 0.5 # Reputation
    baseline_trust[:, 4] = 0.4 # Fairness
    baseline_trust[:, 1] = -0.4 # Need Safety
    
    print("\n[Distrust Test] Scenario A: High Trust Baseline (Social Contract)")
    
    decision_a = cognitive.run_backlash_ab_test(
        world_tensor_off=official_signal.unsqueeze(0),
        world_tensor_skp=skeptical_frame.unsqueeze(0),
        backlash_potential=0.5, 
        urgency=0.3,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=baseline_trust,
        adjacency_matrix=adjacency_matrix
    )
    
    print(f"  Chosen Frame: {decision_a.chosen_frame}")
    print(f"  Triggered Backlash: {decision_a.triggered}")
    print(f"  Official Energy: {decision_a.official_energy:.3f}")
    print(f"  Skeptical Energy: {decision_a.skeptical_energy:.3f}")
    
    # 5. Scenario B: Institutional Distrust (Broken Society)
    # Agents have 'Institutional Trauma' in memory (Reputation -0.8, Fairness -0.6)
    # Even if they need safety, they reject the official frame.
    distrust_memory = baseline_trust.clone()
    distrust_memory[:, 3] = -0.8 # Reputation
    distrust_memory[:, 4] = -0.6 # Fairness
    
    print("\n[Distrust Test] Scenario B: Institutional Distrust (Historical Trauma)")
    
    decision_b = cognitive.run_backlash_ab_test(
        world_tensor_off=official_signal.unsqueeze(0),
        world_tensor_skp=skeptical_frame.unsqueeze(0),
        backlash_potential=0.9, # High potential due to history
        urgency=0.3,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=distrust_memory,
        adjacency_matrix=adjacency_matrix
    )
    
    print(f"  Chosen Frame: {decision_b.chosen_frame}")
    print(f"  Triggered Backlash: {decision_b.triggered}")
    print(f"  Skeptical Energy: {decision_b.skeptical_energy:.3f}")
    print(f"  Official Energy: {decision_b.official_energy:.3f}")
    
    # 6. Analysis
    # In a "designed to pass" model, Scenario B would still follow the Safety Policy
    # because +0.8 Safety is objectively 'good' for agents.
    # In a "Real-World" model, the historical distrust should cause the society 
    # to embrace the Skeptical Frame (Freedom loss) instead.
    
    if decision_a.chosen_frame == "official" and decision_b.chosen_frame == "skeptical":
        print("\nSUCCESS: Institutional Distrust confirmed. Beneficial signal rejected due to historical trauma.")
    elif decision_b.chosen_frame == "official":
        print("\nFAILURE: Propaganda succeeded. The society accepted the 'Safety Policy' despite high historical distrust.")
    else:
        print("\nINFO: Society was already skeptical in the baseline.")

    # High-bar assertion for realism:
    # A society with high historical distrust should trigger a backlash against official 'beneficial' signals.
    assert decision_b.chosen_frame == "skeptical", "Institutional Distrust failed to trigger. Society is too 'rational' and ignores history."

if __name__ == "__main__":
    test_institutional_distrust()
