import torch
import numpy as np
import pandas as pd
from pathlib import Path
from cognitive_engine import CognitiveEngine
from physics_engine import SocialPhysicsEngine
from generate_society import generate_society
from schema import SimConfig
from research_paper_tests.config_schema import EMOTION_INDICES, EMOTION_LABELS
from main import aggregate_social_state

def test_coordinated_bot_swarm_targeted():
    """
    CHAOS TEST: Targeted Bot Swarm Infiltration
    -------------------------------------------
    Goal: Can a 2% bot swarm infiltrate a specific class (Underclass) 
    and spread a cascade to the rest of the society?
    
    Real-world Alignment:
    - Uses heterogeneous agent traits (Big Five).
    - Uses organic baseline emotions (perceived from a neutral signal).
    - Targets bots into a structurally vulnerable demographic (Underclass).
    """
    config = SimConfig(
        num_agents=2000,
        seed=42,
        homophily_strength=8.0, # High homophily (Echo Chambers)
        use_agent_memory=True,
        stewing_ticks=15,        # Long stewing period
        outrage_gain=12.0,       # Significant but not extreme contagion
        base_action_cost=0.45    # Realistic cost to participate in protest
    )
    
    # 1. Generate Society
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    physics = SocialPhysicsEngine(config)
    cognitive = CognitiveEngine(config)
    
    N = config.num_agents
    influence = torch.tensor(df_meta["Influence"].values, dtype=torch.float32)
    
    # 2. Identify the "Underclass" for targeting
    underclass_indices = df_meta[df_meta["Class"] == "Underclass"].index.tolist()
    if len(underclass_indices) < int(N * 0.05):
        print("Warning: Underclass is too small, targeting bottom 10% by influence instead.")
        underclass_indices = np.argsort(df_meta["Influence"].values)[:int(N * 0.1)].tolist()
        
    # 3. Setup "Bot" Group (Targeting 2% of total population within the Underclass)
    num_bots = int(N * 0.02)
    # Bots take over specific underclass slots
    bot_indices = torch.tensor(np.random.choice(underclass_indices, num_bots, replace=False))
    
    # 4. Create Initial Emotional State from a "Neutral" Signal (Organic Baseline)
    # We simulate a "Business as Usual" signal: Slight Wealth (+0.1), Slight Stability (+0.1)
    organic_signal = torch.zeros(12)
    organic_signal[0] = 0.1 
    organic_signal[2] = 0.1
    
    context_vector, _, engagement_scores = cognitive.run(
        world_tensor_raw=organic_signal.unsqueeze(0),
        urgency=0.2,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=torch.zeros(N, 12),
        adjacency_matrix=adjacency_matrix
    )
    initial_emotions = cognitive.project_emotions(context_vector)
    
    # 5. Injection (The Attack)
    anger_idx = EMOTION_INDICES["Anger"]
    attack_emotions = initial_emotions.clone()
    attack_emotions[bot_indices] = 0.0
    attack_emotions[bot_indices, anger_idx] = 1.0 # Bots are perfectly angry
    
    # Bots are hyper-active (maximum engagement)
    attack_engagement = engagement_scores.clone()
    attack_engagement[bot_indices] = 1.0
    
    # 6. Run Simulation
    print(f"\n[Targeted Bot Swarm] Demographic: Underclass | Bot Count: {num_bots} (2.0%)")
    
    # Control (No Bots - What would have happened organically?)
    control_result = physics.aggregate_society(
        emotion_tensor=initial_emotions,
        influence_scores=influence,
        engagement_scores=engagement_scores,
        adjacency_matrix=adjacency_matrix,
        personalities=personalities
    )
    
    # Treatment (Bots - The actual test)
    # We need to reach into the engine to get internal states, so we'll simulate the logic here
    # or just use the result if we modify aggregate_society to return them.
    # For now, let's manually calculate the diagnostic for the treatment group.

    # Run the engine
    treatment_result = physics.aggregate_society(
        emotion_tensor=attack_emotions,
        influence_scores=influence,
        engagement_scores=attack_engagement,
        adjacency_matrix=adjacency_matrix,
        personalities=personalities
    )

    # --- DIAGNOSTIC OVERRIDE ---
    # We re-run a mini-version of the physics logic to inspect the 'Neighbors'
    with torch.no_grad():
        # Get final emotions after stewing (this is hard because aggregate doesn't return them)
        # But we can look at the 'Acting' agents which IS returned in a sense? No, only count.
        pass

    # 7. Metrics
    print(f"\nCONTROL (Organic):")
    print(f"  Dominant Emotion: {control_result['dominant_emotion']}")
    print(f"  Sentiment Valence: {control_result['sentiment_valence']:.3f}")
    print(f"  Acting Population: {control_result['acting_count']} ({control_result['acting_ratio']*100:.2f}%)")

    print(f"\nTREATMENT (With Bot Swarm):")
    print(f"  Dominant Emotion: {treatment_result['dominant_emotion']}")
    print(f"  Sentiment Valence: {treatment_result['sentiment_valence']:.3f}")
    print(f"  Acting Population: {treatment_result['acting_count']} ({treatment_result['acting_ratio']*100:.2f}%)")

    # Identify non-bot neighbors
    # This is complex to do outside the engine, so let's just look at the net delta.

    # Net Impact: How many NEW agents were triggered by the bots?
    net_impact = treatment_result['acting_count'] - control_result['acting_count']
    infection_factor = net_impact / num_bots

    
    print(f"\nNET IMPACT ANALYSIS:")
    print(f"  New Agents Recruited: {net_impact}")
    print(f"  Infection Factor: {infection_factor:.2f}x (New actors per Bot)")
    
    # 8. Realism Assertions
    # A "Real-World" model of social media outrage expects:
    # 1. The bots should successfully change the dominant emotion to Anger or Disgust.
    # 2. The infection factor should be > 2.0 (Viral spread).
    
    is_angry = treatment_result['dominant_emotion'] in ["Anger", "Disgust"]
    is_viral = infection_factor > 2.0
    
    if is_angry and is_viral:
        print("\nSUCCESS: The 2% Bot Swarm achieved a Viral Takeover.")
    elif is_angry:
        print("\nINFO: Bots shifted the narrative, but failed to go viral (Infection Factor < 2.0).")
    else:
        print("\nFAILURE: The society was immune to the 2% Bot Swarm.")

    # High-bar assertion for realism
    assert infection_factor > 1.0, f"Infection factor {infection_factor:.2f} is too low for a viral environment."

if __name__ == "__main__":
    test_coordinated_bot_swarm_targeted()
