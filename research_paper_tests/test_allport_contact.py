import numpy as np
import torch

from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from physics_engine import SocialPhysicsEngine
from research_paper_tests.config_schema import SimConfig
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater

def run_allport_scenario(bridge_strength: float, homophily: float, seed: int):
    """
    Runs a simulation with a specific topology configuration.
    Returns final polarization.
    """
    config = SimConfig(
        num_agents=1000,
        seed=seed,
        homophily_strength=homophily,
        base_connections=15,
        use_agent_memory=True,
        memory_decay_rate=0.78,
        use_backlash_ab_testing=True
    )
    
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    
    cognitive_engine = CognitiveEngine(config)
    physics_engine = SocialPhysicsEngine(config)
    
    N = config.num_agents
    influence = torch.tensor(df_meta["Influence"].values, dtype=torch.float32)
    agent_memory = torch.zeros(N, 12)
    
    # Event: "Controversial Policy" 
    # Positive for Elite (Wealth +0.8), Negative for Fairness (-0.8)
    controversial_event = torch.zeros(12)
    controversial_event[0] = 0.8 # Wealth
    controversial_event[4] = -0.8 # Fairness
    
    final_pol = 0
    
    for i in range(5):
        context_vector, _, engagement_scores = cognitive_engine.run(
            world_tensor_raw=controversial_event.unsqueeze(0),
            urgency=0.5,
            is_personal=False,
            exposures=exposures,
            personalities=personalities,
            agent_affinities=affinities,
            agent_memory=agent_memory,
            adjacency_matrix=adjacency_matrix
        )
        
        emotion_tensor = cognitive_engine.project_emotions(context_vector)
        result = physics_engine.aggregate_society(
            emotion_tensor=emotion_tensor,
            influence_scores=influence,
            engagement_scores=engagement_scores,
            adjacency_matrix=adjacency_matrix,
            personalities=personalities
        )
        
        agent_memory = cognitive_engine.consolidate_memory(
            agent_memory, context_vector, social_rehearsal_factor=result["acting_ratio"]
        )
        
        final_pol = result["polarization"]
        
    return final_pol

def test_allport_intergroup_contact(n_seeds):
    """
    Validates that high homophily (echo chambers) leads to statistically 
    higher polarization than low homophily (integrated societies).
    """
    def runner():
        # Use a seed from the already-seeded numpy to vary generate_society
        seed = np.random.randint(0, 1000000)
        pol_low = run_allport_scenario(bridge_strength=0.5, homophily=3.0, seed=seed)
        pol_high = run_allport_scenario(bridge_strength=0.5, homophily=12.0, seed=seed)
        return {"low": pol_low, "high": pol_high}

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    low_pol = [r["low"] for r in results]
    high_pol = [r["high"] for r in results]
    
    assert_statistically_greater(high_pol, low_pol)
