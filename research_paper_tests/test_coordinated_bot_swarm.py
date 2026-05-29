import torch
import numpy as np
from cognitive_engine import CognitiveEngine
from physics_engine import SocialPhysicsEngine
from generate_society import generate_society
from research_paper_tests.config_schema import EMOTION_INDICES, SimConfig
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_monotonic_relationship,
    assert_statistically_greater
)

def test_bot_swarm_intensity_gradient(n_seeds):
    """
    Validation: Increasing the density of a coordinated bot swarm must 
    monotonically increase the net infection factor (recruitment of organic agents).
    This proves that bots successfully exploit the social physics of contagion.
    """
    # Sweep bot percentage from 0.5% (incipient) to 5.0% (saturation)
    bot_percent_sweep = [0.005, 0.01, 0.02, 0.03, 0.05]
    mean_infection_factors = []

    def get_sim_runner(bot_ratio):
        def runner():
            config = SimConfig(
                num_agents=1000, 
                homophily_strength=8.0,
                use_agent_memory=False, 
                stewing_ticks=10,
                outrage_gain=15.0,
                base_action_cost=0.5
            )
            df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
            physics = SocialPhysicsEngine(config)
            cognitive = CognitiveEngine(config)
            
            N = config.num_agents
            influence = torch.tensor(df_meta["Influence"].values, dtype=torch.float32)
            
            # Setup organic baseline using a neutral signal
            organic_signal = torch.zeros(12)
            organic_signal[0] = 0.1 # Slight Wealth stimulus
            
            context, _, organic_engagement = cognitive.run(
                world_tensor_raw=organic_signal.unsqueeze(0),
                urgency=0.2,
                is_personal=False,
                exposures=exposures,
                personalities=personalities,
                agent_affinities=affinities,
                agent_memory=torch.zeros(N, 12),
                adjacency_matrix=adjacency_matrix
            )
            organic_emotions = cognitive.project_emotions(context)
            
            # 1. Control: Organic outcome without bot interference
            control_res = physics.aggregate_society(
                organic_emotions, influence, organic_engagement, adjacency_matrix, personalities
            )
            
            # 2. Treatment: Inject Coordinated Bots
            num_bots = int(N * bot_ratio)
            # Bots are randomly distributed but perfectly coordinated
            bot_indices = np.random.choice(N, num_bots, replace=False)
            
            attack_emotions = organic_emotions.clone()
            attack_emotions[bot_indices, EMOTION_INDICES["Anger"]] = 1.0 
            
            attack_engagement = organic_engagement.clone()
            attack_engagement[bot_indices] = 1.0 
            
            treatment_res = physics.aggregate_society(
                attack_emotions, influence, attack_engagement, adjacency_matrix, personalities
            )
            
            # Calculate the 'Net Recruitment': Total organic agents who started acting
            net_impact = treatment_res['acting_count'] - control_res['acting_count']
            return float(net_impact)
        return runner

    # Execute Sweep
    for ratio in bot_percent_sweep:
        results = run_monte_carlo(get_sim_runner(ratio), n_seeds=n_seeds)
        mean_impacts = [float(r) for r in results]
        mean_infection_factors.append(np.mean(mean_impacts))

    # Assertion 1: Monotonicity (Bot Density -> Total Viral Recruitment)
    assert_monotonic_relationship(bot_percent_sweep, mean_infection_factors, "positive")

    # Assertion 2: Statistical Significance
    low_results = run_monte_carlo(get_sim_runner(0.005), n_seeds=n_seeds)
    high_results = run_monte_carlo(get_sim_runner(0.05), n_seeds=n_seeds)
    assert_statistically_greater(high_results, low_results)
