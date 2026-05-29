import torch
import numpy as np
from physics_engine import SocialPhysicsEngine
from research_paper_tests.config_schema import SimConfig
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_monotonic_relationship,
    assert_statistically_greater
)

def test_information_fatigue_burnout_gradient(n_seeds):
    """
    Validation: Sustained high arousal must lead to a statistically significant 
    drop in engagement over time due to the refractory period (burnout).
    This proves that agents exhibit "News Avoidance" behavior when overwhelmed.
    """
    def run_burnout_sim():
        config = SimConfig()
        config.use_refractory_period = True
        config.refractory_arousal_threshold = 0.4
        config.refractory_threshold_duration = 3
        config.refractory_engagement_drop = 0.8
        config.stewing_ticks = 1
        
        physics = SocialPhysicsEngine(config)
        N = 100
        # Emotions: [Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation]
        high_arousal_emotions = torch.zeros(N, 8)
        high_arousal_emotions[:, 6] = 1.0 # Sustained Anger
        influence = torch.ones(N)
        
        multipliers = []
        for _ in range(6): # Run a sequence of 6 high-arousal events
            res = physics.aggregate_society(high_arousal_emotions, influence)
            multipliers.append(res['mean_outrage_multiplier'])
        return multipliers

    # 1. Monte Carlo Execution
    results = run_monte_carlo(run_burnout_sim, n_seeds=n_seeds)
    # Aggregate results into mean timeline
    mean_multipliers = np.mean(results, axis=0)
    
    # 2. Gradient Assertion: Engagement must decrease as burnout accumulates
    # We check the tail of the sequence (post-trigger duration)
    # Event 0-2: Building pressure. Event 3-5: Burnout should be visible.
    assert_monotonic_relationship(range(3, 6), mean_multipliers[3:], "negative")

    # 3. Statistical Significance
    # Initial engagement vs Final engagement after burnout
    start_dist = [r[0] for r in results]
    end_dist = [r[-1] for r in results]
    assert_statistically_greater(start_dist, end_dist)
