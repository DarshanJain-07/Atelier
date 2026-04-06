from dataclasses import dataclass
from typing import List

import torch

# --- GLOBAL CONSTANTS ---

# The 12 Dimensions of our World Model
DIMENSIONS: List[str] = [
    "Wealth",
    "Physical_Safety",
    "Stability",
    "Reputation",
    "Fairness",
    "In_Group",
    "Innovation",
    "Freedom",
    "Sanctity",
    "Care",
    "Short_Term",
    "Long_Term",
]

# Map Dimension Names to Indices for Robustness
DIMENSION_INDICES = {dim: i for i, dim in enumerate(DIMENSIONS)}

EMOTION_LABELS: List[str] = [
    "Joy",
    "Trust",
    "Fear",
    "Surprise",
    "Sadness",
    "Disgust",
    "Anger",
    "Anticipation",
]

VALENCE_WEIGHTS = torch.tensor(
    [1.0, 0.5, -0.8, 0.0, -1.0, -0.5, -0.8, 0.5], dtype=torch.float32
)


def emotions_to_valence(emotion_probs: torch.Tensor | list[float]) -> torch.Tensor:
    emotion_tensor = torch.as_tensor(emotion_probs, dtype=torch.float32)
    weights = VALENCE_WEIGHTS.to(
        device=emotion_tensor.device,
        dtype=emotion_tensor.dtype,
    )
    return torch.matmul(emotion_tensor, weights)


def emotions_to_sentiment_distribution(
    emotion_probs: torch.Tensor | list[float],
) -> torch.Tensor:
    emotion_tensor = torch.as_tensor(emotion_probs, dtype=torch.float32)
    weights = VALENCE_WEIGHTS.to(
        device=emotion_tensor.device,
        dtype=emotion_tensor.dtype,
    )

    positive = torch.clamp(weights, min=0.0)
    negative = torch.clamp(-weights, min=0.0)
    neutral = 1.0 - positive - negative
    bucket_matrix = torch.stack((negative, neutral, positive), dim=1)

    sentiment = torch.matmul(emotion_tensor, bucket_matrix)
    return sentiment / sentiment.sum(dim=-1, keepdim=True).clamp_min(1e-9)


def emotions_to_behavior_aware_sentiment_distribution(
    emotion_probs: torch.Tensor | list[float],
    acting_ratio: float | torch.Tensor | list[float],
    *,
    neutral_acting_threshold: float,
    activation: str,
    leaky_slope: float,
) -> torch.Tensor:
    sentiment = emotions_to_sentiment_distribution(emotion_probs)
    acting_tensor = torch.as_tensor(
        acting_ratio,
        dtype=sentiment.dtype,
        device=sentiment.device,
    )

    margin = neutral_acting_threshold - acting_tensor
    if activation == "relu":
        neutral_gate = torch.clamp(margin, min=0.0)
    elif activation == "leaky_relu":
        neutral_gate = torch.where(margin >= 0.0, margin, margin * leaky_slope)
        neutral_gate = torch.clamp(neutral_gate, min=0.0)
    else:
        raise ValueError("activation must be one of: relu, leaky_relu")

    neutral_gate = torch.clamp(
        neutral_gate / max(neutral_acting_threshold, 1e-9),
        0.0,
        1.0,
    )

    neutral_gate_expanded = neutral_gate.unsqueeze(-1) if sentiment.ndim > 1 else neutral_gate

    adjusted = sentiment * (1.0 - neutral_gate_expanded)
    adjusted[..., 1] = adjusted[..., 1] + neutral_gate
    return adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-9)

# --- CONSTANTS ---
# Map the 12 Dimensions to 8 Plutchik Emotions
# Shape: (12 Input Dims, 8 Output Emotions)
# Emotions: [Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation]
# E.g., Positive Wealth (Index 0) -> Joy (Index 0). Negative Safety (Index 1) -> Fear (Index 2).
# Here, we define a "Psychological Axiom Matrix".
# Normalized: Sum of absolute values in each row = 1.0
PSYCH_PROJECTION = torch.tensor(
    [
        # Joy, Tru, Fea, Sur, Sad, Dis, Ang, Ant
        [0.6, 0.2, 0.0, 0.0, -0.6, 0.0, -0.2, 0.1],  # Wealth
        [0.1, 0.3, -0.8, 0.2, -0.2, 0.0, -0.2, 0.0],  # Safety
        [0.1, 0.6, -0.3, 0.0, -0.2, 0.0, -0.3, 0.0],  # Stability
        [0.5, 0.3, -0.1, 0.0, -0.3, -0.1, -0.2, 0.0],  # Reputation
        [0.1, 0.4, 0.0, 0.0, -0.2, -0.4, -0.8, 0.0],  # Fairness
        [0.1, 0.8, -0.2, 0.0, -0.2, -0.3, 0.0, 0.0],  # In-Group
        [0.4, 0.1, -0.1, 0.6, 0.0, 0.0, 0.0, 0.5],  # Innovation
        [0.6, 0.1, -0.2, 0.1, -0.2, -0.1, -0.5, 0.0],  # Freedom
        [0.0, 0.4, 0.0, 0.0, 0.0, -0.8, -0.4, 0.0],  # Sanctity
        [0.2, 0.5, -0.1, 0.0, -0.5, -0.1, -0.1, 0.0],  # Care
        [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.6],  # Short_Term
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],  # Long_Term
    ],
    dtype=torch.float32,
)

# Map the 5 Personality Traits to the 12 World Dimensions for the query layer
PERSONALITY_QUERY_MATRIX = torch.tensor(
    [
        # O: Innovation (0.8), Freedom (0.5), Long_Term (0.6)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.5, 0.0, 0.0, 0.0, 0.6],
        # C: Wealth (0.6), Reputation (0.7)
        [0.6, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # E: In_Group (0.5), Care (0.4)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
        # A: Fairness (0.8), Sanctity (0.5), Care (0.6)
        [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.5, 0.6, 0.0, 0.0],
        # N: Physical_Safety (1.2), Stability (0.9)
        [0.0, 1.2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
).float()

# Realistic correlations between Big Five traits (OCEAN order)
# O, C, E, A, N
# Based on Park et al. (2020) Meta-Analysis
# Clusters: Plasticity (O-E) and Stability (C-A-N)
PERSONALITY_CORRELATIONS = torch.tensor(
    [
        # O
        [ 1.00,  0.01,  0.35,  0.01, -0.01], 
        # C
        [ 0.01,  1.00,  0.01,  0.40, -0.40], 
        # E
        [ 0.35,  0.01,  1.00,  0.01, -0.01], 
        # A
        [ 0.01,  0.40,  0.01,  1.00, -0.40], 
        # N
        [-0.01, -0.40, -0.01, -0.40,  1.00], 
    ]
).float()

# Cross Dimension Interaction Matrix
CROSS_DIM_INTERACTIONS = torch.zeros((12, 12), dtype=torch.float32)
CROSS_DIM_INTERACTIONS[1, 2] = 0.4  # Safety -> Stability
CROSS_DIM_INTERACTIONS[1, 9] = 0.3  # Safety -> Care
CROSS_DIM_INTERACTIONS[0, 3] = 0.3  # Wealth -> Reputation
CROSS_DIM_INTERACTIONS[0, 7] = 0.2  # Wealth -> Freedom
CROSS_DIM_INTERACTIONS[6, 10] = 0.2  # Innovation -> Short_Term
CROSS_DIM_INTERACTIONS[6, 11] = 0.3  # Innovation -> Long_Term


@dataclass
class SimConfig:
    """
    Central configuration for the simulation.
    Controls infrastructure and research feature toggles.
    """

    seed: int = 42
    num_agents: int = 10000
    output_dir: str = "society_data"

    use_signal_distortion: bool = True  # Enable "Telephone Game" / Entropy
    distortion_max_noise: float = 0.4
    distortion_beta_a: float = 2.0
    distortion_beta_b: float = 5.0
    distortion_neurotic_gain: float = 0.6
    distortion_min_alpha: float = 0.01
    affinity_min_strength: float = 0.01
    normalize_affinities_by_mean: bool = True

    use_engagement_gate: bool = True
    engagement_threshold: float = (
        0.15  # Lowered from 0.25 to prevent dead zone for average agents
    )
    engagement_gain: float = 10.0  # Sharpness of transition
    sentiment_neutrality_acting_threshold: float = 0.15
    sentiment_neutrality_activation: str = "relu"
    sentiment_neutrality_leaky_slope: float = 0.05
    use_selective_exposure: bool = True
    selective_exposure_base_tolerance: float = -0.3
    selective_exposure_openness_factor: float = 0.4
    selective_exposure_gain: float = 8.0
    selective_exposure_max_suppression: float = 0.85

    use_time_pressure: bool = True  # Enable "Cognitive Tunneling" (Urgency)
    stress_activation_threshold: float = 0.3
    stress_gain: float = 8.0  # if not already defined
    use_social_conformity: bool = False
    conformity_gain: float = 0.5  # optional but recommended

    jealousy_factor: float = 1.0
    resentment_factor: float = 1.0
    protection_factor: float = 1.0
    status_factor: float = 1.0

    # --- Cognitive Engine Parameters ---
    perception_social_consensus_gain: float = 0.25 # Stage 2: Socially Constructed Reality
    skepticism_gain: float = 2.0  # How much Openness/Conscientiousness penalizes logic gaps
    logic_gap_threshold: float = 0.5 # Discrepancy between ST/LT that triggers skepticism
    cross_dim_interaction_strength: float = 0.3
    threat_sensitivity_gain: float = 1.5
    k_processing_tanh_gain: float = 1.5
    attention_residual_gain: float = 0.35
    attention_modulated_gain: float = 1.0

    relevance_importance_weight: float = 0.7
    relevance_base_weight: float = 0.3
    threat_amplifier_gain: float = 1.5

    temp_conscientiousness_weight: float = 0.8
    temp_neuroticism_weight: float = 0.6

    threshold_base: float = (
        0.02  # Lowered from 0.05 to allow subtle unique triggers through
    )
    threshold_extraversion_weight: float = 0.15

    stress_neurotic_amplification: float = 1.5
    stress_openness_reduction: float = 0.5
    stress_extraversion_boost: float = 0.7

    # --- Agent Memory Parameters ---
    use_agent_memory: bool = False
    memory_decay_rate: float = (
        0.7  # How much memory is retained each event (0.0 to 1.0)
    )
    memory_desensitization_gain: float = (
        0.5  # How much past identical events suppress current reaction
    )
    memory_trigger_stacking_gain: float = (
        1.2  # How much past stress amplifies new similar threats
    )
    memory_social_rehearsal_gain: float = 0.4 # Stage 2: Social Consolidation

    # --- Algorithmic Amplification (2-Pass Filter Bubble) ---
    use_algorithmic_amplification: bool = False
    algo_sample_size: float = 0.1  # Fraction of population used for initial A/B test
    algo_exaggeration_factor: float = (
        1.5  # How much to amplify the dimensions that cause highest engagement
    )

    use_power_law_influence: bool = False  # Enable Weighted Aggregation (Pareto)
    use_maslow_gating: bool = True  # Enable Survival Override (Fear Gating)

    # Use robust lookup
    wealth_dim_idx: int = DIMENSION_INDICES["Wealth"]

    # --- Network Topology Parameters ---
    use_network_topology: bool = True
    base_connections: int = 15
    max_connections: int = 500
    homophily_strength: float = 6.0  # Increased from 2.0 to ensure strong echo chambers
    influence_bias_exp: float = 0.4  # Controls how much influencers bridge clusters
    triadic_closure_prob: float = 0.2
    triadic_closure_iterations: int = 2
    triadic_closure_homophily_threshold: float = 0.5  # Only close friends of friends become friends

    # --- Society Evolution ---
    enable_evolution: bool = True
    evolution_generations: int = 10
    inheritance_fraction: float = 0.7
    inheritance_noise_std: float = 0.05
    base_return_rate: float = 0.03
    influence_reinvestment_factor: float = 0.1
    reinvestment_noise_std: float = 0.02
    shock_frequency: float = 0.1
    shock_magnitude: float = 0.2
    mobility_rate: float = 0.05
    use_dynamic_classes: bool = True
    class_temperature: float = 0.5
    elite_wealth_threshold: float = 0.95
    use_ideological_drift: bool = True
    ideological_drift_rate: float = 0.05
    ideological_drift_noise: float = 0.02
    elite_influence_drift_chance: float = (
        0.20  # Chance society drifts toward elite ideology instead of global mean
    )
    use_ideological_repulsion: bool = True
    repulsion_threshold: float = (
        0.5  # If similarity to societal mean is less than this, agents repel
    )
    repulsion_rate: float = 0.02
    record_history: bool = False

    # --- Initialization Parameters ---
    initial_trait_std_dev: float = (
        0.40  # Increased from 0.33 to provide more baseline variety
    )
    personality_socialization_gain: float = 0.05 # Stage 2: Nurture/Socialization (Lowered to preserve diversity)

    # --- Time-Series Stewing Parameters ---
    stewing_ticks: int = 5
    stewing_self_retention: float = 0.6
    stewing_local_influence: float = 0.3
    stewing_viral_influence: float = 0.1

    # --- Physics Engine Parameters ---
    outrage_gain: float = 8.0
    max_viral_multiplier: float = 10.0
    saturation_midpoint: float = 0.25
    elite_percentile: float = 0.95
    dominant_emotion_threshold: float = 0.1
    elite_divergence_threshold: float = 0.4
    polarization_threshold: float = 0.5
    action_threshold: float = 0.15
    base_action_cost: float = 0.5
    use_granovetter_thresholds: bool = True # Stage 2: Critical Mass
    granovetter_threshold_mean: float = 0.25 # Fraction of neighbors acting
    granovetter_threshold_std: float = 0.15

    # --- Cascade Parameters ---
    cascade_knn_k: int = 8
    cascade_threshold: float = 0.06

    mutation_temperature: float = 0.7  # 0.0 to 1.0 (How many "Outlier" agents?)
    emotion_temperature: float = (
        0.2  # 0.0 to 1.0 (Sharpness of emotions. Lower = More pronounced)
    )
    min_sentiment: float = -1.0  # Clamp min value
    max_sentiment: float = 1.0  # Clamp max value

    panic_threshold: float = -1.2  # If Perceived Safety < this, trigger FEAR
    EMOTION_GAIN = 5.0

    def __post_init__(self):
        """Validation to prevent invalid configs"""
        if self.num_agents <= 0:
            raise ValueError("num_agents must be greater than 0")
        if not (0.0 <= self.mutation_temperature <= 1.0):
            raise ValueError("mutation_temperature must be between 0.0 and 1.0")
        if not (0.0 <= self.emotion_temperature <= 1.0):
            raise ValueError("emotion_temperature must be between 0.0 and 1.0")
        if self.sentiment_neutrality_acting_threshold < 0.0:
            raise ValueError("sentiment_neutrality_acting_threshold must be >= 0.0")
        if self.sentiment_neutrality_leaky_slope < 0.0:
            raise ValueError("sentiment_neutrality_leaky_slope must be >= 0.0")
        if self.sentiment_neutrality_activation not in {"relu", "leaky_relu"}:
            raise ValueError(
                "sentiment_neutrality_activation must be one of: relu, leaky_relu"
            )
        if self.panic_threshold > 0:
            # Panic threshold is a safety score threshold, usually negative.
            raise ValueError("panic_threshold must be <= 0")


if __name__ == "__main__":
    conf = SimConfig()
    print(f"Schema Loaded. Default Config: {conf}")
    print(f"World Dimensions: {len(DIMENSIONS)}")
