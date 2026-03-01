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
        [
            0.35,
            0.05,
            0.0,
            0.0,
            -0.35,
            0.0,
            -0.1,
            0.15,
        ],  # Wealth (+ = Joy, - = Sad)
        [
            0.05,
            0.15,
            -0.35,
            0.15,
            -0.15,
            0.0,
            -0.15,
            0.0,
        ],  # Safety (- = Fear)
        [
            0.1,
            0.35,
            -0.2,
            0.0,
            -0.1,
            0.0,
            -0.25,
            0.0,
        ],  # Stability
        [
            0.3,
            0.2,
            -0.1,
            0.0,
            -0.2,
            -0.1,
            -0.1,
            0.0,
        ],  # Reputation
        [
            0.05,
            0.25,
            0.0,
            0.0,
            -0.1,
            -0.25,
            -0.35,
            0.0,
        ],  # Fairness (- = Anger)
        [
            0.1,
            0.45,
            -0.15,
            0.0,
            -0.1,
            -0.2,
            0.0,
            0.0,
        ],  # In-Group
        [
            0.25,
            0.05,
            -0.05,
            0.35,
            0.0,
            0.0,
            0.0,
            0.3,
        ],  # Innovation (+ = Surprise/Anticipation)
        [
            0.35,
            0.0,
            -0.1,
            0.0,
            -0.15,
            -0.1,
            -0.3,
            0.0,
        ],  # Freedom
        [
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            -0.5,
            -0.25,
            0.0,
        ],  # Sanctity (- = Disgust)
        [
            0.1,
            0.35,
            -0.1,
            0.0,
            -0.3,
            -0.1,
            -0.05,
            0.0,
        ],  # Care (- = Sadness)
        [
            0.0,
            0.0,
            0.15,
            0.05,
            0.0,
            0.0,
            0.1,
            0.7,
        ],  # Short Term
        [0.0, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0, 0.7],  # Long Term
    ]
).float()

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

# Cross Dimension Interaction Matrix
CROSS_DIM_INTERACTIONS = torch.zeros((12, 12), dtype=torch.float32)
CROSS_DIM_INTERACTIONS[1, 2] = 0.4   # Safety -> Stability
CROSS_DIM_INTERACTIONS[1, 9] = 0.3   # Safety -> Care
CROSS_DIM_INTERACTIONS[0, 3] = 0.3   # Wealth -> Reputation
CROSS_DIM_INTERACTIONS[0, 7] = 0.2   # Wealth -> Freedom
CROSS_DIM_INTERACTIONS[6, 10] = 0.2  # Innovation -> Short_Term
CROSS_DIM_INTERACTIONS[6, 11] = 0.3  # Innovation -> Long_Term


@dataclass
class SimConfig:
    """
    Central configuration for the Syntheti-Soc simulation.
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

    use_engagement_gate: bool = True
    engagement_threshold: float = 0.25  # Energy required to meaningfully react
    engagement_gain: float = 10.0  # Sharpness of transition

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
    cross_dim_interaction_strength: float = 0.3
    threat_sensitivity_gain: float = 1.5
    k_processing_tanh_gain: float = 1.5
    
    relevance_importance_weight: float = 0.7
    relevance_base_weight: float = 0.3
    threat_amplifier_gain: float = 1.5

    temp_conscientiousness_weight: float = 0.8
    temp_neuroticism_weight: float = 0.6
    
    threshold_base: float = 0.1
    threshold_extraversion_weight: float = 0.15
    
    stress_neurotic_amplification: float = 1.5
    stress_openness_reduction: float = 0.5
    stress_extraversion_boost: float = 0.7

    use_power_law_influence: bool = False  # Enable Weighted Aggregation (Pareto)
    use_maslow_gating: bool = True  # Enable Survival Override (Fear Gating)
    wealth_dim_idx = DIMENSIONS.index("Wealth")
    
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
    use_dynamic_roles: bool = True
    role_temperature: float = 0.5
    elite_wealth_threshold: float = 0.95
    record_history: bool = False

    # --- Physics Engine Parameters ---
    outrage_gain: float = 2.5
    elite_percentile: float = 0.95

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
        if self.panic_threshold > 0:
            # Panic threshold is a safety score threshold, usually negative.
            raise ValueError("panic_threshold must be <= 0")


if __name__ == "__main__":
    conf = SimConfig()
    print(f"Schema Loaded. Default Config: {conf}")
    print(f"World Dimensions: {len(DIMENSIONS)}")
