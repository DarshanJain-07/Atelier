import numpy as np
import torch

from schema import EMOTION_LABELS, VALENCE_WEIGHTS


class SocialPhysicsEngine:
    """
    Physics Layer
    -----------------
    Nonlinear socio-emotional field model with:
    - Objective state
    - Viral state (nonlinear outrage contagion)
    - Elite state
    - Entropy-based fragmentation
    - Polarization via dispersion and bimodality
    - Active Engagement Weighting (Skin in the Game)
    """

    def __init__(self, config):
        self.config = config

    # ============================================================
    # Core Aggregation
    # ============================================================

    @torch.inference_mode()
    def aggregate_society(
        self,
        emotion_tensor,
        influence_scores,
        engagement_scores=None,
        adjacency_matrix=None,
    ):
        """
        Calculates the socio-emotional metrics for a single event.

        Args:
            emotion_tensor: (N, 8) tensor of emotional states.
            influence_scores: (N,) structural influence weights.
            engagement_scores: (N,) optional tensor from Cognitive Engine. Used to determine
                               active engagement (Skin in the Game).
            adjacency_matrix: (N, N) sparse tensor representing network topology.
        """
        N = emotion_tensor.shape[0]

        # ----------------------------
        # Influence Handling
        # ----------------------------
        if isinstance(influence_scores, (np.ndarray, list)):
            structural_weights = torch.tensor(influence_scores, dtype=torch.float32)
        elif isinstance(influence_scores, torch.Tensor):
            structural_weights = influence_scores.float()
        elif hasattr(influence_scores, "to_numpy"):
            structural_weights = torch.tensor(
                influence_scores.to_numpy(), dtype=torch.float32
            )
        else:
            structural_weights = torch.ones(N, dtype=torch.float32)

        # Influence saturation (prevents oligarch domination)
        structural_weights = torch.log1p(structural_weights)

        # ----------------------------
        # Active Engagement (Skin in the Game)
        # ----------------------------
        if engagement_scores is not None:
            # Agents who aren't paying attention shouldn't dominate the emotional center
            energy = engagement_scores
            # Normalize energy so we don't completely crush baseline influence
            energy = energy / (energy.mean() + 1e-9)

            # Final objective weight is Structural Influence * Active Engagement
            weights = structural_weights * energy
        else:
            weights = structural_weights

        weights = torch.clamp(weights, min=1e-8)
        weights = weights / weights.sum()

        # ============================================================
        # Objective Center of Gravity (Global)
        # ============================================================
        center_of_gravity = (emotion_tensor * weights.unsqueeze(1)).sum(dim=0)

        # ============================================================
        # 1.5️⃣ Topological Context (Local Echo Chambers)
        # ============================================================
        if adjacency_matrix is not None:
            # Each agent's emotional baseline is now the weighted average of their connections
            # adjacency_matrix is row-normalized, so mm acts as a weighted mean
            # Convert to dense to avoid sparse operation limitations on some CPU builds
            local_centers = torch.mm(
                adjacency_matrix.to(emotion_tensor.device).to_dense(), emotion_tensor
            )
        else:
            # If no topology, the local context is just the global context for everyone
            local_centers = center_of_gravity.unsqueeze(0).expand(N, -1)

        # ============================================================
        # Nonlinear Outrage Contagion (Viral State)
        # ============================================================

        # First Principles Virality:
        # 1. Emotional Arousal: Absolute intensity of the emotion.
        # 2. Active Engagement: Is the agent actually paying attention to THIS event?
        # 3. Echo Chamber Amplification: If an agent's highly aroused state is validated by their
        #    immediate peers (local_centers), it creates a denser feedback loop, amplifying virality.

        # Arousal is the raw magnitude of the emotion
        arousal = torch.norm(emotion_tensor, dim=1)

        # If we have cognitive engagement scores, use them as the catalyst
        if engagement_scores is not None:
            # Normalize engagement to act as a multiplier (0.0 to ~2.0)
            normalized_engagement = engagement_scores / (
                engagement_scores.mean() + 1e-9
            )
            viral_energy = arousal * normalized_engagement
        else:
            viral_energy = arousal

        # Network Validation Boost
        # How closely aligned is the agent's emotion with their echo chamber?
        # Cosine similarity between agent emotion and local center emotion
        # (normalized dot product)
        norm_emotion = emotion_tensor / (arousal.unsqueeze(1) + 1e-9)

        # Ensure local_centers is dense before calculating norm
        if local_centers.is_sparse:
            local_centers = local_centers.to_dense()

        local_arousal = torch.norm(local_centers, dim=1)
        norm_local = local_centers / (local_arousal.unsqueeze(1) + 1e-9)

        # Alignment is between -1.0 (opposite) and 1.0 (perfectly aligned)
        alignment = (norm_emotion * norm_local).sum(dim=1)

        # Map alignment from [-1, 1] to [0.0, 2.0]
        # Disagreement dampens the spark (0.0x), fierce agreement fuels it heavily (2.0x)
        validation_multiplier = 1.0 + alignment

        # Apply the validation boost to the viral energy
        viral_energy = viral_energy * validation_multiplier

        outrage_gain = self.config.outrage_gain

        # Sigmoid Saturation Model: simulates algorithm caps and user fatigue.
        # Prevents extreme outliers from generating infinite viral weight.
        max_multiplier = self.config.max_viral_multiplier
        midpoint = self.config.saturation_midpoint

        # Calculate the boost based on the combined viral energy
        outrage_boost = 1.0 + max_multiplier * torch.sigmoid(
            outrage_gain * (viral_energy - midpoint)
        )

        viral_weights = weights * outrage_boost
        viral_weights = viral_weights / viral_weights.sum()

        viral_center = (emotion_tensor * viral_weights.unsqueeze(1)).sum(dim=0)

        # ============================================================
        # Elite Emotional Center
        # ============================================================

        elite_percentile = self.config.elite_percentile
        threshold = torch.quantile(weights, elite_percentile)

        elite_mask = weights >= threshold
        if elite_mask.sum() > 0:
            elite_weights = weights * elite_mask
            elite_weights = elite_weights / elite_weights.sum()
            elite_center = (emotion_tensor * elite_weights.unsqueeze(1)).sum(dim=0)
        else:
            elite_center = center_of_gravity.clone()

        # ============================================================
        # Polarization Metrics
        # ============================================================

        # A. Dispersion-based polarization (Global dispersion)
        global_distances = torch.norm(emotion_tensor - center_of_gravity, dim=1)
        polarization = (global_distances * weights).sum().item()

        # B. Emotional entropy (fragmentation)
        # Normalize center_of_gravity to act as a probability distribution for entropy calculation
        cg_prob = torch.clamp(center_of_gravity, min=1e-9)
        cg_prob = cg_prob / cg_prob.sum()
        entropy = -(cg_prob * torch.log(cg_prob)).sum().item()

        # C. Bimodality proxy (variance along dominant axis)
        dominant_axis = center_of_gravity
        # Normalize dominant axis
        dominant_axis_norm = dominant_axis / (torch.norm(dominant_axis) + 1e-9)
        projections = torch.matmul(emotion_tensor, dominant_axis_norm)
        bimodality = projections.std().item()

        # ============================================================
        # 1D Sentiment/Valence (My Idea)
        # ============================================================
        # Plutchik Valence mapping:
        # Positive: Joy (+1), Trust (+0.5), Anticipation (+0.5), Surprise (0)
        # Negative: Sadness (-1), Disgust (-0.5), Anger (-0.8), Fear (-0.8)
        # Assuming EMOTION_LABELS order: ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]

        # Calculate single valence score for the objective center (-1.0 to 1.0)
        valence_score = torch.dot(center_of_gravity, VALENCE_WEIGHTS).item()

        # ============================================================
        # Dominant Emotion (Viral State)
        # ============================================================

        max_val, dominant_idx = torch.max(viral_center, dim=0)

        # Lower threshold is significantly above random chance (1/8 = 0.125)
        # for an 8-emotion distribution.
        if max_val < self.config.dominant_emotion_threshold:
            dominant_label = "Neutral"
        else:
            dominant_label = EMOTION_LABELS[int(dominant_idx.item())]

        # ============================================================
        # Elite–Population Divergence
        # ============================================================

        elite_divergence = torch.norm(elite_center - center_of_gravity).item()

        # ============================================================
        # Endogenous Event Generation
        # ============================================================

        action_vector = None
        action_name = None

        elite_div_threshold = getattr(self.config, "elite_divergence_threshold", 0.4)
        pol_threshold = getattr(self.config, "polarization_threshold", 0.5)

        if elite_divergence > elite_div_threshold and polarization > pol_threshold:
            # Populist Uprising
            action_vector = [0.0] * 12
            action_vector[1] = -0.5  # Negative Safety
            action_vector[2] = -0.8  # Negative Stability
            action_vector[4] = -0.9  # Negative Fairness
            action_vector[7] = 0.5  # Positive Freedom
            action_name = "Populist Uprising"
        elif elite_divergence > elite_div_threshold:
            # Policy Shift (Elite push through changes)
            action_vector = [0.0] * 12
            action_vector[0] = 0.4  # Wealth impact
            action_vector[4] = -0.3  # Negative Fairness
            action_vector[6] = 0.6  # Positive Innovation
            action_name = "Elite Policy Shift"
        elif polarization > pol_threshold:
            # Protest
            action_vector = [0.0] * 12
            action_vector[2] = -0.6  # Negative Stability
            action_vector[4] = -0.5  # Negative Fairness
            action_vector[5] = 0.7  # Positive In-Group
            action_name = "Civil Protest"

        # ============================================================
        # Final State Object
        # ============================================================

        return {
            # Core states
            "objective_center": center_of_gravity.tolist(),
            "viral_center": viral_center.tolist(),
            "elite_center": elite_center.tolist(),
            # Summaries
            "dominant_emotion": dominant_label,
            "confidence": round(max_val.item(), 3),
            "sentiment_valence": round(valence_score, 3),
            # Virality metrics
            "mean_outrage_multiplier": round(outrage_boost.mean().item(), 3),
            "max_outrage_multiplier": round(outrage_boost.max().item(), 3),
            # Stability metrics
            "polarization": polarization,
            "entropy": entropy,
            "bimodality": bimodality,
            "elite_divergence": elite_divergence,
            "action_vector": action_vector,
            "action_name": action_name,
            "labels": EMOTION_LABELS,
        }


if __name__ == "__main__":
    # Unit Test
    from schema import SimConfig

    conf = SimConfig()
    engine = SocialPhysicsEngine(conf)

    # Fake Data: 10 agents, 8 emotions
    fake_emotions = torch.rand(10, 8)
    fake_influence = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100])  # One whale

    # Fake engagement scores (10 agents)
    fake_engagement = torch.rand(10)

    result = engine.aggregate_society(fake_emotions, fake_influence, fake_engagement)
    print("Dominant:", result["dominant_emotion"])
    print("Polarization:", result["polarization"])
    print("Valence:", result["sentiment_valence"])
