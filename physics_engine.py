import numpy as np
import torch

from schema import EMOTION_LABELS, emotions_to_valence


class SocialPhysicsEngine:
    """Physics Layer
    -----------------
    Nonlinear socio-emotional field model with:
    - Objective state
    - Viral state (nonlinear outrage contagion)
    - Elite state
    - Entropy-based fragmentation
    - Polarization via dispersion and bimodality
    - Active Engagement Weighting (Skin in the Game)
    - Time-Series Stewing (Longitudinal Cascades)
    """

    def __init__(self, config):
        self.config = config
        # Track consecutive high-arousal steps for each agent (if refractory period is enabled)
        self._refractory_counters = None

    @staticmethod
    def _neighbor_average(adjacency_matrix: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(
            adjacency_matrix.coalesce().to(values.device),
            values,
        )

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
        personalities=None,
        is_personal=False,
    ):
        """Calculates the socio-emotional metrics for a single event over time (stewing).

        Args:
            emotion_tensor: (N, 8) tensor of emotional states.
            influence_scores: (N,) structural influence weights.
            engagement_scores: (N,) optional tensor from Cognitive Engine. Used to determine
                               active engagement (Skin in the Game).
            adjacency_matrix: (N, N) sparse tensor representing network topology.

        """
        N = emotion_tensor.shape[0]
        device = emotion_tensor.device

        # Initialize or reset refractory counters
        if getattr(self.config, "use_refractory_period", False):
            if self._refractory_counters is None or self._refractory_counters.shape[0] != N:
                self._refractory_counters = torch.zeros(N, device=device)
        else:
            self._refractory_counters = None

        # ----------------------------
        # Influence Handling
        # ----------------------------
        if isinstance(influence_scores, (np.ndarray, list)):
            structural_weights = torch.tensor(influence_scores, dtype=torch.float32, device=device)
        elif isinstance(influence_scores, torch.Tensor):
            structural_weights = influence_scores.float().to(device)
        elif hasattr(influence_scores, "to_numpy"):
            structural_weights = torch.tensor(
                influence_scores.to_numpy(), dtype=torch.float32, device=device
            )
        else:
            structural_weights = torch.ones(N, dtype=torch.float32, device=device)

        # Influence saturation (prevents oligarch domination)
        structural_weights = torch.log1p(structural_weights)

        # ----------------------------
        # Active Engagement (Skin in the Game)
        # ----------------------------
        current_engagement = engagement_scores if engagement_scores is not None else torch.ones(N, device=device)
        
        # Apply Refractory Period Penalty
        if self._refractory_counters is not None:
            # An agent is in refractory state if they've exceeded threshold for duration
            duration_threshold = getattr(self.config, "refractory_threshold_duration", 5)
            penalty = getattr(self.config, "refractory_engagement_drop", 0.95)
            
            in_refractory = self._refractory_counters >= duration_threshold
            # engagement_multiplier is 1.0 normally, (1.0 - penalty) if in refractory
            engagement_multiplier = torch.ones_like(current_engagement)
            engagement_multiplier[in_refractory] = (1.0 - penalty)
            
            current_engagement = current_engagement * engagement_multiplier

        if engagement_scores is not None or self._refractory_counters is not None:
            # Agents who aren't paying attention shouldn't dominate the emotional center
            energy = current_engagement
            # Normalize energy so we don't completely crush baseline influence
            energy = energy / (energy.mean() + 1e-9)

            # Final objective weight is Structural Influence * Active Engagement
            weights = structural_weights * energy
        else:
            weights = structural_weights

        weights = torch.clamp(weights, min=1e-8)
        weights = weights / weights.sum()

        num_ticks = getattr(self.config, "stewing_ticks", 5)
        current_emotions = emotion_tensor.clone()
        negative_integral = 0.0
        topology = None
        if adjacency_matrix is not None:
            topology = adjacency_matrix.coalesce().to(device)

        for tick in range(num_ticks):
            # ============================================================
            # Objective Center of Gravity (Global)
            # ============================================================
            center_of_gravity = (current_emotions * weights.unsqueeze(1)).sum(dim=0)

            # ============================================================
            # Topological Context (Local Echo Chambers)
            # ============================================================
            if adjacency_matrix is not None:
                # Each agent's emotional baseline is now the weighted average of their connections
                # adjacency_matrix is row-normalized, so mm acts as a weighted mean
                local_centers = self._neighbor_average(topology, current_emotions)
            else:
                # If no topology, the local context is just the global context for everyone
                local_centers = center_of_gravity.unsqueeze(0).expand(N, -1)

            # ============================================================
            # Nonlinear Outrage Contagion (Viral State)
            # ============================================================
            arousal = torch.norm(current_emotions, dim=1)
            
            # Update Refractory Counters based on current arousal
            if self._refractory_counters is not None:
                arousal_threshold = getattr(self.config, "refractory_arousal_threshold", 0.8)
                over_threshold = arousal >= arousal_threshold
                self._refractory_counters[over_threshold] += 1
                self._refractory_counters[~over_threshold] = 0

            # Use raw engagement for viral energy to allow absolute fatigue levels to suppress contagion
            if engagement_scores is not None or self._refractory_counters is not None:
                viral_energy = arousal * current_engagement
            else:
                viral_energy = arousal

            norm_emotion = current_emotions / (arousal.unsqueeze(1) + 1e-9)

            local_arousal = torch.norm(local_centers, dim=1)
            norm_local = local_centers / (local_arousal.unsqueeze(1) + 1e-9)

            alignment = (norm_emotion * norm_local).sum(dim=1)
            validation_multiplier = 1.0 + alignment
            viral_energy = viral_energy * validation_multiplier

            outrage_gain = self.config.outrage_gain
            max_multiplier = self.config.max_viral_multiplier
            midpoint = self.config.saturation_midpoint

            outrage_boost = 1.0 + max_multiplier * torch.sigmoid(
                outrage_gain * (viral_energy - midpoint),
            )

            viral_weights = weights * outrage_boost
            viral_weights = viral_weights / viral_weights.sum()

            viral_center = (current_emotions * viral_weights.unsqueeze(1)).sum(dim=0)

            # ============================================================
            # Time-Series Stewing (Track area under curve for negative emotion)
            # ============================================================
            valence_score = emotions_to_valence(center_of_gravity).item()
            if valence_score < 0:
                negative_integral += abs(valence_score)

            if tick < num_ticks - 1:
                # Get base influence parameters
                self_retention = getattr(self.config, "stewing_self_retention", 0.6)
                local_influence = getattr(self.config, "stewing_local_influence", 0.3)
                viral_influence = getattr(self.config, "stewing_viral_influence", 0.1)

                # --- Realism Fix: Saliency-Dominant Stewing ---
                # Instead of a simple weighted average (which allows the majority to swamp
                # the minority), we use a non-linear blend that prioritizes high-arousal
                # negative emotions (contagion).
                
                # 1. Calculate Component Energy
                self_energy = torch.norm(current_emotions, dim=1, keepdim=True)
                local_energy = torch.norm(local_centers, dim=1, keepdim=True)
                # viral_center is (8,), so we expand it
                viral_energy = torch.norm(viral_center).expand(N, 1)
                
                # 2. Identify Saliency (Negative Outrage)
                # Emotions: [Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation]
                # Outrage: Fear (2), Disgust (5), Anger (6)
                outrage_indices = [2, 5, 6]
                self_outrage = current_emotions[:, outrage_indices].sum(dim=1, keepdim=True)
                local_outrage = local_centers[:, outrage_indices].sum(dim=1, keepdim=True)
                viral_outrage = viral_center[outrage_indices].sum().expand(N, 1)
                
                # 3. Dynamic Weighting (Saliency Boost)
                # If a component has high outrage, we boost its influence
                outrage_boost_gain = getattr(self.config, "stewing_outrage_boost", 2.0)
                
                w_self = self_retention * (1.0 + self_outrage * outrage_boost_gain)
                w_local = local_influence * (1.0 + local_outrage * outrage_boost_gain)
                w_viral = viral_influence * (1.0 + viral_outrage * outrage_boost_gain)
                
                # Normalize weights
                w_total = w_self + w_local + w_viral + 1e-9
                w_self /= w_total
                w_local /= w_total
                w_viral /= w_total
                
                # --- Blend State ---
                new_emotions = (
                    w_self * current_emotions
                    + w_local * local_centers
                    + w_viral * viral_center.unsqueeze(0).expand(N, -1)
                )

                # Restore arousal to prevent energy loss during averaging
                # --- Realism Fix: Energy Contagion ---
                # Old logic: new_emotions = new_emotions * (target_arousal / new_arousal)
                # This prevented energy from spreading. We now allow high-energy 
                # components (local/viral) to boost the agent's arousal.
                
                # Component arousals
                arousal_self = torch.norm(current_emotions, dim=1, keepdim=True)
                arousal_local = torch.norm(local_centers, dim=1, keepdim=True)
                arousal_viral = torch.norm(viral_center).expand(N, 1)
                
                # The 'Infection' arousal: how much energy is in the environment?
                environment_arousal = torch.maximum(arousal_local, arousal_viral)
                
                # If environment is more aroused than me, I catch some of that energy
                energy_contagion_gain = getattr(self.config, "stewing_energy_contagion", 0.3)
                updated_target_arousal = torch.maximum(
                    arousal_self, 
                    arousal_self + (environment_arousal - arousal_self) * energy_contagion_gain
                )
                
                new_arousal = torch.norm(new_emotions, dim=1, keepdim=True) + 1e-9
                new_emotions = new_emotions * (updated_target_arousal / new_arousal)

                current_emotions = new_emotions

        # ============================================================
        # Final State Calculation (End of Stewing)
        # ============================================================

        # Re-calculate center of gravity one last time
        center_of_gravity = (current_emotions * weights.unsqueeze(1)).sum(dim=0)

        # ============================================================
        # Elite Emotional Center
        # ============================================================
        elite_percentile = self.config.elite_percentile
        k_elites = max(1, int(N * (1.0 - elite_percentile)))

        # Get indices of top K weights
        _, top_indices = torch.topk(weights, k=k_elites)

        elite_mask = torch.zeros(N, dtype=torch.bool, device=weights.device)
        elite_mask[top_indices] = True

        elite_weights = weights * elite_mask
        elite_weights = elite_weights / (elite_weights.sum() + 1e-9)
        elite_center = (current_emotions * elite_weights.unsqueeze(1)).sum(dim=0)

        # ============================================================
        # Polarization Metrics (Bimodality Coefficient & Dispersion)
        # ============================================================
        # Instead of Silhouette Score (which fails on consensus/single clusters),
        # we use Sarle's Bimodality Coefficient (BC) along the dominant emotional axis.
        # BC > 0.555 indicates a bimodal (polarized) distribution.

        global_distances = torch.norm(current_emotions - center_of_gravity, dim=1)
        dispersion = (global_distances * weights).sum().item()

        dominant_axis = center_of_gravity
        dominant_axis_norm = dominant_axis / (torch.norm(dominant_axis) + 1e-9)
        projections = torch.matmul(current_emotions, dominant_axis_norm)

        # Calculate Bimodality Coefficient using PyTorch
        mean_proj = projections.mean()
        std_proj = projections.std(unbiased=False)

        if std_proj > 1e-6:
            # Add a small epsilon to denominator and handle small N bias
            n = projections.numel()
            skew = torch.mean(((projections - mean_proj) / std_proj) ** 3)
            kurtosis = torch.mean(((projections - mean_proj) / std_proj) ** 4)

            # Sarle's Bimodality Coefficient: (skew^2 + 1) / kurtosis
            # For a normal distribution, BC = 0.333. For uniform, BC = 0.555.
            # We add a correction factor for small N to avoid over-estimation
            if n > 3:
                # Standard BC
                bimodality_coeff = (skew**2 + 1) / (kurtosis + 1e-6)
            else:
                # Insufficient data for bimodality
                bimodality_coeff = torch.tensor(0.0)
        else:
            bimodality_coeff = torch.tensor(0.0)

        bimodality = torch.clamp(bimodality_coeff, 0.0, 1.0).item()

        # We define structural polarization as the Bimodality Coefficient.
        polarization = bimodality

        cg_prob = torch.clamp(center_of_gravity, min=1e-9)
        cg_prob = cg_prob / cg_prob.sum()
        entropy = -(cg_prob * torch.log(cg_prob)).sum().item()

        # 1D Sentiment/Valence
        valence_score = emotions_to_valence(center_of_gravity).item()

        # Dominant Emotion (Viral State)
        max_val, dominant_idx = torch.max(viral_center, dim=0)
        dominant_label = EMOTION_LABELS[int(dominant_idx.item())] if max_val >= self.config.dominant_emotion_threshold else "Neutral"

        # ============================================================
        # Elite–Population Divergence
        # ============================================================
        elite_divergence = torch.norm(elite_center - center_of_gravity).item()

        # ============================================================
        # Endogenous Event Generation (Action Potential)
        # ============================================================
        action_vector = None
        action_name = None

        # Re-calculate final local centers for validation
        if adjacency_matrix is not None:
            local_centers = self._neighbor_average(topology, current_emotions)
        else:
            local_centers = center_of_gravity.unsqueeze(0).expand(N, -1)

        # --- Stage 2: 2-Stage Action Potential (Asymmetric Contagion) ---
        final_arousal = torch.norm(current_emotions, dim=1)
        norm_emotion = current_emotions / (final_arousal.unsqueeze(1) + 1e-9)
        local_arousal = torch.norm(local_centers, dim=1)
        norm_local = local_centers / (local_arousal.unsqueeze(1) + 1e-9)
        
        # Alignment is how much I agree with my neighbors
        alignment = (norm_emotion * norm_local).sum(dim=1)
        
        # --- Realism Fix: Asymmetric Contagion ---
        # In the old model, social_validation = 1.0 + alignment.
        # This meant negative alignment (conflict) reduced motivation.
        # In the real world, if neighbors are angry/fearful, it increases your stress/arousal
        # regardless of whether you 'agree' with their specific reason.
        
        # We calculate 'Stress Arousal': The contagion of negative energy
        # Emotions: [Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation]
        # Indices 2 (Fear), 5 (Disgust), 6 (Anger) are high-contagion negative emotions
        negative_neighbor_energy = local_centers[:, [2, 5, 6]].sum(dim=1)
        
        # If neighbors are angry/fearful, they provide a 'contagion floor'
        # social_validation should not drop below 1.0 if there is high negative energy nearby.
        contagion_floor = torch.clamp(negative_neighbor_energy * 2.0, min=0.0, max=0.5)
        
        # Dynamic social validation:
        # If I agree (alignment > 0), I am validated (+alignment).
        # If I disagree (alignment < 0), I am stressed/aroused by their energy (+contagion_floor).
        # We take the max of the two to ensure 'conflict' doesn't suppress action.
        social_validation = 1.0 + torch.maximum(alignment, contagion_floor)
        
        base_cost = getattr(self.config, "base_action_cost", 0.5)
        if personalities is not None:
            extraversion = personalities[:, 2].to(current_emotions.device)
            neuroticism = personalities[:, 4].to(current_emotions.device)
            inf = torch.tensor(influence_scores, device=current_emotions.device) if not isinstance(influence_scores, torch.Tensor) else influence_scores.to(current_emotions.device)
            action_cost = base_cost - 0.1 * extraversion - 0.1 * neuroticism - 0.05 * torch.log1p(inf)
        else:
            action_cost = torch.full((N,), base_cost, device=current_emotions.device)

        action_cost = torch.clamp(action_cost, min=0.05)

        # --- Stage 1: Individual Motivation (Internal Willingness) ---
        # Motivation is raw energy minus the cost to act
        individual_motivation = (final_arousal * social_validation) - action_cost

        # --- Stage 2: Critical Mass / Local Thresholds ---
        # Threshold: Emotion must be dominant AND motivation must be positive.
        max_vals, _ = torch.max(current_emotions, dim=1)
        is_motivated = individual_motivation > 0.1
        is_emotional = max_vals >= self.config.dominant_emotion_threshold

        # Filter by engaged population (crucial for is_personal events)
        if is_personal:
            if engagement_scores is not None:
                engaged_mask = engagement_scores > (engagement_scores.max() * 0.5)
            else:
                engaged_mask = torch.rand(N, device=current_emotions.device) > 0.95
        elif engagement_scores is not None:
            # --- Realism Fix: Social Engagement Contagion ---
            # Old logic: engaged_mask = engagement_scores > (engagement_scores.mean() * 0.1)
            # This prevented unengaged people from catching the 'outrage'.
            # New logic: you are engaged if the initial signal engaged you OR
            # if the final social environment is highly aroused (contagion).
            
            initial_engaged_mask = engagement_scores > (engagement_scores.mean() * 0.1)
            
            # Local/Viral arousal (contagion) can re-activate passive agents
            social_arousal = torch.maximum(local_arousal, torch.norm(viral_center))
            social_engagement_threshold = getattr(self.config, "social_engagement_trigger", 0.6)
            contagion_engaged_mask = social_arousal >= social_engagement_threshold
            
            engaged_mask = initial_engaged_mask | contagion_engaged_mask
        else:
            engaged_mask = torch.ones(N, dtype=torch.bool, device=current_emotions.device)

        # Initial set of acting agents: Motivated, Emotional, and Engaged
        acting_agents = (is_motivated & is_emotional & engaged_mask).float()

        if adjacency_matrix is not None and getattr(self.config, "use_granovetter_thresholds", True):
            # Personal thresholds for "following the crowd"
            t_mean = getattr(self.config, "granovetter_threshold_mean", 0.25)
            t_std = getattr(self.config, "granovetter_threshold_std", 0.15)

            if personalities is not None:
                consc = personalities[:, 1].to(current_emotions.device)
                agree = personalities[:, 3].to(current_emotions.device)
                # Conscientiousness and Agreeableness increase the threshold (harder to flip)
                personal_thresholds = t_mean + (consc + agree - 1.0) * t_std
            else:
                personal_thresholds = torch.full((N,), t_mean, device=current_emotions.device)

            personal_thresholds = torch.clamp(personal_thresholds, min=0.01, max=0.9)

            # Iterative activation (Snowball effect)
            for _ in range(3):
                # Calculate fraction of acting neighbors
                neighbor_acting_ratio = self._neighbor_average(
                    topology, acting_agents.unsqueeze(1),
                ).squeeze(1)

                # An agent acts if:
                # 1. They were already acting (Persistence)
                # 2. They are emotional, engaged AND (their motivation is marginal OR neighbors cross threshold)
                marginal_motivation = individual_motivation > -0.1
                social_trigger = neighbor_acting_ratio > personal_thresholds

                new_acting = (is_emotional & engaged_mask & (is_motivated | (marginal_motivation & social_trigger)))
                acting_agents = new_acting.float()

        # Final acting count and ratio
        acting_count = acting_agents.sum().item()
        total_eligible = engaged_mask.sum().item()
        population_size = max(1, N)
        acting_ratio = acting_count / population_size

        elite_div_threshold = getattr(self.config, "elite_divergence_threshold", 0.4)
        pol_threshold = getattr(self.config, "polarization_threshold", 0.5)
        act_threshold = getattr(self.config, "action_threshold", 0.15)

        # Trigger logic refinement
        if acting_ratio > act_threshold:
            # Populist Uprising: High Divergence + High Polarization + Negative Valence
            if elite_divergence > elite_div_threshold and polarization > pol_threshold and valence_score < -0.2:
                action_vector = [0.0] * 12
                action_vector[1] = -0.5  # Negative Safety
                action_vector[2] = -0.8  # Negative Stability
                action_vector[4] = -0.9  # Negative Fairness
                action_vector[7] = 0.5   # Positive Freedom
                action_name = "Populist Uprising"

            # Elite Policy Shift: High Divergence + Elite-specific arousal
            elif elite_divergence > elite_div_threshold and torch.norm(elite_center) > 0.3:
                action_vector = [0.0] * 12
                action_vector[0] = 0.4   # Wealth impact
                action_vector[4] = -0.3  # Negative Fairness
                action_vector[6] = 0.6   # Positive Innovation
                action_name = "Elite Policy Shift"

            # Civil Protest: High Polarization + High Anger/Disgust + Negative Valence
            elif polarization > pol_threshold and (dominant_label in ["Anger", "Disgust", "Sadness"]) and valence_score < -0.1:
                action_vector = [0.0] * 12
                action_vector[2] = -0.6  # Negative Stability
                action_vector[4] = -0.5  # Negative Fairness
                action_vector[5] = 0.7   # Positive In-Group
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
            # Stewing Integrals
            "negative_integral": round(negative_integral, 3),
            # Virality metrics
            "mean_outrage_multiplier": round(outrage_boost.mean().item(), 3),
            "max_outrage_multiplier": round(outrage_boost.max().item(), 3),
            # Stability metrics
            "polarization": polarization,
            "dispersion": dispersion,
            "entropy": entropy,
            "bimodality": bimodality,
            "elite_divergence": elite_divergence,
            "action_vector": action_vector,
            "action_name": action_name,
            "acting_ratio": acting_ratio,
            "acting_count": int(acting_count),
            "total_eligible": total_eligible,
            "population_size": population_size,
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
    print("Negative Integral:", result["negative_integral"])
