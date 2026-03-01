import torch
import torch.nn.functional as F

from attention import AttentionContext
from schema import SimConfig


class CognitiveEngine:
    def __init__(self, config: SimConfig):
        self.config = config

    def distort_signal(
        self,
        event_signal: torch.Tensor,  # (12,)
        personality: torch.Tensor,  # (N,5)
    ) -> torch.Tensor:
        """
        Continuous stochastic signal distortion.
        Respects self.config.use_signal_distortion.
        """

        device = event_signal.device
        N = personality.shape[0]

        # Distortion disabled
        if not self.config.use_signal_distortion:
            return event_signal.unsqueeze(0).expand(N, -1)

        # --- Sample continuous distortion magnitude ---
        beta_dist = torch.distributions.Beta(
            self.config.distortion_beta_a, self.config.distortion_beta_b
        )

        alpha = beta_dist.sample((N,)).to(device)

        # --- Personality modulation (neurotic amplification) ---
        neuroticism = personality[:, 4]
        alpha = alpha * (1.0 + self.config.distortion_neurotic_gain * neuroticism)

        alpha = (
            torch.clamp(alpha, self.config.distortion_min_alpha, 1.0)
            * self.config.distortion_max_noise
        )

        alpha = alpha.unsqueeze(1)  # (N,1)

        # --- Apply Gaussian noise ---
        noise = torch.randn(N, 12, device=device)

        distorted = event_signal.unsqueeze(0) + alpha * noise

        return torch.clamp(distorted, -1.5, 1.5)

    def calculate_attention(
        self,
        exposures: torch.Tensor,
        personalities: torch.Tensor,
        world_tensor: torch.Tensor,
        is_personal: bool,
    ) -> torch.Tensor:
        """
        Runs modular attention pipeline.
        """

        ctx = AttentionContext(
            exposures=exposures,
            personalities=personalities,
            world_tensor=world_tensor,
            is_personal=is_personal,
            config=self.config,
        )

        # Ordered pipeline
        ctx = (
            ctx.rde_layer()
            .personality_query_layer()
            .personal_event_layer()
            .key_processing_layer()
            .cross_dimension_layer()
            .relevance_layer()
            .temperature_layer()
            .threshold_layer()
            .engagement_layer()
        )

        # Final safety check
        if ctx.relevance is None:
            raise ValueError("Attention pipeline failed: relevance was never computed")

        # Extract raw engagement energy before Softmax normalizes it
        raw_energy = torch.norm(ctx.relevance, dim=1)

        # Final Softmax
        attention_weights = F.softmax(ctx.relevance, dim=1)
        return attention_weights, raw_energy

    #
    def apply_stress_bias(self, attention_weights, personality, urgency):
        """
        Stress-Induced Cognitive Bias.
        Personality-dependent amplification and conformity under urgency.
        """

        threshold = self.config.stress_activation_threshold

        if not self.config.use_time_pressure or urgency < threshold:
            return attention_weights

        # Big Five
        openness = personality[:, 0:1]
        extraversion = personality[:, 2:3]
        agreeableness = personality[:, 3:4]
        neuroticism = personality[:, 4:5]

        # Smooth stress scaling
        stress_factor = torch.sigmoid(
            self.config.stress_gain * (urgency - threshold)
        )  # (N,1)

        biased = attention_weights.clone()

        # --------------------------------------------------
        # Neuroticism → amplify dominant interpretation
        # --------------------------------------------------
        dominant_val, dominant_idx = torch.max(biased, dim=1, keepdim=True)
        amplification = 1.0 + stress_factor * neuroticism * 1.5

        biased.scatter_(1, dominant_idx, dominant_val * amplification)

        # --------------------------------------------------
        # Low Openness → reduce diversity
        # --------------------------------------------------
        diversity_scale = 1.0 - stress_factor * (1 - openness) * 0.5
        biased = biased * diversity_scale

        # --------------------------------------------------
        # Extraversion → increase emotional intensity
        # (global scaling of attention sharpness)
        # --------------------------------------------------
        intensity_boost = 1.0 + stress_factor * extraversion * 0.7
        biased = biased * intensity_boost

        # --------------------------------------------------
        # Agreeableness → conformity toward population mean
        # --------------------------------------------------
        if self.config.use_social_conformity:
            population_mean = attention_weights.mean(dim=0, keepdim=True)
            conformity_strength = (
                stress_factor * agreeableness * self.config.conformity_gain
            )
            biased = biased + conformity_strength * population_mean

        # Renormalize
        biased = biased / (biased.sum(dim=1, keepdim=True) + 1e-9)

        return biased

    def run(
        self,
        world_tensor_raw: torch.Tensor,
        urgency: float,
        is_personal: bool,
        exposures: torch.Tensor,
        personalities: torch.Tensor,
        agent_affinities: torch.Tensor,
    ):
        """
        Full cognitive simulation pipeline.
        Returns:
            context_vector: (N, 12)
            attention_weights: (N, 12)
        """

        device = exposures.device

        # ---------------------------------
        # 1. Signal Distortion
        # ---------------------------------
        distorted_world = self.distort_signal(
            world_tensor_raw.to(device),
            personalities,
        )  # (N,12)

        # ---------------------------------
        # 2. Affinity Modulation
        # ---------------------------------
        # Raw volume is scaled by the agent's specific cognitive bandwidth and affinities
        perceived_world = distorted_world * agent_affinities.to(device)

        # ---------------------------------
        # 3. Attention Computation
        # ---------------------------------
        attention_weights, engagement_scores = self.calculate_attention(
            exposures,
            personalities,
            perceived_world,
            is_personal,
        )

        # ---------------------------------
        # 4. Stress Bias
        # ---------------------------------
        attention_weights = self.apply_stress_bias(
            attention_weights,
            personalities,
            urgency,
        )

        # ---------------------------------
        # 5. Context Construction
        # ---------------------------------
        context_vector = perceived_world * attention_weights
        context_vector = torch.clamp(context_vector, -2.0, 2.0)

        return context_vector, attention_weights, engagement_scores
