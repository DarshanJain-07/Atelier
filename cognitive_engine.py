from typing import Optional

import torch
import torch.nn.functional as F

from attention_context import AttentionContext
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

        # Scale distortion magnitude by the signal strength to prevent
        # overwhelming small signals with large additive noise.
        signal_magnitude = torch.norm(event_signal, p=2)
        
        # Ensure alpha scales with signal strength (multiplicative noise effect)
        alpha = alpha * signal_magnitude

        distorted = event_signal.unsqueeze(0) + alpha * noise

        return torch.clamp(distorted, -1.5, 1.5)

    def calculate_attention(
        self,
        exposures: torch.Tensor,
        personalities: torch.Tensor,
        world_tensor: torch.Tensor,
        is_personal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs modular attention pipeline.
        Returns: (attention_weights, raw_energy)
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
            .selective_exposure_layer()
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

        # Final Softmax (multiplied by 5.0 to prevent uniform flat distributions)
        attention_weights = F.softmax(ctx.relevance * 5.0, dim=1)
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
            torch.tensor(
                self.config.stress_gain * (urgency - threshold),
                dtype=torch.float32,
                device=attention_weights.device,
            )
        )  # (N,1)

        biased = attention_weights.clone()

        # --------------------------------------------------
        # Neuroticism → amplify dominant interpretation
        # --------------------------------------------------
        dominant_val, dominant_idx = torch.max(biased, dim=1, keepdim=True)
        amplification = (
            1.0
            + stress_factor * neuroticism * self.config.stress_neurotic_amplification
        )

        biased.scatter_(1, dominant_idx, dominant_val * amplification)

        # --------------------------------------------------
        # Low Openness → reduce diversity
        # --------------------------------------------------
        diversity_scale = (
            1.0 - stress_factor * (1 - openness) * self.config.stress_openness_reduction
        )
        biased = biased * diversity_scale

        # --------------------------------------------------
        # Extraversion → increase emotional intensity
        # (global scaling of attention sharpness)
        # --------------------------------------------------
        intensity_boost = (
            1.0 + stress_factor * extraversion * self.config.stress_extraversion_boost
        )
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

    @torch.inference_mode()
    def run(
        self,
        world_tensor_raw: torch.Tensor,
        urgency: float,
        is_personal: bool,
        exposures: torch.Tensor,
        personalities: torch.Tensor,
        agent_affinities: torch.Tensor,
        agent_memory: Optional[torch.Tensor] = None,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ):
        """
        Full cognitive simulation pipeline.
        Returns:
            context_vector: (N, 12)
            attention_weights: (N, 12)
            engagement_scores: (N,)
            updated_memory: (N, 12) or None
        """

        device = exposures.device

        # ---------------------------------
        # 1. Stage 1: Individual Signal Distortion
        # ---------------------------------
        distorted_world = self.distort_signal(
            world_tensor_raw.squeeze().to(device),
            personalities,
        )  # (N,12)

        # ---------------------------------
        # 2. Stage 2: Socially Constructed Reality (Consensus)
        # ---------------------------------
        # Agents align their misinterpretations with their neighbors.
        # This simulates the "Telephone Game" reaching local consensus.
        if adjacency_matrix is not None and getattr(self.config, "perception_social_consensus_gain", 0.0) > 0:
            social_gain = self.config.perception_social_consensus_gain
            
            # Use the sparse adjacency matrix to calculate the local neighborhood mean distortion
            # adjacency_matrix is expected to be row-normalized (row_sum = 1.0)
            local_consensus = torch.sparse.mm(adjacency_matrix.to(device), distorted_world)
            
            # Blend: (1 - gain) * Individual + (gain) * Neighborhood
            distorted_world = (1.0 - social_gain) * distorted_world + social_gain * local_consensus

        # ---------------------------------
        # 3. Memory Layer (Desensitization & Trigger Stacking)
        # ---------------------------------
        if agent_memory is not None and getattr(self.config, "use_agent_memory", False):
            mem = agent_memory.to(device)

            # Desensitization (Fatigue): If current event aligns with recent memory, reduce impact
            # We use an exponential scaling so repeated hits rapidly approach 1.0 (total fatigue)
            alignment = mem * distorted_world
            fatigue_mask = torch.clamp(alignment, min=0.0)
            fatigue_penalty = 1.0 - torch.exp(
                -fatigue_mask * getattr(self.config, "memory_desensitization_gain", 2.0)
            )

            # Trigger Stacking: Total past stress makes them hyper-reactive to NEW threats
            threat_mask = (distorted_world < 0).float()
            # Calculate total past stress across ALL dimensions
            total_stress = torch.sum(torch.abs(mem), dim=1, keepdim=True)

            # Only stack triggers on dimensions that aren't currently fatiguing them
            fresh_threat_mask = threat_mask * (fatigue_mask < 0.1).float()

            # Logarithmic scaling so it doesn't break the math, but still gives a massive boost
            trigger_stack_boost = (
                torch.log1p(total_stress)
                * fresh_threat_mask
                * getattr(self.config, "memory_trigger_stacking_gain", 3.0)
            )

            # Apply memory modifications:
            # 1. Fatigue multiplies the raw signal down toward 0.
            # 2. Trigger stacking adds an amplification if it's a fresh threat.
            perceived_world = distorted_world * (
                1.0 - torch.clamp(fatigue_penalty, max=0.95)
            ) + (distorted_world * trigger_stack_boost)
        else:
            perceived_world = distorted_world

        # ---------------------------------
        # 3. Affinity Modulation
        # ---------------------------------
        # Raw volume is scaled by the agent's specific cognitive bandwidth and affinities
        perceived_world = perceived_world * agent_affinities.to(device)

        # ---------------------------------
        # 4. Attention Computation
        # ---------------------------------
        attention_weights, engagement_scores = self.calculate_attention(
            exposures,
            personalities,
            perceived_world,
            is_personal,
        )

        # ---------------------------------
        # 5. Stress Bias
        # ---------------------------------
        attention_weights = self.apply_stress_bias(
            attention_weights,
            personalities,
            urgency,
        )

        # ---------------------------------
        # 6. Context Construction
        # ---------------------------------
        context_vector = perceived_world * attention_weights
        context_vector = torch.clamp(context_vector, -2.0, 2.0)

        return context_vector, attention_weights, engagement_scores

    def consolidate_memory(
        self,
        agent_memory: torch.Tensor,
        context_vector: torch.Tensor,
        social_rehearsal_factor: float = 0.0,
    ) -> torch.Tensor:
        """
        Stage 2 Memory Consolidation: Social Rehearsal.
        The decay rate is reduced if the event is globally viral/rehearsed.
        """
        if not getattr(self.config, "use_agent_memory", False):
            return agent_memory

        device = agent_memory.device
        base_decay = getattr(self.config, "memory_decay_rate", 0.7)
        rehearsal_gain = getattr(self.config, "memory_social_rehearsal_gain", 0.4)
        
        # Effective decay: 
        # If rehearsal is high, the memory 'sticks' (decay -> 1.0)
        # If rehearsal is low, the memory fades (decay -> base_decay)
        # We blend the base decay with a higher persistence factor.
        effective_decay = base_decay + (1.0 - base_decay) * (social_rehearsal_factor * rehearsal_gain)
        effective_decay = min(0.99, effective_decay)
        
        # Consolidation Update
        updated_memory = (agent_memory.to(device) * effective_decay) + context_vector.to(device)
        return updated_memory
