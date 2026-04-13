from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from attention_context import AttentionContext
from schema import PSYCH_PROJECTION, SimConfig


@dataclass
class BacklashDecision:
    world_tensor: torch.Tensor
    chosen_frame: str
    triggered: bool
    sample_size: int
    skeptical_count: int
    official_count: int
    skeptical_energy: float
    official_energy: float
    backlash_potential: float
    sample_indices: Optional[torch.Tensor] = None
    sample_context: Optional[torch.Tensor] = None

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "chosen_frame": self.chosen_frame,
            "triggered": self.triggered,
            "sample_size": self.sample_size,
            "skeptical_count": self.skeptical_count,
            "official_count": self.official_count,
            "skeptical_energy": self.skeptical_energy,
            "official_energy": self.official_energy,
            "backlash_potential": self.backlash_potential,
        }


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

    def apply_social_consensus(
        self,
        distorted_world: torch.Tensor,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            adjacency_matrix is None
            or getattr(self.config, "perception_social_consensus_gain", 0.0) <= 0
        ):
            return distorted_world

        social_gain = self.config.perception_social_consensus_gain
        topology = adjacency_matrix.coalesce().to(distorted_world.device)
        local_consensus = torch.sparse.mm(topology, distorted_world)
        return (1.0 - social_gain) * distorted_world + social_gain * local_consensus

    def perceive_world(
        self,
        world_tensor_raw: torch.Tensor,
        personalities: torch.Tensor,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        distorted_world = self.distort_signal(
            world_tensor_raw.squeeze().to(personalities.device),
            personalities,
        )
        return self.apply_social_consensus(distorted_world, adjacency_matrix)

    def project_emotions(self, context_vector: torch.Tensor) -> torch.Tensor:
        projection_matrix = PSYCH_PROJECTION.to(context_vector.device)
        logits = torch.matmul(context_vector, projection_matrix)
        return F.softmax(
            logits / max(0.01, self.config.emotion_temperature),
            dim=1,
        )

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
            .logic_consistency_layer()
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

    def prepare_perceived_world(
        self,
        world_tensor_raw: torch.Tensor,
        personalities: torch.Tensor,
        agent_affinities: torch.Tensor,
        agent_memory: Optional[torch.Tensor] = None,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = personalities.device

        distorted_world = self.perceive_world(
            world_tensor_raw,
            personalities,
            adjacency_matrix=adjacency_matrix,
        )

        if agent_memory is not None and getattr(self.config, "use_agent_memory", False):
            mem = agent_memory.to(device)
            alignment = mem * distorted_world
            fatigue_mask = torch.clamp(alignment, min=0.0)
            fatigue_penalty = 1.0 - torch.exp(
                -fatigue_mask
                * getattr(self.config, "memory_desensitization_gain", 2.0)
            )

            threat_mask = (distorted_world < 0).float()
            total_stress = torch.sum(torch.abs(mem), dim=1, keepdim=True)
            fresh_threat_mask = threat_mask * (fatigue_mask < 0.1).float()
            trigger_stack_boost = (
                torch.log1p(total_stress)
                * fresh_threat_mask
                * getattr(self.config, "memory_trigger_stacking_gain", 3.0)
            )

            perceived_world = distorted_world * (
                1.0 - torch.clamp(fatigue_penalty, max=0.95)
            ) + (distorted_world * trigger_stack_boost)
        else:
            perceived_world = distorted_world

        return perceived_world * agent_affinities.to(device)

    def build_context_vector(
        self,
        perceived_world: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        residual_gain = getattr(self.config, "attention_residual_gain", 0.35)
        modulated_gain = getattr(self.config, "attention_modulated_gain", 1.0)
        context_scale = residual_gain + (attention_weights * modulated_gain)

        context_vector = perceived_world * context_scale

        original_norm = torch.norm(perceived_world, dim=1, keepdim=True)
        new_norm = torch.norm(context_vector, dim=1, keepdim=True) + 1e-9
        context_vector = context_vector * (original_norm / new_norm)

        return torch.clamp(context_vector, -2.0, 2.0)

    def simulate_frame_response(
        self,
        world_tensor_raw: torch.Tensor,
        urgency: float,
        is_personal: bool,
        exposures: torch.Tensor,
        personalities: torch.Tensor,
        agent_affinities: torch.Tensor,
        agent_memory: Optional[torch.Tensor] = None,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        perceived_world = self.prepare_perceived_world(
            world_tensor_raw,
            personalities,
            agent_affinities,
            agent_memory=agent_memory,
            adjacency_matrix=adjacency_matrix,
        )
        attention_weights, engagement_scores = self.calculate_attention(
            exposures,
            personalities,
            perceived_world,
            is_personal,
        )
        attention_weights = self.apply_stress_bias(
            attention_weights,
            personalities,
            urgency,
        )
        context_vector = self.build_context_vector(perceived_world, attention_weights)
        return context_vector, attention_weights, engagement_scores

    def compute_skepticism_scores(self, personalities: torch.Tensor) -> torch.Tensor:
        weights = getattr(self.config, "backlash_trait_routing_weights", {})
        weighted_sum = (
            personalities[:, 0] * float(weights.get("Openness", 0.0))
            + personalities[:, 1] * float(weights.get("Conscientiousness", 0.0))
            + personalities[:, 2] * float(weights.get("Extraversion", 0.0))
            + personalities[:, 3] * float(weights.get("Agreeableness", 0.0))
            + personalities[:, 4] * float(weights.get("Neuroticism", 0.0))
            + float(weights.get("bias", 0.0))
        )
        norm = (
            abs(float(weights.get("Openness", 0.0)))
            + abs(float(weights.get("Conscientiousness", 0.0)))
            + abs(float(weights.get("Extraversion", 0.0)))
            + abs(float(weights.get("Agreeableness", 0.0)))
            + abs(float(weights.get("Neuroticism", 0.0)))
        )
        norm = max(norm, 1.0)
        return torch.sigmoid((weighted_sum / norm) * 4.0)

    def run_backlash_ab_test(
        self,
        world_tensor_off: torch.Tensor,
        world_tensor_skp: Optional[torch.Tensor],
        backlash_potential: float,
        urgency: float,
        is_personal: bool,
        exposures: torch.Tensor,
        personalities: torch.Tensor,
        agent_affinities: torch.Tensor,
        agent_memory: Optional[torch.Tensor] = None,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> BacklashDecision:
        default_decision = BacklashDecision(
            world_tensor=world_tensor_off,
            chosen_frame="official",
            triggered=False,
            sample_size=0,
            skeptical_count=0,
            official_count=0,
            skeptical_energy=0.0,
            official_energy=0.0,
            backlash_potential=float(backlash_potential),
        )
        if (
            world_tensor_skp is None
            or not getattr(self.config, "use_backlash_ab_testing", False)
        ):
            return default_decision

        total_agents = personalities.shape[0]
        device = personalities.device
        sample_size = int(total_agents * self.config.backlash_sample_size)
        if sample_size < 1:
            return default_decision

        sample_indices = torch.randperm(total_agents, device=device)[:sample_size]
        sample_personalities = personalities.index_select(0, sample_indices)
        skepticism_scores = self.compute_skepticism_scores(sample_personalities)
        skeptical_mask = (
            skepticism_scores >= self.config.backlash_skepticism_threshold
        )

        skeptical_local = torch.nonzero(skeptical_mask, as_tuple=False).flatten()
        official_local = torch.nonzero(~skeptical_mask, as_tuple=False).flatten()
        sample_context = torch.zeros(
            sample_size,
            world_tensor_off.shape[1],
            dtype=exposures.dtype,
            device=device,
        )

        skeptical_energy = torch.tensor(0.0, device=device)
        official_energy = torch.tensor(0.0, device=device)

        if skeptical_local.numel() > 0:
            skeptical_indices = sample_indices.index_select(0, skeptical_local)
            skeptical_context, _, skeptical_raw_energy = self.simulate_frame_response(
                world_tensor_skp,
                urgency=urgency,
                is_personal=is_personal,
                exposures=exposures.index_select(0, skeptical_indices),
                personalities=personalities.index_select(0, skeptical_indices),
                agent_affinities=agent_affinities.index_select(0, skeptical_indices),
                agent_memory=(
                    agent_memory.index_select(0, skeptical_indices)
                    if agent_memory is not None
                    else None
                ),
                adjacency_matrix=None,
            )
            sample_context.index_copy_(0, skeptical_local, skeptical_context)
            skeptical_energy = skeptical_raw_energy.mean() * float(backlash_potential)

        if official_local.numel() > 0:
            official_indices = sample_indices.index_select(0, official_local)
            official_context, _, official_raw_energy = self.simulate_frame_response(
                world_tensor_off,
                urgency=urgency,
                is_personal=is_personal,
                exposures=exposures.index_select(0, official_indices),
                personalities=personalities.index_select(0, official_indices),
                agent_affinities=agent_affinities.index_select(0, official_indices),
                agent_memory=(
                    agent_memory.index_select(0, official_indices)
                    if agent_memory is not None
                    else None
                ),
                adjacency_matrix=None,
            )
            sample_context.index_copy_(0, official_local, official_context)
            official_energy = official_raw_energy.mean()

        triggered = bool(
            skeptical_energy
            > official_energy * self.config.backlash_decision_threshold
        )
        chosen_frame = "skeptical" if triggered else "official"
        chosen_world = world_tensor_skp if triggered else world_tensor_off

        return BacklashDecision(
            world_tensor=chosen_world,
            chosen_frame=chosen_frame,
            triggered=triggered,
            sample_size=sample_size,
            skeptical_count=int(skeptical_local.numel()),
            official_count=int(official_local.numel()),
            skeptical_energy=float(skeptical_energy.item()),
            official_energy=float(official_energy.item()),
            backlash_potential=float(backlash_potential),
            sample_indices=sample_indices,
            sample_context=sample_context,
        )

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

        context_vector, attention_weights, engagement_scores = (
            self.simulate_frame_response(
                world_tensor_raw,
                urgency=urgency,
                is_personal=is_personal,
                exposures=exposures,
                personalities=personalities,
                agent_affinities=agent_affinities,
                agent_memory=agent_memory,
                adjacency_matrix=adjacency_matrix,
            )
        )

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
