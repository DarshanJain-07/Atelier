import torch
import torch.nn.functional as F

from schema import CROSS_DIM_INTERACTIONS, PERSONALITY_QUERY_MATRIX

# Global cache for device-specific constants to avoid repeated CPU->GPU transfers
_DEVICE_CACHE = {}


def get_cached_constant(
    name: str, tensor: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    key = (name, device)
    if key not in _DEVICE_CACHE:
        _DEVICE_CACHE[key] = tensor.to(device)
    return _DEVICE_CACHE[key]


class AttentionContext:
    def __init__(self, exposures, personalities, world_tensor, is_personal, config):
        self.exposures = exposures
        self.personalities = personalities
        self.world_tensor = world_tensor
        self.is_personal = is_personal
        self.config = config

        # Intermediate states
        self.Q = None
        self.K = None
        self.relevance = None

    def rde_layer(self):
        exposures = self.exposures
        personalities = self.personalities
        config = self.config

        conscientiousness = personalities[:, 1:2]
        extraversion = personalities[:, 2:3]
        agreeableness = personalities[:, 3:4]
        neuroticism = personalities[:, 4:5]

        normalized_state = (exposures + 1.0) / 2.0
        gaps = 1.0 - normalized_state
        assets = normalized_state

        jealousy_bias = (
            neuroticism * config.jealousy_factor
            + (1.0 - agreeableness) * config.resentment_factor
        )

        protection_bias = (
            conscientiousness * config.protection_factor
            + extraversion * config.status_factor
        )

        rde_sensitivity = (gaps * jealousy_bias) + (assets * protection_bias)

        self.Q = exposures.clone() * torch.clamp(rde_sensitivity, 0.1, 3.0)
        return self

    def personality_query_layer(self):
        personalities = self.personalities
        Q_base = self.Q if self.Q is not None else self.exposures.clone()

        openness = personalities[:, 0:1]
        conscientiousness = personalities[:, 1:2]
        extraversion = personalities[:, 2:3]
        agreeableness = personalities[:, 3:4]
        neuroticism = personalities[:, 4:5]

        # Calculate activations
        O_act = torch.sigmoid(openness * 3.0 - 1.0)
        C_act = F.relu(conscientiousness - 0.3) * 1.5
        E_act = torch.sigmoid(extraversion * 2.0)
        A_act = torch.tanh(agreeableness * 2.0)
        N_act = torch.tanh(neuroticism * 2.5)

        # Shape (N, 5)
        activations = torch.cat([O_act, C_act, E_act, A_act, N_act], dim=1)

        matrix = get_cached_constant(
            "PERSONALITY_QUERY_MATRIX", PERSONALITY_QUERY_MATRIX, Q_base.device,
        )
        personality_mod = torch.matmul(activations, matrix)

        self.Q = torch.tanh(Q_base + personality_mod)
        return self

    def logic_consistency_layer(self):
        """The 'Skepticism' Cognitive Gate.
        If there is a massive discrepancy between Short_Term and Long_Term impacts,
        agents with high Openness/Conscientiousness will penalize the Short_Term signal
        and focus on the Long_Term reality (detecting 'too good to be true' flaws).
        """
        if self.Q is None:
            return self

        # Indices 10 (Short_Term) and 11 (Long_Term)
        st = self.world_tensor[:, 10:11]
        lt = self.world_tensor[:, 11:12]

        # Calculate Logic Gap (discrepancy)
        # If one is very positive and the other very negative, gap is high.
        logic_gap = torch.abs(st - lt)

        # Skepticism is driven by Openness (Intellectual) and Conscientiousness (Analytical)
        openness = self.personalities[:, 0:1]
        conscientiousness = self.personalities[:, 1:2]

        # We blend them for a 'Skepticism' trait
        skepticism_trait = (openness + conscientiousness) / 2.0

        # Sensitivity to the gap
        sensitivity = skepticism_trait * getattr(self.config, "skepticism_gain", 2.0)

        # Penalty applies if the gap is significant
        gap_threshold = getattr(self.config, "logic_gap_threshold", 0.5)

        # Probability of detecting the flaw (Sigmoid response)
        detection_prob = torch.sigmoid(sensitivity * (logic_gap - gap_threshold))

        # Apply suppression to Short_Term attention (Index 10)
        # Higher detection_prob -> lower attention on Short_Term
        new_Q = self.Q.clone()
        new_Q[:, 10:11] = new_Q[:, 10:11] * (1.0 - (detection_prob * 0.85))

        # Boost attention on Long_Term (Index 11) to focus on the 'hidden' reality
        new_Q[:, 11:12] = new_Q[:, 11:12] * (1.0 + (detection_prob * 0.6))

        self.Q = new_Q
        return self

    def personal_event_layer(self):
        if not self.is_personal:
            return self
        if self.Q is None:
            raise ValueError("personal_event_layer called before Q is initialized")

        Q = self.Q
        device = Q.device
        personalities = self.personalities

        # Empathy determines how much an agent cares about others' personal events
        agreeableness = personalities[:, 3:4]

        # Simulate proximity/relevance to the personal event.
        # A highly skewed distribution (pow 10) ensures only a small localized cluster
        # pays high attention to a random personal event.
        proximity = torch.pow(torch.rand(Q.shape[0], 1, device=device), 10.0)

        # Maximum boost scalar
        max_boost = torch.exp(torch.tensor(1.5, device=device))

        # For personal events, baseline interest should be severely suppressed unless proximity is high.
        # Most of the population will ignore the event.
        base_multiplier = 0.05

        # Agent-specific multiplier: low for strangers, high for close ones
        agent_multiplier = base_multiplier + max_boost * proximity * (
            0.5 + 0.5 * agreeableness
        )

        # Scale all dimensions of attention based on relevance/proximity
        Q = Q * agent_multiplier
        self.Q = torch.clamp(Q, -2.5, 2.5)

        return self

    def selective_exposure_layer(self):
        """The 'Triple-Filter-Bubble' Cognitive Gate.
        If the event fundamentally contradicts the agent's core values, and they have low Openness,
        they will strongly suppress it (Confirmation Bias).
        """
        if getattr(self.config, "use_selective_exposure", True) is False:
            return self

        if self.Q is None:
            raise ValueError("selective_exposure_layer called before Q is initialized")

        # Normalize the incoming event (World Tensor) and the agent's worldview (Exposures)
        # Note: self.world_tensor is (N, 12) because of signal distortion
        event_norm = self.world_tensor / (torch.norm(self.world_tensor, dim=1, keepdim=True) + 1e-8)

        # We use a copy of the original exposures to represent their deep-seated worldview
        # self.exposures is (N, 12)
        agent_norms = self.exposures / (torch.norm(self.exposures, dim=1, keepdim=True) + 1e-8)

        # Cosine Similarity between agent worldview and the event per agent
        # Element-wise multiplication followed by sum along dimension 1 -> Shape: (N, 1)
        alignment = (agent_norms * event_norm).sum(dim=1, keepdim=True)

        # Agents with low Openness are more likely to filter out opposing views
        openness = self.personalities[:, 0:1]

        base_tolerance = getattr(
            self.config, "selective_exposure_base_tolerance", -0.3,
        )
        openness_factor = getattr(
            self.config, "selective_exposure_openness_factor", 0.4,
        )
        gain = getattr(self.config, "selective_exposure_gain", 8.0)
        max_suppression = torch.clamp(
            torch.tensor(
                getattr(self.config, "selective_exposure_max_suppression", 0.85),
                dtype=self.Q.dtype,
                device=self.Q.device,
            ),
            0.0,
            1.0,
        )

        # High-openness agents should tolerate even strongly misaligned events,
        # while low-openness agents still heavily suppress contradictory signals.
        # Shift the midpoint slightly left so mid-openness agents are not treated
        # almost like hard blockers in strongly misaligned scenarios.
        openness_midpoint = 0.45
        openness_curve = torch.sigmoid(gain * (openness - openness_midpoint))
        tolerance = base_tolerance - (1.0 + openness_factor) * openness_curve

        # Convert the gate into bounded suppression rather than an unconditional hard block.
        # This preserves a meaningful openness gradient for the figure and the live model.
        suppression_pressure = tolerance - alignment
        suppression = max_suppression * torch.sigmoid(gain * suppression_pressure)

        # --- Realism Fix: Self-Interest Resilience ---
        # If the event directly addresses a 'gap' in the agent's exposures 
        # (e.g., they lack Safety and the signal provides Safety), it should 
        # bypass the confirmation bias filter. Self-interest > Worldview.
        
        # normalized_state is (exposures + 1.0) / 2.0 (0 to 1)
        # gaps = 1.0 - normalized_state
        normalized_state = (self.exposures + 1.0) / 2.0
        gaps = 1.0 - normalized_state
        
        # Benefit: High Gap * High Positive Signal in that dimension
        # world_tensor is (-1.5 to 1.5)
        positive_signal = torch.clamp(self.world_tensor, min=0.0)
        individual_benefit = (gaps * positive_signal).sum(dim=1, keepdim=True)
        
        benefit_threshold = getattr(self.config, "self_interest_resilience_threshold", 0.15)
        # If benefit is high, suppression pressure is reduced
        resilience = torch.sigmoid(10.0 * (individual_benefit - benefit_threshold))
        
        # Final suppression: misaligned events are suppressed, UNLESS they are highly beneficial
        self.Q = self.Q * (1.0 - (suppression * (1.0 - resilience)))

        return self

    def hybrid_attention_layer(self):
        """Inspired by Gemma's hybrid local/global attention.
        Blends the individual agent's attention query (local) with
        the society's mean attention query (global) to simulate
        a broader context awareness (attending to the global zeitgeist).
        """
        if not getattr(self.config, "use_hybrid_attention", False):
            return self

        if self.Q is None:
            raise ValueError("hybrid_attention_layer called before Q is initialized")

        global_weight = getattr(self.config, "hybrid_attention_global_weight", 0.2)

        # Local attention: The agent's current specific Query state (self.Q)
        # Global attention: The society's mean Query state
        Q_global = self.Q.mean(dim=0, keepdim=True)

        # Blend local and global
        self.Q = (1.0 - global_weight) * self.Q + global_weight * Q_global

        return self

    def key_processing_layer(self):
        personalities = self.personalities
        neuroticism = personalities[:, 4:5]

        K = self.world_tensor.clone()

        K_positive = F.relu(K)
        K_negative = F.relu(-K)

        threat_sensitivity = 1.0 + neuroticism * self.config.threat_sensitivity_gain
        K_processed = K_positive - (K_negative * threat_sensitivity)

        self.K = torch.tanh(K_processed * self.config.k_processing_tanh_gain)
        return self

    def cross_dimension_layer(self):
        if self.Q is None:
            raise ValueError("cross_dimension_layer called before Q is initialized")

        Q = self.Q

        influence = Q * self.config.cross_dim_interaction_strength
        matrix = get_cached_constant(
            "CROSS_DIM_INTERACTIONS", CROSS_DIM_INTERACTIONS, Q.device,
        )
        Q_cross = torch.matmul(influence, matrix)

        self.Q = torch.tanh(Q + Q_cross)
        return self

    def relevance_layer(self):
        if self.Q is None:
            raise ValueError("relevance_layer called before Q is initialized")
        if self.K is None:
            raise ValueError("relevance_layer called before K is initialized")

        # --- FIX: Avoid cancellation (e.g. 0.8 Importance - 0.8 Alignment = 0 Attention) ---
        # We ensure absolute importance (magnitude) is the primary driver of attention.
        importance = torch.abs(self.Q) * torch.abs(self.K)
        alignment = self.Q * self.K

        w_imp = getattr(self.config, "relevance_importance_weight", 0.8)
        w_base = getattr(self.config, "relevance_base_weight", 0.2)

        # Even with negative alignment, importance prevents the relevance from hitting zero.
        relevance = (w_imp * importance) + (w_base * alignment)

        is_threat = (self.K < 0).float()
        threat_amplifier = 1.0 + (is_threat * getattr(self.config, "threat_amplifier_gain", 1.5))
        
        # --- Realism Fix: Opportunity Amplification ---
        # If a signal is beneficial (filling a gap), it should also get 
        # an attention boost. Rational agents pay attention to things 
        # that help them survive/prosper.
        normalized_state = (self.exposures + 1.0) / 2.0
        gaps = 1.0 - normalized_state
        positive_signal = torch.clamp(self.world_tensor, min=0.0)
        # benefit_mag = (gaps * positive_signal).sum(dim=1, keepdim=True)
        # We look at per-dimension benefit
        benefit_mask = (gaps * positive_signal) > getattr(self.config, "opportunity_threshold", 0.2)
        
        opportunity_gain = getattr(self.config, "opportunity_amplifier_gain", 1.5)
        opportunity_amplifier = 1.0 + (benefit_mask.float() * opportunity_gain)

        self.relevance = relevance * threat_amplifier * opportunity_amplifier
        return self

    def temperature_layer(self):
        if self.relevance is None:
            raise ValueError("temperature_layer called before relevance_layer")

        personalities = self.personalities
        conscientiousness = personalities[:, 1:2]
        neuroticism = personalities[:, 4:5]

        base_temperature = 1.0
        temp_modulation = (
            base_temperature
            + (conscientiousness - 0.5) * self.config.temp_conscientiousness_weight
            - (neuroticism - 0.5) * self.config.temp_neuroticism_weight
        )
        temp_modulation = torch.clamp(temp_modulation, 0.5, 2.0)

        self.relevance = self.relevance / temp_modulation
        return self

    def threshold_layer(self):
        if self.relevance is None:
            raise ValueError("threshold_layer called before relevance_layer")

        personalities = self.personalities
        extraversion = personalities[:, 2:3]

        threshold = (
            self.config.threshold_base
            - (extraversion - 0.5) * self.config.threshold_extraversion_weight
        )
        threshold = torch.clamp(threshold, 0.0, 0.3)

        mask = (torch.abs(self.relevance) > threshold).float()
        self.relevance = self.relevance * mask
        return self

    def engagement_layer(self):
        if self.relevance is None:
            raise ValueError("engagement_layer called before relevance_layer")

        energy = torch.norm(self.relevance, dim=1, keepdim=True)

        engagement = torch.sigmoid(
            self.config.engagement_gain * (energy - self.config.engagement_threshold),
        )

        self.relevance = self.relevance * engagement
        return self
