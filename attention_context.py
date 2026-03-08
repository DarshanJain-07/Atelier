import torch
import torch.nn.functional as F
from schema import PERSONALITY_QUERY_MATRIX, CROSS_DIM_INTERACTIONS

# Global cache for device-specific constants to avoid repeated CPU->GPU transfers
_DEVICE_CACHE = {}

def get_cached_constant(name: str, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
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

        matrix = get_cached_constant("PERSONALITY_QUERY_MATRIX", PERSONALITY_QUERY_MATRIX, Q_base.device)
        personality_mod = torch.matmul(activations, matrix)

        self.Q = torch.tanh(Q_base + personality_mod)
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
        agent_multiplier = base_multiplier + max_boost * proximity * (0.5 + 0.5 * agreeableness)
        
        # Scale all dimensions of attention based on relevance/proximity
        Q = Q * agent_multiplier
        self.Q = torch.clamp(Q, -2.5, 2.5)

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
        matrix = get_cached_constant("CROSS_DIM_INTERACTIONS", CROSS_DIM_INTERACTIONS, Q.device)
        Q_cross = torch.matmul(influence, matrix)

        self.Q = torch.tanh(Q + Q_cross)
        return self

    def relevance_layer(self):
        if self.Q is None:
            raise ValueError("relevance_layer called before Q is initialized")
        if self.K is None:
            raise ValueError("relevance_layer called before K is initialized")

        Q = self.Q
        K = self.K

        importance = torch.abs(Q) * torch.abs(K)
        base_relevance = Q * K

        is_threat = (K < 0).float()
        threat_amplifier = 1.0 + (is_threat * self.config.threat_amplifier_gain)

        self.relevance = (self.config.relevance_importance_weight * importance + self.config.relevance_base_weight * base_relevance) * threat_amplifier
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

        threshold = self.config.threshold_base - (extraversion - 0.5) * self.config.threshold_extraversion_weight
        threshold = torch.clamp(threshold, 0.0, 0.3)

        mask = (torch.abs(self.relevance) > threshold).float()
        self.relevance = self.relevance * mask
        return self

    def engagement_layer(self):
        if self.relevance is None:
            raise ValueError("engagement_layer called before relevance_layer")

        energy = torch.norm(self.relevance, dim=1, keepdim=True)

        engagement = torch.sigmoid(
            self.config.engagement_gain * (energy - self.config.engagement_threshold)
        )

        self.relevance = self.relevance * engagement
        return self
