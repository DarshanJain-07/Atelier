import torch
import torch.nn.functional as F


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

        personality_mod = torch.zeros_like(Q_base)

        openness_boost = torch.sigmoid(openness * 3.0 - 1.0)
        personality_mod[:, 6] += openness_boost.squeeze() * 0.8
        personality_mod[:, 7] += openness_boost.squeeze() * 0.5
        personality_mod[:, 11] += openness_boost.squeeze() * 0.6

        neurotic_concern = torch.tanh(neuroticism * 2.5)
        personality_mod[:, 1] += neurotic_concern.squeeze() * 1.2
        personality_mod[:, 2] += neurotic_concern.squeeze() * 0.9

        consc_mod = F.relu(conscientiousness - 0.3) * 1.5
        personality_mod[:, 0] += consc_mod.squeeze() * 0.6
        personality_mod[:, 3] += consc_mod.squeeze() * 0.7

        extra_mod = torch.sigmoid(extraversion * 2.0)
        personality_mod[:, 5] += extra_mod.squeeze() * 0.5
        personality_mod[:, 9] += extra_mod.squeeze() * 0.4

        agree_mod = torch.tanh(agreeableness * 2.0)
        personality_mod[:, 4] += agree_mod.squeeze() * 0.8
        personality_mod[:, 9] += agree_mod.squeeze() * 0.6
        personality_mod[:, 8] += agree_mod.squeeze() * 0.5

        self.Q = torch.tanh(Q_base + personality_mod)
        return self

    def personal_event_layer(self):
        if not self.is_personal:
            return self
        if self.Q is None:
            raise ValueError("cross_dimension_layer called before Q is initialized")

        Q = self.Q
        personal_mask = torch.zeros_like(Q)
        personal_mask[:, 0] = 1.0
        personal_mask[:, 1] = 1.0
        personal_mask[:, 3] = 1.0

        personal_boost = 1.0 + torch.exp(torch.tensor(1.5))
        Q = Q * (1.0 + personal_mask * (personal_boost - 1.0))
        self.Q = torch.clamp(Q, -2.5, 2.5)

        return self

    def key_processing_layer(self):
        personalities = self.personalities
        neuroticism = personalities[:, 4:5]

        K = self.world_tensor.clone()

        K_positive = F.relu(K)
        K_negative = F.relu(-K)

        threat_sensitivity = 1.0 + neuroticism * 1.5
        K_processed = K_positive - (K_negative * threat_sensitivity)

        self.K = torch.tanh(K_processed * 1.5)
        return self

    def cross_dimension_layer(self):
        if self.Q is None:
            raise ValueError("cross_dimension_layer called before Q is initialized")

        Q = self.Q
        interaction_strength = 0.3

        safety_influence = Q[:, 1:2] * interaction_strength
        Q[:, 2] += safety_influence.squeeze() * 0.4
        Q[:, 9] += safety_influence.squeeze() * 0.3

        wealth_influence = Q[:, 0:1] * interaction_strength
        Q[:, 3] += wealth_influence.squeeze() * 0.3
        Q[:, 7] += wealth_influence.squeeze() * 0.2

        innovation_influence = Q[:, 6:7] * interaction_strength
        Q[:, 11] += innovation_influence.squeeze() * 0.3
        Q[:, 10] += innovation_influence.squeeze() * 0.2

        self.Q = torch.tanh(Q)
        return self

    def relevance_layer(self):
        if self.Q is None:
            raise ValueError("cross_dimension_layer called before Q is initialized")
        if self.K is None:
            raise ValueError("cross_dimension_layer called before K is initialized")

        Q = self.Q
        K = self.K

        importance = torch.abs(Q) * torch.abs(K)
        base_relevance = Q * K

        is_threat = (K < 0).float()
        threat_amplifier = 1.0 + (is_threat * 1.5)

        self.relevance = (0.7 * importance + 0.3 * base_relevance) * threat_amplifier
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
            + (conscientiousness - 0.5) * 0.8
            - (neuroticism - 0.5) * 0.6
        )
        temp_modulation = torch.clamp(temp_modulation, 0.5, 2.0)

        self.relevance = self.relevance / temp_modulation
        return self

    def threshold_layer(self):
        if self.relevance is None:
            raise ValueError("threshold_layer called before relevance_layer")

        personalities = self.personalities
        extraversion = personalities[:, 2:3]

        threshold_base = 0.1
        threshold = threshold_base - (extraversion - 0.5) * 0.15
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
