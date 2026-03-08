"""
The Root Cause
The `personal_event_layer` inside `attention_context.py` was previously adding a boost to the attention/relevance tensor (`Q`) only for the localized cluster of people (those with high `proximity`). However, for the rest of the general population (strangers with `proximity` close to 0), their baseline attention (`Q`) was left completely untouched from earlier layers (such as `personality_query_layer`).

Because their baseline attention was never suppressed for personal events, their natural engagement levels were still high enough to cross the activation threshold, causing the entire population to react just as they would to a non-personal event.

The Fix
I updated the `personal_event_layer` to fundamentally change how personal events are processed:
1. Severe Suppression for Strangers: By default, if an event `is_personal`, the baseline attention (`Q`) is now severely suppressed for the entire population (by scaling it down using a very low `base_multiplier` of `0.05`).
2. Targeted Amplification: Only the small, localized cluster (people with high `proximity` and empathy to the event) receives a large positive multiplier that overcomes this suppression.

This change ensures that a personal event is essentially ignored by the general population while still triggering strong reactions among the small number of people directly related to it.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from attention_context import AttentionContext
from schema import DIMENSIONS


class DummyConfig:
    def __init__(self):
        self.jealousy_factor = 1.0
        self.resentment_factor = 1.0
        self.protection_factor = 1.0
        self.status_factor = 1.0
        self.cross_dim_interaction_strength = 0.1
        self.threat_sensitivity_gain = 0.5
        self.k_processing_tanh_gain = 1.0
        self.relevance_importance_weight = 0.5
        self.relevance_base_weight = 0.5
        self.threat_amplifier_gain = 1.0
        self.temp_conscientiousness_weight = 0.1
        self.temp_neuroticism_weight = 0.1
        self.threshold_base = 0.1
        self.threshold_extraversion_weight = 0.05
        self.engagement_gain = 2.0
        self.engagement_threshold = 0.5


def test_personal_event():
    torch.manual_seed(42)
    n_population = 100
    exposures = torch.rand(n_population, len(DIMENSIONS))
    personalities = torch.rand(n_population, 5)
    world_tensor = torch.rand(1, len(DIMENSIONS))

    config = DummyConfig()

    # Non-personal event
    ctx_non_personal = AttentionContext(
        exposures, personalities, world_tensor, False, config
    )
    ctx_non_personal.rde_layer().personality_query_layer().personal_event_layer().key_processing_layer().cross_dimension_layer().relevance_layer().temperature_layer().threshold_layer().engagement_layer()
    reacting_non_personal = (
        (torch.norm(ctx_non_personal.relevance, dim=1) > 0).float().sum().item()
    )

    # Personal event
    ctx_personal = AttentionContext(
        exposures, personalities, world_tensor, True, config
    )
    ctx_personal.rde_layer().personality_query_layer().personal_event_layer().key_processing_layer().cross_dimension_layer().relevance_layer().temperature_layer().threshold_layer().engagement_layer()
    reacting_personal = (
        (torch.norm(ctx_personal.relevance, dim=1) > 0).float().sum().item()
    )

    print(
        f"Non-personal event: {reacting_non_personal} people reacting out of {n_population}"
    )
    print(f"Personal event: {reacting_personal} people reacting out of {n_population}")


if __name__ == "__main__":
    test_personal_event()
