# Testing Guide

This repository includes a large scenario-driven pytest suite under `research_paper_tests/`. The suite is used both as engineering regression coverage and as a research-validation harness for the behaviors described in the project documentation and paper materials.

## Test Runner

The canonical entrypoint is:

```bash
./run_all_tests.sh
```

The script:

- sets `PYTHONPATH=.`
- selects the best available local pytest executable
- runs `research_paper_tests` as a suite
- optionally sets evolution-mode environment variables when requested

### Evolution Matrix Modes

```bash
./run_all_tests.sh --evolution with
./run_all_tests.sh --evolution without
./run_all_tests.sh --evolution both
```

These modes drive:

- `RESEARCH_TEST_EVOLUTION_MODE`
- `RESEARCH_TEST_EVOLUTION_MATRIX`

The matrix-specific smoke test module is `test_evolution_matrix.py`.

## Direct Pytest Usage

Examples:

```bash
python3 -m pytest research_paper_tests/test_run_profile_contract.py -q
python3 -m pytest research_paper_tests/test_runtime_regressions.py -q
python3 -m pytest research_paper_tests/test_trait_sweeps.py -q
```

## Test Harness Structure

| Path | Role |
| --- | --- |
| `research_paper_tests/config_schema.py` | Central scenario definitions, live-default adapters, helper builders, and reusable constants. |
| `research_paper_tests/_metrics.py` | Graph/statistics helpers used by multiple tests. |
| `research_paper_tests/conftest.py` | Ensures the repo root is importable during pytest collection. |
| `research_paper_tests/generated/` | Output folder for figure-generating tests. |
| `run_all_tests.sh` | Preferred suite runner. |

### Scenario Philosophy

Most tests do not hard-code arbitrary tensors inline. Instead they:

1. look up a named scenario in `config_schema.py`
2. derive a live `SimConfig` from current runtime defaults
3. build a world, a synthetic society, or both
4. assert directional or structural behavior

This keeps the suite aligned with the app as defaults evolve.

## Test Catalog

The catalog below documents every current `research_paper_tests/test_*.py` file and the test functions it exposes.

### Contracts And Suite Integrity

| File | Test function(s) | What it verifies | Artifacts |
| --- | --- | --- | --- |
| `test_collection_contract.py` | `test_research_paper_files_expose_pytest_tests` | Every `test_*.py` file actually exports at least one pytest test. | None |
| `test_run_profile_contract.py` | `test_run_profile_defaults_follow_sim_config_defaults`; `test_run_profile_accepts_sim_config_aliases` | `RunProfile` stays aligned with `SimConfig`, including alias handling for public request fields. | None |
| `test_evolution_mode_override.py` | `test_session_evolution_override_updates_default_sim_config`; `test_session_evolution_override_updates_default_run_profile`; `test_session_evolution_override_respects_locked_baseline_pairs` | Session-level evolution overrides behave predictably without breaking explicitly paired baseline/evolved scenarios. | None |
| `test_evolution_matrix.py` | `test_generated_society_cases_support_requested_evolution_modes` | Scenario generation works across requested evolution modes and keeps topology/memory shapes coherent. | None |

### Society Generation, Structure, And Topology

| File | Test function(s) | What it verifies | Artifacts |
| --- | --- | --- | --- |
| `test_trait_distribution.py` | `test_generated_trait_distributions_are_well_formed` | Personalities stay bounded, wealth exposure has spread, and influence remains positive. | None |
| `test_personality_correlations.py` | `test_generated_personalities_follow_target_correlations` | Generated Big Five correlations track the target correlation matrix. | None |
| `test_personalities_for_clustering.py` | `test_personality_distribution_keeps_high_and_low_neuroticism_tails` | The personality sampler preserves meaningful tail mass for clustering work. | None |
| `test_clusters.py` | `test_personality_clusters_show_nontrivial_trait_spread` | K-means personality clusters are not degenerate and separate on trait axes. | None |
| `test_network_topology.py` | `test_network_topology_is_normalized_and_homophilous` | Sparse adjacency rows are normalized and connected agents are sufficiently similar. | None |
| `test_network_clustering.py` | `test_triadic_closure_increases_average_clustering` | Triadic closure increases average graph clustering over the raw backbone. | None |
| `test_echo_chambers.py` | `test_homophilous_topology_forms_stronger_echo_chambers` | Higher homophily produces stronger similarity across connected pairs. | None |
| `test_louvain_modularity.py` | `test_homophily_raises_louvain_modularity` | Higher homophily also increases community modularity under Louvain partitioning. | None |
| `test_personality_socialization.py` | `test_personality_socialization_reduces_neighbor_friction` | Post-topology socialization makes connected agents more alike in trait space. | None |
| `test_cascade_power_law.py` | `test_power_law_influence_creates_heavier_tail_than_lognormal` | Power-law influence settings create a heavier-tailed influence distribution than the flatter alternative. | None |
| `test_ideological_influence_gini.py` | `test_power_law_influence_increases_influence_inequality` | Influence inequality rises when power-law influence is enabled. | None |
| `test_influence_susceptibility_ratio.py` | `test_structural_influence_improves_realized_reach` | Agents with more structural influence convert their position into higher realized reach. | None |

### Evolution, Inequality, And Class Structure

| File | Test function(s) | What it verifies | Artifacts |
| --- | --- | --- | --- |
| `test_wealth_gini.py` | `test_evolution_increases_wealth_inequality` | Evolved societies differ measurably from baseline societies on wealth inequality. | None |
| `test_population_segmentation.py` | `test_same_event_produces_distinct_subgroup_response_profiles`; `test_generate_population_segmentation_figure` | The same event produces materially different engagement/action profiles across classes. | `generated/population_segmentation.png` |

### Cognitive Processing And Perception

| File | Test function(s) | What it verifies | Artifacts |
| --- | --- | --- | --- |
| `test_signal_distortion.py` | `test_signal_distortion_scales_with_neuroticism` | More neurotic agents distort threatening signals more strongly. | None |
| `test_perception_social_consensus.py` | `test_social_consensus_aligns_neighbor_perceptions` | Local-consensus blending reduces distance between neighbors’ perceived worlds. | None |
| `test_divergence.py` | `test_emotional_divergence_tracks_neuroticism` | Neuroticism increases fear-heavy downstream emotional response. | None |
| `test_cognitive_gate.py` | `test_cognitive_gate_blocks_misaligned_low_openness_agents`; `test_cognitive_gate_retains_engagement_for_high_openness_gradient` | Selective exposure suppresses low-openness agents more strongly, while high openness restores engagement. | Included in summary figure coverage |
| `test_truth_refinement.py` | `test_truth_refinement_prioritizes_long_term_for_skeptical_agents` | Skeptical agents reweight attention from short-term framing toward long-term implications. | None |
| `test_relative_deprivation.py` | `test_relative_deprivation_hits_marginalized_agents_harder` | The same event produces stronger anger in structurally marginalized agents than in elites. | Included in summary figure coverage |
| `test_personal.py` | `test_personal_events_stay_more_localized_than_general_events` | Personal events stay more localized than identical non-personal events. | None |
| `test_trait_sweeps.py` | `test_trait_sweeps_reveal_monotonic_behavioral_gradients`; `test_generate_trait_sweeps_figure` | Trait sweeps produce directional gradients in engagement, attention, and action cost. | `generated/trait_sweeps.png` |

### Memory, Amplification, And Social Contagion

| File | Test function(s) | What it verifies | Artifacts |
| --- | --- | --- | --- |
| `test_memory_rehearsal.py` | `test_memory_rehearsal_slows_decay` | Social rehearsal slows memory decay relative to isolated retention. | Included in summary figure coverage |
| `test_agent_memory.py` | `test_agent_memory_accumulates_and_stacks_new_threats` | Repeated threats build memory and amplify later related engagement. | Included in summary figure coverage |
| `test_algorithmic_filter_bubble.py` | `test_algorithmic_filter_bubble_mutates_feed_and_boosts_engagement` | The algorithmic amplification pass mutates the feed and increases engagement. | Included in summary figure coverage |
| `test_viral_scaling.py` | `test_viral_scaling_has_sigmoid_regime_and_cap`; `test_generate_viral_scaling_figure` | Viral amplification follows a nonlinear growth regime and respects its cap. | `generated/viral_scaling.png` |
| `test_maximum_virality.py` | `test_virality_multiplier_stays_bounded_by_config` | Maximum virality remains bounded by config-defined limits. | None |
| `test_r0_basic_reproduction.py` | `test_r0_estimate_finds_nonzero_secondary_engagement` | Seeding thought contagion produces non-zero secondary engagement. | None |
| `test_granovetter_cascade.py` | `test_granovetter_thresholds_increase_collective_action` | Threshold-style cascade logic increases or preserves collective action relative to baseline aggregation. | Included in summary figure coverage |

### Emotion Mapping, Validation, And Outcome Boundaries

| File | Test function(s) | What it verifies | Artifacts |
| --- | --- | --- | --- |
| `test_semantic_alignment.py` | `test_semantic_alignment_rewards_matching_sentiment_baselines` | Prosperity and threat worlds map to matching sentiment profiles better than mismatched baselines. | Included in summary figure coverage |
| `test_accuracy_metrics.py` | `test_accuracy_metrics_prefer_matching_baseline` | Validation metrics reward a matching baseline over an intentionally mismatched one. | None |
| `test_response_boundaries.py` | `test_event_magnitude_monotonically_increases_engagement_and_action`; `test_low_salience_worlds_keep_reaction_bounded_before_escalation`; `test_generate_response_boundaries_figure` | Reaction strength scales with event magnitude while low-salience worlds remain bounded before escalation. | `generated/response_boundaries.png` |
| `test_emotion_direction_and_bridge_diffusion.py` | `test_world_direction_changes_which_emotion_dominates`; `test_bridge_agents_expand_cross_cluster_diffusion`; `test_generate_emotion_direction_and_bridge_diffusion_figure` | Different world directions produce different dominant emotions, and bridge agents spread emotion across otherwise separate communities. | `generated/emotion_direction_and_bridge_diffusion.png` |
| `test_endogenous_events.py` | `test_endogenous_events_fire_only_for_unstable_societies` | Endogenous follow-on events are emitted only under unstable/highly polarized states. | None |
| `test_bimodality_polarization.py` | `test_bimodality_coefficient_detects_polarized_distribution` | Bimodality coefficient distinguishes polarized distributions from normal ones. | Included in summary figure coverage |

### Runtime, Cache, And Resource Regressions

| File | Test function(s) | What it verifies | Artifacts |
| --- | --- | --- | --- |
| `test_runtime_regressions.py` | `test_prepare_society_cache_isolates_topology_variants`; `test_prepare_society_cache_isolates_evolution_variants`; `test_prepare_society_returns_fresh_request_memory`; `test_generated_topology_handles_small_populations`; `test_acting_ratio_uses_scoped_population_size` | Cache entries do not bleed across incompatible variants, request memory is fresh, topology works for small populations, and action ratios use the correct scoped denominator. | None |
| `test_ram_usage.py` | `test_full_debug_pipeline_stays_within_reasonable_memory` | The in-memory debug pipeline stays below a defined peak memory budget. | None |

### Research Figure Suites

| File | Test function(s) | What it verifies | Artifacts |
| --- | --- | --- | --- |
| `test_research_paper_figures.py` | `test_generate_research_paper_summary_figure`; `test_generate_research_paper_advanced_visualizations`; `test_generate_research_paper_multiseed_debug_figure` | Composite figure generation covering multiple mechanisms, plus advanced and multi-seed visualization outputs used for research presentation. | `generated/research_paper_summary.png`; `generated/research_paper_advanced_visualizations.png`; `generated/research_paper_multiseed_debug.png` |

## Figure-Generating Tests

The suite currently includes explicit figure-generation tests for:

| Test file | Output |
| --- | --- |
| `test_response_boundaries.py` | `generated/response_boundaries.png` |
| `test_trait_sweeps.py` | `generated/trait_sweeps.png` |
| `test_viral_scaling.py` | `generated/viral_scaling.png` |
| `test_emotion_direction_and_bridge_diffusion.py` | `generated/emotion_direction_and_bridge_diffusion.png` |
| `test_population_segmentation.py` | `generated/population_segmentation.png` |
| `test_research_paper_figures.py` | `generated/research_paper_summary.png`, `generated/research_paper_advanced_visualizations.png`, `generated/research_paper_multiseed_debug.png` |

## Practical Notes

### Tests That Touch External Models

- `input_layer.py` depends on the Gemini API for full end-to-end HTTP simulation requests.
- `validation.py` loads a Hugging Face sentiment model on demand.

The research tests mostly use in-memory debug helpers and scenario fixtures, which reduces dependence on external services, but full application runs may still require credentials and model downloads.

### Why `config_schema.py` Matters

`research_paper_tests/config_schema.py` is effectively the suite’s scenario registry. It:

- reads live runtime defaults instead of duplicating them
- exposes helper builders like `build_world`
- provides reusable index maps for traits, emotions, and sentiment
- supports evolution-mode overrides for the matrix runs

That file is the best place to extend the suite when adding a new behavior-oriented test.

### Keeping The Suite Healthy

When adding a new test file:

1. prefer scenario-driven setup over hard-coded magic tensors
2. reuse `_metrics.py` helpers when possible
3. add figure output to the catalog if the test writes artifacts
4. make sure the file exports at least one `test_*` function, or `test_collection_contract.py` will fail
