# API Reference

This page documents the executable surface of the application in `main.py`: HTTP endpoints, request and response models, debug helpers, and operational behavior. It intentionally focuses on structure and semantics rather than duplicating raw schema defaults.

## Runtime Surface

`main.py` exposes two HTTP endpoints through FastAPI:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Lightweight liveness probe. Returns `{ "status": "ok" }`. |
| `POST` | `/simulate` | Main simulation endpoint used by the frontend and by external callers. |
| `GET` | `/docs` | Human-facing documentation browser backed by the repo markdown files. |
| `GET` | `/api/docs` | Swagger/OpenAPI UI for the HTTP API. |

The app also mounts the static frontend at `/`, serving files from `frontend/`.

## Documentation-Specific Routes

The in-app documentation browser uses a small docs content API:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/docs/pages` | Returns the available markdown-backed docs pages and the default slug. |
| `GET` | `/docs/{slug}` | Loads the docs browser UI for a specific page slug. |

## `/simulate` Request Model

The request body is modeled by `SimulationRequest`:

```json
{
  "news_text": "string",
  "runs": [
    {
      "seed": 42,
      "social_class": "All"
    }
  ]
}
```

### Top-Level Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `news_text` | `string` | Raw event, policy, or situation text to analyze. |
| `runs` | `RunProfile[]` | One or more simulation configurations to execute against the same input event. |

### `RunProfile`

`RunProfile` is the public request-facing configuration model. It mirrors the simulation config in `schema.py`, but presents a frontend/API-friendly shape and supports aliases for several core fields.

#### Core Execution Fields

| Field | Meaning |
| --- | --- |
| `seed` | Deterministic seed for society generation and runtime randomness. |
| `temperature` / `mutation_temperature` | Controls society mutation and diversity. |
| `agent_count` / `num_agents` | Requested population size for the run. |
| `social_class` | Optional class filter applied before simulation. |
| `emotion_temperature` | Controls emotional projection sharpness. |

#### Core Behavior Toggles

| Field | Meaning |
| --- | --- |
| `use_distortion` / `use_signal_distortion` | Enables signal distortion in perception. |
| `use_pressure` / `use_time_pressure` | Enables urgency-driven cognitive tunneling. |
| `use_power_law` / `use_power_law_influence` | Enables more unequal influence weighting. |
| `enable_evolution` | Enables long-horizon society evolution before the event run. |

#### Time-Series And Emotional State Fields

| Field | Meaning |
| --- | --- |
| `stewing_ticks` | Number of post-event social-steeping iterations. |
| `stewing_self_retention` | How much each agent retains its own prior state per tick. |
| `stewing_local_influence` | Weight of local-neighbor state during stewing. |
| `stewing_viral_influence` | Weight of viral/global state during stewing. |
| `sentiment_neutrality_acting_threshold` | Threshold used when mapping emotion to sentiment with behavior-awareness. |
| `sentiment_neutrality_activation` | Activation style for neutrality shaping. |
| `sentiment_neutrality_leaky_slope` | Leaky activation slope if that mode is used. |

#### Algorithmic Amplification Fields

| Field | Meaning |
| --- | --- |
| `use_algorithmic_amplification` | Enables the two-pass attention/engagement amplification flow. |
| `algo_sample_size` | Population fraction used for the first-pass probe. |
| `algo_exaggeration_factor` | Strength of the feed mutation applied to top attention dimensions. |

#### Memory Fields

| Field | Meaning |
| --- | --- |
| `use_agent_memory` | Enables persistent memory between runs on the same society object. |
| `memory_decay_rate` | Retention/decay strength for memory traces. |
| `memory_desensitization_gain` | Repetition-based suppression effect. |
| `memory_trigger_stacking_gain` | Threat-stacking sensitization effect. |
| `memory_social_rehearsal_gain` | Salience-dependent reinforcement effect. |

#### Network And Cascade Fields

| Field | Meaning |
| --- | --- |
| `use_network_topology` | Enables explicit sparse adjacency topology. |
| `homophily_strength` | Strength of similarity-based edge formation. |
| `influence_bias_exp` | Bias toward influential nodes in topology creation. |
| `triadic_closure_prob` | Probability of closing friend-of-friend triangles. |
| `triadic_closure_iterations` | Number of triadic closure refinement passes. |
| `triadic_closure_homophily_threshold` | Similarity requirement for closure. |
| `use_granovetter_thresholds` | Enables threshold-style collective action gating. |
| `granovetter_threshold_mean` | Mean threshold location across agents. |
| `granovetter_threshold_std` | Threshold spread across agents. |
| `personality_socialization_gain` | Post-topology trait drift toward network neighbors. |

#### Cognitive Fields

| Field | Meaning |
| --- | --- |
| `use_hybrid_attention` | Enables hybrid local/global attention blending. |
| `hybrid_attention_global_weight` | Blending ratio for the global societal mean attention. |
| `cross_dim_interaction_strength` | Coupling strength across world dimensions. |
| `threat_sensitivity_gain` | Amplifies negative-signal perception. |
| `k_processing_tanh_gain` | Controls nonlinearity in key processing. |
| `attention_residual_gain` | Baseline contribution in attention computation. |
| `attention_modulated_gain` | Strength of gated modulation in attention. |
| `relevance_importance_weight` | Importance-driven contribution to relevance. |
| `relevance_base_weight` | Base dot-product contribution to relevance. |
| `threat_amplifier_gain` | Final amplification of negative/threatening input. |
| `stress_neurotic_amplification` | Stress-driven neurotic amplification. |
| `stress_openness_reduction` | Stress-driven openness suppression. |
| `stress_extraversion_boost` | Stress-driven extraversion contribution. |

#### Physics And Distortion Fields

| Field | Meaning |
| --- | --- |
| `outrage_gain` | Controls virality/outage growth. |
| `max_viral_multiplier` | Cap for viral amplification. |
| `saturation_midpoint` | Midpoint of nonlinear virality growth. |
| `distortion_max_noise` | Maximum distortion amplitude. |
| `distortion_neurotic_gain` | Neurotic amplification of distortion. |
| `perception_social_consensus_gain` | Blend factor for socially constructed perception. |
| `affinity_min_strength` | Lower bound on affinity weights. |
| `normalize_affinities_by_mean` | Whether affinity normalization is mean-based. |

#### Evolution Fields

| Field | Meaning |
| --- | --- |
| `evolution_generations` | Number of generations in the evolution step. |
| `inheritance_fraction` | Wealth retention across generations. |
| `shock_frequency` | Frequency of macro shocks. |
| `shock_magnitude` | Magnitude of macro shocks. |

## `/simulate` Response Model

The endpoint returns:

```json
{
  "status": "success",
  "results": []
}
```

Each element in `results` corresponds to one requested run.

### Per-Run Result Fields

| Field | Meaning |
| --- | --- |
| `run_index` | Index of the run in the request payload. |
| `config` | Echoed `RunProfile` payload for that run. |
| `dominant_emotion` | Population-level dominant emotion label. |
| `polarization` | Rounded polarization metric from social aggregation. |
| `divergence` | Backward-compatible alias of Wasserstein distance. |
| `wasserstein_distance` | Distance between system sentiment and baseline sentiment. |
| `kl_divergence` | KL divergence between baseline and system sentiment distributions. |
| `validation_details` | Full divergence payload plus stewing interpretation. |
| `explainability` | Human-readable breakdown from `ExplainabilityEngine`. |
| `agent_states` | Dominant emotion label for each returned agent. |
| `agent_influence` | Influence value for each returned agent. |
| `agent_metadata` | Agent records containing id, class, region, and Big Five vector. |
| `endogenous_event` | Name of the follow-on event if one was generated. |
| `detected_biases` | Bias labels from the input-layer analysis. |
| `reasoning` | LLM-generated reasoning trace for the world model step. |
| `negative_integral` | Area-under-curve style negativity over stewing ticks. |
| `acting_ratio` | Share of the scoped population crossing the action threshold. |
| `total_eligible` | Count of agents eligible for the action calculation. |
| `population_size` | Scoped population size after filtering and truncation. |
| `warnings` | Generation/evolution warnings accumulated during preparation. |

### `explainability` Object

`ExplainabilityEngine.generate_explanation(...)` composes the qualitative narrative returned to the frontend. The object includes story-style summaries for:

- cognitive drivers
- shift between objective, viral, and elite centers
- viral dynamics
- societal structure
- tug-of-war / fragmentation
- long-term impact
- endogenous event framing
- demographic archetypes

## Debug And Programmatic Helper Functions

`main.py` also exposes a useful set of importable helpers that the research test suite relies on directly:

| Function | Purpose |
| --- | --- |
| `create_sim_config` | Builds a simulation config with required derived fields. |
| `clone_sim_config` | Copies a config while preserving derived fields. |
| `run_profile_to_sim_config_kwargs` | Converts request-facing config into `SimConfig` kwargs. |
| `build_debug_society` | Builds an in-memory society object from supplied tensors. |
| `prepare_society_for_debug` | Generates a real test/debug society from config. |
| `evolve_society_for_debug` | Runs the evolution pass explicitly. |
| `distort_world_signal` | Runs only the perception distortion stage. |
| `run_cognitive_cycle` | Runs the cognitive engine and returns context, attention, and engagement. |
| `consolidate_agent_memory` | Applies memory consolidation without full simulation. |
| `project_emotions` | Maps context vectors into emotion space. |
| `aggregate_social_state` | Runs the social physics aggregation stage only. |
| `map_emotions_to_sentiment` | Converts emotion output into sentiment buckets. |
| `calculate_validation_metrics` | Computes divergence against a baseline sentiment profile. |
| `create_topology_for_debug` | Creates topology from supplied exposures, personalities, and influence. |
| `apply_triadic_closure_for_debug` | Applies triadic closure to a supplied adjacency matrix. |
| `run_debug_simulation` | Executes the full in-memory simulation pipeline without HTTP. |

## Operational Behavior

### Concurrency

For each `/simulate` request, `main.py` concurrently launches:

- the LLM world-model task
- the baseline sentiment-model task
- one society preparation task per requested run

This keeps latency lower when multiple runs reuse the same input event.

### Society Cache

Generated societies are cached in an in-memory LRU cache keyed by the effective config. The cache excludes transient fields like the output directory and keeps topology/evolution variants isolated.

### Class Filtering And Subsetting

If `social_class` is not `All`, the run:

1. filters metadata and tensors to the matching class
2. reslices the sparse adjacency matrix to that subgroup
3. optionally truncates to `agent_count`

This means all returned metrics are scoped to the actual filtered population.

### Algorithmic Amplification

If enabled, the system performs:

1. a first pass on a small sample
2. dimension ranking by engagement-weighted attention
3. world-tensor mutation on the most engaging dimensions
4. a second full-population broadcast using the mutated tensor

### Endogenous Events

If the aggregated social state crosses action conditions, `physics_engine.py` emits an `action_vector` and `action_name`. The orchestrator immediately feeds that new event back into the cognitive engine as a follow-on personal event.

### Memory Consolidation

When memory is enabled, the post-run memory update uses both the current context vector and a rehearsal factor derived from social confidence and acting ratio.

### Cleanup

Each request:

- clears memory pressure where possible
- creates a short-lived temp output directory
- schedules old temp directories for background cleanup

## Related Docs

- [README.md](../README.md)
- [development.md](./development.md)
- [testing.md](./testing.md)
- [orchestration.md](./orchestration.md)
