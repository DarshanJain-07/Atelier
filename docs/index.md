# ATELIER Documentation Hub

ATELIER is a hybrid neuro-symbolic multi-agent simulation system for modeling how a shared event becomes divergent personal perception, emotional contagion, class-sensitive reaction, and collective social behavior.

This `docs/` directory now serves two audiences:

- contributors who need to understand how the application is structured and how to work on it
- researchers who need to understand what each subsystem and test suite component is validating

## Start Here

If you are new to the repository, read in this order:

1. [README.md](../README.md)
2. [development.md](./development.md)
3. [api-reference.md](./api-reference.md)
4. the subsystem pages most relevant to your change
5. [testing.md](./testing.md)

## Documentation Map

| Document | Focus |
| --- | --- |
| [README.md](../README.md) | Project overview, architecture, setup, and navigation. |
| [development.md](./development.md) | Local setup, runtime workflow, and contributor-oriented codebase map. |
| [api-reference.md](./api-reference.md) | HTTP endpoints, request/response structure, debug helpers, and orchestration behavior. |
| [testing.md](./testing.md) | Test-running guide and a full catalog of the research test suite. |
| [orchestration.md](./orchestration.md) | Conceptual overview of orchestration behavior in `main.py`. |
| [input-layer.md](./input-layer.md) | LLM-backed perception layer and world-state extraction. |
| [cognitive-engine.md](./cognitive-engine.md) | Cognitive processing, distortion, memory, and emotion projection. |
| [attention-context.md](./attention-context.md) | Attention and gating internals used by the cognitive engine. |
| [physics-engine.md](./physics-engine.md) | Social aggregation, virality, polarization, and endogenous events. |
| [society-generation.md](./society-generation.md) | Society generation, topology, and structural initialization. |
| [society-evolution.md](./society-evolution.md) | Evolutionary dynamics across generations. |

## System Summary

At the highest level, the platform works like this:

1. A natural-language event enters the perception layer.
2. The event is translated into a structured world representation with urgency and bias metadata.
3. A synthetic population is generated or reused from cache.
4. The cognitive engine transforms the same event into agent-specific context and emotion.
5. The physics engine aggregates those reactions into societal metrics and possible endogenous events.
6. Validation and explainability layers package the results for both researchers and end users.

## Where To Look For Specific Work

| You want to change... | Start with... |
| --- | --- |
| FastAPI request handling or response payloads | [api-reference.md](./api-reference.md), `main.py` |
| UI controls or interaction flow | [development.md](./development.md), `frontend/script.js`, `frontend/index.html` |
| Event interpretation | [input-layer.md](./input-layer.md), `input_layer.py` |
| Per-agent attention, relevance, or memory | [cognitive-engine.md](./cognitive-engine.md), [attention-context.md](./attention-context.md) |
| Network or society generation | [society-generation.md](./society-generation.md), `generate_society.py` |
| Evolution and inequality dynamics | [society-evolution.md](./society-evolution.md), `society_evolution.py` |
| Collective metrics or cascades | [physics-engine.md](./physics-engine.md), `physics_engine.py` |
| Regression coverage or research figures | [testing.md](./testing.md), `research_paper_tests/` |

## Notes

- The project is intentionally transparent and research-friendly; the docs aim to expose the full behavior of the app rather than only the user-facing path.
- Configuration defaults live in code and evolve over time, so these docs emphasize behavior, structure, and field purpose more than frozen literal values.
