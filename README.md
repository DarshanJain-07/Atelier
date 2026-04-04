# ATELIER

ATELIER is a hybrid neuro-symbolic multi-agent simulation platform for studying how public sentiment, emotional contagion, polarization, and backlash evolve inside a modeled society. It combines an LLM-based perception layer with deterministic PyTorch simulation so the same input event can be translated into structured world-state signals and then propagated through large populations with memory, topology, inequality, and social-threshold effects.

## What The App Does

Given a piece of text such as a news event, policy change, or public statement, ATELIER:

1. Interprets the event into a structured world-state representation.
2. Generates a synthetic society with exposures, personalities, influence, classes, memory, and optional network topology.
3. Runs a cognitive pipeline that turns the same event into different subjective reactions for different agents.
4. Aggregates those reactions into social metrics such as dominant emotion, polarization, virality, elite divergence, collective action, and endogenous events.
5. Returns explainability summaries and agent-level state for the web UI and for research/debug workflows.

## Architecture At A Glance

The main runtime path is:

1. `input_layer.py`
   Converts raw text into the structured world signal, urgency score, bias labels, and reasoning trace.
2. `generate_society.py`
   Builds the initial society, influence distribution, class structure, and optional network topology.
3. `society_evolution.py`
   Optionally evolves the generated society before the event is simulated.
4. `cognitive_engine.py` and `attention_context.py`
   Distort, filter, gate, and internalize the incoming signal per agent.
5. `physics_engine.py`
   Aggregates agent emotion into viral, elite, and population-level social outcomes.
6. `explainability.py` and `validation.py`
   Translate outputs into human-readable narratives and compare them against baseline sentiment models.
7. `main.py`
   Orchestrates FastAPI endpoints, caching, class slicing, simulation runs, and the static frontend.

## Repository Map

| Path | Purpose |
| --- | --- |
| `main.py` | FastAPI app, orchestration, request models, debug helpers, and `/simulate` endpoint. |
| `schema.py` | Shared simulation configuration, dimensions, emotion mappings, and conversion utilities. |
| `input_layer.py` | LLM-backed event interpretation layer. |
| `generate_society.py` | Society generation, topology creation, triadic closure, and structural metadata. |
| `society_evolution.py` | Multi-generation evolution of wealth, class, ideology, and mobility. |
| `cognitive_engine.py` | Perception, relevance, engagement, memory, and emotion projection logic. |
| `attention_context.py` | Attention and gating sub-pipeline used by the cognitive engine. |
| `physics_engine.py` | Aggregation, virality, polarization, action thresholds, and endogenous events. |
| `validation.py` | Baseline-model loading plus divergence and stewing validation metrics. |
| `explainability.py` | Human-readable interpretation of simulation outputs. |
| `frontend/` | Static single-page UI served by FastAPI. |
| `docs/` | Contributor-facing documentation and subsystem guides. |
| `research_paper_tests/` | Scenario-driven pytest suite for behavior, regressions, and figure generation. |
| `run_all_tests.sh` | Canonical test runner for the research suite. |

## Getting Started

### Prerequisites

- Python 3.10+
- A virtual environment
- A `GEMINI_API_KEY` in `.env` for the LLM-backed input layer
- Network access the first time the app downloads the Hugging Face baseline sentiment model used by `validation.py`

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` with:

```env
GEMINI_API_KEY=your_key_here
```

### Run The App

```bash
source .venv/bin/activate
python3 main.py
```

The FastAPI server listens on `http://localhost:8000` and serves the frontend from the same process. Visiting that URL loads the web UI in `frontend/`.

Documentation is now available inside the running app at `http://localhost:8000/docs`, and the OpenAPI/Swagger docs live at `http://localhost:8000/api/docs`.

## API And Frontend Workflow

The primary API surface is:

- `GET /health`
- `POST /simulate`
- `GET /docs`
- `GET /api/docs`

`/simulate` accepts a `news_text` string plus one or more run profiles. Each run profile can toggle cognitive, topology, amplification, evolution, and memory features while also scoping the population by social class. The response includes per-run metrics, explainability, validation output, agent states, and any endogenous event that was triggered.

The frontend uses that endpoint to:

- submit experiments
- compare batched runs
- inspect explainability summaries
- browse agent metadata and emotion state
- adjust both core and research-facing controls

Full request and response details live in [docs/api-reference.md](docs/api-reference.md).

## Testing

The `research_paper_tests/` suite is not a token smoke-test folder. It is a scenario-driven validation harness covering:

- config and API contracts
- society generation and topology behavior
- cognitive and emotional dynamics
- social-physics and cascade behavior
- evolution and inequality behavior
- runtime and memory regressions
- figure generation for research outputs

Run the whole suite with:

```bash
./run_all_tests.sh
```

Optional evolution matrix modes:

```bash
./run_all_tests.sh --evolution with
./run_all_tests.sh --evolution without
./run_all_tests.sh --evolution both
```

A full test catalog lives in [docs/testing.md](docs/testing.md).

## Documentation Map

- [docs/index.md](docs/index.md): documentation hub and reading order
- [docs/api-reference.md](docs/api-reference.md): endpoints, payloads, orchestration helpers, and operational behavior
- [docs/development.md](docs/development.md): setup, local workflow, and codebase map
- [docs/testing.md](docs/testing.md): test-running guide plus test-case inventory
- [docs/input-layer.md](docs/input-layer.md): perception layer details
- [docs/cognitive-engine.md](docs/cognitive-engine.md): cognitive processing details
- [docs/attention-context.md](docs/attention-context.md): attention and gating details
- [docs/physics-engine.md](docs/physics-engine.md): aggregation and collective behavior details
- [docs/society-generation.md](docs/society-generation.md): population and topology generation details
- [docs/society-evolution.md](docs/society-evolution.md): long-horizon socioeconomic evolution details
- [docs/orchestration.md](docs/orchestration.md): orchestration concepts in `main.py`

## Notes For Contributors

- The source of truth for runtime configuration is `schema.py`, but the public request surface is `RunProfile` in `main.py`.
- `research_paper_tests/config_schema.py` centralizes reusable scenarios, test-world builders, and helper defaults for the research suite.
- Documentation in this repo avoids hard-coding schema default values unless a page is specifically about a mechanism, so docs stay useful as the simulation evolves.
