# Development Guide

This page is the contributor-facing quickstart for working on ATELIER locally. It focuses on how the codebase is organized, how to run it, and what to know before making changes.

## Local Setup

### Prerequisites

- Python 3.10+
- `pip`
- virtual environment support
- a valid `GEMINI_API_KEY`

### Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_key_here
```

`input_layer.py` reads this key at startup. If it is missing, the server still starts but event analysis will fail when a simulation request reaches the LLM step.

## Running The Application

### Backend And Frontend

```bash
source .venv/bin/activate
python3 main.py
```

What happens:

- FastAPI starts in `main.py`
- the frontend is mounted as static files from `frontend/`
- the app listens on `0.0.0.0:8000`
- visiting `http://localhost:8000` loads the UI

### Health Check

```bash
curl http://localhost:8000/health
```

### Example Simulation Request

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "news_text": "A major employer announces AI-driven layoffs across several regions.",
    "runs": [
      {
        "seed": 42,
        "social_class": "All",
        "agent_count": 5000
      }
    ]
  }'
```

Use the [API reference](./api-reference.md) for the full request surface.

## Codebase Map

| File | Role |
| --- | --- |
| `main.py` | HTTP API, orchestration, debug helpers, and frontend mounting. |
| `schema.py` | Shared dimensions, labels, config, and conversion helpers. |
| `input_layer.py` | LLM-backed world-state extraction from natural language. |
| `generate_society.py` | Population generation, influence, class structure, topology, and triadic closure. |
| `society_evolution.py` | Cross-generation evolution of the generated population. |
| `cognitive_engine.py` | Per-agent perception, relevance, engagement, memory, and emotion projection. |
| `attention_context.py` | Internal attention/gating pipeline used by the cognitive engine. |
| `physics_engine.py` | Population-level aggregation, virality, collective action, and endogenous events. |
| `validation.py` | Baseline sentiment model and divergence calculations. |
| `explainability.py` | Narrative interpretation of quantitative outputs. |
| `frontend/index.html` | Main application shell and control panel structure. |
| `frontend/script.js` | UI state, batching, visualization, and API wiring. |
| `frontend/style.css` | Styling for the single-page interface. |

## Runtime Data Flow

The practical execution flow when debugging a request is:

1. `news_text` enters `input_layer.get_world_state`
2. `main.py` creates or reuses societies for each run
3. `generate_society.py` and optionally `society_evolution.py` prepare the population
4. `cognitive_engine.py` transforms the event into agent-specific context
5. `physics_engine.py` aggregates those contexts into social metrics
6. `validation.py` compares the output to a baseline sentiment model
7. `explainability.py` translates the results for the UI
8. `frontend/script.js` renders the returned metrics and agent state

## Frontend Notes

The UI is a single-page app with no separate build step in this repository.

Key behaviors in `frontend/script.js`:

- batched run creation and editing
- toggles for core and research controls
- explainability panel rendering
- history and filmstrip management
- agent-canvas rendering
- tooltips and telemetry display

Because the frontend is served directly by FastAPI, most development changes only require restarting `python3 main.py`.

## Research And Debug Workflow

The repository intentionally exposes debug-friendly helpers in `main.py` so contributors can work below the HTTP layer. The research suite calls these helpers directly to test specific behaviors in isolation.

Useful imports for notebook or pytest work:

- `prepare_society_for_debug`
- `build_debug_society`
- `run_debug_simulation`
- `run_cognitive_cycle`
- `aggregate_social_state`
- `distort_world_signal`
- `create_topology_for_debug`
- `apply_triadic_closure_for_debug`

## Testing Workflow

Run the whole suite:

```bash
./run_all_tests.sh
```

Run specific tests:

```bash
python3 -m pytest research_paper_tests/test_run_profile_contract.py -q
python3 -m pytest research_paper_tests/test_runtime_regressions.py -q
```

Run the evolution matrix explicitly:

```bash
./run_all_tests.sh --evolution with
./run_all_tests.sh --evolution without
./run_all_tests.sh --evolution both
```

The full suite guide is in [testing.md](./testing.md).

## Important Implementation Notes

### Configuration Surfaces

- `SimConfig` in `schema.py` is the source of truth for simulation parameters.
- `RunProfile` in `main.py` is the API-friendly request model layered on top of it.
- `research_paper_tests/config_schema.py` builds scenario configs from the live runtime defaults, which helps the tests stay aligned with the real app.

### Caching

Society generation can be expensive, so `main.py` maintains an in-memory LRU cache keyed by effective config. If you change cache-relevant generation logic, review the cache key behavior.

### External Dependencies

- `input_layer.py` calls the Gemini API.
- `validation.py` downloads and uses a Hugging Face sentiment model the first time it is loaded.

That means fully end-to-end local testing may require network access even when the codebase itself is local.

### Generated Artifacts

Several figure tests write `.png` files to `research_paper_tests/generated/`. Those outputs are documented in [testing.md](./testing.md).
