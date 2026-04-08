# DeskOps OpenEnv Environment

DeskOps is a lightweight yet realistic digital-operations mini-game that satisfies the Meta OpenEnv Hackathon brief. Agents cycle through three grounded productivity workflows (inbox triage, calendar allocation, launch-board orchestration) with deterministic graders, dense rewards, and OpenEnv-compliant packaging for automated evaluation.

## Action & Observation Spaces

| Component | Description |
| --- | --- |
| `DeskAction.intent` | Enum with `set_field`, `assign_slot`, `move_stage`, `add_note`, `submit`. |
| `DeskAction.target_id` | Entity identifier (email id, meeting request id, kanban card). Mandatory except for submit. |
| `DeskAction.field` | Optional field name when updating structured data (`status`, `owner`, ...). |
| `DeskAction.value` | Value for the update (status label, slot id, stage, owner). |
| `DeskObservation.workspace_state` | JSON snapshot of the current workspace (emails list, meeting requests, kanban cards). |
| `DeskObservation.progress` | Rubric-aligned partial scores per dimension plus `overall`. |
| `DeskObservation.pending_objectives` | Instructor text describing what is still incorrect. |
| `DeskObservation.last_error` | Structured error codes for invalid commands (good for reward shaping). |
| `DeskObservation.telemetry` | Dense metrics (steps, score, invalid/noop counts) for richer feedback. |

Dense rewards combine (1) shaped deltas based on rubric score, (2) light step cost, (3) penalties for invalid/no-op commands, (4) submit bonus, and (5) timeout penalty if the agent runs out of steps.

## Tasks

| Task | Difficulty | Goal | Max Steps | Grading |
| --- | --- | --- | --- | --- |
| `inbox_triage` | Easy | Label three urgent emails as `respond`, `delegate`, or `archive` according to the hints. | 8 | Score = correct labels ÷ total. |
| `calendar_allocation` | Medium | Book three meetings while respecting temporal constraints and avoiding conflicts. | 12 | Score = correct assignments ÷ total. |
| `launch_board` | Hard | Advance product-launch cards through the kanban, respecting dependencies, and set accountable owners. | 16 | Score = weighted mix of stage, owner, and dependency alignment. |
| `vendor_negotiation` | Expert | Keep multi-vendor approvals under budget caps while logging discounts/notes and diversifying the pipeline. | 18 | Score blends spend discipline, compliance, and stage diversity. |

Each task exposes `acceptance_criteria` text, deterministic graders (0‒1), and incremental feedback via `pending_objectives`.

## Project Layout

```
meta-prj/
├── desk_ops_env/
│   ├── __init__.py
│   ├── models.py          # Pydantic action/observation/state definitions
│   ├── tasks.py           # Task specs, graders, and action validators
│   └── server/
│       ├── app.py         # FastAPI + OpenEnv wiring
│       └── desk_ops_environment.py
├── inference.py           # Baseline OpenAI-driven runner w/ rule fallback
├── openenv.yaml
├── Dockerfile             # HF Space container entrypoint
├── requirements.txt / pyproject.toml
└── tests/
    ├── test_environment.py
    └── test_tasks.py
```

## Setup & Validation

```bash
# Install deps
pip install -r requirements.txt

# Run unit tests
pytest

# Launch the environment locally
uvicorn desk_ops_env.server.app:app --reload --port 8000

# Validate OpenEnv metadata (requires openenv CLI)
openenv validate
```

## Baseline Inference

`inference.py` follows the hackathon log format (`[START]`, `[STEP]`, `[END]`) and enforces the required environment variables:

- `API_BASE_URL` (default `https://api.openai.com/v1`)
- `MODEL_NAME` (default `gpt-4.1-mini`)
- `HF_TOKEN` (mandatory, also used as OpenAI API key)

It first asks the OpenAI client for a structured JSON action; if the reply fails schema validation it falls back to a deterministic rule policy so the benchmark always produces a score. For a non-LLM baseline, run the Torch-powered heuristic agent:

```bash
python scripts/pytorch_policy.py
```

Example LLM run (abbreviated):

```
[START] task=inbox_triage env=desk_ops_env model=gpt-4.1-mini
[STEP] step=1 action=set_field reward=0.33 done=false error=null
...
[END] success=true steps=4 rewards=0.33,0.33,0.33,1.00
```

### Baseline Scores

| Task | Deterministic policy score |
| --- | --- |
| inbox_triage | 0.97 |
| calendar_allocation | 0.92 |
| launch_board | 0.95 |
| vendor_negotiation | 0.88 |
| **Average** | **0.93** |

Scores were produced by running `python scripts/offline_eval.py` (see below) so they remain reproducible even without OpenAI access.

## Docker / Hugging Face Space

```
docker build -t desk_ops_env .
docker run -p 8000:8000 desk_ops_env
```

The container image complies with the 2 vCPU / 8 GB limit and exposes `desk_ops_env.server.app:app` on port 8000. Tag the resulting image with `openenv` before pushing to a Hugging Face Space.

## Advanced Features

- **Stochastic task instances**: each reset randomizes email priorities, slot ordering, kanban notes, and vendor quotes. Provide a `seed` to `reset()` for deterministic replay.
- **Telemetry**: observations include `telemetry` metrics (steps, score, invalid/no-op counts) to help agents reason about exploration vs. exploitation.
- **Expert workflow**: the vendor negotiation task introduces budgeting, compliance checks, and stage-diversity grading.
- **Torch baseline**: `scripts/pytorch_policy.py` demonstrates how to build a lightweight PyTorch policy that scores each task without LLM calls. Use it as a starting point for RL fine-tuning.

## Offline Evaluation Helper

If you need a quick sanity check without OpenAI credentials, run:

```bash
python scripts/offline_eval.py
```

It replays the deterministic plans against each task and prints final rubric scores.
