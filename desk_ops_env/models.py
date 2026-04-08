from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:  # pragma: no cover
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:  # pragma: no cover
    class _Base(BaseModel):
        model_config = dict(extra="forbid", validate_assignment=True)

    class Action(_Base):
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(_Base):
        done: bool = False
        reward: float | int | bool | None = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(_Base):
        episode_id: Optional[str] = None
        step_count: int = 0


class DeskActionType(str, Enum):
    """High-level intents the agent can use inside DeskOps."""

    SET_FIELD = "set_field"
    ASSIGN_SLOT = "assign_slot"
    MOVE_STAGE = "move_stage"
    ADD_NOTE = "add_note"
    SUBMIT = "submit"


class DeskAction(Action):
    """Structured command that keeps the interface deterministic."""

    intent: DeskActionType = Field(..., description="Action the agent wants to execute")
    target_id: Optional[str] = Field(
        default=None,
        description="Primary entity identifier (email id, meeting request id, backlog card id, ...)",
    )
    field: Optional[str] = Field(
        default=None,
        description="Field that should be updated when using set_field/add_note intents.",
    )
    value: Optional[str] = Field(
        default=None, description="Value the agent would like to set or the slot to assign"
    )


class DeskObservation(Observation):
    """Observation surfaced to the agent after every step."""

    task_id: str = Field(..., description="Current task identifier")
    difficulty: str = Field(..., description="Difficulty bucket")
    instructions: str = Field(..., description="Natural-language task brief")
    workspace_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Compact machine-readable snapshot of the workspace",
    )
    progress: Dict[str, float] = Field(
        default_factory=dict,
        description="Progress scores per rubric dimension, normalized to 0-1",
    )
    pending_objectives: List[str] = Field(
        default_factory=list, description="Outstanding objectives that still need work"
    )
    last_action: Optional[str] = Field(
        default=None, description="Echo of the last action that the environment processed"
    )
    last_error: Optional[str] = Field(
        default=None, description="Validation error for the most recent action, if any"
    )
    telemetry: Dict[str, float] = Field(
        default_factory=dict,
        description="Signals such as steps taken, invalid/noop counts, score deltas",
    )
    reward: float = Field(0.0, description="Dense reward after the action was applied")
    done: bool = Field(False, description="Whether the episode has terminated")


class DeskOpsState(State):
    """Internal state exposed via the OpenEnv state() method."""

    task_id: Optional[str] = Field(default=None, description="Currently loaded task id")
    difficulty: Optional[str] = Field(
        default=None, description="Difficulty of the active task for quick introspection"
    )
    steps_taken: int = Field(0, ge=0, description="Number of actions applied in the episode")
    partial_score: float = Field(0.0, ge=0.0, le=1.0, description="Latest rubric score")


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
