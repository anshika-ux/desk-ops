from __future__ import annotations

import copy
import uuid
import random
from typing import Dict, List, Optional

from ..models import DeskAction, DeskObservation, DeskOpsState, DeskActionType
from ..tasks import TASK_DEFINITIONS, TASK_LIBRARY, TaskDefinition, GraderResult

try:  # pragma: no cover
    from openenv.core.env_server import Environment
except ImportError:  # pragma: no cover
    class Environment:  # type: ignore
        def __init__(self, *_, **__):
            ...

        def __class_getitem__(cls, _):
            return cls


class DeskOpsEnvironment(Environment[DeskAction, DeskObservation, DeskOpsState]):
    """Mini-game style environment that simulates real desk operations."""

    def __init__(self):
        super().__init__()
        self._state = DeskOpsState(episode_id=str(uuid.uuid4()), step_count=0)
        self._task_pointer = 0
        self._current_task: Optional[TaskDefinition] = None
        self._workspace: Dict | None = None
        self._last_pending: List[str] = []
        self._last_action_summary: Optional[str] = None
        self._last_error: Optional[str] = None
        self._metrics: Dict[str, int] = {"invalid": 0, "noop": 0}
        self._rng = random.Random()
        self._reward_cfg = {
            "step_cost": 0.01,
            "invalid_penalty": 0.1,
            "no_change_penalty": 0.05,
            "shaping_scale": 1.0,
            "submit_bonus": 1.0,
            "timeout_penalty": 0.2,
        }

    # ------------------------------------------------------------------
    # Core OpenEnv API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs,
    ) -> DeskObservation:
        """Load a new task (optionally filtered by id/difficulty)."""

        if seed is not None:
            self._rng.seed(seed)
        else:
            self._rng.seed(uuid.uuid4().int & 0xFFFFFFFF)

        task = self._select_task(task_id=task_id, difficulty=difficulty)
        self._current_task = task
        self._workspace = task.initial_state_factory(self._rng)
        self._state.step_count = 0
        self._state.steps_taken = 0
        self._state.episode_id = episode_id or str(uuid.uuid4())
        self._state.task_id = task.task_id
        self._state.difficulty = task.difficulty.value
        self._state.partial_score = 0.0
        self._last_pending = task.acceptance_criteria.copy()
        self._last_action_summary = None
        self._last_error = None
        self._metrics = {"invalid": 0, "noop": 0}

        result = task.score_fn(self._workspace)
        self._state.partial_score = result.score
        return self._build_observation(result, reward=0.0, done=False)

    def step(self, action: DeskAction, timeout_s: Optional[float] = None, **kwargs) -> DeskObservation:
        if self._current_task is None or self._workspace is None:
            raise RuntimeError("Call reset() before step().")

        task = self._current_task
        self._state.step_count += 1
        self._state.steps_taken = self._state.step_count

        base_result = task.score_fn(self._workspace)
        reward = -self._reward_cfg["step_cost"]
        done = False
        self._last_error = None

        if action.intent == DeskActionType.SUBMIT:
            reward += base_result.score * self._reward_cfg["submit_bonus"]
            done = True
            self._last_action_summary = "submit"
            self._last_pending = base_result.pending
            self._state.partial_score = base_result.score
            return self._build_observation(base_result, reward=reward, done=done)

        outcome = task.handle_action(self._workspace, action)
        self._last_action_summary = outcome.message
        if outcome.error:
            self._last_error = outcome.error
            reward -= self._reward_cfg["invalid_penalty"]
            self._metrics["invalid"] += 1
        elif not outcome.changed:
            reward -= self._reward_cfg["no_change_penalty"]
            self._metrics["noop"] += 1

        updated_result = task.score_fn(self._workspace)
        delta = updated_result.score - base_result.score
        reward += delta * self._reward_cfg["shaping_scale"]
        self._state.partial_score = updated_result.score
        self._last_pending = updated_result.pending

        if self._state.step_count >= task.max_steps:
            done = True
            reward -= self._reward_cfg["timeout_penalty"]
            reward += updated_result.score * 0.5

        return self._build_observation(updated_result, reward=reward, done=done)

    @property
    def state(self) -> DeskOpsState:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _select_task(
        self, task_id: Optional[str] = None, difficulty: Optional[str] = None
    ) -> TaskDefinition:
        if task_id:
            if task_id not in TASK_DEFINITIONS:
                raise ValueError(f"Unknown task_id {task_id}")
            return TASK_DEFINITIONS[task_id]

        if difficulty:
            filtered = [t for t in TASK_LIBRARY if t.difficulty.value == difficulty]
            if not filtered:
                raise ValueError(f"No tasks for difficulty {difficulty}")
        else:
            filtered = list(TASK_LIBRARY)

        task = filtered[self._task_pointer % len(filtered)]
        self._task_pointer = (self._task_pointer + 1) % len(TASK_LIBRARY)
        return task

    def _build_observation(
        self,
        grader_result: GraderResult,
        reward: float,
        done: bool,
    ) -> DeskObservation:
        if self._current_task is None or self._workspace is None:
            raise RuntimeError("Observation requested before reset")

        workspace_snapshot = copy.deepcopy(self._workspace)
        pending = grader_result.pending or self._last_pending
        return DeskObservation(
            task_id=self._current_task.task_id,
            difficulty=self._current_task.difficulty.value,
            instructions=self._current_task.instructions,
            workspace_state=workspace_snapshot,
            progress=grader_result.breakdown,
            pending_objectives=pending,
            last_action=self._last_action_summary,
            last_error=self._last_error,
            telemetry={
                "steps": float(self._state.step_count),
                "score": grader_result.score,
                "invalid_actions": float(self._metrics.get("invalid", 0)),
                "noop_actions": float(self._metrics.get("noop", 0)),
            },
            reward=reward,
            done=done,
        )
