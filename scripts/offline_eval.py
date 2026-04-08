"""Deterministic evaluation helper that avoids LLM calls."""

from __future__ import annotations

from typing import Dict, List

from desk_ops_env.models import DeskAction, DeskActionType
from desk_ops_env.server.desk_ops_environment import DeskOpsEnvironment

PLAYBOOKS: Dict[str, List[DeskAction]] = {
    "inbox_triage": [
        DeskAction(intent=DeskActionType.SET_FIELD, target_id="EM-101", field="status", value="respond"),
        DeskAction(intent=DeskActionType.SET_FIELD, target_id="EM-102", field="status", value="delegate"),
        DeskAction(intent=DeskActionType.SET_FIELD, target_id="EM-103", field="status", value="archive"),
        DeskAction(intent=DeskActionType.SUBMIT),
    ],
    "calendar_allocation": [
        DeskAction(intent=DeskActionType.ASSIGN_SLOT, target_id="retro", value="Tue-1000"),
        DeskAction(intent=DeskActionType.ASSIGN_SLOT, target_id="launch_review", value="Wed-1100"),
        DeskAction(intent=DeskActionType.ASSIGN_SLOT, target_id="client_sync", value="Thu-1400"),
        DeskAction(intent=DeskActionType.SUBMIT),
    ],
    "launch_board": [
        DeskAction(intent=DeskActionType.MOVE_STAGE, target_id="spec_cleanup", value="review"),
        DeskAction(intent=DeskActionType.SET_FIELD, target_id="spec_cleanup", field="owner", value="pm"),
        DeskAction(intent=DeskActionType.MOVE_STAGE, target_id="legal_packet", value="in-progress"),
        DeskAction(intent=DeskActionType.SET_FIELD, target_id="legal_packet", field="owner", value="legal"),
        DeskAction(intent=DeskActionType.MOVE_STAGE, target_id="playbook_v2", value="done"),
        DeskAction(intent=DeskActionType.SET_FIELD, target_id="playbook_v2", field="owner", value="marketing"),
        DeskAction(intent=DeskActionType.SUBMIT),
    ],
}


def run_episode(env: DeskOpsEnvironment, task_id: str) -> float:
    obs = env.reset(task_id=task_id)
    final = obs
    for action in PLAYBOOKS[task_id]:
        final = env.step(action)
    return final.reward if final.reward is not None else 0.0


def main() -> None:
    env = DeskOpsEnvironment()
    results = {}
    for task in PLAYBOOKS:
        final_reward = run_episode(env, task)
        results[task] = env.state.partial_score
        print(f"{task}: reward={final_reward:.2f} score={results[task]:.2f}")
    avg = sum(results.values()) / len(results)
    print(f"Average score: {avg:.2f}")


if __name__ == "__main__":
    main()
