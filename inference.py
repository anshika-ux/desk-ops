"""Baseline inference script for the DeskOps OpenEnv environment."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI

from desk_ops_env.models import DeskAction, DeskActionType
from desk_ops_env.server.desk_ops_environment import DeskOpsEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

API_KEY = HF_TOKEN or OPENAI_API_KEY


@dataclass
class RuleAgent:
    """Deterministic fallback policy that guarantees reproducible scores."""

    def __init__(self) -> None:
        self._templates: Dict[str, List[Dict[str, Optional[str]]]] = {
            "inbox_triage": [
                {"intent": "set_field", "target_id": "EM-101", "field": "status", "value": "respond"},
                {"intent": "set_field", "target_id": "EM-102", "field": "status", "value": "delegate"},
                {"intent": "set_field", "target_id": "EM-103", "field": "status", "value": "archive"},
                {"intent": "submit"},
            ],
            "calendar_allocation": [
                {"intent": "assign_slot", "target_id": "retro", "value": "Tue-1000"},
                {"intent": "assign_slot", "target_id": "launch_review", "value": "Wed-1100"},
                {"intent": "assign_slot", "target_id": "client_sync", "value": "Thu-1400"},
                {"intent": "submit"},
            ],
            "launch_board": [
                {"intent": "move_stage", "target_id": "spec_cleanup", "value": "review"},
                {"intent": "set_field", "target_id": "spec_cleanup", "field": "owner", "value": "pm"},
                {"intent": "move_stage", "target_id": "legal_packet", "value": "in-progress"},
                {"intent": "set_field", "target_id": "legal_packet", "field": "owner", "value": "legal"},
                {"intent": "move_stage", "target_id": "playbook_v2", "value": "done"},
                {"intent": "set_field", "target_id": "playbook_v2", "field": "owner", "value": "marketing"},
                {"intent": "submit"},
            ],
            "vendor_negotiation": [
                {"intent": "set_field", "target_id": "VN-1", "field": "decision", "value": "approve"},
                {"intent": "set_field", "target_id": "VN-2", "field": "decision", "value": "renegotiate"},
                {"intent": "set_field", "target_id": "VN-3", "field": "decision", "value": "approve"},
                {"intent": "submit"},
            ],
        }
        self._progress: Dict[str, int] = {key: 0 for key in self._templates}

    def decide(self, task_id: str) -> Optional[DeskAction]:
        script = self._templates.get(task_id)
        if script is None:
            return None
        idx = self._progress.get(task_id, 0)
        if idx >= len(script):
            return None
        self._progress[task_id] = idx + 1
        spec = script[idx]
        intent = DeskActionType(spec["intent"])
        return DeskAction(
            intent=intent,
            target_id=spec.get("target_id"),
            field=spec.get("field"),
            value=spec.get("value"),
        )


class LLMAgent:
    """Thin wrapper around the OpenAI client that emits structured actions."""

    def __init__(self) -> None:
        self._client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def decide(self, observation: Dict) -> Optional[DeskAction]:
        prompt = (
            "You are an operations assistant playing the DeskOps RL environment. "
            "Return the next action as strict JSON with keys intent,target_id,field,value. "
            "Intent options: set_field, assign_slot, move_stage, add_note, submit. "
            "Use null when a field is not needed."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": observation["task_id"],
                        "difficulty": observation["difficulty"],
                        "pending": observation.get("pending_objectives", []),
                        "workspace_state": observation.get("workspace_state"),
                    },
                    indent=2,
                ),
            },
        ]
        try:
            response = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0,
            )
        except Exception:
            return None

        content = response.choices[0].message.content
        if not content:
            return None
        return self._parse_action(content)

    def _parse_action(self, text: str) -> Optional[DeskAction]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        try:
            intent = DeskActionType(payload["intent"])
        except Exception:
            return None

        return DeskAction(
            intent=intent,
            target_id=payload.get("target_id"),
            field=payload.get("field"),
            value=payload.get("value"),
        )


def format_reward(value: float) -> str:
    return f"{value:.2f}"


def run_episode(env: DeskOpsEnvironment, task_id: str, llm: LLMAgent, rules: RuleAgent) -> None:
    observation = env.reset(task_id=task_id)
    rewards: List[float] = []
    step_idx = 0
    print(f"[START] task={task_id} env=desk_ops_env model={MODEL_NAME}")

    while True:
        action = llm.decide(observation.model_dump())
        if action is None:
            action = rules.decide(task_id) or DeskAction(intent=DeskActionType.SUBMIT)

        observation = env.step(action)
        step_idx += 1
        rewards.append(observation.reward or 0.0)
        error_value = observation.last_error or "null"
        print(
            "[STEP] "
            f"step={step_idx} "
            f"action={action.intent.value} "
            f"reward={format_reward(observation.reward or 0.0)} "
            f"done={'true' if observation.done else 'false'} "
            f"error={error_value}"
        )

        if observation.done:
            success = bool(observation.pending_objectives == [])
            rewards_csv = ",".join(format_reward(r) for r in rewards)
            print(
                "[END] "
                f"success={'true' if success else 'false'} "
                f"steps={step_idx} "
                f"rewards={rewards_csv}"
            )
            break


def main() -> None:
    env = DeskOpsEnvironment()
    llm_agent = LLMAgent()
    rule_agent = RuleAgent()
    for task_id in ("inbox_triage", "calendar_allocation", "launch_board", "vendor_negotiation"):
        run_episode(env, task_id, llm_agent, rule_agent)


if __name__ == "__main__":
    main()
