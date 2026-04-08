"""Torch-based heuristic agent for DeskOps."""

from __future__ import annotations

import torch

from desk_ops_env.models import DeskAction, DeskActionType
from desk_ops_env.server.desk_ops_environment import DeskOpsEnvironment


def _submit_action() -> DeskAction:
    return DeskAction(intent=DeskActionType.SUBMIT)


class TorchHeuristicPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("priority_weights", torch.tensor([0.7, 0.3]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (features * self.priority_weights).sum(dim=-1)

    def act(self, observation: dict) -> DeskAction:
        task_id = observation["task_id"]
        if task_id == "inbox_triage":
            return self._inbox_action(observation)
        if task_id == "calendar_allocation":
            return self._calendar_action(observation)
        if task_id == "launch_board":
            return self._board_action(observation)
        if task_id == "vendor_negotiation":
            return self._vendor_action(observation)
        return _submit_action()

    def _inbox_action(self, observation: dict) -> DeskAction:
        emails = observation["workspace_state"]["emails"]
        if all(email["status"] != "unlabeled" for email in emails):
            return _submit_action()
        feats = []
        for email in emails:
            handled = 0.0 if email["status"] == "unlabeled" else 1.0
            feats.append([float(email["priority"]), 1.0 - handled])
        scores = self.forward(torch.tensor(feats))
        idx = torch.argmax(scores).item()
        email = emails[idx]
        hint = email["hints"].lower()
        if "delegate" in hint:
            value = "delegate"
        elif "archive" in hint or "close" in hint:
            value = "archive"
        else:
            value = "respond"
        return DeskAction(
            intent=DeskActionType.SET_FIELD,
            target_id=email["id"],
            field="status",
            value=value,
        )

    def _calendar_action(self, observation: dict) -> DeskAction:
        requests = observation["workspace_state"]["requests"]
        slots = {slot["id"]: slot for slot in observation["workspace_state"]["slots"]}
        for req in requests:
            if req.get("slot"):
                continue
            candidates = []
            for slot_id, slot in slots.items():
                vec = torch.tensor(
                    [
                        1.0 if "morning" in slot["labels"] else 0.0,
                        1.0 / slot["cost"],
                    ]
                )
                candidates.append(vec)
            stacked = torch.stack(candidates)
            scores = stacked.sum(dim=-1)
            best_slot = list(slots.keys())[torch.argmax(scores).item()]
            return DeskAction(
                intent=DeskActionType.ASSIGN_SLOT,
                target_id=req["id"],
                value=best_slot,
            )
        return _submit_action()

    def _board_action(self, observation: dict) -> DeskAction:
        cards = observation["workspace_state"]["cards"]
        for card in cards:
            if not card.get("owner"):
                owner = "pm" if "spec" in card["id"] else "legal"
                return DeskAction(
                    intent=DeskActionType.SET_FIELD,
                    target_id=card["id"],
                    field="owner",
                    value=owner,
                )
        for card in cards:
            if card["stage"] != "done":
                next_stage = {
                    "backlog": "in-progress",
                    "in-progress": "review",
                    "review": "done",
                }[card["stage"]]
                return DeskAction(
                    intent=DeskActionType.MOVE_STAGE,
                    target_id=card["id"],
                    value=next_stage,
                )
        return _submit_action()

    def _vendor_action(self, observation: dict) -> DeskAction:
        vendors = observation["workspace_state"]["vendors"]
        spend_cap = observation["workspace_state"]["spend_cap"]
        net_spend = sum(max(0, v["quote"] - v["discount"]) for v in vendors if v["status"] == "approve")
        for vendor in vendors:
            if vendor["status"] == "pending":
                net_cost = vendor["quote"] - vendor["discount"]
                decision = "approve" if net_cost <= vendor["budget_ceiling"] and net_spend + net_cost <= spend_cap else "renegotiate"
                return DeskAction(
                    intent=DeskActionType.SET_FIELD,
                    target_id=vendor["id"],
                    field="decision",
                    value=decision,
                )
        return _submit_action()


def run_policy() -> None:
    env = DeskOpsEnvironment()
    policy = TorchHeuristicPolicy()
    for task_id in ("inbox_triage", "calendar_allocation", "launch_board", "vendor_negotiation"):
        obs = env.reset(task_id=task_id, seed=42)
        done = False
        steps = 0
        while not done and steps < 25:
            action = policy.act(obs.model_dump())
            obs = env.step(action)
            steps += 1
        print(f"{task_id}: score={env.state.partial_score:.2f} steps={steps}")


if __name__ == "__main__":
    run_policy()
