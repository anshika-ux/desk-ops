from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .models import DeskAction, DeskActionType, TaskDifficulty

BOUND_EPSILON = 1e-3


def _bounded(score: float) -> float:
    """Clamp grader scores to the OpenEnv allowed range (0, 1)."""
    if score <= 0.0:
        return BOUND_EPSILON
    if score >= 1.0:
        return 1.0 - BOUND_EPSILON
    return score


@dataclass
class ActionOutcome:
    changed: bool
    message: str
    error: str | None = None


@dataclass
class GraderResult:
    score: float
    breakdown: Dict[str, float]
    pending: List[str]


@dataclass
class TaskDefinition:
    task_id: str
    title: str
    difficulty: TaskDifficulty
    instructions: str
    acceptance_criteria: List[str]
    max_steps: int
    initial_state_factory: Callable[[random.Random], Dict]
    handle_action: Callable[[Dict, DeskAction], ActionOutcome]
    score_fn: Callable[[Dict], GraderResult]


EMAIL_SUBJECTS = [
    "Product brief needs sign-off",
    "Vendor invoices stuck in queue",
    "Conference invite follow-up",
    "Security questionnaire pending",
]

MEETING_NOTES = [
    "Needs legal + product present.",
    "Avoid clashes with focus block.",
    "Prefer afternoon to catch US team.",
    "Stakeholders want hybrid setup.",
]

KANBAN_NOTES = [
    "Needs PM approval before Eng.",
    "Legal must sign by Wednesday.",
    "Ops drafted skeleton. Marketing owns final copy.",
    "Security audit waiting on vendor NDA.",
]

VENDOR_NAMES = ["AeroSync", "NovaOps", "FoundryWorks", "LumenByte"]


# ---------------------------------------------------------------------------
# Task 1: Inbox Triage (Easy)
# ---------------------------------------------------------------------------

INBOX_EXPECTED = {
    "EM-101": "respond",
    "EM-102": "delegate",
    "EM-103": "archive",
}


def _inbox_initial_state(rng: random.Random) -> Dict:
    subjects = rng.sample(EMAIL_SUBJECTS, k=3)
    emails = []
    for idx, template in enumerate(subjects, start=1):
        email_id = f"EM-10{idx}"
        priority = rng.randint(1, 3)
        sla = rng.choice([20, 30, 45])
        emails.append(
            {
                "id": email_id,
                "subject": template,
                "sender": rng.choice(["Priya", "Finance Bot", "Events", "Security"]),
                "status": "unlabeled",
                "priority": priority,
                "sla_minutes": sla,
                "hints": "Customer deadline is tonight."
                if priority == 3
                else rng.choice(MEETING_NOTES),
            }
        )
    return {"emails": emails, "action_history": []}


def _inbox_handle_action(state: Dict, action: DeskAction) -> ActionOutcome:
    if action.intent not in {DeskActionType.SET_FIELD}:
        return ActionOutcome(False, "Inbox triage expects set_field", "UNSUPPORTED_INTENT")
    if action.target_id is None:
        return ActionOutcome(False, "target_id is required", "MISSING_TARGET")

    label = (action.value or "").strip().lower()
    if label not in {"respond", "delegate", "archive"}:
        return ActionOutcome(
            False,
            "Status must be respond|delegate|archive",
            "INVALID_VALUE",
        )

    email = next((e for e in state["emails"] if e["id"] == action.target_id), None)
    if email is None:
        return ActionOutcome(False, f"Email {action.target_id} not found", "UNKNOWN_EMAIL")

    changed = email["status"].lower() != label
    email["status"] = label
    history = state.setdefault("action_history", [])
    history.append({"email": action.target_id, "priority": email["priority"]})
    return ActionOutcome(changed, f"Labeled {action.target_id} as {label}")


def _inbox_score(state: Dict) -> GraderResult:
    total = len(state["emails"])
    matches = sum(1 for e in state["emails"] if e["status"].lower() == INBOX_EXPECTED.get(e["id"], ""))
    pending = [
        f"{email['id']} should be {INBOX_EXPECTED[email['id']]}"
        for email in state["emails"]
        if email["status"].lower() != INBOX_EXPECTED.get(email["id"], "")
    ]
    label_score = matches / total if total else 0.0
    history = state.get("action_history", [])
    sequence_score = 0.0
    if history:
        priorities = [h["priority"] for h in history]
        ideal = sorted(priorities, reverse=True)
        sequence_score = sum(1 for a, b in zip(priorities, ideal) if a == b) / len(priorities)
    sla_hits = sum(
        1
        for email in state["emails"]
        if email["sla_minutes"] <= 30 and email["status"].lower() == INBOX_EXPECTED.get(email["id"], "")
    )
    sla_total = sum(1 for email in state["emails"] if email["sla_minutes"] <= 30) or 1
    sla_score = sla_hits / sla_total
    score = _bounded(0.6 * label_score + 0.2 * sequence_score + 0.2 * sla_score)
    breakdown = {
        "overall": score,
        "labels": label_score,
        "sequence": sequence_score,
        "sla_focus": sla_score,
    }
    if sequence_score < 1.0 and history:
        pending.append("Handle highest-priority emails first.")
    return GraderResult(score=score, breakdown=breakdown, pending=pending)


# ---------------------------------------------------------------------------
# Task 2: Calendar Allocation (Medium)
# ---------------------------------------------------------------------------

CALENDAR_EXPECTED = {
    "client_sync": "Thu-1400",
    "retro": "Tue-1000",
    "launch_review": "Wed-1100",
}

CALENDAR_ALLOWED = {
    "client_sync": {"Thu-1400"},
    "retro": {"Tue-0930", "Tue-1000"},
    "launch_review": {"Wed-1100", "Wed-1500"},
}


def _calendar_initial_state(rng: random.Random) -> Dict:
    slots = [
        {"id": "Tue-0930", "labels": ["morning", "focus"], "booked": False, "cost": 1.2},
        {"id": "Tue-1000", "labels": ["morning"], "booked": False, "cost": 1.0},
        {"id": "Wed-1100", "labels": ["late-morning", "legal"], "booked": False, "cost": 0.8},
        {"id": "Wed-1500", "labels": ["design"], "booked": False, "cost": 1.4},
        {"id": "Thu-1400", "labels": ["client", "pm"], "booked": False, "cost": 0.9},
    ]
    rng.shuffle(slots)
    return {
        "requests": [
            {
                "id": "client_sync",
                "duration": 60,
                "stakeholders": ["Accounts", "Client"],
                "notes": "Needs late-week PM availability and no overlap with Ops standup.",
                "slot": None,
            },
            {
                "id": "retro",
                "duration": 30,
                "stakeholders": ["Engineering", "Design"],
                "notes": "Must be morning and wrap before focus block.",
                "slot": None,
            },
            {
                "id": "launch_review",
                "duration": 45,
                "stakeholders": ["Product", "Legal", "Marketing"],
                "notes": "Needs Design + Legal and should avoid lunch hour.",
                "slot": None,
            },
        ],
        "slots": slots,
        "audit_log": [],
    }


def _calendar_handle_action(state: Dict, action: DeskAction) -> ActionOutcome:
    if action.intent != DeskActionType.ASSIGN_SLOT:
        return ActionOutcome(False, "Use assign_slot for calendar work", "UNSUPPORTED_INTENT")
    if not action.target_id or not action.value:
        return ActionOutcome(False, "target_id and value required", "MISSING_FIELDS")

    request = next((r for r in state["requests"] if r["id"] == action.target_id), None)
    if request is None:
        return ActionOutcome(False, f"Request {action.target_id} missing", "UNKNOWN_REQUEST")

    slot = next((s for s in state["slots"] if s["id"] == action.value), None)
    if slot is None:
        return ActionOutcome(False, f"Slot {action.value} missing", "UNKNOWN_SLOT")
    if slot["booked"] and request.get("slot") != slot["id"]:
        return ActionOutcome(False, f"Slot {slot['id']} already booked", "SLOT_TAKEN")

    allowed = CALENDAR_ALLOWED[action.target_id]
    if slot["id"] not in allowed:
        return ActionOutcome(False, f"Slot {slot['id']} violates constraints", "INVALID_SLOT")

    if request.get("slot") and request["slot"] != slot["id"]:
        old_slot = next(s for s in state["slots"] if s["id"] == request["slot"])
        old_slot["booked"] = False

    changed = request.get("slot") != slot["id"]
    request["slot"] = slot["id"]
    slot["booked"] = True
    state.setdefault("audit_log", []).append({"request": request["id"], "slot": slot["id"], "cost": slot["cost"]})
    return ActionOutcome(changed, f"Booked {slot['id']} for {request['id']}")


def _calendar_score(state: Dict) -> GraderResult:
    total = len(state["requests"])
    matches = 0
    pending: List[str] = []
    for req in state["requests"]:
        assigned = req.get("slot")
        if assigned == CALENDAR_EXPECTED[req["id"]]:
            matches += 1
        else:
            pending.append(
                f"{req['id']} expected {CALENDAR_EXPECTED[req['id']]} but got {assigned or 'None'}"
            )
    score_assign = matches / total if total else 0.0
    budget_cost = sum(entry["cost"] for entry in state.get("audit_log", []))
    normalized_cost = max(0.0, 1.0 - max(0.0, budget_cost - 3.5) * 0.1)
    score = _bounded(0.7 * score_assign + 0.3 * normalized_cost)
    breakdown = {
        "overall": score,
        "scheduling": score_assign,
        "budget_efficiency": normalized_cost,
    }
    return GraderResult(score=score, breakdown=breakdown, pending=pending)


# ---------------------------------------------------------------------------
# Task 3: Launch Board Orchestration (Hard)
# ---------------------------------------------------------------------------

BOARD_EXPECTED = {
    "spec_cleanup": {"stage": "review", "owner": "pm"},
    "legal_packet": {"stage": "in-progress", "owner": "legal"},
    "playbook_v2": {"stage": "done", "owner": "marketing"},
}

STAGE_ORDER = ["backlog", "in-progress", "review", "done"]
ALLOWED_OWNERS = {"pm", "legal", "marketing", "ops"}


def _board_initial_state(rng: random.Random) -> Dict:
    cards = [
        {
            "id": "spec_cleanup",
            "stage": "backlog",
            "owner": None,
            "notes": rng.choice(KANBAN_NOTES),
        },
        {
            "id": "legal_packet",
            "stage": "backlog",
            "owner": None,
            "notes": rng.choice(KANBAN_NOTES),
        },
        {
            "id": "playbook_v2",
            "stage": "in-progress",
            "owner": "ops",
            "notes": rng.choice(KANBAN_NOTES),
        },
    ]
    rng.shuffle(cards)
    return {"cards": cards}


def _board_handle_action(state: Dict, action: DeskAction) -> ActionOutcome:
    if action.target_id is None:
        return ActionOutcome(False, "target_id required", "MISSING_TARGET")

    card = next((c for c in state["cards"] if c["id"] == action.target_id), None)
    if card is None:
        return ActionOutcome(False, f"Card {action.target_id} missing", "UNKNOWN_CARD")

    if action.intent == DeskActionType.MOVE_STAGE:
        new_stage = (action.value or "").strip().lower()
        if new_stage not in STAGE_ORDER:
            return ActionOutcome(False, "Invalid stage", "BAD_STAGE")
        card_stage = card["stage"].lower()
        current_idx = STAGE_ORDER.index(card_stage)
        target_idx = STAGE_ORDER.index(new_stage)
        if target_idx < current_idx:
            return ActionOutcome(False, "Cannot move backwards", "REGRESSION_BLOCKED")
        dependencies = {
            "legal_packet": ["spec_cleanup"],
            "playbook_v2": ["legal_packet"],
        }
        blocked = []
        for dep in dependencies.get(card["id"], []):
            dep_card = next(c for c in state["cards"] if c["id"] == dep)
            if STAGE_ORDER.index(dep_card["stage"]) < STAGE_ORDER.index("review"):
                blocked.append(dep)
        if blocked:
            return ActionOutcome(False, f"Dependency {'/'.join(blocked)} not ready", "DEPENDENCY_BLOCKED")
        changed = card_stage != new_stage
        card["stage"] = new_stage
        return ActionOutcome(changed, f"Moved {card['id']} to {new_stage}")

    if action.intent in {DeskActionType.SET_FIELD, DeskActionType.ADD_NOTE}:
        if (action.field or "").lower() != "owner":
            return ActionOutcome(False, "Only owner field can be updated", "FIELD_LOCKED")
        owner = (action.value or "").strip().lower()
        if owner not in ALLOWED_OWNERS:
            return ActionOutcome(False, "Owner not recognized", "BAD_OWNER")
        changed = (card.get("owner") or "").lower() != owner
        card["owner"] = owner
        return ActionOutcome(changed, f"Set owner for {card['id']} to {owner}")

    return ActionOutcome(False, "Unsupported intent for board", "UNSUPPORTED_INTENT")


def _board_score(state: Dict) -> GraderResult:
    total_cards = len(state["cards"])
    if total_cards == 0:
        return GraderResult(score=0.0, breakdown={}, pending=["No cards present"])
    pending: List[str] = []
    stage_hits = 0
    owner_hits = 0
    dependency_hits = 0

    for card in state["cards"]:
        expected = BOARD_EXPECTED[card["id"]]
        if card["stage"].lower() == expected["stage"]:
            stage_hits += 1
        else:
            pending.append(f"{card['id']} stage should be {expected['stage']}")
        if (card.get("owner") or "").lower() == expected["owner"]:
            owner_hits += 1
        else:
            pending.append(f"{card['id']} owner should be {expected['owner']}")
        if card["id"] == "playbook_v2":
            legal = next(c for c in state["cards"] if c["id"] == "legal_packet")
            if STAGE_ORDER.index(legal["stage"]) >= STAGE_ORDER.index("review"):
                dependency_hits += 1
        elif card["id"] == "legal_packet":
            cleanup = next(c for c in state["cards"] if c["id"] == "spec_cleanup")
            if STAGE_ORDER.index(cleanup["stage"]) >= STAGE_ORDER.index("review"):
                dependency_hits += 1
        else:
            dependency_hits += 1

    stage_alignment = stage_hits / total_cards
    owner_alignment = owner_hits / total_cards
    dependency_score = dependency_hits / total_cards
    score = _bounded(0.4 * stage_alignment + 0.4 * owner_alignment + 0.2 * dependency_score)
    breakdown = {
        "overall": score,
        "stage_alignment": stage_alignment,
        "owner_alignment": owner_alignment,
        "dependency_health": dependency_score,
    }
    return GraderResult(score=score, breakdown=breakdown, pending=pending)


# ---------------------------------------------------------------------------
# Task 4: Vendor Negotiation (Expert)
# ---------------------------------------------------------------------------


def _vendor_initial_state(rng: random.Random) -> Dict:
    vendors = []
    for idx, name in enumerate(rng.sample(VENDOR_NAMES, k=3), start=1):
        quote = rng.randint(80_000, 150_000)
        ceiling = rng.randint(90_000, 130_000)
        vendors.append(
            {
                "id": f"VN-{idx}",
                "name": name,
                "quote": quote,
                "status": "pending",
                "discount": 0,
                "stage": "evaluation",
                "budget_ceiling": ceiling,
                "notes": "",
            }
        )
    return {"vendors": vendors, "spend_cap": 280_000, "negotiation_log": []}


def _vendor_handle_action(state: Dict, action: DeskAction) -> ActionOutcome:
    if action.target_id is None:
        return ActionOutcome(False, "target_id required", "MISSING_TARGET")
    vendor = next((v for v in state["vendors"] if v["id"] == action.target_id), None)
    if vendor is None:
        return ActionOutcome(False, f"Vendor {action.target_id} missing", "UNKNOWN_VENDOR")

    if action.intent == DeskActionType.SET_FIELD:
        field = (action.field or "").lower()
        if field == "decision":
            decision = (action.value or "").strip().lower()
            if decision not in {"approve", "renegotiate", "drop"}:
                return ActionOutcome(False, "Decision must be approve|renegotiate|drop", "BAD_DECISION")
            changed = vendor["status"] != decision
            vendor["status"] = decision
            return ActionOutcome(changed, f"{vendor['id']} set to {decision}")
        if field == "discount":
            try:
                discount = int(action.value or "0")
            except ValueError:
                return ActionOutcome(False, "Discount must be int", "BAD_DISCOUNT")
            vendor["discount"] = discount
            return ActionOutcome(True, f"Discount {discount} for {vendor['id']}")

    if action.intent == DeskActionType.ADD_NOTE:
        vendor["notes"] = (action.value or "").strip()
        return ActionOutcome(True, f"Noted '{vendor['notes']}' for {vendor['id']}")

    if action.intent == DeskActionType.MOVE_STAGE:
        stage = (action.value or "").strip().lower()
        if stage not in {"evaluation", "negotiation", "approved"}:
            return ActionOutcome(False, "Invalid stage", "BAD_STAGE")
        vendor["stage"] = stage
        return ActionOutcome(True, f"{vendor['id']} moved to {stage}")

    return ActionOutcome(False, "Unsupported intent for vendor task", "UNSUPPORTED_INTENT")


def _vendor_score(state: Dict) -> GraderResult:
    approved = [v for v in state["vendors"] if v["status"] == "approve"]
    spend = sum(max(0, v["quote"] - v["discount"]) for v in approved)
    spend_ratio = spend / max(state["spend_cap"], 1)
    budget_score = 1.0 if spend <= state["spend_cap"] else max(0.0, 1.0 - (spend_ratio - 1.0))
    compliance_hits = sum(
        1
        for v in state["vendors"]
        if (v["status"] != "approve" or max(0, v["quote"] - v["discount"]) <= v["budget_ceiling"])
    )
    compliance_score = compliance_hits / len(state["vendors"])
    diversification = len({v["stage"] for v in state["vendors"]}) / 3
    overall = _bounded(0.5 * budget_score + 0.3 * compliance_score + 0.2 * diversification)
    pending = []
    if spend > state["spend_cap"]:
        pending.append("Reduce approvals or increase discounts to stay within spend cap.")
    for vendor in state["vendors"]:
        if vendor["status"] == "approve" and max(0, vendor["quote"] - vendor["discount"]) > vendor["budget_ceiling"]:
            pending.append(f"{vendor['id']} exceeds its ceiling.")
    return GraderResult(
        score=overall,
        breakdown={
            "overall": overall,
            "budget": budget_score,
            "compliance": compliance_score,
            "pipeline_diversity": diversification,
        },
        pending=pending,
    )


TASK_LIBRARY: Tuple[TaskDefinition, ...] = (
    TaskDefinition(
        task_id="inbox_triage",
        title="Inbox Cleanup",
        difficulty=TaskDifficulty.EASY,
        instructions=(
            "Assign the right handling plan for each priority email. Choices: respond, delegate, archive. "
            "Read the hint field carefully and finish within 8 steps. Submit once all statuses match."
        ),
        acceptance_criteria=[
            "Each email has a final status",
            "Status aligns with the operations runbook",
        ],
        max_steps=8,
        initial_state_factory=_inbox_initial_state,
        handle_action=_inbox_handle_action,
        score_fn=_inbox_score,
    ),
    TaskDefinition(
        task_id="calendar_allocation",
        title="Calendar Allocation",
        difficulty=TaskDifficulty.MEDIUM,
        instructions=(
            "Book time for three competing meetings without violating timing constraints. Use assign_slot and "
            "only confirm a slot after validating the notes."
        ),
        acceptance_criteria=[
            "Each request mapped to exactly one slot",
            "Slots remain conflict-free",
            "Assignments respect the constraint hints",
        ],
        max_steps=12,
        initial_state_factory=_calendar_initial_state,
        handle_action=_calendar_handle_action,
        score_fn=_calendar_score,
    ),
    TaskDefinition(
        task_id="launch_board",
        title="Launch Board Orchestration",
        difficulty=TaskDifficulty.HARD,
        instructions=(
            "Advance launch-critical cards through the kanban while locking proper owners. You may only move "
            "cards forward and owners must reflect the accountable team."
        ),
        acceptance_criteria=[
            "No backward stage moves",
            "Final stage + owner match launch plan",
        ],
        max_steps=16,
        initial_state_factory=_board_initial_state,
        handle_action=_board_handle_action,
        score_fn=_board_score,
    ),
    TaskDefinition(
        task_id="vendor_negotiation",
        title="Vendor Negotiation War Room",
        difficulty=TaskDifficulty.EXPERT,
        instructions=(
            "Approve, renegotiate, or drop vendors to stay within the spend cap. Record discounts and notes, and "
            "only approve vendors whose net quote fits their ceiling."
        ),
        acceptance_criteria=[
            "Total approved spend remains under the cap",
            "Vendors meeting ceilings are approved, others renegotiated/dropped",
            "Pipeline maintains healthy stage diversity",
        ],
        max_steps=18,
        initial_state_factory=_vendor_initial_state,
        handle_action=_vendor_handle_action,
        score_fn=_vendor_score,
    ),
)

TASK_DEFINITIONS = {task.task_id: task for task in TASK_LIBRARY}
