import random

import pytest

from desk_ops_env.tasks import TASK_DEFINITIONS


def test_inbox_score_reaches_one():
    task = TASK_DEFINITIONS["inbox_triage"]
    rng = random.Random(0)
    state = task.initial_state_factory(rng)
    expected_map = {"EM-101": "respond", "EM-102": "delegate", "EM-103": "archive"}
    for email in state["emails"]:
        email_id = email["id"]
        email["status"] = expected_map[email_id]
    sorted_emails = sorted(state["emails"], key=lambda e: e["priority"], reverse=True)
    state["action_history"] = [{"email": e["id"], "priority": e["priority"]} for e in sorted_emails]
    result = task.score_fn(state)
    assert result.score == pytest.approx(1.0)


def test_calendar_allocation_partial_credit():
    task = TASK_DEFINITIONS["calendar_allocation"]
    state = task.initial_state_factory(random.Random(1))
    req = state["requests"][0]
    req["slot"] = "Thu-1400"
    state["slots"][-1]["booked"] = True
    result = task.score_fn(state)
    assert 0 < result.score < 1


def test_vendor_budget_penalty():
    task = TASK_DEFINITIONS["vendor_negotiation"]
    state = task.initial_state_factory(random.Random(2))
    for vendor in state["vendors"]:
        vendor["status"] = "approve"
        vendor["discount"] = 0
    result = task.score_fn(state)
    assert result.breakdown["budget"] < 1.0
