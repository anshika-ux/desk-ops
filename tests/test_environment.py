from desk_ops_env.models import DeskAction, DeskActionType
from desk_ops_env.server.desk_ops_environment import DeskOpsEnvironment


def test_environment_step_and_submit():
    env = DeskOpsEnvironment()
    obs = env.reset(task_id="inbox_triage", seed=123)
    assert obs.task_id == "inbox_triage"
    first_email = obs.workspace_state["emails"][0]["id"]
    action = DeskAction(
        intent=DeskActionType.SET_FIELD,
        target_id=first_email,
        field="status",
        value="respond",
    )
    obs = env.step(action)
    assert obs.reward is not None
    assert "steps" in obs.telemetry
    submit_obs = env.step(DeskAction(intent=DeskActionType.SUBMIT))
    assert submit_obs.done is True
