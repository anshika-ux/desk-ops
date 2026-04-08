"""FastAPI entrypoint for the DeskOps environment."""

try:
    from openenv.core.env_server import create_app
except ImportError:  # pragma: no cover
    def create_app(*_, **__):  # type: ignore
        raise ImportError("openenv is required to create the DeskOps FastAPI app")

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..models import DeskAction, DeskObservation
from .desk_ops_environment import DeskOpsEnvironment

app = create_app(
    DeskOpsEnvironment,
    DeskAction,
    DeskObservation,
    env_name="desk_ops_env",
)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"message": "DeskOps OpenEnv is running."})


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
