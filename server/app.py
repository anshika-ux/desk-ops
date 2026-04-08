"""Entry point shim required by OpenEnv validator."""

from desk_ops_env.server.app import app, main  # re-export FastAPI app and main

__all__ = ["app", "main"]
