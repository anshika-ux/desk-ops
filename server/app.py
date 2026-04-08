"""Entry point shim required by OpenEnv validator."""

from desk_ops_env.server.app import app as fastapi_app
from desk_ops_env.server.app import main as fastapi_main

app = fastapi_app


def main() -> None:
    fastapi_main()


if __name__ == "__main__":
    main()
