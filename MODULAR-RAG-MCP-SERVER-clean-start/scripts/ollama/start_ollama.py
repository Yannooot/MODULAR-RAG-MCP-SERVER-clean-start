"""Start the local Ollama service and verify an embedding model is installed."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def get_tags(base_url: str) -> dict[str, Any] | None:
    try:
        with urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def start_service(ollama_path: str) -> None:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [ollama_path, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )


def wait_until_ready(base_url: str, timeout_seconds: int) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        tags = get_tags(base_url)
        if tags is not None:
            return tags
        time.sleep(1)
    return None


def model_is_installed(tags: dict[str, Any], model: str) -> bool:
    models = tags.get("models")
    if not isinstance(models, list):
        return False
    names = [item.get("name") for item in models if isinstance(item, dict)]
    return any(name == model or name.startswith(f"{model}:") for name in names if name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout < 1:
        print("Error: --timeout must be greater than zero.", file=sys.stderr)
        return 1

    ollama_path = shutil.which("ollama")
    if ollama_path is None:
        print("Error: Ollama is not installed or is not available in PATH.", file=sys.stderr)
        return 1

    tags = get_tags(args.base_url)
    if tags is None:
        print(f"Starting Ollama service at {args.base_url} ...")
        start_service(ollama_path)
        tags = wait_until_ready(args.base_url, args.timeout)

    if tags is None:
        print(
            f"Error: Ollama did not become ready within {args.timeout} seconds.",
            file=sys.stderr,
        )
        return 1

    print(f"Ollama service is ready at {args.base_url}.")
    if model_is_installed(tags, args.model):
        print(f"Embedding model '{args.model}' is installed.")
    else:
        print(f"Embedding model '{args.model}' is not installed.")
        print(f"Run: ollama pull {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
