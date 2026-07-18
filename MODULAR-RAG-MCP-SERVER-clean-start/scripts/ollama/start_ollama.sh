#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"

cd "$project_root"

if [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
elif [[ -f ".venv/Scripts/activate" ]]; then
    source ".venv/Scripts/activate"
else
    echo "Error: virtual environment .venv was not found." >&2
    exit 1
fi

python "scripts/ollama/start_ollama.py" "$@"
