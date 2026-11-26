#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY_BIN="python3"
if ! command -v python3 >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PY_BIN="python"
    else
        echo "[Error] Python 3.10+ was not found in PATH."
        exit 1
    fi
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "[System] Initializing centralized AI Environment. This will take a few minutes..."
    "$PY_BIN" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip --quiet
fi

if ! .venv/bin/python -c "import hashlib, pathlib, sys; cur=hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest(); st=pathlib.Path('.venv/.requirements.sha256'); sys.exit(0 if st.exists() and st.read_text(encoding='utf-8').strip()==cur else 1)"; then
    echo "[System] Installing/updating dependencies from requirements.txt..."
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -c "import hashlib, pathlib; pathlib.Path('.venv/.requirements.sha256').write_text(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest(), encoding='utf-8')"
fi

exec .venv/bin/python web_app.py "$@"
