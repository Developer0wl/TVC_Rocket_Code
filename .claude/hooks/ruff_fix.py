"""PostToolUse hook: run ruff (fix + format) on a Python file just edited by Claude."""

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    path = (payload.get("tool_input") or {}).get("file_path", "")
    if not path.endswith(".py") or not Path(path).exists():
        return 0
    root = Path(__file__).resolve().parents[2]
    ruff = root / ".venv" / "Scripts" / "ruff.exe"
    if not ruff.exists():
        return 0
    subprocess.run([str(ruff), "check", "--fix", "--quiet", path], cwd=root)
    subprocess.run([str(ruff), "format", "--quiet", path], cwd=root)
    # Surface remaining lint errors to Claude via exit code 2 + stderr.
    res = subprocess.run([str(ruff), "check", path], cwd=root, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
