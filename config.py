import os
from pathlib import Path

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SECRET_KEY        = os.environ.get("SECRET_KEY", "haven-dev-secret")
PORT              = int(os.environ.get("PORT", 5000))
DEBUG             = os.environ.get("DEBUG", "true").lower() == "true"
VIDEO_ENABLED     = os.environ.get("VIDEO_ENABLED", "false") == "true"

if not ANTHROPIC_API_KEY:
    print("[Haven] WARNING: ANTHROPIC_API_KEY not set. Remi will use fallback responses.")
