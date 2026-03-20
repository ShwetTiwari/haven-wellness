import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from backend.app import app

if __name__ == "__main__":
    from config import PORT, DEBUG
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)