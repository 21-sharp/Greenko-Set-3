import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from backend.app import app
except ImportError:
    from app import app

def handler(event, context):
    try:
        from mangum import Mangum
        return Mangum(app, lifespan="off")(event, context)
    except Exception:
        return app(event, context)
