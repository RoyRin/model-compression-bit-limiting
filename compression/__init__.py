from pathlib import Path
from decimal import Decimal

#   BASE_DIR = Path(__file__).parent.parent
BASE_DIR = Path("/workspace/code/model-compression-bit-limiting")
DEBUG = False
ZERO = Decimal(0)
ONE = Decimal(1)
# Debug flag


def debug_print(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")


models = {
    "70m": "EleutherAI/pythia-70m",
    "160m": "EleutherAI/pythia-160m",
    "410m": "EleutherAI/pythia-410m",
    "1b": "EleutherAI/pythia-1b",
    "1.4b": "EleutherAI/pythia-1.4b",
    "2.8b": "EleutherAI/pythia-2.8b",
    "6.9b": "EleutherAI/pythia-6.9b",
}
