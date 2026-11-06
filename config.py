from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "fuel", "retail", "travel",
    "electronics", "utilities", "entertainment", "gift_cards", "crypto",
]
CHANNELS = ["pos", "web", "mobile", "atm"]
COUNTRIES = ["US", "GB", "DE", "IN", "BR", "NG", "RU", "CA"]

CATEGORY_CODES = {name: i for i, name in enumerate(MERCHANT_CATEGORIES)}
CHANNEL_CODES = {name: i for i, name in enumerate(CHANNELS)}

VELOCITY_WINDOW = 3600            # seconds
DEFAULT_AVG_AMOUNT = 60.0         # used for users we haven't seen yet
MAX_GAP_SECONDS = 30 * 24 * 3600
RATIO_CAP = 50.0

RANDOM_STATE = 42
MODEL_VERSION = "rf-v1"
