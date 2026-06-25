"""Shared paths and constants for the SmartBidder package."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR / "models"
DATA_DIR = PACKAGE_DIR.parent / "data"

ENCODER_PATH = MODELS_DIR / "encoder.joblib"
CTR_MODEL_PATH = MODELS_DIR / "ctr_model.joblib"
CTR_CALIB_PATH = MODELS_DIR / "ctr_calib.joblib"
CVR_MODEL_PATH = MODELS_DIR / "cvr_model.joblib"
CVR_CALIB_PATH = MODELS_DIR / "cvr_calib.joblib"
WINRATE_MODEL_PATH = MODELS_DIR / "winrate_model.joblib"
WINRATE_CALIB_PATH = MODELS_DIR / "winrate_calib.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Extra feature appended (after the base feature block) for the win-rate model.
LOG_BID_FEATURE = "log_bid"
