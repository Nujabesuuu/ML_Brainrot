from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_CSV = DATA_DIR / "index.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "brainrot_mobilenetv2.keras"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# every image is stored as a 256x256 JPEG, the model sees a 224x224 crop
STORED_SIZE = 256
IMG_SIZE = 224
JPEG_QUALITY = 90

CLASS_NAMES = ["not_brainrot", "brainrot"]

# how many images we take from each source (class 0 = 5000, class 1 = 5000)
N_AI_NORMAL = 500
N_PHOTO_HF = 500
N_PHOTO_IMAGENETTE = 4000
