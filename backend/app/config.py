from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = BACKEND_ROOT / "weights"
LOGS_DIR = BACKEND_ROOT / "logs"
EXPORTS_DIR = BACKEND_ROOT / "exports"

# Model config
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEVICE = "cuda"
INFERENCE_SIZE = 512  # pixels

# Training config
TRAIN_SAMPLES = 200
TRAIN_EPOCHS = 10
TRAIN_BATCH_SIZE = 8
TRAIN_LR = 5e-5
TRAIN_IMG_SIZE = 512

# Validation
MAX_UPLOAD_SIZE_MB = 100
MIN_IMAGE_DIM = 64
MAX_IMAGE_DIM = 8192
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
