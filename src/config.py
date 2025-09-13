from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/FER2013"
OUT_DIR = ROOT / "data/prepared"

TRAIN_PCT = 0.8
VAL_PCT = 0.1

CLASSES = [ 
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
]

MODEL_NAME = "yolov8n-cls.pt"
EPOCHS = 3
BATCH = 64
IMG_SIZE = 224

DEVICE_PRIORITY = ["mps", "cuda", "cpu"]
