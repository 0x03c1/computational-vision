import torch
from ultralytics import YOLO
from pathlib import Path
from rich import print
from config import OUT_DIR, MODEL_NAME, EPOCHS, BATCH, IMG_SIZE, DEVICE_PRIORITY, CLASSES

DEVICE = "cpu"
for d in DEVICE_PRIORITY:
    if d == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"; break
    if d == "cuda" and torch.cuda.is_available():
        DEVICE = "cuda"; break
print(f"[i] Usando device: {DEVICE}")

def list_dir_classes(dirpath: Path):
    return sorted([p.name for p in dirpath.iterdir() if p.is_dir()])

if __name__ == "__main__":
    data_dir = OUT_DIR
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"
    assert train_dir.exists() and val_dir.exists(), "Execute prepare_fer2013.py primeiro"

    train_classes = list_dir_classes(train_dir)
    val_classes   = list_dir_classes(val_dir)

    print(f"[i] Classes (train): {train_classes}")
    print(f"[i] Classes (val)  : {val_classes}")

    assert set(train_classes) == set(val_classes), \
        f"Conjunto de classes difere entre train e val.\ntrain={train_classes}\nval={val_classes}"
    assert set(train_classes) == set(CLASSES), \
        f"As classes detectadas não batem com CLASSES do config.py.\nDetectadas={train_classes}\nEsperadas={CLASSES}"

    for cache in data_dir.rglob("*.cache"):
        print(f"[!] Removendo cache antigo: {cache}")
        cache.unlink(missing_ok=True)

    # Treino
    model = YOLO(MODEL_NAME)
    results = model.train(
        data=str(data_dir),      
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project="runs",
        name="yolo-cls-fer2013",
        batch=BATCH,
        workers=0,                
        verbose=True
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights/best.pt"
    out = Path("models/yolo-cls"); out.mkdir(parents=True, exist_ok=True)
    dest = out / "best_fer2013.pt"
    dest.write_bytes(best.read_bytes())
    print(f"✔ Modelo salvo em: {dest}")
