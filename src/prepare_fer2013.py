import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import RAW_DIR, OUT_DIR, CLASSES, TRAIN_PCT, VAL_PCT

RAW_SUBDIRS = [".", "train", "test", "val"]          # procura em todos
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def find_images_for_class(class_name: str):
    """Varre RAW_DIR e subdirs conhecidos (train/test/val) atrás de imagens daquela classe."""
    found = []
    for sub in RAW_SUBDIRS:
        base = (RAW_DIR / sub / class_name).resolve()
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    found.append(p)
    return found

def gather_class_images():
    """Coleta (path, label) para todas as classes, aceitando múltiplas origens."""
    items = []
    counts = {}
    for c in CLASSES:
        imgs = find_images_for_class(c)
        if not imgs:
            print(f"[!] Aviso: classe '{c}' não encontrada em {RAW_DIR} (ou está vazia).")
        else:
            print(f"[i] {c}: {len(imgs)} imagens")
            items += [(p, c) for p in imgs]
        counts[c] = len(imgs)
    total = sum(counts.values())
    if total == 0:
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada. Verifique se o dataset está em {RAW_DIR} "
            f"com subpastas de classes (ex.: train/{CLASSES[0]}, test/{CLASSES[0]})."
        )
    return items, counts

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "val").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "test").mkdir(parents=True, exist_ok=True)

    (items, counts) = gather_class_images()
    print("\nResumo por classe:", counts)
    paths, labels = zip(*items)

    # split estratificado
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        paths, labels, train_size=TRAIN_PCT, stratify=labels, random_state=42
    )
    val_size = VAL_PCT / (1 - TRAIN_PCT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1 - val_size, stratify=y_tmp, random_state=42
    )

    # garante diretórios de destino
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # copia, preservando nome do arquivo
    def copy_many(X, y, split):
        for src, cls in zip(X, y):
            dst = OUT_DIR / split / cls / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    copy_many(X_train, y_train, "train")
    copy_many(X_val,   y_val,   "val")
    copy_many(X_test,  y_test,  "test")

    print(f"\n✔ Dados preparados em: {OUT_DIR}")
    print("   - train/", sum(1 for _ in (OUT_DIR / 'train').rglob("*") if _.is_file()), "arquivos")
    print("   - val/  ", sum(1 for _ in (OUT_DIR / 'val').rglob("*") if _.is_file()), "arquivos")
    print("   - test/ ", sum(1 for _ in (OUT_DIR / 'test').rglob("*") if _.is_file()), "arquivos")

if __name__ == "__main__":
    main()
