import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
from config import OUT_DIR, CLASSES

MODEL_PATH = Path("models/yolo-cls/best_fer2013.pt")

def plot_cm(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

if __name__ == '__main__':
    model = YOLO(str(MODEL_PATH))
    test_dir = OUT_DIR / 'test'
    y_true, y_pred = [], []

    for ci, cls in enumerate(CLASSES):
        for img in (test_dir / cls).glob('*'):
            res = model.predict(source=str(img), verbose=False)
            # top-1 da primeira predição
            probs = res[0].probs
            pred_idx = int(probs.top1)
            y_true.append(ci)
            y_pred.append(pred_idx)

    cm = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    plot_cm(cm, CLASSES, normalize=False, title='Confusion Matrix (counts)')
    plt.show()
