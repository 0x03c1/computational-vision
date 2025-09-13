import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
from config import DEVICE_PRIORITY

MODEL_PATH = Path("models/yolo-cls/best_fer2013.pt")
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

DEVICE = 'cpu'
for d in DEVICE_PRIORITY:
    if d == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = 'mps'; break
    if d == 'cuda' and torch.cuda.is_available():
        DEVICE = 'cuda'; break

def main():
    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Não foi possível abrir a câmera')

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, 1.2, 5, minSize=(64,64))

        for (x,y,w,h) in faces:
            roi = frame[y:y+h, x:x+w]
            res = model.predict(source=roi, device=DEVICE, verbose=False)
            probs = res[0].probs
            label = res[0].names[int(probs.top1)]
            conf = float(probs.top1conf)

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow('YOLO-CLS: Expressões', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
