# Expressões Faciais usando YOLO

Este projeto demonstra como treinar e avaliar um modelo para **classificação de expressões faciais** utilizando **YOLO (Ultralytics, modo Classificação)** em cima do dataset [FER2013](https://www.kaggle.com/datasets/msambare/fer2013).

---

## 1. Preparar o ambiente

Crie um ambiente virtual e ative:

```bash
python -m venv .venv
```

Ativação no macOS/Linux:

```bash
source .venv/bin/activate
```

Ativação no Windows PowerShell:

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## 2. Preparar o dataset

Baixe o dataset **FER2013** no link acima e coloque em `data/raw/FER2013/`.
Em seguida, prepare a base de treinamento:

```bash
python src/prepare_fer2013.py
```

---

## 3. Treinar o modelo

Execute o script de treino:

```bash
python src/train_cls.py
```

Isso irá salvar os pesos treinados em `models/yolo-cls/best_fer2013.pt`.

---

## 4. Avaliar o modelo

Para gerar a matriz de confusão e calcular a acurácia por classe:

```bash
python src/evaluate_cls.py
```

---

## 5. Inferência em tempo real

Para executar a inferência utilizando a webcam:

```bash
python src/infer_video.py
```

---

## Estrutura do projeto

* `src/prepare_fer2013.py`: prepara o dataset FER2013 em `train/val/test`.
* `src/train_cls.py`: treino do modelo YOLO em modo classificação.
* `src/evaluate_cls.py`: avaliação do modelo (matriz de confusão e métricas).
* `src/infer_video.py`: inferência em tempo real usando webcam.
