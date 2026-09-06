import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import DataLoader

from src.dataset import RWF2000Dataset
from src.model import VideoClassifier

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Avaliacao no conjunto de teste cego")
    parser.add_argument("--threshold", type=float, default=0.50, help="Limiar de probabilidade para classificar como Fight (padrao: 0.50)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando avaliacao no dispositivo: {device} | Limiar (th): {args.threshold:.2f}")

    # 1. Carrega o modelo com os melhores pesos
    model = VideoClassifier(num_classes=2).to(device)
    checkpoint_path = "models/best_model.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pesos nao encontrados em {checkpoint_path}. Execute src/train.py primeiro.")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. Dataset e DataLoader do conjunto de TESTE cego
    test_ds = RWF2000Dataset("data/splits/test.csv", num_frames=16, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    all_targets = []
    all_probs = []

    print(f"Avaliando {len(test_ds)} videos de teste...")
    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(device)
            logits = model(videos)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_preds = (all_probs >= args.threshold).astype(int)

    # 3. Calculo das metricas obrigatorias
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print("\n================ RELATORIO DE AVALIACAO (TESTE) ================")
    print(f"Limiar de Decisao (th): {args.threshold:.2f}")
    print(f"Acuracia (Accuracy):    {acc * 100:.2f}%")
    print(f"Precisao (Precision):   {precision * 100:.2f}%")
    print(f"Revogacao (Recall):     {recall * 100:.2f}%")
    print(f"F1-Score:               {f1 * 100:.2f}%")
    print("\nMatriz de Confusao:")
    print(cm)
    print("\nDetalhamento por Classe:")
    print(classification_report(all_targets, all_preds, target_names=["NonFight", "Fight"], zero_division=0))
    print("=================================================================\n")

    # 4. Salva metricas em JSON
    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }
    with open("reports/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 5. Gera e salva imagem da Matriz de Confusao
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Matriz de Confusao - Conjunto de Teste")
    plt.colorbar()
    classes = ["NonFight", "Fight"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Rotulo Real")
    plt.xlabel("Rotulo Previsto")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=300)
    print("Matriz de confusao salva em 'reports/confusion_matrix.png'")

if __name__ == "__main__":
    main()

