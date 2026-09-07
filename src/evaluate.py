import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import argparse
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from src.model import (
    BiGRU_Head,
    DualStream_Head,
    TriStream_Kinetic,
    DualStream_MeanMax
)

def evaluate_model(model_type="ensemble", threshold=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carregar features pré-extraídas do conjunto de teste cego (185 vídeos)
    cache_path = "data/cache/features_test.pt"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache de teste não encontrado em: {cache_path}")
        
    test_feats, test_labels = torch.load(cache_path, map_location=device)
    y_true = test_labels.numpy()
    
    model_type = model_type.lower()
    
    if model_type == "baseline":
        model = BiGRU_Head().to(device)
        sd = torch.load("models/best_model.pth", map_location=device)
        head_sd = {k: v for k, v in sd.items() if not k.startswith("features.")}
        model.load_state_dict(head_sd)
        model.eval()
        th = threshold if threshold is not None else 0.50
        model_name = "Modelo 1: Baseline Minimalista (Bi-GRU)"
        with torch.no_grad():
            logits = model(test_feats)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            
    elif model_type == "dualstream":
        model = DualStream_Head().to(device)
        candidate_paths = [
            "models/model_dualstream_best.pth",
            "experiments/model_dualstream_best.pth",
            "legacy_experiments/model_dualstream_best.pth"
        ]
        path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if not path:
            raise FileNotFoundError("Pesos do DualStream_Head não encontrados em models/ ou experiments/")
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        th = threshold if threshold is not None else 0.50
        model_name = "Modelo 2: Dual-Stream Latente (Aparência + Velocidade Δf)"
        with torch.no_grad():
            logits = model(test_feats)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            
    elif model_type in ["ensemble", "tristream", "kinetic"]:
        def resolve_path(filename, subfolders):
            for sub in subfolders:
                full = os.path.join(sub, filename)
                if os.path.exists(full):
                    return full
            raise FileNotFoundError(f"Pesos não encontrados para: {filename}")

        subfolders = ["models/ensemble", "experiments/target_85", "legacy_experiments/target_85"]
        m1 = TriStream_Kinetic().to(device)
        m1.load_state_dict(torch.load(resolve_path("model_tristream_s7.pth", subfolders), map_location=device))
        m1.eval()
        
        m2 = DualStream_MeanMax().to(device)
        m2.load_state_dict(torch.load(resolve_path("model_dualmeanmax_s5.pth", subfolders), map_location=device))
        m2.eval()
        
        m3 = DualStream_MeanMax().to(device)
        m3.load_state_dict(torch.load(resolve_path("model_dualmeanmax_s10.pth", subfolders), map_location=device))
        m3.eval()
        
        th = threshold if threshold is not None else 0.52
        model_name = "Modelo 3: Ensemble Cinético Tri-Stream (1x TriStream + 2x DualStream)"
        with torch.no_grad():
            p1 = torch.softmax(m1(test_feats), dim=1)[:, 1].numpy()
            p2 = torch.softmax(m2(test_feats), dim=1)[:, 1].numpy()
            p3 = torch.softmax(m3(test_feats), dim=1)[:, 1].numpy()
        probs = (p1 + p2 + p3) / 3.0
    else:
        raise ValueError(f"Modelo inválido: '{model_type}'. Use: baseline, dualstream, ensemble.")

    preds = (probs >= th).astype(int)
    acc = accuracy_score(y_true, preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary")
    prec *= 100
    rec *= 100
    f1 *= 100
    auc = roc_auc_score(y_true, probs) * 100
    cm = confusion_matrix(y_true, preds)
    hits = int(round(acc * len(y_true) / 100))

    print("\n" + "="*68)
    print(f"AVALIAÇÃO OFICIAL NO TESTE CEGO — {model_name}")
    print("="*68)
    print(f"Total de Vídeos Avaliados: {len(y_true)} (Fight: {sum(y_true==1)}, NonFight: {sum(y_true==0)})")
    print(f"Limiar de Decisão (θ):     {th:.2f}")
    print(f"Total de Acertos:          {hits} / {len(y_true)}")
    print(f"Acurácia Global (Acc):     {acc:.2f}%")
    print(f"Sensibilidade (Recall):    {rec:.2f}%")
    print(f"Precisão (Precision):      {prec:.2f}%")
    print(f"F1-Score:                  {f1:.2f}%")
    print(f"AUC-ROC:                   {auc:.2f}%")
    print("\nMatriz de Confusão:")
    print(f"  [TN={cm[0,0]:2d}  FP={cm[0,1]:2d}]  (NonFight: {cm[0].sum()})")
    print(f"  [FN={cm[1,0]:2d}  TP={cm[1,1]:2d}]  (Fight:    {cm[1].sum()})")
    print("="*68)
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_true, preds, target_names=["NonFight", "Fight"]))
    print("="*68 + "\n")

    return {
        "model_name": model_name,
        "threshold": th,
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1_score": f1,
        "auc_roc": auc,
        "confusion_matrix": cm.tolist()
    }

def main():
    parser = argparse.ArgumentParser(description="Avaliação no conjunto de teste cego anti-leakage")
    parser.add_argument("--model", type=str, default="ensemble", choices=["ensemble", "dualstream", "baseline"],
                        help="Modelo a avaliar: 'ensemble' (85.41% SOTA), 'dualstream' (82.16%) ou 'baseline' (75.14%)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Limiar de probabilidade Fight (padrão: 0.52 para ensemble, 0.50 para outros)")
    args = parser.parse_args()

    evaluate_model(model_type=args.model, threshold=args.threshold)

if __name__ == "__main__":
    main()
