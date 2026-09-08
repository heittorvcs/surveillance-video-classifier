"""
Avaliacao sobre um split, a partir das features cacheadas do backbone congelado.

Pre-requisito: python src/build_cache.py

Uso:
    python src/evaluate.py --model ensemble
    python src/evaluate.py --model dualstream --split val
    python src/evaluate.py --model ensemble --threshold 0.50

Sobre o limiar do ensemble: o valor padrao 0.52 foi originalmente escolhido
observando o proprio split de teste, o que caracteriza selecao de modelo com
informacao do teste. Para uma estimativa nao enviesada, recalibre na validacao
com src/calibrate_threshold.py e passe o valor obtido em --threshold.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import os

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.model import (
    DEFAULT_THRESHOLDS,
    BiGRU_Head,
    DualStream_Head,
    load_ensemble_head,
)

CACHE_DIR = "data/cache"

MODEL_NAMES = {
    "baseline": "Modelo 1: Baseline Minimalista (Bi-GRU)",
    "dualstream": "Modelo 2: Dual-Stream Latente (Aparencia + Velocidade)",
    "ensemble": "Modelo 3: Ensemble Cinetico Tri-Stream (1x TriStream + 2x DualStream)",
}

BASELINE_WEIGHTS = ["models/best_model.pth", "models/baseline_seed42.pth"]
DUALSTREAM_WEIGHTS = [
    "models/model_dualstream_best.pth",
    "models/dualstream_seed42.pth",
    "experiments/model_dualstream_best.pth",
    "legacy_experiments/model_dualstream_best.pth",
]


def load_features(split, device):
    path = os.path.join(CACHE_DIR, f"features_{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache de features ausente: {path}\n"
            f"Gere-o com: python src/build_cache.py --splits {split}"
        )
    feats, labels = torch.load(path, map_location="cpu")
    return feats.to(device), labels


def load_head(model, candidates, device):
    """
    Carrega pesos de cabeca a partir do primeiro caminho existente.

    Aceita tanto checkpoints de cabeca isolada quanto checkpoints do modelo
    completo (nesse caso descarta as chaves do backbone congelado).
    """
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"Pesos nao encontrados em: {', '.join(candidates)}")

    state = torch.load(path, map_location=device)
    head_state = {k: v for k, v in state.items() if not k.startswith("features.")}
    model.load_state_dict(head_state)
    model.eval()
    return path


@torch.no_grad()
def predict_probs(model_type, feats, device):
    """Probabilidade da classe Fight para cada video, como numpy array."""
    if model_type == "baseline":
        model = BiGRU_Head().to(device)
        weights = load_head(model, BASELINE_WEIGHTS, device)
        probs = torch.softmax(model(feats), dim=1)[:, 1]

    elif model_type == "dualstream":
        model = DualStream_Head().to(device)
        weights = load_head(model, DUALSTREAM_WEIGHTS, device)
        probs = torch.softmax(model(feats), dim=1)[:, 1]

    else:
        model = load_ensemble_head(device=device)
        weights = "models/ensemble/ (3 membros)"
        probs = model(feats)[:, 1]

    return probs.detach().cpu().numpy(), weights


def evaluate_model(model_type="ensemble", threshold=None, split="test", save_json=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_type = model_type.lower()
    if model_type not in MODEL_NAMES:
        raise ValueError(f"Modelo invalido: '{model_type}'. Use: {', '.join(MODEL_NAMES)}.")

    feats, labels = load_features(split, device)
    y_true = labels.detach().cpu().numpy()

    probs, weights = predict_probs(model_type, feats, device)
    th = threshold if threshold is not None else DEFAULT_THRESHOLDS[model_type]
    preds = (probs >= th).astype(int)

    acc = accuracy_score(y_true, preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    prec, rec, f1 = prec * 100, rec * 100, f1 * 100
    auc = roc_auc_score(y_true, probs) * 100
    cm = confusion_matrix(y_true, preds)
    hits = int((preds == y_true).sum())

    split_label = {"test": "TESTE CEGO", "val": "VALIDACAO", "train": "TREINO"}[split]

    print("\n" + "=" * 68)
    print(f"AVALIACAO NO {split_label} - {MODEL_NAMES[model_type]}")
    print("=" * 68)
    print(f"Pesos:                     {weights}")
    print(f"Videos avaliados:          {len(y_true)} (Fight: {int((y_true == 1).sum())}, "
          f"NonFight: {int((y_true == 0).sum())})")
    print(f"Limiar de decisao:         {th:.2f}")
    print(f"Acertos:                   {hits} / {len(y_true)}")
    print(f"Acuracia:                  {acc:.2f}%")
    print(f"Recall (Fight):            {rec:.2f}%")
    print(f"Precisao (Fight):          {prec:.2f}%")
    print(f"F1-Score:                  {f1:.2f}%")
    print(f"AUC-ROC:                   {auc:.2f}%")
    print("\nMatriz de Confusao:")
    print(f"  [TN={cm[0, 0]:3d}  FP={cm[0, 1]:3d}]  (NonFight: {cm[0].sum()})")
    print(f"  [FN={cm[1, 0]:3d}  TP={cm[1, 1]:3d}]  (Fight:    {cm[1].sum()})")
    print("=" * 68)
    print("\nRelatorio de Classificacao:")
    print(classification_report(y_true, preds, target_names=["NonFight", "Fight"], zero_division=0))

    if model_type == "ensemble" and threshold is None and split == "test":
        print("NOTA: este limiar (0.52) e o trio de sementes entregue foram selecionados")
        print("      observando o split de teste. Trate 85.41% como melhor checkpoint,")
        print("      nao como estimativa nao enviesada — ver README, secao 5.")
    print("=" * 68 + "\n")

    result = {
        "model": model_type,
        "model_name": MODEL_NAMES[model_type],
        "split": split,
        "threshold": th,
        "num_videos": int(len(y_true)),
        "hits": hits,
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1_score": f1,
        "auc_roc": auc,
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
    }

    if save_json:
        os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Metricas salvas em: {save_json}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Avaliacao sobre features cacheadas")
    parser.add_argument("--model", type=str, default="ensemble",
                        choices=["ensemble", "dualstream", "baseline"], help="Modelo a avaliar")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"],
                        help="Split a avaliar (padrao: test)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Limiar de probabilidade Fight (padrao: ver DEFAULT_THRESHOLDS em src/model.py)")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Caminho para salvar as metricas em JSON")
    args = parser.parse_args()

    evaluate_model(model_type=args.model, threshold=args.threshold,
                   split=args.split, save_json=args.save_json)


if __name__ == "__main__":
    main()
