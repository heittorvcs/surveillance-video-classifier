"""
Calibra o limiar de decisao sobre o split de VALIDACAO.

Motivacao: o limiar 0.52 usado no ensemble foi originalmente escolhido observando
o split de teste. Isso torna a metrica reportada uma estimativa otimista, porque o
teste deixa de ser cego para a etapa de selecao. Este script move a calibracao para
a validacao (215 videos) e so entao, opcionalmente, toca o teste uma unica vez.

Criterios disponiveis:
    f1                   maximiza F1 da classe Fight
    recall_at_precision  maximiza recall sujeito a precisao >= --min_precision
    youden               maximiza (sensibilidade + especificidade - 1)

Uso:
    python src/calibrate_threshold.py --model ensemble
    python src/calibrate_threshold.py --model ensemble --criterion recall_at_precision --min_precision 0.80
    python src/calibrate_threshold.py --model ensemble --eval_test
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

import numpy as np
import torch

from src.evaluate import evaluate_model, load_features, predict_probs


def sweep(y_true, probs, thresholds):
    """
    Metricas da classe Fight para cada limiar candidato.

    Vetorizado: a busca de ensembles avalia milhares de trios, e chamar o sklearn
    uma vez por limiar tornava o script inutilizavel na pratica (centenas de
    milhares de chamadas). Aqui a matriz de confusao dos m limiares e calculada
    de uma vez sobre as n amostras.
    """
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(probs, dtype=float)
    th = np.asarray(thresholds, dtype=float)

    preds = p[None, :] >= th[:, None]           # (m, n)
    tp = (preds & y[None, :]).sum(axis=1)
    fp = (preds & ~y[None, :]).sum(axis=1)
    fn = (~preds & y[None, :]).sum(axis=1)
    tn = (~preds & ~y[None, :]).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
        specificity = np.where(tn + fp > 0, tn / np.maximum(tn + fp, 1), 0.0)

    accuracy = (tp + tn) / len(y)

    return [{
        "threshold": float(th[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "f1": float(f1[i]),
        "youden": float(recall[i] + specificity[i] - 1.0),
        "accuracy": float(accuracy[i]),
        "false_negative": int(fn[i]),
        "false_positive": int(fp[i]),
    } for i in range(len(th))]


def pick(rows, criterion, min_precision):
    if criterion == "f1":
        return max(rows, key=lambda r: (r["f1"], r["recall"]))

    if criterion == "youden":
        return max(rows, key=lambda r: (r["youden"], r["recall"]))

    feasible = [r for r in rows if r["precision"] >= min_precision]
    if not feasible:
        best_prec = max(rows, key=lambda r: r["precision"])
        print(f"AVISO: nenhum limiar atinge precisao >= {min_precision:.2f} na validacao "
              f"(maxima observada: {best_prec['precision']:.4f}). Caindo para o criterio F1.")
        return max(rows, key=lambda r: (r["f1"], r["recall"]))

    return max(feasible, key=lambda r: (r["recall"], r["f1"]))


def main():
    parser = argparse.ArgumentParser(description="Calibracao do limiar sobre o split de validacao")
    parser.add_argument("--model", type=str, default="ensemble",
                        choices=["ensemble", "dualstream", "baseline"])
    parser.add_argument("--criterion", type=str, default="recall_at_precision",
                        choices=["f1", "recall_at_precision", "youden"])
    parser.add_argument("--min_precision", type=float, default=0.80,
                        help="Precisao minima exigida no criterio recall_at_precision")
    parser.add_argument("--step", type=float, default=0.01, help="Passo da varredura de limiares")
    parser.add_argument("--eval_test", action="store_true",
                        help="Apos calibrar, avalia o teste UMA vez com o limiar escolhido")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Caminho para salvar o resultado da calibracao")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_feats, val_labels = load_features("val", device)
    y_true = val_labels.detach().cpu().numpy()
    probs, weights = predict_probs(args.model, val_feats, device)

    thresholds = np.arange(0.05, 0.95 + args.step / 2, args.step)
    rows = sweep(y_true, probs, thresholds)
    best = pick(rows, args.criterion, args.min_precision)

    print("\n" + "=" * 72)
    print(f"CALIBRACAO DE LIMIAR NA VALIDACAO - modelo '{args.model}'")
    print("=" * 72)
    print(f"Pesos:            {weights}")
    print(f"Videos:           {len(y_true)} (Fight: {int((y_true == 1).sum())}, "
          f"NonFight: {int((y_true == 0).sum())})")
    print(f"Criterio:         {args.criterion}"
          + (f" (precisao minima {args.min_precision:.2f})" if args.criterion == "recall_at_precision" else ""))
    print("-" * 72)
    print(f"{'theta':>7} {'prec':>8} {'recall':>8} {'F1':>8} {'acc':>8} {'FN':>5} {'FP':>5}")
    for r in rows:
        if abs(r["threshold"] * 100 - round(r["threshold"] * 100)) < 1e-6 and round(r["threshold"] * 100) % 5 == 0:
            marker = "  <-- escolhido" if abs(r["threshold"] - best["threshold"]) < 1e-9 else ""
            print(f"{r['threshold']:7.2f} {r['precision']:8.4f} {r['recall']:8.4f} "
                  f"{r['f1']:8.4f} {r['accuracy']:8.4f} {r['false_negative']:5d} "
                  f"{r['false_positive']:5d}{marker}")
    print("-" * 72)
    print(f"LIMIAR CALIBRADO: theta = {best['threshold']:.2f}")
    print(f"  Na validacao -> precisao {best['precision']*100:.2f}% | recall {best['recall']*100:.2f}% | "
          f"F1 {best['f1']*100:.2f}% | acuracia {best['accuracy']*100:.2f}% | "
          f"FN {best['false_negative']} | FP {best['false_positive']}")
    print("=" * 72 + "\n")

    result = {
        "model": args.model,
        "criterion": args.criterion,
        "min_precision": args.min_precision,
        "calibrated_on": "val",
        "threshold": best["threshold"],
        "validation_metrics": best,
        "sweep": rows,
    }

    if args.eval_test:
        print("Avaliando o teste cego uma unica vez com o limiar calibrado na validacao...")
        result["test_metrics"] = evaluate_model(
            model_type=args.model, threshold=best["threshold"], split="test"
        )

    out = args.save_json or f"reports/threshold_calibration_{args.model}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Resultado salvo em: {out}")


if __name__ == "__main__":
    main()
