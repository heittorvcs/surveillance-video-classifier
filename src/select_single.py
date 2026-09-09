"""
Seleciona a semente e o limiar de um modelo individual usando APENAS a validacao.

Serve ao mesmo proposito de src/select_ensemble.py, mas para os Modelos 1 e 2.
Os tres precisam passar pelo mesmo criterio de selecao: se um for escolhido de um
jeito e outro de outro, a comparacao entre eles mede o procedimento de selecao, e
nao a arquitetura.

Uso:
    python src/select_single.py --arch baseline   --export
    python src/select_single.py --arch dualstream --export --eval_test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch

from src.calibrate_threshold import pick, sweep
from src.evaluate import evaluate_model, load_features
from src.model import build_head

# Onde src/evaluate.py procura os pesos de cada modelo individual.
EXPORT_TARGETS = {
    "baseline": "models/best_model.pth",
    "dualstream": "models/model_dualstream_best.pth",
}


@torch.no_grad()
def candidate_probs(arch, paths, feats, device):
    out = {}
    for path in sorted(paths):
        model, _ = build_head(arch)
        model = model.to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        out[path] = torch.softmax(model(feats), dim=1)[:, 1].detach().cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser(description="Selecao de semente e limiar na validacao")
    parser.add_argument("--arch", type=str, required=True, choices=["baseline", "dualstream"])
    parser.add_argument("--pool_glob", type=str, default=None,
                        help="Padrao dos candidatos (padrao: models/pool/<arch>_s*.pth)")
    parser.add_argument("--criterion", type=str, default="recall_at_precision",
                        choices=["f1", "recall_at_precision", "youden"])
    parser.add_argument("--min_precision", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--fixed_threshold", type=float, default=None,
                        help="Fixa o limiar em vez de varre-lo. Selecionar semente E limiar sobre 215 videos superajusta a validacao; com o limiar fixo, a selecao mede a arquitetura e nao o ponto de operacao.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--export", action="store_true",
                        help="Copia a semente escolhida para o caminho lido por src/evaluate.py")
    parser.add_argument("--eval_test", action="store_true",
                        help="Avalia o teste UMA vez com a semente e o limiar escolhidos")
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    pool_glob = args.pool_glob or f"models/pool/{args.arch}_s*.pth"
    paths = sorted(glob.glob(pool_glob))
    if not paths:
        raise FileNotFoundError(
            f"Nenhum candidato em '{pool_glob}'.\n"
            f"Treine o pool: for s in 1..20; "
            f"python src/train.py --arch {args.arch} --seed $s --out models/pool/{args.arch}_s$s.pth"
        )

    val_feats, val_labels = load_features("val", device)
    y_val = val_labels.detach().cpu().numpy()
    probs = candidate_probs(args.arch, paths, val_feats, device)

    thresholds = (np.array([args.fixed_threshold]) if args.fixed_threshold is not None
                  else np.arange(0.05, 0.95 + args.step / 2, args.step))
    key = {"f1": "f1", "recall_at_precision": "recall", "youden": "youden"}[args.criterion]

    results = []
    for path, p in probs.items():
        best = pick(sweep(y_val, p, thresholds), args.criterion, args.min_precision)
        results.append({"checkpoint": path, **best})
    results.sort(key=lambda r: (r[key], r["f1"]), reverse=True)
    champion = results[0]

    accs = np.array([r["accuracy"] for r in results]) * 100

    print("\n" + "=" * 78)
    print(f"SELECAO NA VALIDACAO - {args.arch} ({len(paths)} candidatos, criterio {args.criterion})")
    print("=" * 78)
    for i, r in enumerate(results[:args.top_k], start=1):
        print(f"{i:2d}. theta={r['threshold']:.2f} | prec {r['precision']*100:5.2f}% | "
              f"rec {r['recall']*100:5.2f}% | F1 {r['f1']*100:5.2f}% | "
              f"acc {r['accuracy']*100:5.2f}% | {Path(r['checkpoint']).stem}")
    print("-" * 78)
    print(f"Distribuicao na validacao: {accs.mean():.2f}% +- {accs.std(ddof=1):.2f}% "
          f"[{accs.min():.2f}% - {accs.max():.2f}%]")
    print(f"ESCOLHIDO: {champion['checkpoint']} com theta = {champion['threshold']:.2f}")
    print("=" * 78 + "\n")

    payload = {
        "arch": args.arch,
        "criterion": args.criterion,
        "min_precision": args.min_precision,
        "selected_on": "val",
        "num_candidates": len(paths),
        "champion": champion,
        "validation_distribution": {
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std(ddof=1)),
            "min_accuracy": float(accs.min()),
            "max_accuracy": float(accs.max()),
        },
        "ranking": results[:args.top_k],
    }

    if args.export:
        target = EXPORT_TARGETS[args.arch]
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        shutil.copyfile(champion["checkpoint"], target)
        print(f"[export] {champion['checkpoint']} -> {target}")
        print(f"Atualize DEFAULT_THRESHOLDS['{args.arch}'] em src/model.py para "
              f"{champion['threshold']:.2f}.\n")

    if args.eval_test:
        print("Avaliando o teste cego uma unica vez com a semente e o limiar da validacao...")
        payload["test_metrics"] = evaluate_model(
            model_type=args.arch, threshold=champion["threshold"], split="test"
        )

    out = args.save_json or f"reports/selection_{args.arch}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Resultado salvo em: {out}")


if __name__ == "__main__":
    main()
