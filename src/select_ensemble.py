"""
Seleciona os membros do ensemble e o limiar usando APENAS o split de validacao.

Este script existe para corrigir o vies de selecao do checkpoint campeao original:
o trio de sementes (TriStream s7 + DualMeanMax s5 + DualMeanMax s10) e o limiar 0.52
foram escolhidos observando o split de teste, o que torna 85.41% um melhor-de-N
medido no proprio conjunto de avaliacao, e nao uma estimativa nao enviesada.

Fluxo correto:
    1) Treinar um pool de candidatos (o teste nunca e consultado):
         for s in 1..20:
           python src/train.py --arch tristream   --seed $s --out models/pool/tristream_s$s.pth
           python src/train.py --arch dualmeanmax --seed $s --out models/pool/dualmeanmax_s$s.pth

    2) Selecionar trio + limiar na validacao:
         python src/select_ensemble.py --export

    3) Avaliar o teste UMA unica vez com o que foi escolhido:
         python src/select_ensemble.py --eval_test

Uso:
    python src/select_ensemble.py
    python src/select_ensemble.py --criterion recall_at_precision --min_precision 0.80
    python src/select_ensemble.py --export --eval_test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import glob
import itertools
import json
import os
import shutil

import numpy as np
import torch

from src.calibrate_threshold import pick, sweep
from src.evaluate import evaluate_model, load_features
from src.model import ENSEMBLE_MEMBER_FILES, DualStream_MeanMax, TriStream_Kinetic

DEFAULT_TRISTREAM_GLOB = "models/pool/tristream_s*.pth"
DEFAULT_DUALSTREAM_GLOB = "models/pool/dualmeanmax_s*.pth"


@torch.no_grad()
def member_probs(cls, paths, feats, device):
    """Probabilidade de Fight na validacao para cada checkpoint do pool."""
    out = {}
    for path in sorted(paths):
        model = cls().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        out[path] = torch.softmax(model(feats), dim=1)[:, 1].detach().cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser(description="Selecao de membros e limiar do ensemble na validacao")
    parser.add_argument("--tristream_glob", type=str, default=DEFAULT_TRISTREAM_GLOB)
    parser.add_argument("--dualstream_glob", type=str, default=DEFAULT_DUALSTREAM_GLOB)
    parser.add_argument("--criterion", type=str, default="recall_at_precision",
                        choices=["f1", "recall_at_precision", "youden"])
    parser.add_argument("--min_precision", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--fixed_threshold", type=float, default=None,
                        help="Fixa o limiar em vez de varre-lo. Selecionar semente E limiar sobre 215 videos superajusta a validacao; com o limiar fixo, a selecao mede a arquitetura e nao o ponto de operacao.")
    parser.add_argument("--top_k", type=int, default=10, help="Quantos trios listar no ranking")
    parser.add_argument("--export", action="store_true",
                        help="Copia o trio escolhido para models/ensemble/ com os nomes esperados")
    parser.add_argument("--eval_test", action="store_true",
                        help="Avalia o teste UMA vez com o trio e o limiar escolhidos")
    parser.add_argument("--save_json", type=str, default="reports/ensemble_selection.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tri_paths = glob.glob(args.tristream_glob)
    dual_paths = glob.glob(args.dualstream_glob)
    if not tri_paths or len(dual_paths) < 2:
        raise FileNotFoundError(
            "Pool de candidatos insuficiente.\n"
            f"  TriStream  ('{args.tristream_glob}'): {len(tri_paths)} encontrados\n"
            f"  DualStream ('{args.dualstream_glob}'): {len(dual_paths)} encontrados\n"
            "Treine o pool primeiro — ver o cabecalho deste arquivo."
        )

    val_feats, val_labels = load_features("val", device)
    y_true = val_labels.detach().cpu().numpy()
    print(f"Validacao: {len(y_true)} videos | pool: {len(tri_paths)} TriStream, {len(dual_paths)} DualStream")

    tri_probs = member_probs(TriStream_Kinetic, tri_paths, val_feats, device)
    dual_probs = member_probs(DualStream_MeanMax, dual_paths, val_feats, device)

    thresholds = (np.array([args.fixed_threshold]) if args.fixed_threshold is not None
                  else np.arange(0.05, 0.95 + args.step / 2, args.step))
    results = []

    for tri, (d1, d2) in itertools.product(tri_paths, itertools.combinations(sorted(dual_paths), 2)):
        ens = (tri_probs[tri] + dual_probs[d1] + dual_probs[d2]) / 3.0
        best = pick(sweep(y_true, ens, thresholds), args.criterion, args.min_precision)
        results.append({"members": [tri, d1, d2], **best})

    key = {"f1": "f1", "recall_at_precision": "recall", "youden": "youden"}[args.criterion]
    results.sort(key=lambda r: (r[key], r["f1"]), reverse=True)
    champion = results[0]

    print("\n" + "=" * 78)
    print(f"RANKING DOS TRIOS NA VALIDACAO (criterio: {args.criterion}) - {len(results)} combinacoes")
    print("=" * 78)
    for i, r in enumerate(results[:args.top_k], start=1):
        names = " + ".join(Path(m).stem for m in r["members"])
        print(f"{i:2d}. theta={r['threshold']:.2f} | prec {r['precision']*100:5.2f}% | "
              f"rec {r['recall']*100:5.2f}% | F1 {r['f1']*100:5.2f}% | "
              f"acc {r['accuracy']*100:5.2f}% | {names}")
    print("=" * 78)

    accs = np.array([r["accuracy"] for r in results]) * 100
    print(f"Distribuicao na validacao: {accs.mean():.2f}% +- {accs.std(ddof=1):.2f}% "
          f"[{accs.min():.2f}% - {accs.max():.2f}%]")
    print(f"\nTRIO ESCOLHIDO (apenas com a validacao): theta = {champion['threshold']:.2f}")
    for m in champion["members"]:
        print(f"  - {m}")
    print()

    payload = {
        "criterion": args.criterion,
        "min_precision": args.min_precision,
        "selected_on": "val",
        "champion": champion,
        "validation_distribution": {
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std(ddof=1)),
            "min_accuracy": float(accs.min()),
            "max_accuracy": float(accs.max()),
            "num_combinations": len(results),
        },
        "ranking": results[:args.top_k],
    }

    if args.export:
        os.makedirs("models/ensemble", exist_ok=True)
        targets = [name for name, _ in ENSEMBLE_MEMBER_FILES]
        for src_path, target in zip(champion["members"], targets):
            dst = os.path.join("models/ensemble", target)
            shutil.copyfile(src_path, dst)
            print(f"[export] {src_path} -> {dst}")
        print("\nAtualize DEFAULT_THRESHOLDS['ensemble'] em src/model.py para "
              f"{champion['threshold']:.2f}.\n")

    if args.eval_test:
        print("Avaliando o teste cego uma unica vez com trio e limiar definidos na validacao...")
        payload["test_metrics"] = evaluate_model(
            model_type="ensemble", threshold=champion["threshold"], split="test"
        )

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Resultado salvo em: {args.save_json}")


if __name__ == "__main__":
    main()
