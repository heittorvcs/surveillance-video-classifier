"""
Quantifica o vies otimista de selecionar modelo e limiar sobre o conjunto de teste.

A primeira versao deste projeto escolheu o trio de membros e o limiar do ensemble
maximizando a acuracia no proprio split de teste. Este script mede, com o mesmo
pool de candidatos, quanta acuracia esse procedimento "ganha" sem que exista
qualquer ganho real de poder discriminativo.

Tres protocolos sao comparados sobre exatamente os mesmos modelos:

  A. Oraculo no teste   - escolhe trio e limiar maximizando acuracia no TESTE.
                          E o procedimento original. O numero resultante e um
                          maximo sobre milhares de configuracoes e nao generaliza.
  B. Protocolo correto  - escolhe trio e limiar apenas na VALIDACAO e avalia o
                          teste uma unica vez.
  C. Distribuicao       - acuracia no teste de todos os trios no limiar calibrado
                          na validacao. E a referencia do que esperar do metodo.

Pre-requisitos:
    python src/build_cache.py
    for s in 1..20: python src/train.py --arch tristream   --seed $s --out models/pool/tristream_s$s.pth
    for s in 1..20: python src/train.py --arch dualmeanmax --seed $s --out models/pool/dualmeanmax_s$s.pth

Uso:
    python reports/selection_bias_analysis.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.calibrate_threshold import pick, sweep
from src.evaluate import load_features
from src.model import DualStream_MeanMax, TriStream_Kinetic


@torch.no_grad()
def member_probs(cls, paths, feats, device):
    out = {}
    for path in sorted(paths):
        model = cls().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        out[path] = torch.softmax(model(feats), dim=1)[:, 1].detach().cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser(description="Analise do vies de selecao no teste")
    parser.add_argument("--tristream_glob", type=str, default="models/pool/tristream_s*.pth")
    parser.add_argument("--dualstream_glob", type=str, default="models/pool/dualmeanmax_s*.pth")
    parser.add_argument("--criterion", type=str, default="recall_at_precision",
                        choices=["f1", "recall_at_precision", "youden"])
    parser.add_argument("--min_precision", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--honest_threshold", type=float, default=0.50,
                        help="Limiar fixo do protocolo correto (o braco oraculo sempre varre)")
    parser.add_argument("--save_json", type=str, default="reports/selection_bias_analysis.json")
    parser.add_argument("--save_png", type=str, default="reports/selection_bias_analysis.png")
    args = parser.parse_args()

    device = torch.device("cpu")
    tri_paths = sorted(glob.glob(args.tristream_glob))
    dual_paths = sorted(glob.glob(args.dualstream_glob))
    if not tri_paths or len(dual_paths) < 2:
        raise FileNotFoundError("Pool de candidatos insuficiente — ver o cabecalho deste arquivo.")

    val_feats, val_labels = load_features("val", device)
    test_feats, test_labels = load_features("test", device)
    y_val = val_labels.detach().cpu().numpy()
    y_test = test_labels.detach().cpu().numpy()

    tri_val = member_probs(TriStream_Kinetic, tri_paths, val_feats, device)
    dual_val = member_probs(DualStream_MeanMax, dual_paths, val_feats, device)
    tri_test = member_probs(TriStream_Kinetic, tri_paths, test_feats, device)
    dual_test = member_probs(DualStream_MeanMax, dual_paths, test_feats, device)

    thresholds = np.arange(0.05, 0.95 + args.step / 2, args.step)
    trios = list(itertools.product(tri_paths, itertools.combinations(dual_paths, 2)))
    n_configs = len(trios) * len(thresholds)

    print("=" * 78)
    print("ANALISE DO VIES DE SELECAO SOBRE O CONJUNTO DE TESTE")
    print("=" * 78)
    print(f"Pool:              {len(tri_paths)} TriStream + {len(dual_paths)} DualMeanMax")
    print(f"Trios possiveis:   {len(trios)}")
    print(f"Limiares:          {len(thresholds)} (de {thresholds[0]:.2f} a {thresholds[-1]:.2f})")
    print(f"Configuracoes:     {n_configs:,}".replace(",", "."))
    print(f"Validacao:         {len(y_val)} videos | Teste: {len(y_test)} videos")
    print("-" * 78)

    key = {"f1": "f1", "recall_at_precision": "recall", "youden": "youden"}[args.criterion]
    oracle_candidates = []
    honest_candidates = []
    test_probs = {}

    for tri, (d1, d2) in trios:
        members = (tri, d1, d2)
        ens_val = (tri_val[tri] + dual_val[d1] + dual_val[d2]) / 3.0
        ens_test = (tri_test[tri] + dual_test[d1] + dual_test[d2]) / 3.0
        test_probs[members] = ens_test

        # A. Oraculo: maximiza acuracia diretamente no TESTE (procedimento original)
        best_test = max(sweep(y_test, ens_test, thresholds), key=lambda r: r["accuracy"])
        oracle_candidates.append({**best_test, "members": list(members)})

        # B. Protocolo correto: escolhe apenas na VALIDACAO
        best_val = pick(sweep(y_val, ens_val, np.array([args.honest_threshold])),
                        args.criterion, args.min_precision)
        honest_candidates.append({"members": members, "validation": best_val})

    oracle = max(oracle_candidates, key=lambda c: c["accuracy"])
    honest = max(honest_candidates, key=lambda c: (c["validation"][key], c["validation"]["f1"]))
    theta = honest["validation"]["threshold"]

    # C. Distribuicao no teste de todos os trios, no limiar calibrado na validacao
    dist = np.array([float(((probs >= theta).astype(int) == y_test).mean() * 100)
                     for probs in test_probs.values()])

    honest_test = sweep(y_test, test_probs[honest["members"]], np.array([theta]))[0]
    honest["members"] = list(honest["members"])

    gap = oracle["accuracy"] * 100 - honest_test["accuracy"] * 100

    print(f"A. ORACULO NO TESTE   acuracia {oracle['accuracy']*100:6.2f}% "
          f"(theta {oracle['threshold']:.2f}, FN {oracle['false_negative']})")
    print(f"   melhor de {n_configs:,} configuracoes avaliadas no proprio teste".replace(",", "."))
    print()
    print(f"B. PROTOCOLO CORRETO  acuracia {honest_test['accuracy']*100:6.2f}% "
          f"(theta {theta:.2f}, FN {honest_test['false_negative']})")
    print(f"   trio e limiar definidos apenas nos {len(y_val)} videos de validacao")
    print()
    print(f"C. DISTRIBUICAO       acuracia {dist.mean():6.2f}% +- {dist.std(ddof=1):.2f}% "
          f"[{dist.min():.2f}% - {dist.max():.2f}%]")
    print(f"   todos os {len(trios)} trios no teste, no limiar theta = {theta:.2f}")
    print("-" * 78)
    print(f"VIES DE SELECAO (A - B): {gap:.2f} pontos percentuais de otimismo")
    z = (honest_test["accuracy"] * 100 - dist.mean()) / dist.std(ddof=1)
    print(f"O resultado honesto fica a {z:+.2f} desvios da media da distribuicao — "
          f"ou seja, e um trio tipico.")
    z_oracle = (oracle["accuracy"] * 100 - dist.mean()) / dist.std(ddof=1)
    print(f"O oraculo fica a {z_oracle:+.2f} desvios — a distancia e artefato da busca, "
          f"nao de capacidade.")
    print("=" * 78)

    payload = {
        "pool": {"tristream": len(tri_paths), "dualmeanmax": len(dual_paths)},
        "num_trios": len(trios),
        "num_thresholds": len(thresholds),
        "num_configurations": int(n_configs),
        "oracle_on_test": {k: v for k, v in oracle.items() if k != "ens_test"},
        "honest_protocol": {
            "members": honest["members"],
            "threshold": theta,
            "validation": honest["validation"],
            "test": honest_test,
        },
        "test_distribution_at_val_threshold": {
            "mean": float(dist.mean()),
            "std": float(dist.std(ddof=1)),
            "min": float(dist.min()),
            "max": float(dist.max()),
            "n": int(len(dist)),
        },
        "selection_bias_pp": float(gap),
    }
    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResultado salvo em: {args.save_json}")

    # Figura: onde cada protocolo cai na distribuicao real
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    ax.hist(dist, bins=30, color="#7f8c8d", alpha=0.75, edgecolor="white",
            label=f"Todos os {len(trios)} trios no teste (θ={theta:.2f})")
    ax.axvline(dist.mean(), color="#2c3e50", lw=2, linestyle="-",
               label=f"Média da distribuição: {dist.mean():.2f}%")
    ax.axvline(honest_test["accuracy"] * 100, color="#27ae60", lw=2.5, linestyle="--",
               label=f"Seleção na validação: {honest_test['accuracy']*100:.2f}%")
    ax.axvline(oracle["accuracy"] * 100, color="#c0392b", lw=2.5, linestyle="--",
               label=f"Oráculo no teste: {oracle['accuracy']*100:.2f}%")
    ax.set_xlabel("Acurácia no conjunto de teste cego (%)", fontsize=11)
    ax.set_ylabel("Número de trios", fontsize=11)
    ax.set_title("Viés de seleção: escolher no teste desloca o número, não a capacidade",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.save_png, bbox_inches="tight")
    plt.close()
    print(f"Figura salva em:    {args.save_png}")


if __name__ == "__main__":
    main()
