"""
Analise dos erros de classificacao no conjunto de teste.

Contar 13 falsos negativos e 21 falsos positivos diz pouco sobre o que o modelo
erra. Este script responde tres perguntas que mudam a decisao de engenharia:

  1. Os erros estao perto ou longe da fronteira de decisao? Erros perto do limiar
     sao recuperaveis calibrando o ponto de operacao; erros distantes sao falhas
     de representacao e exigem mudanca de modelo ou de dados.
  2. Os erros se espalham ou se concentram? Se um punhado de cameras concentra a
     maioria, o problema e de cobertura de cenario, nao de capacidade do modelo.
  3. Algum limiar resolveria? Se nao, o ganho tem de vir de outro lugar.

Saida: reports/error_analysis.json e um relatorio no terminal.

Uso:
    python reports/error_analysis.py
    python reports/error_analysis.py --model dualstream --top 10
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
import pandas as pd
import torch

from src.create_splits import group_id
from src.evaluate import load_features, predict_probs
from src.model import DEFAULT_THRESHOLDS


def build_table(model_type, threshold, device):
    """Um DataFrame com rotulo, probabilidade, grupo de camera e margem por video."""
    feats, labels = load_features("test", device)
    y = labels.detach().cpu().numpy()
    probs, _ = predict_probs(model_type, feats, device)

    df = pd.read_csv("data/splits/test.csv")
    df["y"] = y
    df["prob"] = probs
    df["pred"] = (probs >= threshold).astype(int)
    df["errou"] = df["pred"] != df["y"]
    df["grupo"] = df["video_path"].map(group_id)
    df["arquivo"] = df["video_path"].map(os.path.basename)
    # Distancia ate a fronteira de decisao, em pontos percentuais
    df["margem"] = (df["prob"] - threshold).abs() * 100
    return df


def main():
    parser = argparse.ArgumentParser(description="Analise dos erros no conjunto de teste")
    parser.add_argument("--model", type=str, default="ensemble",
                        choices=["ensemble", "dualstream", "baseline"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top", type=int, default=6, help="Quantos erros mais confiantes listar")
    parser.add_argument("--save_json", type=str, default="reports/error_analysis.json")
    args = parser.parse_args()

    device = torch.device("cpu")
    th = args.threshold if args.threshold is not None else DEFAULT_THRESHOLDS[args.model]
    df = build_table(args.model, th, device)

    err = df[df["errou"]]
    fn = err[err["y"] == 1]
    fp = err[err["y"] == 0]

    print("=" * 78)
    print(f"ANALISE DOS ERROS - {args.model} (theta = {th:.2f}, {len(df)} videos de teste)")
    print("=" * 78)
    print(f"Erros: {len(err)}  ({len(fn)} falsos negativos, {len(fp)} falsos positivos)")

    # 1) Distancia ate a fronteira
    print("\n1. OS ERROS ESTAO PERTO DA FRONTEIRA DE DECISAO?")
    print(f"   Margem mediana dos erros:   {err['margem'].median():5.1f} p.p.")
    print(f"   Margem mediana dos acertos: {df[~df['errou']]['margem'].median():5.1f} p.p.")
    for lim in (5, 10, 20, 30):
        n = int((err["margem"] <= lim).sum())
        print(f"   erros a ate {lim:2d} p.p. do limiar: {n:2d} ({n / max(len(err), 1) * 100:.0f}%)")
    print(f"\n   Falsos negativos: margem mediana {fn['margem'].median():5.1f} p.p.")
    print(f"   Falsos positivos: margem mediana {fp['margem'].median():5.1f} p.p.")

    # 2) Concentracao por camera
    g = df.groupby("grupo").agg(clipes=("errou", "size"), erros=("errou", "sum"))
    multi = g[g["erros"] >= 2].sort_values("erros", ascending=False)
    limpos = int((g["erros"] == 0).sum())

    print("\n2. OS ERROS SE CONCENTRAM EM POUCAS CAMERAS?")
    print(f"   {len(err)} erros distribuidos em {err['grupo'].nunique()} grupos, de {len(g)} no teste")
    print(f"   {limpos} grupos ({limpos / len(g) * 100:.0f}%) nao produzem nenhum erro")
    if len(multi):
        share = int(multi["erros"].sum()) / max(len(err), 1) * 100
        print(f"   {len(multi)} grupos com 2 ou mais erros concentram {int(multi['erros'].sum())} "
              f"dos {len(err)} erros ({share:.0f}%)")
        print(f"\n   {'grupo':18s} {'erros':>6s} {'clipes':>7s}  classes presentes")
        for grp, row in multi.iterrows():
            classes = "/".join(sorted(df[df["grupo"] == grp]["label_name"].unique()))
            print(f"   {grp:18s} {int(row['erros']):6d} {int(row['clipes']):7d}  {classes}")

    piores = g.nlargest(3, "erros").index.tolist()
    resto = df[~df["grupo"].isin(piores)]
    acc_geral = (~df["errou"]).mean() * 100
    acc_resto = (~resto["errou"]).mean() * 100
    print(f"\n   Acuracia geral:                                  {acc_geral:5.2f}%")
    print(f"   Excluindo os 3 grupos piores ({len(df) - len(resto):2d} clipes): {acc_resto:5.2f}%")

    # 3) Algum limiar resolveria?
    print("\n3. ALGUM OUTRO LIMIAR RESOLVERIA?")
    y = df["y"].to_numpy()
    probs = df["prob"].to_numpy()
    sweep = []
    for t in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        p = (probs >= t).astype(int)
        row = {"threshold": t, "accuracy": float((p == y).mean() * 100),
               "false_negative": int(((y == 1) & (p == 0)).sum()),
               "false_positive": int(((y == 0) & (p == 1)).sum())}
        sweep.append(row)
        marca = "  <-- em uso" if abs(t - th) < 1e-9 else ""
        print(f"   theta={t:.2f}  acuracia={row['accuracy']:5.2f}%  "
              f"FN={row['false_negative']:2d}  FP={row['false_positive']:2d}{marca}")

    # 4) Os erros mais confiantes
    print(f"\n4. OS {args.top} ERROS MAIS CONFIANTES")
    for _, r in err.nlargest(args.top, "margem").iterrows():
        tipo = "FN" if r["y"] == 1 else "FP"
        print(f"   {tipo}  {r['arquivo']:24s} real={r['label_name']:9s} "
              f"P(Fight)={r['prob'] * 100:5.1f}%  margem={r['margem']:4.1f} p.p.")

    print("=" * 78)

    payload = {
        "model": args.model,
        "threshold": th,
        "num_videos": int(len(df)),
        "num_errors": int(len(err)),
        "false_negatives": int(len(fn)),
        "false_positives": int(len(fp)),
        "margin": {
            "median_errors": float(err["margem"].median()),
            "median_correct": float(df[~df["errou"]]["margem"].median()),
            "median_fn": float(fn["margem"].median()) if len(fn) else None,
            "median_fp": float(fp["margem"].median()) if len(fp) else None,
            "errors_within_pp": {str(l): int((err["margem"] <= l).sum()) for l in (5, 10, 20, 30)},
        },
        "clustering": {
            "groups_total": int(len(g)),
            "groups_with_errors": int(err["grupo"].nunique()),
            "groups_error_free": limpos,
            "groups_with_2plus_errors": int(len(multi)),
            "errors_in_those_groups": int(multi["erros"].sum()) if len(multi) else 0,
            "worst_groups": [
                {"grupo": grp, "erros": int(r["erros"]), "clipes": int(r["clipes"]),
                 "classes": sorted(df[df["grupo"] == grp]["label_name"].unique())}
                for grp, r in multi.iterrows()
            ],
            "accuracy_all": acc_geral,
            "accuracy_excluding_worst3": acc_resto,
        },
        "threshold_sweep": sweep,
        "most_confident_errors": [
            {"arquivo": r["arquivo"], "tipo": "FN" if r["y"] == 1 else "FP",
             "real": r["label_name"], "prob_fight": float(r["prob"]), "margem_pp": float(r["margem"])}
            for _, r in err.nlargest(args.top, "margem").iterrows()
        ],
    }

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Resultado salvo em: {args.save_json}")


if __name__ == "__main__":
    main()
