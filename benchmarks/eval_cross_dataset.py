"""
Validacao externa: modelo treinado no RWF-2000 avaliado no SCVD, sem retreinar.

SCVD (Smart-City CCTV Violence Detection) e um dataset independente de CFTV urbano,
com 3 classes. Nenhum video dele participou do treino, da validacao ou da selecao
de modelo -- e uma avaliacao puramente externa.

Mapeamento para o problema binario deste projeto:
    Normal      -> NonFight (0)
    Violence    -> Fight (1)
    Weaponized  -> Fight (1)   [apenas na versao A]

Duas versoes, para separar violencia corporal de violencia armada:
    A) COM armada:  Normal vs (Violence + Weaponized)
    B) SEM armada:  Normal vs Violence

Fonte do SCVD: https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd
Os videos NAO sao redistribuidos aqui. Baixe o dataset e aponte --scvd_root para a
pasta SCVD_converted (a variante *_sec_split tem clipes de 1 segundo, curtos demais
para um modelo treinado em janelas de 5 s).

Uso:
    python benchmarks/eval_cross_dataset.py --scvd_root caminho/para/SCVD_converted
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from tqdm import tqdm

from src.calibrate_threshold import sweep
from src.dataset import frames_to_tensor, sample_frames
from src.evaluate import load_features, predict_probs
from src.model import DEFAULT_THRESHOLDS, BiGRU_Head, DualStream_Head, load_ensemble_head

CLASSES = ("Normal", "Violence", "Weaponized")


def listar(scvd_root):
    rows = []
    for split in ("Train", "Test"):
        for cls in CLASSES:
            for f in sorted(glob.glob(os.path.join(scvd_root, split, cls, "*.avi"))):
                rows.append({"path": f, "classe": cls, "split_origem": split,
                             "arquivo": os.path.basename(f)})
    if not rows:
        raise FileNotFoundError(
            f"Nenhum video encontrado em {scvd_root}.\n"
            "Esperado: <scvd_root>/{Train,Test}/{Normal,Violence,Weaponized}/*.avi"
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def extrair_features(df, device, cache_path):
    """Features do backbone congelado, uma vez por video."""
    if cache_path and os.path.exists(cache_path):
        print(f"Carregando features de {cache_path}")
        return torch.load(cache_path)

    base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    backbone = base.features.to(device).eval()
    avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device).eval()

    feats = []
    for p in tqdm(df["path"], desc="Extraindo SCVD"):
        x = frames_to_tensor(sample_frames(p, 16), 16).to(device)
        feats.append(avgpool(backbone(x)).flatten(1).unsqueeze(0).cpu())
    out = torch.cat(feats, dim=0)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(out, cache_path)
    return out


@torch.no_grad()
def probs_modelo(nome, feats, device):
    if nome == "ensemble":
        return load_ensemble_head(device=device)(feats)[:, 1].numpy()
    cls, ckpt = {"baseline": (BiGRU_Head, "models/best_model.pth"),
                 "dualstream": (DualStream_Head, "models/model_dualstream_best.pth")}[nome]
    m = cls().to(device)
    state = torch.load(ckpt, map_location=device)
    m.load_state_dict({k: v for k, v in state.items() if not k.startswith("features.")})
    m.eval()
    return torch.softmax(m(feats), dim=1)[:, 1].numpy()


def metricas(y, p, th):
    pred = (p >= th).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {"n": int(len(y)), "n_fight": int((y == 1).sum()), "n_nonfight": int((y == 0).sum()),
            "accuracy": float(accuracy_score(y, pred) * 100), "precision": float(prec * 100),
            "recall": float(rec * 100), "f1": float(f1 * 100),
            "auc": float(roc_auc_score(y, p) * 100) if len(set(y)) > 1 else None,
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}


def versoes(df, coluna):
    """(rotulos, probabilidades) para as versoes A (com armada) e B (sem armada)."""
    yA = df["classe"].map({"Normal": 0, "Violence": 1, "Weaponized": 1}).to_numpy()
    mask = (df["classe"] != "Weaponized").to_numpy()
    yB = df.loc[mask, "classe"].map({"Normal": 0, "Violence": 1}).to_numpy()
    return (yA, df[coluna].to_numpy()), (yB, df.loc[mask, coluna].to_numpy())


def figura(df, device, out_png):
    """Distribuicao de P(Fight) e sensibilidade ao limiar."""
    f_rwf, l_rwf = load_features("test", device)
    p_rwf, _ = predict_probs("ensemble", f_rwf, device)
    y_rwf = l_rwf.detach().cpu().numpy()
    p = df["prob_ensemble"].to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)
    bins = np.linspace(0, 1, 26)

    a = ax[0]
    a.hist(p_rwf[y_rwf == 0], bins=bins, alpha=.55, color="#2980b9",
           label=f"RWF-2000 NonFight (mediana {np.median(p_rwf[y_rwf == 0]):.2f})")
    normal = df.loc[df.classe == "Normal", "prob_ensemble"]
    a.hist(normal, bins=bins, alpha=.55, color="#c0392b",
           label=f"SCVD Normal (mediana {normal.median():.2f})")
    a.hist(p_rwf[y_rwf == 1], bins=bins, histtype="step", lw=2.5, color="#16a085",
           label=f"RWF-2000 Fight (mediana {np.median(p_rwf[y_rwf == 1]):.2f})")
    viol = df.loc[df.classe == "Violence", "prob_ensemble"]
    a.hist(viol, bins=bins, histtype="step", lw=2.5, ls="--", color="#27ae60",
           label=f"SCVD Violence (mediana {viol.median():.2f})")
    a.axvline(.50, color="k", ls=":", lw=2, label="limiar 0.50")
    a.set_xlabel("P(Fight)"); a.set_ylabel("nº de vídeos")
    a.set_title("A classe positiva transfere; a negativa desloca", fontweight="bold", fontsize=12)
    a.legend(fontsize=8.5); a.grid(alpha=.3)

    b = ax[1]
    ths = np.arange(0.05, 0.96, 0.01)
    for nome, (y, pr), cor in [("A · com armada", versoes(df, "prob_ensemble")[0], "#8e44ad"),
                               ("B · sem armada", versoes(df, "prob_ensemble")[1], "#e67e22")]:
        accs = [r["accuracy"] * 100 for r in sweep(y, pr, ths)]
        b.plot(ths, accs, lw=2.5, color=cor, label=nome)
        i = int(np.argmax(accs))
        b.scatter([ths[i]], [accs[i]], color=cor, s=70, zorder=5)
        b.annotate(f"{accs[i]:.1f}%\nθ={ths[i]:.2f}", (ths[i], accs[i]), textcoords="offset points",
                   xytext=(6, -22), fontsize=9, color=cor, fontweight="bold")
    b.axvline(.50, color="k", ls=":", lw=2, label="limiar calibrado no RWF-2000")
    b.axhline(81.62, color="#16a085", ls="--", lw=1.5, label="acurácia in-domain (RWF-2000)")
    b.set_xlabel("limiar θ"); b.set_ylabel("acurácia no SCVD (%)")
    b.set_title("O que falta é calibração, não capacidade", fontweight="bold", fontsize=12)
    b.legend(fontsize=8.5, loc="lower center"); b.grid(alpha=.3)

    plt.suptitle(f"Transferência RWF-2000 → SCVD ({len(df)} vídeos de CFTV urbano)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validacao externa no SCVD")
    parser.add_argument("--scvd_root", type=str, required=True,
                        help="Pasta SCVD_converted, com Train/ e Test/ dentro")
    parser.add_argument("--cache", type=str, default="data/cache/features_scvd.pt",
                        help="Cache das features do SCVD (nao versionado)")
    parser.add_argument("--save_json", type=str, default="reports/cross_dataset_scvd.json")
    parser.add_argument("--save_png", type=str, default="reports/cross_dataset_scvd.png")
    parser.add_argument("--save_csv", type=str, default=None,
                        help="Opcional: predicao por video")
    args = parser.parse_args()

    device = torch.device("cpu")
    df = listar(args.scvd_root)
    print(f"SCVD: {len(df)} videos")
    for cls, n in df.groupby("classe").size().items():
        print(f"  {cls:11s} {n:4d}")

    feats = extrair_features(df, device, args.cache)

    resultados = {"dataset": "SCVD (Smart-City CCTV Violence Detection)",
                  "num_videos": int(len(df)),
                  "classes": {c: int((df.classe == c).sum()) for c in CLASSES},
                  "modelos": {}}

    for nome in ("baseline", "dualstream", "ensemble"):
        p = probs_modelo(nome, feats, device)
        df[f"prob_{nome}"] = p
        th = DEFAULT_THRESHOLDS[nome]
        (yA, pA), (yB, pB) = versoes(df, f"prob_{nome}")
        mA, mB = metricas(yA, pA, th), metricas(yB, pB, th)

        det = {c: float((df.loc[df.classe == c, f"prob_{nome}"] >= th).mean() * 100) for c in CLASSES}

        print("\n" + "=" * 78)
        print(f"MODELO: {nome}  (theta = {th:.2f})")
        print("=" * 78)
        for titulo, m in [("VERSAO A - COM violencia armada", mA), ("VERSAO B - SEM violencia armada", mB)]:
            print(f"\n  {titulo}")
            print(f"    n={m['n']} (Fight {m['n_fight']}, NonFight {m['n_nonfight']})")
            print(f"    Acuracia {m['accuracy']:6.2f}%  Recall {m['recall']:6.2f}%  "
                  f"Precisao {m['precision']:6.2f}%  F1 {m['f1']:6.2f}%  AUC {m['auc']:6.2f}%")
            print(f"    TN={m['tn']:3d}  FP={m['fp']:3d}  FN={m['fn']:3d}  TP={m['tp']:3d}")
        print("\n    Deteccao como Fight por classe do SCVD:")
        for c in CLASSES:
            print(f"      {c:11s} {det[c]:6.2f}%")

        resultados["modelos"][nome] = {"threshold": th, "versao_A_com_armada": mA,
                                       "versao_B_sem_armada": mB, "deteccao_por_classe": det}

    # Ganho obtido apenas recalibrando o limiar no dominio novo
    ths = np.arange(0.05, 0.96, 0.01)
    for chave, (y, pr) in [("versao_A_com_armada", versoes(df, "prob_ensemble")[0]),
                           ("versao_B_sem_armada", versoes(df, "prob_ensemble")[1])]:
        rows = sweep(y, pr, ths)
        best = max(rows, key=lambda r: r["accuracy"])
        atual = min(rows, key=lambda r: abs(r["threshold"] - DEFAULT_THRESHOLDS["ensemble"]))
        resultados["modelos"]["ensemble"][chave]["recalibrado"] = {
            "threshold": best["threshold"],
            "accuracy": best["accuracy"] * 100,
            "recall": best["recall"] * 100,
            "precision": best["precision"] * 100,
            "f1": best["f1"] * 100,
            "ganho_acuracia_pp": (best["accuracy"] - atual["accuracy"]) * 100,
        }

    figura(df, device, args.save_png)
    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    if args.save_csv:
        df.drop(columns=["path"]).to_csv(args.save_csv, index=False)

    print(f"\nResultado salvo em: {args.save_json}")
    print(f"Figura salva em:    {args.save_png}")


if __name__ == "__main__":
    main()
