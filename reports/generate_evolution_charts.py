"""
Gera as figuras comparativas dos tres modelos no teste cego:
  reports/matrizes_confusao_3_modelos.png
  reports/curvas_roc_3_modelos.png
  reports/jornada_evolutiva_3_modelos.png

As arquiteturas e o carregamento de pesos vem de src/, sem redefinicao local.
Pre-requisito: python src/build_cache.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.evaluate import load_features, predict_probs
from src.model import DEFAULT_THRESHOLDS

# Configurar estilo visual
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

device = torch.device("cpu")
test_feats, test_labels = load_features("test", device)
y_true = test_labels.detach().cpu().numpy()
n_fight = int((y_true == 1).sum())

# Probabilidades dos tres modelos no teste cego
probs1, _ = predict_probs("baseline", test_feats, device)
probs2, _ = predict_probs("dualstream", test_feats, device)
probs3, _ = predict_probs("ensemble", test_feats, device)

preds1 = (probs1 >= DEFAULT_THRESHOLDS["baseline"]).astype(int)
preds2 = (probs2 >= DEFAULT_THRESHOLDS["dualstream"]).astype(int)
preds3 = (probs3 >= DEFAULT_THRESHOLDS["ensemble"]).astype(int)

cm1 = confusion_matrix(y_true, preds1)
cm2 = confusion_matrix(y_true, preds2)
cm3 = confusion_matrix(y_true, preds3)

auc1 = roc_auc_score(y_true, probs1) * 100
auc2 = roc_auc_score(y_true, probs2) * 100
auc3 = roc_auc_score(y_true, probs3) * 100

print(f"Modelo 1: Acc={accuracy_score(y_true, preds1)*100:.2f}% | Rec={cm1[1,1]/n_fight*100:.2f}% | FN={cm1[1,0]} | AUC={auc1:.2f}%")
print(f"Modelo 2: Acc={accuracy_score(y_true, preds2)*100:.2f}% | Rec={cm2[1,1]/n_fight*100:.2f}% | FN={cm2[1,0]} | AUC={auc2:.2f}%")
print(f"Modelo 3: Acc={accuracy_score(y_true, preds3)*100:.2f}% | Rec={cm3[1,1]/n_fight*100:.2f}% | FN={cm3[1,0]} | AUC={auc3:.2f}%")

# ==========================================
# FIGURA 1: MATRIZES DE CONFUSÃO LADO A LADO
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
def cm_title(nome, cm, preds):
    """Titulo com metricas calculadas na hora, nunca escritas a mao."""
    acc = accuracy_score(y_true, preds) * 100
    rec = cm[1, 1] / n_fight * 100
    return f"{nome}\n(Acc: {acc:.2f}% | Rec: {rec:.2f}% | {cm[1, 0]} FN)"


models_info = [
    (cm_title("Modelo 1: Baseline Bi-GRU", cm1, preds1), cm1, "#1f77b4"),
    (cm_title("Modelo 2: Dual-Stream Latente", cm2, preds2), cm2, "#ff7f0e"),
    (cm_title("Modelo 3: Ensemble Cinético", cm3, preds3), cm3, "#2ca02c")
]

for idx, (title, cm, color) in enumerate(models_info):
    ax = axes[idx]
    caxes = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    classes = ["NonFight", "Fight"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predição do Modelo", fontsize=11)
    if idx == 0:
        ax.set_ylabel("Classe Real (Ground Truth)", fontsize=11)
    
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            if i == 1 and j == 0:
                ax.text(j, i, f"{val}\n(FN)", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="#d62728")
            elif i == 1 and j == 1:
                ax.text(j, i, f"{val}\n(TP)", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="white" if val > thresh else "black")
            elif i == 0 and j == 0:
                ax.text(j, i, f"{val}\n(TN)", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="white" if val > thresh else "black")
            else:
                ax.text(j, i, f"{val}\n(FP)", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="black")

plt.tight_layout()
plt.savefig("reports/matrizes_confusao_3_modelos.png", bbox_inches="tight")
plt.close()
print("Salvo: reports/matrizes_confusao_3_modelos.png")

# ==========================================
# FIGURA 2: CURVAS ROC LADO A LADO E COMPARATIVO
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

fpr1, tpr1, _ = roc_curve(y_true, probs1)
fpr2, tpr2, _ = roc_curve(y_true, probs2)
fpr3, tpr3, _ = roc_curve(y_true, probs3)

ax.plot(fpr1, tpr1, color="#1f77b4", lw=2.2, linestyle="--", label=f"M1: Baseline Bi-GRU (AUC = {auc1/100:.3f})")
ax.plot(fpr2, tpr2, color="#ff7f0e", lw=2.5, linestyle="-.", label=f"M2: Dual-Stream Latente (AUC = {auc2/100:.3f})")
ax.plot(fpr3, tpr3, color="#2ca02c", lw=3.0, label=f"M3: Ensemble Cinético Tri-Stream (AUC = {auc3/100:.3f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle=":", label="Acaso (AUC = 0.500)")

fn3 = cm3[1, 0]
tp3 = cm3[1, 1]
fp3 = cm3[0, 1]
tn3 = cm3[0, 0]
opr_fpr = fp3 / (fp3 + tn3)
opr_tpr = tp3 / (tp3 + fn3)
ax.scatter([opr_fpr], [opr_tpr], color="#d62728", s=90, zorder=5, label=f"M3 no limiar entregue (θ={DEFAULT_THRESHOLDS['ensemble']:.2f}, Rec={opr_tpr*100:.1f}%)")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("Taxa de Falsos Positivos (FPR / Falsos Alarmes)", fontsize=11, fontweight="bold")
ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR / Recall Fight)", fontsize=11, fontweight="bold")
ax.set_title("Curvas ROC: Poder Discriminativo ao Longo da Evolução", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=10, frameon=True, facecolor="white", framealpha=0.9)
ax.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig("reports/curvas_roc_3_modelos.png", bbox_inches="tight")
plt.close()
print("Salvo: reports/curvas_roc_3_modelos.png")

# ==========================================
# FIGURA 3: DASHBOARD MASTER CONSOLIDADO (4 PAINÉIS)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

# Painel A: Redução Drástica de Falsos Negativos
ax_a = axes[0, 0]
models_label = ["M1: Baseline\n(Bi-GRU)", "M2: Dual-Stream\n(Aparência + Δf)", "M3: Ensemble\nCinético (Δ²f)"]
fns = [cm1[1, 0], cm2[1, 0], cm3[1, 0]]
colors_bar = ["#4a90e2", "#f5a623", "#27ae60"]
bars = ax_a.bar(models_label, fns, color=colors_bar, width=0.55, edgecolor="#333333", linewidth=1)
ax_a.set_title("(A) Falsos Negativos (Lutas Não Detectadas)", fontsize=12, fontweight="bold")
ax_a.set_ylabel(f"Vídeos perdidos (em {len(y_true)})", fontsize=10)
ax_a.set_ylim(0, max(26, max(fns) * 1.2))
for bar, fn in zip(bars, fns):
    yval = bar.get_height()
    ax_a.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{fn} FN", ha="center", va="bottom", fontsize=11, fontweight="bold")
_reducao = (1 - fns[2] / fns[0]) * 100 if fns[0] else 0.0
ax_a.annotate(f"Redução de {_reducao:.1f}% nos FN\n(melhor checkpoint: {fns[2]} FN)",
             xy=(2, fns[2]), xytext=(1.0, 16),
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2),
             fontsize=10, fontweight="bold", color="#c0392b",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#e67e22", lw=1))
ax_a.grid(axis="y", alpha=0.3)

# Painel B: Subida do Recall e F1-Score
ax_b = axes[0, 1]
x_pos = np.arange(len(models_label))
width = 0.3
recalls = [cm1[1, 1]/n_fight * 100, cm2[1, 1]/n_fight * 100, cm3[1, 1]/n_fight * 100]
accuracies = [accuracy_score(y_true, preds1)*100, accuracy_score(y_true, preds2)*100, accuracy_score(y_true, preds3)*100]

b1 = ax_b.bar(x_pos - width/2, accuracies, width, label="Acurácia Global (%)", color="#2980b9", edgecolor="#333333")
b2 = ax_b.bar(x_pos + width/2, recalls, width, label="Recall Fight (%)", color="#27ae60", edgecolor="#333333")
ax_b.set_title("(B) Evolução de Acurácia e Sensibilidade Operacional", fontsize=12, fontweight="bold")
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(models_label, fontsize=9, fontweight="bold")
ax_b.set_ylabel("Percentual (%)", fontsize=10)
ax_b.set_ylim(65, 100)
ax_b.legend(loc="lower right", fontsize=9)
for b in b1:
    ax_b.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.6, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
for b in b2:
    ax_b.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.6, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
ax_b.grid(axis="y", alpha=0.3)

# Painel C: Curvas ROC Sobrepostas
ax_c = axes[1, 0]
ax_c.plot(fpr1, tpr1, color="#2980b9", lw=2.0, linestyle="--", label=f"M1: Baseline (AUC {auc1:.1f}%)")
ax_c.plot(fpr2, tpr2, color="#e67e22", lw=2.2, linestyle="-.", label=f"M2: DualStream (AUC {auc2:.1f}%)")
ax_c.plot(fpr3, tpr3, color="#27ae60", lw=2.8, label=f"M3: Ensemble (AUC {auc3:.1f}%)")
ax_c.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
ax_c.set_title("(C) Curvas ROC (Poder Discriminativo)", fontsize=12, fontweight="bold")
ax_c.set_xlabel("FPR (Taxa de Falsos Alarmes)", fontsize=10)
ax_c.set_ylabel("TPR (Recall Fight)", fontsize=10)
ax_c.legend(loc="lower right", fontsize=9)
ax_c.grid(True, alpha=0.3)

# Painel D: Latência em Edge AI e Throughput
ax_d = axes[1, 1]
# Latencias lidas do benchmark unificado; nunca escritas a mao.
# Gere com: python benchmarks/benchmark_detailed_latency.py
LATENCY_JSON = "reports/latency_benchmark.json"
if os.path.exists(LATENCY_JSON):
    with open(LATENCY_JSON, encoding="utf-8") as f:
        _lat = json.load(f)
    lats = [m["forward_ms"] for m in _lat["models"]]
    lat_note = f"medido em {(_lat['hardware']['processor'] or 'CPU')[:40]}"
else:
    lats = None
    lat_note = "execute benchmarks/benchmark_detailed_latency.py"

if lats is None:
    ax_d.axis("off")
    ax_d.set_title("(D) Latência em CPU — não medida", fontsize=12, fontweight="bold")
    ax_d.text(0.5, 0.5, "Sem reports/latency_benchmark.json.\n" + lat_note,
              ha="center", va="center", fontsize=10, transform=ax_d.transAxes)
else:
    bars_lat = ax_d.bar(models_label, lats, color=["#34495e", "#7f8c8d", "#16a085"],
                        width=0.55, edgecolor="#333333")
    ax_d.set_title(f"(D) Forward em CPU, 16 quadros 224x224\n({lat_note})",
                   fontsize=11, fontweight="bold")
    ax_d.set_ylabel("Tempo de forward (ms por clipe)", fontsize=10)
    ax_d.set_ylim(0, max(105, max(lats) * 1.25))
    for bar, lat in zip(bars_lat, lats):
        ax_d.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.2, f"{lat:.1f} ms",
                  ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_d.axhline(100, color="#e74c3c", linestyle=":", label="Orçamento de borda (100 ms)")
    ax_d.legend(loc="upper left", fontsize=9)
    ax_d.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("reports/jornada_evolutiva_3_modelos.png", bbox_inches="tight")
plt.close()
print("Salvo: reports/jornada_evolutiva_3_modelos.png")

