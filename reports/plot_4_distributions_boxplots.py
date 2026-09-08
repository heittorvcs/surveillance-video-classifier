"""
Boxplots das distribuicoes de 20 runs por arquitetura e 20 comites de ensemble.

O ponto vermelho marca o melhor checkpoint entregue. Ele fica FORA da distribuicao
dos 20 comites porque o trio de sementes e o limiar foram escolhidos observando o
proprio split de teste — e por isso esta rotulado como tal, e nao como resultado
esperado do metodo.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# Carregar detalhes dos modelos individuais e dos ensembles
df_ind = pd.read_csv("reports/benchmark_3_modelos_20_runs_detalhes.csv")
df_ens = pd.read_csv("reports/benchmark_20_ensembles_detalhes.csv")

m1_acc = df_ind[df_ind["model"].str.contains("Baseline")]["accuracy"].values
m2_acc = df_ind[df_ind["model"].str.contains("Dual-Stream")]["accuracy"].values
m3_acc = df_ind[df_ind["model"].str.contains("Tri-Stream")]["accuracy"].values
ens_acc = df_ens["accuracy"].values

m1_rec = df_ind[df_ind["model"].str.contains("Baseline")]["recall"].values
m2_rec = df_ind[df_ind["model"].str.contains("Dual-Stream")]["recall"].values
m3_rec = df_ind[df_ind["model"].str.contains("Tri-Stream")]["recall"].values
ens_rec = df_ens["recall"].values

m1_fn = df_ind[df_ind["model"].str.contains("Baseline")]["false_negative"].values
m2_fn = df_ind[df_ind["model"].str.contains("Dual-Stream")]["false_negative"].values
m3_fn = df_ind[df_ind["model"].str.contains("Tri-Stream")]["false_negative"].values
ens_fn = df_ens["false_negative"].values

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
labels = ["M1: Baseline\n(Bi-GRU)", "M2: Dual-Stream\n(Aparência+Δf)", "M3: Tri-Stream\n(Cinético Δ²f)", "Comitês Ensemble\n(20 Ensembles Triplos)"]
palette = ["#2980b9", "#e67e22", "#16a085", "#27ae60"]

# 1. Acurácia
bp1 = axes[0].boxplot([m1_acc, m2_acc, m3_acc, ens_acc], patch_artist=True, tick_labels=labels, medianprops=dict(color="black", lw=2))
for patch, color in zip(bp1["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[0].set_title("Distribuição de Acurácia Global (%) em 20 Runs", fontsize=11, fontweight="bold", pad=10)
axes[0].set_ylabel("Acurácia no Teste Cego (%)", fontsize=10)
axes[0].grid(True, alpha=0.3)
# Destacar o Ensemble Campeão de Produção
axes[0].scatter([4], [85.41], color="#c0392b", s=100, zorder=5, label="Melhor checkpoint: 85.41% (selecao no teste)")
axes[0].legend(loc="lower right", fontsize=8)

# 2. Recall
bp2 = axes[1].boxplot([m1_rec, m2_rec, m3_rec, ens_rec], patch_artist=True, tick_labels=labels, medianprops=dict(color="black", lw=2))
for patch, color in zip(bp2["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1].set_title("Distribuição de Recall Fight (%) em 20 Runs", fontsize=11, fontweight="bold", pad=10)
axes[1].set_ylabel("Sensibilidade (Recall Fight %)", fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].scatter([4], [92.05], color="#c0392b", s=100, zorder=5, label="Melhor checkpoint: 92.05% (selecao no teste)")
axes[1].legend(loc="lower right", fontsize=8)

# 3. Falsos Negativos
bp3 = axes[2].boxplot([m1_fn, m2_fn, m3_fn, ens_fn], patch_artist=True, tick_labels=labels, medianprops=dict(color="black", lw=2))
for patch, color in zip(bp3["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[2].set_title("Falsos Negativos (Lutas Perdidas) em 20 Runs", fontsize=11, fontweight="bold", pad=10)
axes[2].set_ylabel("Quantidade de Vídeos Perdidos (em 185)", fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].scatter([4], [7], color="#c0392b", s=100, zorder=5, label="Melhor checkpoint: 7 FN (selecao no teste)")
axes[2].legend(loc="upper right", fontsize=8)

plt.suptitle("Validação Estatística Total: 20 Execuções por Arquitetura e 20 Comitês de Ensemble", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()

plt.savefig("reports/benchmark_3_modelos_20_runs_boxplots.png", bbox_inches="tight")
plt.close()
print("Boxplots salvos em reports/benchmark_3_modelos_20_runs_boxplots.png")

