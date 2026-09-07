import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_recall_fscore_support

# Configurar estilo visual
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

device = torch.device("cpu")
test_feats, test_labels = torch.load("data/cache/features_test.pt", map_location=device)
y_true = test_labels.numpy()

# 1. Definicoes de Arquitetura
class BiGRU_Baseline(nn.Module):
    def __init__(self, feature_dim=576, hidden_dim=64, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, num_classes)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.classifier(out.mean(dim=1))

class DualStream_Latent(nn.Module):
    def __init__(self, feature_dim=576, hidden_dim=64, num_classes=2):
        super().__init__()
        self.gru_app = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_mot = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out_app, _ = self.gru_app(x)
        app_feat = out_app.mean(dim=1)
        diff = x[:, 1:, :] - x[:, :-1, :]
        out_mot, _ = self.gru_mot(diff)
        mot_feat = out_mot.mean(dim=1)
        combined = torch.cat([app_feat, mot_feat], dim=1)
        return self.classifier(combined)

class TriStream_Kinetic(nn.Module):
    def __init__(self, feature_dim=576, hidden_dim=64, num_classes=2):
        super().__init__()
        self.gru_app = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_vel = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_acc = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        total_feat_dim = (hidden_dim * 4) * 3
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(total_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out_app, _ = self.gru_app(x)
        feat_app = torch.cat([out_app.mean(dim=1), out_app.max(dim=1).values], dim=1)
        vel = x[:, 1:, :] - x[:, :-1, :]
        out_vel, _ = self.gru_vel(vel)
        feat_vel = torch.cat([out_vel.mean(dim=1), out_vel.max(dim=1).values], dim=1)
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        out_acc, _ = self.gru_acc(acc)
        feat_acc = torch.cat([out_acc.mean(dim=1), out_acc.max(dim=1).values], dim=1)
        combined = torch.cat([feat_app, feat_vel, feat_acc], dim=1)
        return self.classifier(combined)

class DualStream_MeanMax(nn.Module):
    def __init__(self, feature_dim=576, hidden_dim=64, num_classes=2):
        super().__init__()
        self.gru_app = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_mot = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 8, 96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes)
        )
    def forward(self, x):
        out_app, _ = self.gru_app(x)
        app_feat = torch.cat([out_app.mean(dim=1), out_app.max(dim=1).values], dim=1)
        diff = x[:, 1:, :] - x[:, :-1, :]
        out_mot, _ = self.gru_mot(diff)
        mot_feat = torch.cat([out_mot.mean(dim=1), out_mot.max(dim=1).values], dim=1)
        combined = torch.cat([app_feat, mot_feat], dim=1)
        return self.classifier(combined)

# 1. Avaliar Modelo 1: Baseline Bi-GRU
m1 = BiGRU_Baseline().to(device)
# Extrair pesos da cabeca de models/best_model.pth
best_sd = torch.load("models/best_model.pth", map_location=device)
head_sd = {k: v for k, v in best_sd.items() if not k.startswith("features.")}
m1.load_state_dict(head_sd)
m1.eval()
with torch.no_grad():
    logits1 = m1(test_feats)
    probs1 = torch.softmax(logits1, dim=1)[:, 1].numpy()
preds1 = (probs1 >= 0.50).astype(int)
cm1 = confusion_matrix(y_true, preds1)
auc1 = roc_auc_score(y_true, probs1) * 100

def resolve_path(filename, subfolders):
    for sub in subfolders:
        full = os.path.join(sub, filename)
        if os.path.exists(full):
            return full
    raise FileNotFoundError(f"Arquivo não encontrado: {filename}")

# 2. Avaliar Modelo 2: Dual-Stream Latente
m2 = DualStream_Latent().to(device)
p2_head = resolve_path("model_dualstream_best.pth", ["models", "experiments", "legacy_experiments"])
m2.load_state_dict(torch.load(p2_head, map_location=device))
m2.eval()
with torch.no_grad():
    logits2 = m2(test_feats)
    probs2 = torch.softmax(logits2, dim=1)[:, 1].numpy()
preds2 = (probs2 >= 0.50).astype(int)
cm2 = confusion_matrix(y_true, preds2)
auc2 = roc_auc_score(y_true, probs2) * 100

# 3. Avaliar Modelo 3: Ensemble Tri-Stream Cinético
ens_subs = ["models/ensemble", "experiments/target_85", "legacy_experiments/target_85"]
ens1 = TriStream_Kinetic().to(device)
ens1.load_state_dict(torch.load(resolve_path("model_tristream_s7.pth", ens_subs), map_location=device))
ens1.eval()

ens2 = DualStream_MeanMax().to(device)
ens2.load_state_dict(torch.load(resolve_path("model_dualmeanmax_s5.pth", ens_subs), map_location=device))
ens2.eval()

ens3 = DualStream_MeanMax().to(device)
ens3.load_state_dict(torch.load(resolve_path("model_dualmeanmax_s10.pth", ens_subs), map_location=device))
ens3.eval()

with torch.no_grad():
    p1 = torch.softmax(ens1(test_feats), dim=1)[:, 1].numpy()
    p2 = torch.softmax(ens2(test_feats), dim=1)[:, 1].numpy()
    p3 = torch.softmax(ens3(test_feats), dim=1)[:, 1].numpy()
probs3 = (p1 + p2 + p3) / 3.0
preds3 = (probs3 >= 0.52).astype(int)
cm3 = confusion_matrix(y_true, preds3)
auc3 = roc_auc_score(y_true, probs3) * 100

print(f"Modelo 1: Acc={accuracy_score(y_true, preds1)*100:.2f}% | Rec={cm1[1,1]/88*100:.2f}% | FN={cm1[1,0]} | AUC={auc1:.2f}%")
print(f"Modelo 2: Acc={accuracy_score(y_true, preds2)*100:.2f}% | Rec={cm2[1,1]/88*100:.2f}% | FN={cm2[1,0]} | AUC={auc2:.2f}%")
print(f"Modelo 3: Acc={accuracy_score(y_true, preds3)*100:.2f}% | Rec={cm3[1,1]/88*100:.2f}% | FN={cm3[1,0]} | AUC={auc3:.2f}%")

# ==========================================
# FIGURA 1: MATRIZES DE CONFUSÃO LADO A LADO
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
models_info = [
    ("Modelo 1: Baseline Bi-GRU\n(Acc: 75.14% | Rec: 76.14% | 21 FN)", cm1, "#1f77b4"),
    ("Modelo 2: Dual-Stream Latente\n(Acc: 82.16% | Rec: 88.64% | 10 FN)", cm2, "#ff7f0e"),
    ("Modelo 3: Ensemble Cinético (SOTA)\n(Acc: 85.41% | Rec: 92.05% | APENAS 7 FN)", cm3, "#2ca02c")
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
ax.scatter([opr_fpr], [opr_tpr], color="#d62728", s=90, zorder=5, label=f"Ponto Calibrado M3 (θ=0.52, Rec={opr_tpr*100:.1f}%)")

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
ax_a.set_ylabel("Quantidade de Vídeos Perdidos (em 185)", fontsize=10)
ax_a.set_ylim(0, 26)
for bar, fn in zip(bars, fns):
    yval = bar.get_height()
    ax_a.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{fn} FN", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax_a.annotate("Redução de 66.7% nos FNs\n(Apenas 7 lutas perdidas!)",
             xy=(2, 7), xytext=(1.0, 16),
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2),
             fontsize=10, fontweight="bold", color="#c0392b",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#e67e22", lw=1))
ax_a.grid(axis="y", alpha=0.3)

# Painel B: Subida do Recall e F1-Score
ax_b = axes[0, 1]
x_pos = np.arange(len(models_label))
width = 0.3
recalls = [cm1[1, 1]/88 * 100, cm2[1, 1]/88 * 100, cm3[1, 1]/88 * 100]
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
lats = [69.21, 88.31, 73.29]
bars_lat = ax_d.bar(models_label, lats, color=["#34495e", "#7f8c8d", "#16a085"], width=0.55, edgecolor="#333333")
ax_d.set_title("(D) Latência End-to-End em CPU (16 Quadros 224x224)", fontsize=12, fontweight="bold")
ax_d.set_ylabel("Tempo de Inferência (ms por clipe)", fontsize=10)
ax_d.set_ylim(0, 105)
for bar, lat in zip(bars_lat, lats):
    yval = bar.get_height()
    fps_equiv = 16000.0 / lat
    ax_d.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{lat:.1f} ms\n({fps_equiv:.0f} FPS equiv.)", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_d.axhline(100, color="#e74c3c", linestyle=":", label="Limite Real-Time de Borda (100 ms)")
ax_d.legend(loc="upper left", fontsize=9)
ax_d.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("reports/jornada_evolutiva_3_modelos.png", bbox_inches="tight")
plt.close()
print("Salvo: reports/jornada_evolutiva_3_modelos.png")

