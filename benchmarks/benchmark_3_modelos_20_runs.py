import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt

from src.model import BiGRU_Head, DualStream_Head, TriStream_Kinetic
from src.train import FlipAugmentedFeatures

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_DIR = "data/cache"


def load_cache(name):
    path = os.path.join(CACHE_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache de features ausente: {path}\n"
            f"Gere-o com: python src/build_cache.py"
        )
    feats, labels = torch.load(path, map_location="cpu")
    return feats.to(device), labels.to(device)


# Carregar caches de features
train_feats, train_labels = load_cache("features_train.pt")
val_feats, val_labels = load_cache("features_val.pt")
test_feats, test_labels = load_cache("features_test.pt")
y_true = test_labels.detach().cpu().numpy()

# Data augmentation: mesmo protocolo de src/train.py (flip sorteado por época)
FLIP_PATH = os.path.join(CACHE_DIR, "features_train_flip.pt")
if os.path.exists(FLIP_PATH):
    _flip_feats, _ = torch.load(FLIP_PATH, map_location="cpu")
    TRAIN_DATASET = FlipAugmentedFeatures(train_feats, _flip_feats.to(device), train_labels)
    AUGMENT = True
else:
    TRAIN_DATASET = TensorDataset(train_feats, train_labels)
    AUGMENT = False

print(f"Caches carregados: Treino={len(train_labels)}, Val={len(val_labels)}, Teste={len(test_labels)}")
print(f"Dispositivo de Execução: {device} | Data augmentation: {'ativo' if AUGMENT else 'ausente'}")

# ==============================================================================
# 1. Arquiteturas — importadas de src/model.py (fonte única, sem redefinição)
# ==============================================================================

Model1_Baseline_BiGRU = BiGRU_Head          # Aparência estática f_t
Model2_DualStream_Latente = DualStream_Head  # f_t + velocidade Δf_t
Model3_TriStream_Cinetico = TriStream_Kinetic  # f_t + Δf_t + aceleração Δ²f_t

# ==============================================================================
# 2. Protocolo de Treinamento e Avaliação Padronizado
# ==============================================================================

def train_and_eval_single_run(model_cls, seed, fight_weight=1.35, max_epochs=25, patience=5, threshold=0.50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model_cls().to(device)
    train_loader = DataLoader(TRAIN_DATASET, batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=32, shuffle=False)
    
    class_weights = torch.tensor([1.0, fight_weight], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    best_loss = float("inf")
    best_weights = None
    patience_cnt = 0
    stopped_epoch = max_epochs
    
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for f, y in train_loader:
            optimizer.zero_grad()
            out = model(f)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        v_loss, v_tot = 0.0, 0
        with torch.no_grad():
            for f, y in val_loader:
                out = model(f)
                v_loss += criterion(out, y).item() * y.size(0)
                v_tot += y.size(0)
        v_loss /= v_tot
        
        if v_loss < best_loss:
            best_loss = v_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                stopped_epoch = epoch
                break
                
    train_time = time.perf_counter() - t0
    
    # Avaliação no Teste Cego
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        logits = model(test_feats)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    prec *= 100
    rec *= 100
    f1 *= 100
    auc = roc_auc_score(y_true, probs) * 100
    cm = confusion_matrix(y_true, preds)
    
    return {
        "seed": seed,
        "stopped_epoch": stopped_epoch,
        "train_time_s": train_time,
        "best_val_loss": best_loss,
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1_score": f1,
        "auc_roc": auc,
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
        "probs": probs
    }

# ==============================================================================
# 3. Execução do Benchmark Padronizado (20 Sementes por Modelo = 60 Treinamentos)
# ==============================================================================

NUM_SEEDS = 20
architectures = [
    ("Modelo 1: Baseline Bi-GRU", Model1_Baseline_BiGRU, 0.50),
    ("Modelo 2: Dual-Stream Latente", Model2_DualStream_Latente, 0.50),
    ("Modelo 3: Tri-Stream Cinético", Model3_TriStream_Cinetico, 0.50)
]

all_detailed_records = []
all_model_probs = {name: [] for name, _, _ in architectures}

print("\n" + "="*75)
print(f"INICIANDO BENCHMARK PADRONIZADO: 3 MODELOS x {NUM_SEEDS} RUNS INDEPENDENTES")
print("="*75)

for arch_name, arch_cls, default_th in architectures:
    print(f"\n>>> Executando {NUM_SEEDS} runs para: {arch_name}...")
    t_start_arch = time.perf_counter()
    
    for s in range(1, NUM_SEEDS + 1):
        res = train_and_eval_single_run(arch_cls, seed=s, threshold=default_th)
        all_model_probs[arch_name].append(res["probs"])
        
        record = {
            "model": arch_name,
            "seed": s,
            "stopped_epoch": res["stopped_epoch"],
            "accuracy": res["accuracy"],
            "recall": res["recall"],
            "precision": res["precision"],
            "f1_score": res["f1_score"],
            "auc_roc": res["auc_roc"],
            "false_negative": res["false_negative"],
            "false_positive": res["false_positive"],
            "true_positive": res["true_positive"],
            "true_negative": res["true_negative"],
            "best_val_loss": res["best_val_loss"],
            "train_time_s": res["train_time_s"]
        }
        all_detailed_records.append(record)
        print(f"  [Seed {s:2d}/{NUM_SEEDS}] Acc: {res['accuracy']:5.2f}% | Rec: {res['recall']:5.2f}% | Prec: {res['precision']:5.2f}% | FN: {res['false_negative']:2d} | Ep: {res['stopped_epoch']:2d} ({res['train_time_s']:.2f}s)")
        
    dur = time.perf_counter() - t_start_arch
    print(f" Concluído {arch_name} em {dur:.1f}s")

# Salvar detalhes das 60 execuções
df_details = pd.DataFrame(all_detailed_records)
os.makedirs("reports", exist_ok=True)
df_details.to_csv("reports/benchmark_3_modelos_20_runs_detalhes.csv", index=False)
print("\nDetalhes salvos em: reports/benchmark_3_modelos_20_runs_detalhes.csv")

# ==============================================================================
# 4. Cálculo do Sumário Estatístico (Média +- Desvio Padrão)
# ==============================================================================

summary_rows = []
for arch_name, _, _ in architectures:
    sub = df_details[df_details["model"] == arch_name]
    summary_rows.append({
        "Modelo": arch_name,
        "Acurácia Média (%)": f"{sub['accuracy'].mean():.2f} ± {sub['accuracy'].std():.2f}",
        "Acurácia Mín (%)": f"{sub['accuracy'].min():.2f}",
        "Acurácia Máx (%)": f"{sub['accuracy'].max():.2f}",
        "Recall Médio (%)": f"{sub['recall'].mean():.2f} ± {sub['recall'].std():.2f}",
        "Precisão Média (%)": f"{sub['precision'].mean():.2f} ± {sub['precision'].std():.2f}",
        "F1-Score Médio (%)": f"{sub['f1_score'].mean():.2f} ± {sub['f1_score'].std():.2f}",
        "AUC-ROC Médio (%)": f"{sub['auc_roc'].mean():.2f} ± {sub['auc_roc'].std():.2f}",
        "Média Falsos Negativos (FN)": f"{sub['false_negative'].mean():.1f} ± {sub['false_negative'].std():.1f}",
        "Mín Falsos Negativos (FN)": int(sub['false_negative'].min()),
        "Média Falsos Positivos (FP)": f"{sub['false_positive'].mean():.1f} ± {sub['false_positive'].std():.1f}",
        "Época Média Parada": f"{sub['stopped_epoch'].mean():.1f}"
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("reports/benchmark_3_modelos_20_runs_sumario.csv", index=False)
print("Sumário salvo em: reports/benchmark_3_modelos_20_runs_sumario.csv")

print("\n" + "="*85)
print("TABELA ESTATÍSTICA OFICIAL (20 RUNS INDEPENDENTES POR ARQUITETURA)")
print("="*85)
print(df_summary.to_string(index=False))
print("="*85)

# ==============================================================================
# 5. Geração de Gráfico de Boxplots Comparativos (Alta Resolução)
# ==============================================================================

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=300)

palette = ["#2980b9", "#e67e22", "#27ae60"]
labels_clean = ["M1: Baseline\n(Bi-GRU)", "M2: Dual-Stream\n(Aparência + Δf)", "M3: Tri-Stream\n(Δf + Δ²f Cinético)"]

# Subplot 1: Acurácia Global
data_acc = [df_details[df_details["model"] == m]["accuracy"].values for m, _, _ in architectures]
bp1 = axes[0].boxplot(data_acc, patch_artist=True, tick_labels=labels_clean, medianprops=dict(color="black", lw=2))
for patch, color in zip(bp1["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[0].set_title("Distribuição da Acurácia Global (%)", fontsize=12, fontweight="bold", pad=10)
axes[0].set_ylabel("Acurácia no Teste Cego (%)", fontsize=11)
axes[0].grid(True, alpha=0.3)

# Subplot 2: Recall Fight (Sensibilidade)
data_rec = [df_details[df_details["model"] == m]["recall"].values for m, _, _ in architectures]
bp2 = axes[1].boxplot(data_rec, patch_artist=True, tick_labels=labels_clean, medianprops=dict(color="black", lw=2))
for patch, color in zip(bp2["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1].set_title("Distribuição do Recall Fight (%)", fontsize=12, fontweight="bold", pad=10)
axes[1].set_ylabel("Sensibilidade (Recall Fight %)", fontsize=11)
axes[1].grid(True, alpha=0.3)

# Subplot 3: Falsos Negativos (Lutas Perdidas)
data_fn = [df_details[df_details["model"] == m]["false_negative"].values for m, _, _ in architectures]
bp3 = axes[2].boxplot(data_fn, patch_artist=True, tick_labels=labels_clean, medianprops=dict(color="black", lw=2))
for patch, color in zip(bp3["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[2].set_title("Falsos Negativos (Lutas Perdidas)", fontsize=12, fontweight="bold", pad=10)
axes[2].set_ylabel("Quantidade de Vídeos Perdidos", fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.suptitle("Validação Estatística de Robustez: 20 Execuções Independentes por Arquitetura", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

boxplot_path = "reports/benchmark_3_modelos_20_runs_boxplots.png"
plt.savefig(boxplot_path, bbox_inches="tight")
plt.close()

print(f"\nGráfico de Boxplots salvo com sucesso em: {boxplot_path}")
print("Benchmark de 20 runs concluído com sucesso!")

