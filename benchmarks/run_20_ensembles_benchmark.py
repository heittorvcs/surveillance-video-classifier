"""
Treina pools de sementes e avalia 20 comites triplos formados SEM selecao.

O objetivo deste benchmark e medir a distribuicao honesta do ensemble: os trios
sao formados por uma regra fixa de sementes, nao escolhidos pelo desempenho no
teste. E o contraste entre esta distribuicao e o "melhor checkpoint" que revela
quanto do numero de vitrine vem de selecao.

Pre-requisito: python src/build_cache.py
"""

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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.model import DualStream_MeanMax, TriStream_Kinetic
from src.train import FlipAugmentedFeatures

device = torch.device("cpu")
CACHE_DIR = "data/cache"


def load_cache(name):
    path = os.path.join(CACHE_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Cache de features ausente: " + path + " | Gere-o com: python src/build_cache.py"
        )
    feats, labels = torch.load(path, map_location="cpu")
    return feats.to(device), labels.to(device)


train_feats, train_labels = load_cache("features_train.pt")
val_feats, val_labels = load_cache("features_val.pt")
test_feats, test_labels = load_cache("features_test.pt")
y_true = test_labels.detach().cpu().numpy()

FLIP_PATH = os.path.join(CACHE_DIR, "features_train_flip.pt")
if os.path.exists(FLIP_PATH):
    _flip_feats, _ = torch.load(FLIP_PATH, map_location="cpu")
    TRAIN_DATASET = FlipAugmentedFeatures(train_feats, _flip_feats.to(device), train_labels)
    AUGMENT = True
else:
    TRAIN_DATASET = TensorDataset(train_feats, train_labels)
    AUGMENT = False

print(f"Caches carregados: Treino={len(train_labels)}, Val={len(val_labels)}, Teste={len(test_labels)}")
print(f"Data augmentation: {'ativo' if AUGMENT else 'ausente'}")

# As arquiteturas vem de src/model.py — nenhuma redefinicao local.

# 2. Função de Treinamento
def train_model(model_cls, seed, fight_weight=1.35, max_epochs=25, patience=5):
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
                break
                
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        logits = model(test_feats)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
    return probs, best_loss

# 3. Treinar Pool de 20 Sementes para TriStream e 20 para DualStream
NUM_RUNS = 20
tristream_probs = {}
dualstream_probs = {}

print("\n" + "="*70)
print(f"1. Treinando Pool de {NUM_RUNS} Sementes para TriStream_Kinetic...")
print("="*70)
t0 = time.perf_counter()
for s in range(1, NUM_RUNS + 1):
    p, vloss = train_model(TriStream_Kinetic, seed=s)
    tristream_probs[s] = p
    print(f"  TriStream Seed {s:2d} treinado (ValLoss: {vloss:.4f})", flush=True)

print(f"TriStream concluído em {time.perf_counter() - t0:.1f}s")

print("\n" + "="*70)
print(f"2. Treinando Pool de {NUM_RUNS} Sementes para DualStream_MeanMax...")
print("="*70)
t0 = time.perf_counter()
for s in range(1, NUM_RUNS + 1):
    p, vloss = train_model(DualStream_MeanMax, seed=s)
    dualstream_probs[s] = p
    print(f"  DualStream Seed {s:2d} treinado (ValLoss: {vloss:.4f})", flush=True)

print(f"DualStream concluído em {time.perf_counter() - t0:.1f}s")

# 4. Formar e Avaliar 20 Ensembles Triplos Independentes
# Cada Ensemble k combina 1 TriStream e 2 DualStreams com sementes alternadas
print("\n" + "="*70)
print(f"3. Avaliando 20 Ensembles Triplos Independentes (Limiar θ = 0.52)...")
print("="*70)

ensemble_records = []
# ATENCAO: 0.52 e o limiar herdado da selecao original feita sobre o teste.
# Para uma leitura nao enviesada, calibre na validacao com
# src/calibrate_threshold.py e passe o valor obtido em THRESHOLD.
th = float(os.environ.get("ENSEMBLE_THRESHOLD", 0.52))

for k in range(1, NUM_RUNS + 1):
    # Seleção dos 3 membros com diversidade de sementes
    s_tri = k
    s_dual1 = ((k + 4) % NUM_RUNS) + 1
    s_dual2 = ((k + 9) % NUM_RUNS) + 1
    
    p1 = tristream_probs[s_tri]
    p2 = dualstream_probs[s_dual1]
    p3 = dualstream_probs[s_dual2]
    
    ens_prob = (p1 + p2 + p3) / 3.0
    preds = (ens_prob >= th).astype(int)
    
    acc = accuracy_score(y_true, preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    prec *= 100
    rec *= 100
    f1 *= 100
    auc = roc_auc_score(y_true, ens_prob) * 100
    cm = confusion_matrix(y_true, preds)
    
    rec_dict = {
        "ensemble_id": k,
        "combo": f"Tri(s{s_tri}) + Dual(s{s_dual1}) + Dual(s{s_dual2})",
        "threshold": th,
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1_score": f1,
        "auc_roc": auc,
        "false_negative": int(cm[1, 0]),
        "false_positive": int(cm[0, 1]),
        "true_positive": int(cm[1, 1]),
        "true_negative": int(cm[0, 0])
    }
    ensemble_records.append(rec_dict)
    print(f"  Ensemble {k:2d}: Acc={acc:5.2f}% | Rec={rec:5.2f}% | Prec={prec:5.2f}% | F1={f1:5.2f}% | FN={cm[1,0]:2d} | FP={cm[0,1]:2d} ({rec_dict['combo']})", flush=True)

df_ens = pd.DataFrame(ensemble_records)
os.makedirs("reports", exist_ok=True)
df_ens.to_csv("reports/benchmark_20_ensembles_detalhes.csv", index=False)

print("\n" + "="*75)
print("SUMÁRIO ESTATÍSTICO DOS 20 ENSEMBLES:")
print("="*75)
print(f"Acurácia Média:       {df_ens['accuracy'].mean():.2f}% ± {df_ens['accuracy'].std():.2f}% (Faixa: [{df_ens['accuracy'].min():.2f}% - {df_ens['accuracy'].max():.2f}%])")
print(f"Recall Médio:         {df_ens['recall'].mean():.2f}% ± {df_ens['recall'].std():.2f}% (Faixa: [{df_ens['recall'].min():.2f}% - {df_ens['recall'].max():.2f}%])")
print(f"Precisão Média:       {df_ens['precision'].mean():.2f}% ± {df_ens['precision'].std():.2f}%")
print(f"F1-Score Médio:       {df_ens['f1_score'].mean():.2f}% ± {df_ens['f1_score'].std():.2f}%")
print(f"AUC-ROC Média:        {df_ens['auc_roc'].mean():.2f}% ± {df_ens['auc_roc'].std():.2f}%")
print(f"Falsos Negativos (FN): {df_ens['false_negative'].mean():.1f} ± {df_ens['false_negative'].std():.1f} (Mínimo: {int(df_ens['false_negative'].min())} FN)")
print(f"Falsos Positivos (FP): {df_ens['false_positive'].mean():.1f} ± {df_ens['false_positive'].std():.1f}")
print("="*75)

