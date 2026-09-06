import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import time
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import RWF2000Dataset
from src.model import VideoClassifier

def extract_features(dataset, backbone, avgpool, device, desc="Extraindo Features"):
    """
    Passa os videos pelo backbone congelado e extrai vetores espaciais (T=16, D=576).
    Isso acelera o treinamento da GRU em 100x no CPU.
    """
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    all_features = []
    all_labels = []
    
    backbone.eval()
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc=desc):
            b, t, c, h, w = videos.shape
            x = videos.view(b * t, c, h, w).to(device)
            feat = backbone(x)
            feat = avgpool(feat)
            feat = torch.flatten(feat, 1).view(b, t, -1)  # (b, 16, 576)
            all_features.append(feat.cpu())
            all_labels.append(labels)
            
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

def main():
    parser = argparse.ArgumentParser(description="Treinamento do Classificador de Violencia")
    parser.add_argument("--epochs", type=int, default=20, help="Numero de epocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do lote")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado")
    parser.add_argument("--fight_weight", type=float, default=1.35, help="Peso da classe Fight na loss para mitigar falsos negativos")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade cientifica")
    parser.add_argument("--patience", type=int, default=5, help="Paciencia para Early Stopping baseado na val_loss")
    args = parser.parse_args()

    # Reprodutibilidade estrita
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de execucao: {device} | Seed: {args.seed}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)

    # 1. Instancia o modelo completo
    model = VideoClassifier(num_classes=2, freeze_backbone=True).to(device)

    # 2. Caching de features espaciais para treino ultra-rapido
    train_cache_path = "data/cache/features_train.pt"
    val_cache_path = "data/cache/features_val.pt"

    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        print("Carregando features em cache...")
        train_feats, train_labels = torch.load(train_cache_path)
        val_feats, val_labels = torch.load(val_cache_path)
    else:
        print("Extraindo features do MobileNetV3 (uma unica vez)...")
        train_ds = RWF2000Dataset("data/splits/train.csv", num_frames=16, is_train=True)
        val_ds = RWF2000Dataset("data/splits/val.csv", num_frames=16, is_train=False)

        train_feats, train_labels = extract_features(train_ds, model.features, model.avgpool, device, "Treino")
        val_feats, val_labels = extract_features(val_ds, model.features, model.avgpool, device, "Validacao")

        torch.save((train_feats, train_labels), train_cache_path)
        torch.save((val_feats, val_labels), val_cache_path)
        print(f"Features salvas em data/cache/ (Treino: {train_feats.shape}, Val: {val_feats.shape})")

    train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=args.batch_size, shuffle=False)

    # 3. Otimizacao e Criterio (com peso maior em Fight para penalizar falsos negativos)
    class_weights = torch.tensor([1.0, args.fight_weight], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        list(model.gru.parameters()) + list(model.classifier.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\nIniciando treinamento ({args.epochs} epocas max, patience={args.patience})...")
    for epoch in range(1, args.epochs + 1):
        # Treino
        model.gru.train()
        model.classifier.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            
            gru_out, _ = model.gru(feats)
            logits = model.classifier(gru_out.mean(dim=1))
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        # Validacao
        model.gru.eval()
        model.classifier.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                gru_out, _ = model.gru(feats)
                logits = model.classifier(gru_out.mean(dim=1))
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoca [{epoch:02d}/{args.epochs:02d}] - "
              f"Treino Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc*100:.1f}%")

        # Checkpoint baseado na menor val_loss (impede overfitting de epocas tardias)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"  --> [Checkpoint] Novo melhor modelo salvo! (Val Loss: {best_val_loss:.4f}, Acc: {best_val_acc*100:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] Parada antecipada acionada apos {args.patience} epocas consecutivas sem melhora na perda de validacao.")
                break

    print(f"\nMelhor Modelo - Val Loss: {best_val_loss:.4f} | Val Acc: {best_val_acc*100:.2f}%")
    print("Modelo salvo em 'models/best_model.pth'")

    # 4. Salva historico em JSON e Graficos
    with open("reports/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Treino")
    plt.plot(history["val_loss"], label="Validacao")
    plt.title("Curva de Loss")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([a * 100 for a in history["train_acc"]], label="Treino")
    plt.plot([a * 100 for a in history["val_acc"]], label="Validacao")
    plt.title("Curva de Acuracia (%)")
    plt.xlabel("Epoca")
    plt.ylabel("Acuracia (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("reports/training_curves.png", dpi=300)
    print("Graficos salvos em 'reports/training_curves.png'")

if __name__ == "__main__":
    main()

