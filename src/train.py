"""
Treina qualquer uma das cabecas temporais sobre as features do backbone congelado.

Pre-requisito: os caches de features. Gere-os antes com:
    python src/build_cache.py

Exemplos:
    python src/train.py --arch baseline
    python src/train.py --arch dualstream --seed 7
    python src/train.py --arch tristream  --seed 7 --out models/ensemble/model_tristream_s7.pth
    python src/train.py --arch dualmeanmax --seed 5 --out models/ensemble/model_dualmeanmax_s5.pth

Data augmentation: se data/cache/features_train_flip.pt existir, cada amostra e
sorteada a cada epoca entre a versao original e a espelhada. Sem esse cache o
treino roda sem augmentation (e avisa).
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.model import build_head

CACHE_DIR = "data/cache"


class FlipAugmentedFeatures(Dataset):
    """
    Sorteia, por amostra e a cada acesso, entre as features originais e as espelhadas.

    E isto que torna o flip um data augmentation de fato: com o backbone congelado,
    aplicar o flip apenas na extracao produziria uma perturbacao fixa por video,
    identica em todas as epocas.
    """

    def __init__(self, feats, feats_flip, labels, p=0.5):
        assert feats.shape == feats_flip.shape
        self.feats = feats
        self.feats_flip = feats_flip
        self.labels = labels
        self.p = p

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        source = self.feats_flip if torch.rand(1).item() < self.p else self.feats
        return source[idx], self.labels[idx]


def require_cache(name):
    path = os.path.join(CACHE_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cache de features ausente: {path}\n"
            f"Gere-o com: python src/build_cache.py"
        )
    return path


def load_cache(name, device):
    feats, labels = torch.load(require_cache(name), map_location="cpu")
    return feats.to(device), labels.to(device)


def build_train_loader(args, device):
    train_feats, train_labels = load_cache("features_train.pt", device)

    flip_path = os.path.join(CACHE_DIR, "features_train_flip.pt")
    if os.path.exists(flip_path) and not args.no_augment:
        flip_feats, _ = torch.load(flip_path, map_location="cpu")
        dataset = FlipAugmentedFeatures(train_feats, flip_feats.to(device), train_labels)
        print("Data augmentation ativo: flip horizontal sorteado por amostra a cada epoca.")
    else:
        dataset = TensorDataset(train_feats, train_labels)
        if args.no_augment:
            print("Data augmentation desativado por --no_augment.")
        else:
            print("AVISO: features_train_flip.pt ausente - treinando SEM data augmentation.\n"
                  "       Gere o cache espelhado com: python src/build_cache.py --splits train")

    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True), len(train_labels)


def run_epoch(model, loader, criterion, optimizer=None):
    """Uma passada completa. Com optimizer treina; sem optimizer avalia."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for feats, labels in loader:
            if training:
                optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def save_curves(history, out_png):
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
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Treinamento das cabecas temporais sobre features cacheadas")
    parser.add_argument("--arch", type=str, default="baseline",
                        choices=["baseline", "dualstream", "tristream", "dualmeanmax"],
                        help="Arquitetura da cabeca temporal")
    parser.add_argument("--epochs", type=int, default=25, help="Numero maximo de epocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do lote")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado")
    parser.add_argument("--fight_weight", type=float, default=1.35,
                        help="Peso da classe Fight na loss (custo assimetrico FN vs FP; o dataset e balanceado)")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatoria")
    parser.add_argument("--patience", type=int, default=5, help="Paciencia do early stopping (val_loss)")
    parser.add_argument("--no_augment", action="store_true", help="Desativa o augmentation por flip")
    parser.add_argument("--out", type=str, default=None,
                        help="Caminho do checkpoint (padrao: models/<arch>_seed<seed>.pth)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | Arquitetura: {args.arch} | Seed: {args.seed}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    out_path = args.out or f"models/{args.arch}_seed{args.seed}.pth"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    train_loader, n_train = build_train_loader(args, device)
    val_feats, val_labels = load_cache("features_val.pt", device)
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=args.batch_size, shuffle=False)
    print(f"Treino: {n_train} videos | Validacao: {len(val_labels)} videos")

    model, desc = build_head(args.arch)
    model = model.to(device)

    class_weights = torch.tensor([1.0, args.fight_weight], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch, patience_counter = 0, 0

    print(f"\n{desc} - treinando ate {args.epochs} epocas (patience={args.patience})\n")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoca [{epoch:02d}/{args.epochs:02d}] "
              f"Treino Loss {train_loss:.4f} Acc {train_acc*100:5.1f}% | "
              f"Val Loss {val_loss:.4f} Acc {val_acc*100:5.1f}%")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            torch.save(model.state_dict(), out_path)
            print(f"  -> checkpoint salvo em {out_path} (val_loss {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Early Stopping] {args.patience} epocas sem melhora na val_loss.")
                break

    print(f"\nMelhor epoca: {best_epoch} | val_loss {best_val_loss:.4f} | checkpoint: {out_path}")

    # Artefatos por semente vao para reports/pool/ (nao versionado): treinar um pool
    # de 20 sementes por arquitetura geraria 160 arquivos no meio de reports/.
    # As curvas dos modelos efetivamente selecionados ficam em reports/selected/.
    tag = f"{args.arch}_seed{args.seed}"
    os.makedirs("reports/pool", exist_ok=True)
    history_path = f"reports/pool/training_history_{tag}.json"
    curves_path = f"reports/pool/training_curves_{tag}.png"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "arch": args.arch,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "augmentation": not args.no_augment,
            **history,
        }, f, indent=2)

    save_curves(history, curves_path)
    print(f"Historico: {history_path}")
    print(f"Curvas:    {curves_path}")
    print("\nProximo passo: python src/evaluate.py --model baseline|dualstream|ensemble")


if __name__ == "__main__":
    main()
