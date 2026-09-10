"""
Extrai as features espaciais do backbone congelado e as materializa em disco.

Este passo e pre-requisito de src/train.py, src/evaluate.py, src/calibrate_threshold.py
e dos scripts em benchmarks/. Como o MobileNetV3 fica congelado, as features de um
video nunca mudam: extrai-las uma vez transforma o treino das cabecas recorrentes
em segundos por epoca.

Caches produzidos em data/cache/:
  features_train.pt       features do split de treino
  features_train_flip.pt  mesmas features com flip horizontal (data augmentation)
  features_val.pt         split de validacao (selecao de modelo e limiar)
  features_test.pt        split de teste cego (avaliacao final)

Uso:
    python src/build_cache.py                 # todos os splits
    python src/build_cache.py --splits test   # apenas um split
    python src/build_cache.py --force         # reextrai mesmo se o cache existir
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from tqdm import tqdm

from src.dataset import RWF2000Dataset

CACHE_DIR = "data/cache"


def build_backbone(device):
    """MobileNetV3-Small pre-treinado no ImageNet, congelado, em modo eval."""
    base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    backbone = base.features.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device).eval()
    return backbone, avgpool


@torch.no_grad()
def extract_features(dataset, backbone, avgpool, device, batch_size=8, desc="Extraindo"):
    """Retorna (features (N, 16, 576), labels (N,)) em CPU."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_features, all_labels = [], []

    for videos, labels in tqdm(loader, desc=desc):
        b, t, c, h, w = videos.shape
        x = videos.view(b * t, c, h, w).to(device)
        feat = avgpool(backbone(x)).flatten(1).view(b, t, -1)
        all_features.append(feat.cpu())
        all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def build(split, flip, backbone, avgpool, device, force=False, root=None):
    suffix = "_flip" if flip else ""
    out_path = os.path.join(CACHE_DIR, f"features_{split}{suffix}.pt")

    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} ja existe (use --force para reextrair)")
        return

    csv_path = f"data/splits/{split}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Split nao encontrado: {csv_path}. Rode 'python src/create_splits.py' "
            f"com o dataset RWF-2000 em archive/RWF-2000/."
        )

    dataset = RWF2000Dataset(csv_path, num_frames=16, flip=flip, root=root)
    label = f"{split}{' (flip)' if flip else ''}"
    feats, labels = extract_features(dataset, backbone, avgpool, device, desc=label)

    torch.save((feats, labels), out_path)
    print(f"[ok] {out_path} — features {tuple(feats.shape)}, labels {tuple(labels.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Extracao e cache das features do backbone congelado")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"], help="Splits a processar")
    parser.add_argument("--no_flip_cache", action="store_true",
                        help="Nao gera o cache espelhado do treino (desativa o data augmentation)")
    parser.add_argument("--force", action="store_true", help="Reextrai mesmo se o cache ja existir")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Raiz alternativa do dataset, substituindo o prefixo 'archive/' dos CSVs. "
                             "Use quando o dataset estiver fora do projeto ou quando os caminhos "
                             "estourarem o limite de 260 caracteres do Windows.")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de extracao: {device}")

    backbone, avgpool = build_backbone(device)

    for split in args.splits:
        build(split, flip=False, backbone=backbone, avgpool=avgpool, device=device,
              force=args.force, root=args.dataset_root)
        if split == "train" and not args.no_flip_cache:
            build(split, flip=True, backbone=backbone, avgpool=avgpool, device=device,
                  force=args.force, root=args.dataset_root)

    print("\nCaches disponiveis em data/cache/:")
    for name in sorted(os.listdir(CACHE_DIR)):
        size_mb = os.path.getsize(os.path.join(CACHE_DIR, name)) / 1e6
        print(f"  {name:28s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
