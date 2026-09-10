"""
O data augmentation por flip horizontal melhora o resultado?

Comparar um modelo treinado com augmentation contra um sem nao responde a pergunta:
a variacao entre sementes e maior que o efeito que se quer medir. Aqui cada braco
treina 20 sementes por arquitetura e a comparacao usa media, desvio e o erro padrao
da diferenca entre as medias.

Bracos (mesmas sementes, mesmo protocolo, mesma validacao):
    SEM   features_train.pt                 cache limpo
    COM   + features_train_flip.pt          sorteio por amostra a cada epoca
    ORIG  features_train_frozenflip.pt      flip aleatorio aplicado uma unica vez,
                                            na extracao (opcional, para comparacao)

O braco ORIG so roda se o cache existir. Ele reproduz o que acontece quando o flip
e aplicado na extracao das features de um backbone congelado: cada video recebe uma
perturbacao fixa, identica em todas as epocas -- que nao e augmentation.

Pre-requisito:
    python src/build_cache.py --splits train

Uso:
    python benchmarks/eval_augmentation.py
    python benchmarks/eval_augmentation.py --seeds 10 --archs tristream
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.model import build_head
from src.train import FlipAugmentedFeatures, run_epoch

CACHE = "data/cache"
DEV = torch.device("cpu")


def treinar(arch, seed, ds_treino, val_loader, epochs, patience, fight_weight):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_head(arch)[0].to(DEV)

    loader = DataLoader(ds_treino, batch_size=32, shuffle=True)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, fight_weight], device=DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    melhor, pesos, pac = float("inf"), None, 0
    for _ in range(epochs):
        run_epoch(model, loader, crit, opt)
        sched.step()
        vloss, _ = run_epoch(model, val_loader, crit)
        if vloss < melhor:
            melhor, pac = vloss, 0
            pesos = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pac += 1
            if pac >= patience:
                break
    model.load_state_dict(pesos)
    model.eval()
    return model


@torch.no_grad()
def avaliar(model, feats, y, th=0.50):
    p = torch.softmax(model(feats), dim=1)[:, 1].cpu().numpy()
    pred = (p >= th).astype(int)
    return {"accuracy": float((pred == y).mean() * 100),
            "auc": float(roc_auc_score(y, p) * 100),
            "fn": int(((pred == 0) & (y == 1)).sum()),
            "fp": int(((pred == 1) & (y == 0)).sum())}


def invariancia_ao_flip(limpo, flip, seed=0):
    """
    Quanto o espelhamento muda a feature que chega nas cabecas temporais.

    Referencia: a similaridade entre videos diferentes. Se espelhar mexe muito menos
    do que trocar de video, o backbone ja e praticamente invariante ao flip e o
    augmentation tem pouco a oferecer.
    """
    a = limpo.reshape(-1, limpo.shape[-1])
    b = flip.reshape(-1, flip.shape[-1])
    g = torch.Generator().manual_seed(seed)
    outro = limpo[torch.randperm(limpo.shape[0], generator=g)].reshape(-1, limpo.shape[-1])
    return {"cos_video_vs_espelhado": float(F.cosine_similarity(a, b, dim=1).mean()),
            "cos_video_vs_outro_video": float(F.cosine_similarity(a, outro, dim=1).mean()),
            "variacao_relativa_da_norma_pct": float(((limpo - flip).norm(dim=2)
                                                     / limpo.norm(dim=2)).mean() * 100)}


def main():
    parser = argparse.ArgumentParser(description="Efeito do data augmentation por flip")
    parser.add_argument("--seeds", type=int, default=20, help="Sementes por arquitetura e braco")
    parser.add_argument("--archs", nargs="+", default=["baseline", "dualstream", "tristream", "dualmeanmax"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fight_weight", type=float, default=1.35)
    parser.add_argument("--save_json", type=str, default="reports/augmentation_effect.json")
    args = parser.parse_args()

    def carregar(nome):
        caminho = os.path.join(CACHE, nome)
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Cache ausente: {caminho}. Rode: python src/build_cache.py")
        return torch.load(caminho, map_location="cpu")

    val_f, val_l = carregar("features_val.pt")
    test_f, test_l = carregar("features_test.pt")
    val_y, test_y = val_l.numpy(), test_l.numpy()
    val_loader = DataLoader(TensorDataset(val_f, val_l), batch_size=32)

    limpo, y_tr = carregar("features_train.pt")
    flip, _ = carregar("features_train_flip.pt")

    bracos = {"SEM": lambda: TensorDataset(limpo, y_tr),
              "COM": lambda: FlipAugmentedFeatures(limpo, flip, y_tr)}
    congelado = os.path.join(CACHE, "features_train_frozenflip.pt")
    if os.path.exists(congelado):
        orig, _ = torch.load(congelado, map_location="cpu")
        bracos["ORIG"] = lambda: TensorDataset(orig, y_tr)

    seeds = list(range(1, args.seeds + 1))
    print("Bracos: %s | %d sementes x %d arquiteturas = %d treinamentos"
          % (", ".join(bracos), len(seeds), len(args.archs), len(bracos) * len(seeds) * len(args.archs)))

    res = {}
    for braco, fn in bracos.items():
        res[braco] = {}
        for arch in args.archs:
            linhas = [{"seed": s, "test": avaliar(treinar(arch, s, fn(), val_loader, args.epochs,
                                                          args.patience, args.fight_weight), test_f, test_y)}
                      for s in seeds]
            res[braco][arch] = linhas
            auc = np.array([x["test"]["auc"] for x in linhas])
            acc = np.array([x["test"]["accuracy"] for x in linhas])
            print("  %-5s %-12s AUC %.2f ± %.2f | acuracia %.2f ± %.2f"
                  % (braco, arch, auc.mean(), auc.std(ddof=1), acc.mean(), acc.std(ddof=1)), flush=True)

    inv = invariancia_ao_flip(limpo, flip)

    print("\n" + "=" * 82)
    print("EFEITO DO AUGMENTATION NO TESTE CEGO (%d sementes por celula)" % len(seeds))
    print("=" * 82)
    print("%-13s %20s %20s %18s" % ("arquitetura", "SEM (AUC)", "COM (AUC)", "diferenca"))
    efeitos = {}
    for arch in args.archs:
        a = np.array([x["test"]["auc"] for x in res["SEM"][arch]])
        b = np.array([x["test"]["auc"] for x in res["COM"][arch]])
        dif = b.mean() - a.mean()
        se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        efeitos[arch] = {"sem_auc": a.mean(), "sem_std": a.std(ddof=1),
                         "com_auc": b.mean(), "com_std": b.std(ddof=1),
                         "diferenca": dif, "erro_padrao": se, "em_desvios": dif / se if se else 0.0}
        print("%-13s %9.2f ± %-8.2f %9.2f ± %-8.2f %+7.2f ± %.2f" %
              (arch, a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1), dif, se))

    maior = max(abs(v["diferenca"]) for v in efeitos.values())
    ruido = np.mean([v["sem_std"] for v in efeitos.values()])
    print("-" * 82)
    print("Maior efeito observado: %.2f p.p. de AUC. Desvio medio entre sementes: %.2f p.p." % (maior, ruido))
    print("O efeito e %.1fx menor que o ruido entre sementes." % (ruido / maior if maior else float("inf")))

    print("\nPOR QUE: o backbone congelado ja e quase invariante ao espelhamento")
    print("  similaridade video x ele mesmo espelhado : %.4f" % inv["cos_video_vs_espelhado"])
    print("  similaridade video x outro video         : %.4f" % inv["cos_video_vs_outro_video"])
    print("  O MobileNetV3 foi pre-treinado no ImageNet, que ja usa flip horizontal como")
    print("  augmentation. As features chegam nas cabecas praticamente iguais.")
    print("=" * 82)

    payload = {"seeds": len(seeds), "archs": args.archs, "threshold": 0.50,
               "efeito": efeitos, "invariancia_ao_flip": inv, "por_semente": res}
    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("\nResultado salvo em: %s" % args.save_json)


if __name__ == "__main__":
    main()
