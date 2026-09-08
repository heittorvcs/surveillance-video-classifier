"""
Gera as particoes treino / validacao / teste a partir do RWF-2000 bruto.

Fonte do dataset: RWF-2000 (Cheng, Cai & Li, 2020) —
https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
Espelho comum: https://www.kaggle.com/datasets/vulamnguyen/rwf2000

Layout esperado apos a extracao:
    archive/RWF-2000/train/Fight/*.avi
    archive/RWF-2000/train/NonFight/*.avi
    archive/RWF-2000/val/Fight/*.avi
    archive/RWF-2000/val/NonFight/*.avi

Anti-leakage: no RWF-2000 varios clipes vem de cortes temporais do mesmo video
original — mesma camera, mesmo fundo, mesma iluminacao. Um split aleatorio faria
o modelo memorizar cenarios. Aqui o identificador do video original e derivado do
nome do arquivo (prefixo antes do sufixo `_<n>.avi`) e usado como grupo:

  - treino:            os 1.600 videos da pasta oficial `train`;
  - validacao/teste:   os 400 videos da pasta oficial `val`, divididos 50/50 por
                       GRUPO com GroupShuffleSplit, para que nenhuma camera
                       apareca dos dois lados.

Ao final o script verifica explicitamente que a intersecao de grupos entre os tres
splits e vazia e falha se nao for.

Uso:
    python src/create_splits.py
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd

CLASSES = {"NonFight": 0, "Fight": 1}
GROUP_SUFFIX = re.compile(r"_\d+\.avi$")


def group_id(filename):
    """Identificador do video original: nome do arquivo sem o sufixo `_<n>.avi`."""
    return GROUP_SUFFIX.sub("", os.path.basename(filename))


def get_video_records(dir_path):
    records = []
    for class_name, label in CLASSES.items():
        folder = Path(dir_path) / class_name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Pasta do dataset nao encontrada: {folder}\n"
                "Baixe o RWF-2000 e extraia em archive/RWF-2000/ — ver o cabecalho deste arquivo."
            )
        for video_file in sorted(folder.glob("*.avi")):
            records.append({
                "video_path": str(video_file.as_posix()),
                "label": label,
                "label_name": class_name,
                "group_id": group_id(video_file.name),
            })
    if not records:
        raise RuntimeError(f"Nenhum .avi encontrado em {dir_path}")
    return pd.DataFrame(records)


def assert_no_leakage(splits):
    """Falha se qualquer grupo (video original) aparecer em mais de um split."""
    groups = {name: set(df["group_id"]) for name, df in splits.items()}
    files = {name: set(df["video_path"].map(os.path.basename)) for name, df in splits.items()}

    problems = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared_groups = groups[a] & groups[b]
        shared_files = files[a] & files[b]
        if shared_groups:
            problems.append(f"{a}/{b}: {len(shared_groups)} grupos compartilhados "
                            f"(ex.: {sorted(shared_groups)[:3]})")
        if shared_files:
            problems.append(f"{a}/{b}: {len(shared_files)} arquivos compartilhados")

    print("\nVerificacao anti-leakage:")
    for name in ("train", "val", "test"):
        print(f"  {name:6s} {len(splits[name]):5d} videos | {len(groups[name]):4d} grupos unicos")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"  intersecao {a}/{b}: {len(groups[a] & groups[b])} grupos")

    if problems:
        print("\nFALHA DE ANTI-LEAKAGE:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    print("  OK: nenhum grupo ou arquivo atravessa a fronteira dos splits.\n")


def main():
    # Import tardio: assim group_id() e assert_no_leakage() podem ser reutilizados
    # (pelos testes, por exemplo) sem exigir scikit-learn instalado.
    from sklearn.model_selection import GroupShuffleSplit

    root = Path("archive/RWF-2000")
    os.makedirs("data/splits", exist_ok=True)

    df_train = get_video_records(root / "train")

    # A pasta oficial `val` (400 videos) vira validacao + teste, dividida por grupo.
    df_val_pool = get_video_records(root / "val")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss.split(df_val_pool, df_val_pool["label"], groups=df_val_pool["group_id"]))
    df_val = df_val_pool.iloc[val_idx].copy()
    df_test = df_val_pool.iloc[test_idx].copy()

    splits = {"train": df_train, "val": df_val, "test": df_test}
    assert_no_leakage(splits)

    for name, df in splits.items():
        csv_path = f"data/splits/{name}.csv"
        df[["video_path", "label", "label_name"]].to_csv(csv_path, index=False)
        print(f"[{name.upper():5s}] {csv_path} - {len(df)} videos "
              f"(Fight: {(df['label'] == 1).sum()}, NonFight: {(df['label'] == 0).sum()})")

    print("\nProximo passo: python src/build_cache.py")


if __name__ == "__main__":
    main()
