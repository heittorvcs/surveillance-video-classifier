"""
Testes das particoes anti-leakage.

Rodam sobre os CSVs versionados em data/splits/, sem precisar do dataset bruto.

    pytest tests/            # ou
    python tests/test_splits.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.create_splits import group_id

SPLITS_DIR = Path("data/splits")
SPLIT_NAMES = ("train", "val", "test")
EXPECTED_SIZES = {"train": 1600, "val": 215, "test": 185}


def load_split(name):
    df = pd.read_csv(SPLITS_DIR / f"{name}.csv")
    df["group_id"] = df["video_path"].map(group_id)
    df["basename"] = df["video_path"].map(os.path.basename)
    return df


def test_splits_exist_and_have_expected_sizes():
    for name in SPLIT_NAMES:
        df = load_split(name)
        assert len(df) == EXPECTED_SIZES[name], (
            f"{name}: esperado {EXPECTED_SIZES[name]} videos, encontrado {len(df)}"
        )


def test_labels_are_binary_and_consistent():
    for name in SPLIT_NAMES:
        df = load_split(name)
        assert set(df["label"].unique()) <= {0, 1}
        mapping = {"NonFight": 0, "Fight": 1}
        assert (df["label_name"].map(mapping) == df["label"]).all(), (
            f"{name}: label_name e label divergem"
        )


def test_no_group_leakage_between_splits():
    """Nenhum video original (grupo/camera) pode aparecer em dois splits."""
    groups = {name: set(load_split(name)["group_id"]) for name in SPLIT_NAMES}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = groups[a] & groups[b]
        assert not shared, f"Vazamento entre {a} e {b}: {len(shared)} grupos, ex. {sorted(shared)[:3]}"


def test_no_file_leakage_between_splits():
    files = {name: set(load_split(name)["basename"]) for name in SPLIT_NAMES}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = files[a] & files[b]
        assert not shared, f"Arquivos repetidos entre {a} e {b}: {len(shared)}"


def test_both_classes_present_in_every_split():
    for name in SPLIT_NAMES:
        counts = load_split(name)["label"].value_counts()
        assert counts.get(0, 0) > 0 and counts.get(1, 0) > 0, f"{name}: falta uma das classes"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} testes passaram.")
    sys.exit(1 if failures else 0)
