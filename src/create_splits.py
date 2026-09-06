import os
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def get_video_records(dir_path):
    records = []
    classes = {"NonFight": 0, "Fight": 1}
    for class_name, label in classes.items():
        folder = Path(dir_path) / class_name
        for video_file in folder.glob("*.avi"):
            # O prefixo antes do ultimo underline e o ID do video original (anti-leakage)
            base_id = re.sub(r"_\d+\.avi$", "", video_file.name)
            records.append({
                "video_path": str(video_file.as_posix()),
                "label": label,
                "label_name": class_name,
                "group_id": base_id
            })
    return pd.DataFrame(records)

def main():
    root = Path("archive/RWF-2000")
    os.makedirs("data/splits", exist_ok=True)

    # 1. Conjunto de Treino (1600 videos originais do train)
    df_train = get_video_records(root / "train")
    
    # 2. Conjunto de Val oficial (400 videos) -> Dividido 50/50 em Val e Teste sem data leakage
    df_val_pool = get_video_records(root / "val")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    train_idx, test_idx = next(gss.split(df_val_pool, df_val_pool["label"], groups=df_val_pool["group_id"]))
    
    df_val = df_val_pool.iloc[train_idx].copy()
    df_test = df_val_pool.iloc[test_idx].copy()

    # Salva os splits em CSV
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        csv_path = f"data/splits/{name}.csv"
        df[["video_path", "label", "label_name"]].to_csv(csv_path, index=False)
        print(f"[{name.upper()}] Salvo em {csv_path} - Total: {len(df)} | "
              f"Fight: {(df['label'] == 1).sum()}, NonFight: {(df['label'] == 0).sum()}")

if __name__ == "__main__":
    main()
