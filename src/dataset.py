import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

class RWF2000Dataset(Dataset):
    """
    Dataset simples e conciso para carregar videos do RWF-2000.
    Amostra 16 frames equidistantes, redimensiona para 224x224 e normaliza.
    """
    def __init__(self, csv_file, num_frames=16, is_train=False):
        self.df = pd.read_csv(csv_file)
        self.num_frames = num_frames
        self.is_train = is_train
        
        # Normalizacao padrao ImageNet (C, H, W)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            total_frames = self.num_frames

        # Indices equidistantes para cobrir todo o clipe (5s)
        indices = np.linspace(0, max(0, total_frames - 1), self.num_frames, dtype=int)
        indices_set = set(indices)

        frames = []
        frame_idx = 0
        while cap.isOpened() and len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            frame_idx += 1
        cap.release()

        # Fallback caso falte frames (duplica o ultimo)
        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return np.array(frames[:self.num_frames])  # (16, 224, 224, 3)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = self._load_frames(row["video_path"])  # shape: (16, 224, 224, 3)

        # Data Augmentation: Flip horizontal consistente em todos os 16 frames
        if self.is_train and np.random.rand() > 0.5:
            frames = np.flip(frames, axis=2).copy()

        # Converte para Tensor (T, C, H, W) e normaliza
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        for t in range(self.num_frames):
            tensor[t] = self.normalize(tensor[t])

        label = torch.tensor(row["label"], dtype=torch.long)
        return tensor, label
