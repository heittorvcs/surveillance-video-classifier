import os

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sample_frames(video_path, num_frames=16):
    """
    Amostra num_frames quadros equidistantes de um video.

    Usa grab() para avancar o cursor sem decodificar os quadros descartados e
    retrieve()/read() apenas nos instantes selecionados. Retorna um array
    (num_frames, 224, 224, 3) em RGB uint8.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        total_frames = num_frames

    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    indices_set = set(indices.tolist())

    frames = []
    frame_idx = 0
    while cap.isOpened() and len(frames) < num_frames:
        if frame_idx in indices_set:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        else:
            if not cap.grab():
                break
        frame_idx += 1
    cap.release()

    # Fallback caso o container falhe ou o clipe seja mais curto que o esperado
    if len(frames) == 0:
        frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.array(frames[:num_frames])


def frames_to_tensor(frames, num_frames=16):
    """Converte (T, H, W, C) uint8 RGB em tensor (T, C, H, W) normalizado (ImageNet)."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    for t in range(num_frames):
        tensor[t] = normalize(tensor[t])
    return tensor


class RWF2000Dataset(Dataset):
    """
    Dataset do RWF-2000: 16 quadros equidistantes, 224x224, normalizacao ImageNet.

    O flip horizontal e DETERMINISTICO (parametro `flip`), nao aleatorio. O motivo
    e que o backbone e congelado e as features sao extraidas uma unica vez para
    cache; um flip aleatorio nessa passada geraria uma perturbacao fixa por video,
    nao um data augmentation. O augmentation efetivo e obtido materializando dois
    caches (original e espelhado) e sorteando entre eles a cada epoca — ver
    src/build_cache.py e src/train.py.
    """

    def __init__(self, csv_file, num_frames=16, flip=False, root=None):
        self.df = pd.read_csv(csv_file)
        self.num_frames = num_frames
        self.flip = flip
        self.root = root

    def __len__(self):
        return len(self.df)

    def resolve(self, video_path):
        """
        Reescreve o caminho do CSV sob `root`, quando informado.

        Os CSVs guardam caminhos relativos comecando por "archive/". Apontar para
        outra raiz serve a dois casos: dataset guardado fora da pasta do projeto, e
        caminhos longos demais para o limite de 260 caracteres do Windows -- 16 dos
        1.600 videos de treino do RWF-2000 tem nomes que estouram esse limite a
        partir de uma pasta profunda, e o OpenCV nao aceita o prefixo \?\.
        """
        if not self.root:
            return video_path
        partes = video_path.replace("\\", "/").split("/", 1)
        return os.path.join(self.root, partes[1] if len(partes) > 1 else partes[0])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = sample_frames(self.resolve(row["video_path"]), self.num_frames)

        # Flip horizontal aplicado de forma identica aos 16 quadros do clipe
        if self.flip:
            frames = np.flip(frames, axis=2)

        tensor = frames_to_tensor(frames, self.num_frames)
        label = torch.tensor(row["label"], dtype=torch.long)
        return tensor, label
