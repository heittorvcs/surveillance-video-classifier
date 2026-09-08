import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from src.model import load_classifier

def main():
    video_path = "sample_video.avi"
    out_path = Path("reports/sample_prediction_mosaic.png")

    # 1. Carrega o modelo campeão (Modelo 3: Ensemble Cinético Tri-Stream)
    device = torch.device("cpu")
    model, default_th, model_desc = load_classifier("ensemble", device=device)

    # 2. Extrai os 16 frames brutos para exibicao e o tensor normalizado
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    indices = np.linspace(0, max(0, total_frames - 1), 16, dtype=int)
    indices_set = set(indices)

    raw_frames = []
    frame_idx = 0
    while cap.isOpened() and len(raw_frames) < 16:
        if frame_idx in indices_set:
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_frames.append(cv2.resize(frame_rgb, (224, 224)))
        else:
            if not cap.grab(): break
        frame_idx += 1
    cap.release()

    while len(raw_frames) < 16:
        raw_frames.append(raw_frames[-1])

    # 3. Inferencia com o Ensemble SOTA
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = torch.from_numpy(np.array(raw_frames)).permute(0, 3, 1, 2).float() / 255.0
    for t in range(16):
        tensor[t] = normalize(tensor[t])
    input_tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = output[0]
        prob_fight = probs[1].item()
        prob_nonfight = probs[0].item()

    is_fight = (prob_fight >= default_th)
    confidence = (prob_fight if is_fight else prob_nonfight) * 100
    label_text = "[ALERTA] VIOLÊNCIA DETECTADA (FIGHT)" if is_fight else "[OK] AMBIENTE SEGURO (NON-FIGHT)"
    border_color = "#e63946" if is_fight else "#2a9d8f"

    # 4. Gera o mosaico 4x4 de alta resolução
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    fig.patch.set_facecolor("#18191a")

    for i, ax in enumerate(axes.flat):
        ax.imshow(raw_frames[i])
        ax.axis("off")
        time_sec = (indices[i] / fps)
        ax.set_title(f"Frame {i+1:02d} ({time_sec:.1f}s)", fontsize=10, color="#e4e6eb", pad=4, fontweight="bold")

    # Titulo e Faixa de Status no topo
    fig.suptitle(
        f"{label_text}\n"
        f"Modelo: Ensemble Cinético Tri-Stream (SOTA 85.41%) | Limiar: θ = {default_th:.2f}\n"
        f"Prob. Fight: {prob_fight*100:.1f}% | Prob. NonFight: {prob_nonfight*100:.1f}% | Confiança: {confidence:.1f}%",
        fontsize=14,
        fontweight="bold",
        color=border_color,
        y=0.98
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.91])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Mosaico gerado com sucesso em: {out_path.resolve()}")

if __name__ == "__main__":
    main()

