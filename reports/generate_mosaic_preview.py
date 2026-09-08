"""
Gera o mosaico 4x4 dos 16 quadros amostrados e a predicao do modelo sobre eles.

Uso:
    python reports/generate_mosaic_preview.py
    python reports/generate_mosaic_preview.py --video meu_clipe.mp4 --label Fight --model dualstream
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import frames_to_tensor, sample_frames
from src.model import load_classifier


def main():
    parser = argparse.ArgumentParser(description="Mosaico de predicao sobre um video")
    parser.add_argument("--video", type=str, default="sample_video.avi")
    parser.add_argument("--model", type=str, default="ensemble",
                        choices=["ensemble", "dualstream", "baseline"])
    parser.add_argument("--label", type=str, default="Fight", choices=["Fight", "NonFight"],
                        help="Rotulo real do video, exibido no cabecalho")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out", type=str, default="reports/sample_prediction_mosaic.png")
    args = parser.parse_args()

    out_path = Path(args.out)
    device = torch.device("cpu")
    model, default_th, model_desc = load_classifier(args.model, device=device)
    threshold = args.threshold if args.threshold is not None else default_th

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    indices = np.linspace(0, max(0, total_frames - 1), 16, dtype=int)

    raw_frames = sample_frames(args.video, 16)
    input_tensor = frames_to_tensor(raw_frames, 16).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = output[0] if args.model == "ensemble" else torch.softmax(output, dim=1)[0]

    prob_fight = probs[1].item()
    prob_nonfight = probs[0].item()
    is_fight = prob_fight >= threshold
    confidence = (prob_fight if is_fight else prob_nonfight) * 100
    margin = abs(prob_fight - threshold) * 100

    predicted = "Fight" if is_fight else "NonFight"
    correct = predicted == args.label
    label_text = "[ALERTA] VIOLENCIA DETECTADA (FIGHT)" if is_fight else "[OK] AMBIENTE SEGURO (NON-FIGHT)"
    border_color = "#e63946" if is_fight else "#2a9d8f"

    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    fig.patch.set_facecolor("#18191a")

    for i, ax in enumerate(axes.flat):
        ax.imshow(raw_frames[i])
        ax.axis("off")
        ax.set_title(f"Frame {i + 1:02d} ({indices[i] / fps:.1f}s)", fontsize=10,
                     color="#e4e6eb", pad=4, fontweight="bold")

    fig.suptitle(
        f"{label_text}\n"
        f"{model_desc} | limiar θ = {threshold:.2f}\n"
        f"P(Fight) {prob_fight * 100:.1f}% | P(NonFight) {prob_nonfight * 100:.1f}% | "
        f"confiança {confidence:.1f}% | margem {margin:.1f} p.p.\n"
        f"Rótulo real: {args.label} — {'ACERTO' if correct else 'ERRO'}",
        fontsize=13,
        fontweight="bold",
        color=border_color,
        y=0.985,
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Mosaico gerado em: {out_path.resolve()}")
    print(f"Predicao: {predicted} (real: {args.label}) | margem {margin:.1f} p.p.")


if __name__ == "__main__":
    main()
