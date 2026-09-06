import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import time
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms

from src.model import VideoClassifier

def preprocess_video(video_path, num_frames=16):
    """
    Extrai 16 frames equidistantes, redimensiona para 224x224 e normaliza.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = num_frames

    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    indices_set = set(indices)

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

    if len(frames) == 0:
        frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = torch.from_numpy(np.array(frames[:num_frames])).permute(0, 3, 1, 2).float() / 255.0
    for t in range(num_frames):
        tensor[t] = normalize(tensor[t])

    return tensor.unsqueeze(0)  # Shape (1, 16, 3, 224, 224)

def export_onnx(model, device, output_path="models/model.onnx"):
    """
    Exporta o modelo para formato ONNX para inferencia otimizada em borda (Edge AI).
    """
    model.eval()
    dummy_input = torch.randn(1, 16, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_video"],
        output_names=["logits"],
        opset_version=14,
        dynamo=False
    )
    print(f"[ONNX] Modelo exportado com sucesso para '{output_path}'")

def main():
    parser = argparse.ArgumentParser(description="Inferencia em video de teste")
    parser.add_argument("--video", type=str, default=None, help="Caminho do video para teste (.avi, .mp4)")
    parser.add_argument("--threshold", type=float, default=0.50, help="Limiar de probabilidade para classificar como Fight (padrao: 0.50)")
    parser.add_argument("--weights", type=str, default="models/best_model.pth", help="Caminho dos pesos salvos")
    parser.add_argument("--export_onnx", action="store_true", help="Exporta modelo treinado para ONNX")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Carrega modelo treinado
    model = VideoClassifier(num_classes=2).to(device)
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Pesos nao encontrados em {args.weights}. Treine o modelo primeiro.")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    if args.export_onnx:
        export_onnx(model, device)
        if not args.video:
            return

    # Se nenhum video for passado, seleciona o primeiro video do conjunto de teste
    if not args.video:
        import pandas as pd
        df_test = pd.read_csv("data/splits/test.csv")
        sample_video = df_test.iloc[0]
        video_path = sample_video["video_path"]
        ground_truth = sample_video["label_name"]
    else:
        video_path = args.video
        ground_truth = "Desconhecido"

    print(f"\nProcessando video: {video_path}")
    input_tensor = preprocess_video(video_path).to(device)

    # 2. Execucao com medicao de latencia
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        prob_fight = probs[1].item()
        pred_idx = 1 if prob_fight >= args.threshold else 0
    latency_ms = (time.perf_counter() - t0) * 1000

    classes = ["NonFight (Nao Violento)", "Fight (Violento)"]
    pred_label = classes[pred_idx]
    confidence = probs[pred_idx].item() * 100

    print("\n================ RESULTADO DA INFERENCIA ================")
    print(f"Video analisado:       {os.path.basename(video_path)}")
    print(f"Rotulo Real (GT):      {ground_truth}")
    print(f"Predicao do Modelo:    {pred_label}")
    print(f"Confianca:             {confidence:.2f}%")
    print(f"Prob. Fight:           {prob_fight*100:.2f}% | Prob. NonFight: {probs[0].item()*100:.2f}%")
    print(f"Limiar de Decisao (th): {args.threshold:.2f}")
    print(f"Latencia (Inferencia): {latency_ms:.1f} ms (~{1000/latency_ms:.1f} FPS)")
    print("=========================================================\n")

if __name__ == "__main__":
    main()

