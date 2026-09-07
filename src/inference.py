import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import time
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms

from src.model import load_classifier, VideoClassifierDualStream

def preprocess_video(video_path, num_frames=16):
    """
    Decodificação eficiente com amostragem uniforme de 16 frames,
    redimensionamento para 224x224 e normalização ImageNet.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o arquivo de vídeo: {video_path}")

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

    return tensor.unsqueeze(0)  # Shape: (1, 16, 3, 224, 224)

def export_onnx(model, device, output_path="models/model_dualstream.onnx"):
    """
    Exporta modelo para o padrão aberto ONNX para inferência otimizada em Edge AI.
    """
    model.eval()
    dummy_input = torch.randn(1, 16, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_video"],
        output_names=["logits"],
        dynamic_axes={"input_video": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=14
    )
    print(f"[ONNX] Grafo exportado com sucesso para '{output_path}'")

def main():
    parser = argparse.ArgumentParser(description="Classificador de Violência em Vigilância (Edge AI)")
    parser.add_argument("--video", type=str, default="sample_video.avi", help="Caminho do vídeo para teste (.avi, .mp4)")
    parser.add_argument("--model", type=str, default="ensemble", choices=["ensemble", "dualstream", "baseline"],
                        help="Modelo a executar: 'ensemble' (85.41% SOTA), 'dualstream' (82.16%) ou 'baseline' (75.14%)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Limiar de probabilidade Fight (padrão: 0.52 para ensemble, 0.50 para outros)")
    parser.add_argument("--weights", type=str, default=None, help="Caminho customizado para pesos")
    parser.add_argument("--export_onnx", action="store_true", help="Exporta modelo Dual-Stream para formato ONNX")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.export_onnx:
        m_dual = VideoClassifierDualStream().to(device)
        export_onnx(m_dual, device)
        if not args.video:
            return

    # 1. Carregar Modelo Selecionado
    model, default_th, model_desc = load_classifier(args.model, device=device, custom_path=args.weights)
    threshold = args.threshold if args.threshold is not None else default_th

    # 2. Obter Vídeo
    video_path = args.video
    ground_truth = "Desconhecido / Em Produção"
    if not os.path.exists(video_path):
        # Fallback para primeiro vídeo do conjunto de teste se o arquivo não existir
        if os.path.exists("data/splits/test.csv"):
            import pandas as pd
            df_test = pd.read_csv("data/splits/test.csv")
            sample = df_test.iloc[0]
            video_path = sample["video_path"]
            ground_truth = f"{sample['label_name']} (Split de Teste)"
        else:
            raise FileNotFoundError(f"Vídeo de entrada não encontrado: {video_path}")

    print(f"\n" + "="*65)
    print(f"SURVEILLANCE VIDEO CLASSIFIER — INFERÊNCIA OPERACIONAL")
    print("="*65)
    print(f"Dispositivo de Execução: {device}")
    print(f"Modelo Ativo:            {model_desc}")
    print(f"Arquivo de Vídeo:        {video_path}")
    print(f"Rótulo Real (Ground Truth): {ground_truth}")

    # 3. Pré-processamento
    input_tensor = preprocess_video(video_path).to(device)

    # 4. Inferência com Cronometragem Rigorosa
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
        # Se for o Ensemble, o retorno já é a probabilidade do softmax; se for logits, aplica softmax
        if args.model == "ensemble":
            probs = output[0]
        else:
            probs = torch.softmax(output, dim=1)[0]
            
        prob_fight = probs[1].item()
        prob_nonfight = probs[0].item()
        pred_idx = 1 if prob_fight >= threshold else 0
        
    latency_ms = (time.perf_counter() - t0) * 1000
    fps_equiv = 16000.0 / latency_ms

    classes = ["NonFight (Normal / Sem Agressão)", "Fight (Violência Detectada!)"]
    pred_label = classes[pred_idx]
    confidence = (prob_fight if pred_idx == 1 else prob_nonfight) * 100

    print("\n---------------- RELATÓRIO DE INFERÊNCIA ----------------")
    print(f"Decisão do Sistema:      {pred_label}")
    print(f"Confiança da Decisão:    {confidence:.2f}%")
    print(f"Probabilidade Fight:     {prob_fight * 100:.2f}%")
    print(f"Probabilidade NonFight:  {prob_nonfight * 100:.2f}%")
    print(f"Limiar de Decisão (θ):   {threshold:.2f}")
    print(f"Latência End-to-End:     {latency_ms:.2f} ms")
    print(f"Throughput Estimado:     {1000.0/latency_ms:.1f} vídeos/s ({fps_equiv:.1f} FPS equiv.)")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
