"""
Inferencia operacional sobre um arquivo de video.

Uso:
    python src/inference.py --video sample_video.avi
    python src/inference.py --video sample_video.avi --model dualstream --threshold 0.50
    python src/inference.py --video meu_clipe.mp4 --label Fight
    python src/inference.py --export_onnx --onnx_model ensemble

A latencia e reportada separando decodificacao/pre-processamento da inferencia,
porque so a soma das duas representa o custo real por clipe em producao.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import os
import time

import torch

from src.dataset import frames_to_tensor, sample_frames
from src.model import load_classifier

ONNX_OUTPUTS = {
    "dualstream": "models/model_dualstream.onnx",
    "baseline": "models/model.onnx",
    "ensemble": "models/model_ensemble.onnx",
}


def preprocess_video(video_path, num_frames=16):
    """Decodifica 16 quadros equidistantes e devolve o tensor (1, 16, 3, 224, 224)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Arquivo de video nao encontrado: {video_path}")
    frames = sample_frames(video_path, num_frames)
    return frames_to_tensor(frames, num_frames).unsqueeze(0)


def export_onnx(model_name, device, output_path=None):
    """
    Exporta para ONNX o modelo JA TREINADO indicado por `model_name`.

    Os pesos sao sempre carregados via load_classifier antes da exportacao, e a
    paridade numerica do grafo resultante deve ser conferida com
    benchmarks/verify_onnx_parity.py.
    """
    model, _, desc = load_classifier(model_name, device=device)
    model.eval()

    output_path = output_path or ONNX_OUTPUTS[model_name]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dummy_input = torch.randn(1, 16, 3, 224, 224, device=device)
    # O ensemble devolve probabilidades (media dos softmax); os demais devolvem logits.
    output_name = "probs" if model_name == "ensemble" else "logits"

    # Exportador legado (dynamo=False): produz um grafo unico, sem arquivo .onnx.data
    # ao lado, e aceita `dynamic_axes` diretamente. O exportador dynamo do torch 2.x
    # grava o arquivo e so entao falha ao converter para opset 14, deixando um
    # artefato valido junto de um traceback.
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_video"],
        output_names=[output_name],
        dynamic_axes={"input_video": {0: "batch_size"}, output_name: {0: "batch_size"}},
        opset_version=14,
        dynamo=False,
    )
    print(f"[ONNX] {desc} exportado para '{output_path}'")
    print(f"[ONNX] Verifique a paridade numerica com: python benchmarks/verify_onnx_parity.py "
          f"--model {model_name}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Classificador de violencia em video de vigilancia")
    parser.add_argument("--video", type=str, default="sample_video.avi",
                        help="Caminho do video (.avi, .mp4)")
    parser.add_argument("--model", type=str, default="ensemble",
                        choices=["ensemble", "dualstream", "baseline"], help="Modelo a executar")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Limiar de probabilidade Fight (padrao: ver DEFAULT_THRESHOLDS em src/model.py)")
    parser.add_argument("--label", type=str, default=None, choices=["Fight", "NonFight"],
                        help="Rotulo real do video, se conhecido (apenas para exibicao)")
    parser.add_argument("--weights", type=str, default=None, help="Caminho customizado para pesos")
    parser.add_argument("--export_onnx", action="store_true", help="Exporta o modelo treinado para ONNX")
    parser.add_argument("--onnx_model", type=str, default=None,
                        choices=["ensemble", "dualstream", "baseline"],
                        help="Modelo a exportar (padrao: o mesmo de --model)")
    parser.add_argument("--no_infer", action="store_true", help="Apenas exporta, sem rodar inferencia")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.export_onnx:
        export_onnx(args.onnx_model or args.model, device)
        if args.no_infer:
            return

    model, default_th, model_desc = load_classifier(args.model, device=device, custom_path=args.weights)
    threshold = args.threshold if args.threshold is not None else default_th

    print("\n" + "=" * 65)
    print("SURVEILLANCE VIDEO CLASSIFIER - INFERENCIA OPERACIONAL")
    print("=" * 65)
    print(f"Dispositivo:   {device}")
    print(f"Modelo:        {model_desc}")
    print(f"Video:         {args.video}")
    print(f"Rotulo real:   {args.label if args.label else 'nao informado (use --label)'}")

    t_pre = time.perf_counter()
    input_tensor = preprocess_video(args.video).to(device)
    preprocess_ms = (time.perf_counter() - t_pre) * 1000

    t_inf = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
        # O ensemble ja devolve probabilidades; os demais devolvem logits.
        probs = output[0] if args.model == "ensemble" else torch.softmax(output, dim=1)[0]
    inference_ms = (time.perf_counter() - t_inf) * 1000

    prob_fight = probs[1].item()
    prob_nonfight = probs[0].item()
    pred_idx = 1 if prob_fight >= threshold else 0

    classes = ["NonFight (sem agressao)", "Fight (violencia detectada)"]
    confidence = (prob_fight if pred_idx == 1 else prob_nonfight) * 100
    margin = abs(prob_fight - threshold) * 100
    total_ms = preprocess_ms + inference_ms

    print("\n---------------- RELATORIO DE INFERENCIA ----------------")
    print(f"Decisao:                 {classes[pred_idx]}")
    print(f"Confianca:               {confidence:.2f}%")
    print(f"Probabilidade Fight:     {prob_fight * 100:.2f}%")
    print(f"Probabilidade NonFight:  {prob_nonfight * 100:.2f}%")
    print(f"Limiar:                  {threshold:.2f} (margem: {margin:.1f} p.p.)")
    if args.label:
        veredito = "ACERTO" if classes[pred_idx].startswith(args.label) else "ERRO"
        print(f"Rotulo real:             {args.label} -> {veredito}")
    print("---------------------------------------------------------")
    print(f"Decodificacao + pre-proc: {preprocess_ms:7.2f} ms")
    print(f"Forward do modelo:        {inference_ms:7.2f} ms")
    print(f"Total por clipe:          {total_ms:7.2f} ms  ({1000.0 / total_ms:.1f} clipes/s)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
