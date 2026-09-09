"""
Verifica que o grafo ONNX exportado reproduz numericamente o modelo PyTorch,
e mede a latencia em ONNX Runtime sob o mesmo protocolo.

Sem esta verificacao, "exportado para ONNX" nao e uma afirmacao de Edge AI: um
grafo pode existir, carregar sem erro e ainda assim produzir valores errados.

Uso:
    python src/inference.py --export_onnx --onnx_model dualstream --no_infer
    python benchmarks/verify_onnx_parity.py --model dualstream
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
import os
import time

import numpy as np
import torch

from src.inference import ONNX_OUTPUTS
from src.model import load_classifier


def timeit(fn, runs, warmup=5):
    """
    Retorna (mediana, p95) em milissegundos.

    Mediana, e nao media: em CPU compartilhada uma unica pausa do escalonador
    desloca a media em dezenas de porcento, enquanto a mediana permanece estavel.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    arr = np.array(samples)
    return float(np.median(arr)), float(np.percentile(arr, 95))


def main():
    parser = argparse.ArgumentParser(description="Paridade numerica e latencia ONNX Runtime")
    parser.add_argument("--model", type=str, default="dualstream",
                        choices=["ensemble", "dualstream", "baseline"])
    parser.add_argument("--onnx_path", type=str, default=None)
    parser.add_argument("--samples", type=int, default=8, help="Entradas aleatorias comparadas")
    parser.add_argument("--runs", type=int, default=30, help="Repeticoes na medicao de latencia")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Diferenca maxima absoluta aceita")
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime nao instalado. Rode: pip install onnxruntime")
        return 1

    onnx_path = args.onnx_path or ONNX_OUTPUTS[args.model]
    if not os.path.exists(onnx_path):
        print(f"Grafo ONNX nao encontrado: {onnx_path}")
        print(f"Gere-o com: python src/inference.py --export_onnx --onnx_model {args.model} --no_infer")
        return 1

    device = torch.device("cpu")
    model, _, desc = load_classifier(args.model, device=device)
    model.eval()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print("=" * 72)
    print(f"PARIDADE ONNX x PYTORCH - {desc}")
    print("=" * 72)
    print(f"Grafo:      {onnx_path}")
    print(f"Amostras:   {args.samples} | Tolerancia: {args.tolerance:g}")

    max_diff = 0.0
    torch.manual_seed(0)
    with torch.no_grad():
        for i in range(args.samples):
            x = torch.randn(1, 16, 3, 224, 224)
            torch_out = model(x).cpu().numpy()
            onnx_out = session.run(None, {input_name: x.numpy()})[0]
            diff = float(np.abs(torch_out - onnx_out).max())
            max_diff = max(max_diff, diff)
            print(f"  amostra {i + 1}: diferenca maxima {diff:.3e}")

    ok = max_diff <= args.tolerance
    print("-" * 72)
    print(f"Diferenca maxima global: {max_diff:.3e} -> {'PARIDADE OK' if ok else 'DIVERGENCIA'}")

    dummy = torch.randn(1, 16, 3, 224, 224)
    dummy_np = dummy.numpy()
    with torch.no_grad():
        torch_ms, torch_p95 = timeit(lambda: model(dummy), args.runs)
    onnx_ms, onnx_p95 = timeit(lambda: session.run(None, {input_name: dummy_np}), args.runs)

    print("-" * 72)
    print(f"PyTorch CPU:      {torch_ms:7.2f} ms (p95 {torch_p95:.2f})")
    print(f"ONNX Runtime CPU: {onnx_ms:7.2f} ms (p95 {onnx_p95:.2f})")
    speedup = torch_ms / onnx_ms if onnx_ms else float("nan")
    print(f"Ganho ONNX Runtime: {speedup:.2f}x")
    print("=" * 72)

    payload = {
        "model": args.model,
        "onnx_path": onnx_path,
        "max_abs_diff": max_diff,
        "tolerance": args.tolerance,
        "parity_ok": ok,
        "torch_ms": torch_ms,
        "onnx_ms": onnx_ms,
        "speedup": speedup,
    }
    out = args.save_json or f"reports/onnx_parity_{args.model}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Resultado salvo em: {out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
