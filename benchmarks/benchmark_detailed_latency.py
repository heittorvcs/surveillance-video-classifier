"""
Perfil de latencia em CPU dos tres modelos, medidos sob o MESMO protocolo.

Os tres modelos passam pelo mesmo laco, na mesma maquina e na mesma execucao:
medicoes feitas em execucoes distintas nao sao comparaveis entre si. O custo de
decodificacao do video e medido separadamente do forward, porque so a soma dos dois
representa o custo real por clipe.

Saida: reports/latency_benchmark.json e uma tabela no terminal.

Uso:
    python benchmarks/benchmark_detailed_latency.py
    python benchmarks/benchmark_detailed_latency.py --runs 50 --video sample_video.avi
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
import platform
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from src.dataset import frames_to_tensor, sample_frames
from src.model import (
    BiGRU_Head,
    DualStream_Head,
    load_ensemble_head,
)


def hardware_profile():
    """Contexto da medicao: numeros de latencia sem isso nao sao comparaveis."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "desconhecido",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
    }


def timeit(fn, runs, warmup=5):
    """
    Retorna (mediana, desvio, p95) em milissegundos.

    Mediana, e nao media: em CPU compartilhada uma unica pausa do escalonador
    desloca a media em dezenas de porcento, enquanto a mediana permanece estavel.
    O desvio continua sendo reportado para tornar a dispersao visivel.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    arr = np.array(samples)
    return float(np.median(arr)), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, float(np.percentile(arr, 95))


def count_params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


def main():
    parser = argparse.ArgumentParser(description="Perfil de latencia em CPU sob protocolo unico")
    parser.add_argument("--runs", type=int, default=30, help="Repeticoes cronometradas por medicao")
    parser.add_argument("--video", type=str, default="sample_video.avi",
                        help="Video real usado para medir a decodificacao")
    parser.add_argument("--save_json", type=str, default="reports/latency_benchmark.json")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cpu")

    hw = hardware_profile()
    print("=" * 72)
    print("PERFIL DE LATENCIA EM CPU - PROTOCOLO UNICO")
    print("=" * 72)
    for k, v in hw.items():
        print(f"  {k:16s} {v}")
    print(f"  {'runs':16s} {args.runs} (warmup 5)")
    print("=" * 72)

    base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    backbone = base.features.to(device).eval()
    avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device).eval()

    heads = {}

    m_base = BiGRU_Head().to(device).eval()
    baseline_ckpt = "models/best_model.pth"
    if os.path.exists(baseline_ckpt):
        state = torch.load(baseline_ckpt, map_location=device)
        m_base.load_state_dict({k: v for k, v in state.items() if not k.startswith("features.")})
    heads["Modelo 1: Baseline"] = m_base

    m_dual = DualStream_Head().to(device).eval()
    dual_ckpt = "models/model_dualstream_best.pth"
    if os.path.exists(dual_ckpt):
        m_dual.load_state_dict(torch.load(dual_ckpt, map_location=device))
    heads["Modelo 2: Dual-Stream"] = m_dual

    try:
        heads["Modelo 3: Ensemble"] = load_ensemble_head(device=device)
    except FileNotFoundError as exc:
        print(f"AVISO: ensemble indisponivel ({exc}); seguindo sem ele.")

    dummy_video = torch.randn(1, 16, 3, 224, 224, device=device)

    def run_backbone():
        b, t, c, h, w = dummy_video.shape
        x = dummy_video.view(b * t, c, h, w)
        return avgpool(backbone(x)).flatten(1).view(b, t, -1)

    # 1) Decodificacao + pre-processamento de um video real
    decode_ms = decode_std = decode_p95 = None
    if os.path.exists(args.video):
        def run_decode():
            frames = sample_frames(args.video, 16)
            return frames_to_tensor(frames, 16)
        decode_ms, decode_std, decode_p95 = timeit(run_decode, args.runs)
        print(f"\nDecodificacao + pre-proc ({args.video}): "
              f"{decode_ms:.2f} +- {decode_std:.2f} ms (p95 {decode_p95:.2f})")
    else:
        print(f"\nAVISO: '{args.video}' nao encontrado; decodificacao nao medida.")

    # 2) Backbone isolado (compartilhado por todos os modelos)
    backbone_ms, backbone_std, backbone_p95 = timeit(run_backbone, args.runs)
    backbone_params = count_params(backbone)
    print(f"Backbone MobileNetV3-Small (16 quadros): "
          f"{backbone_ms:.2f} +- {backbone_std:.2f} ms (p95 {backbone_p95:.2f})")

    feat = run_backbone()

    # 3) Modelos medidos de forma INTERCALADA
    #
    # Medir cada modelo num bloco separado torna a comparacao invalida: uma pausa do
    # escalonador durante o bloco de um modelo o penaliza sozinho, e ja produziu aqui
    # o resultado fisicamente impossivel de o baseline aparecer mais lento que o
    # ensemble, que roda o mesmo backbone mais duas cabecas. Intercalando, qualquer
    # contencao transitoria atinge os tres na mesma proporcao e a comparacao e pareada.
    names = list(heads)

    def run_e2e(head):
        b, t, c, h, w = dummy_video.shape
        x = dummy_video.view(b * t, c, h, w)
        f = avgpool(backbone(x)).flatten(1).view(b, t, -1)
        return head(f)

    for _ in range(5):  # aquecimento
        for name in names:
            heads[name](feat)
            run_e2e(heads[name])

    head_samples = {n: [] for n in names}
    e2e_samples = {n: [] for n in names}
    for _ in range(args.runs):
        for name in names:
            head = heads[name]
            t0 = time.perf_counter()
            head(feat)
            head_samples[name].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            run_e2e(head)
            e2e_samples[name].append((time.perf_counter() - t0) * 1000)

    rows = []
    for name in names:
        head_arr = np.array(head_samples[name])
        e2e_arr = np.array(e2e_samples[name])
        e2e_ms = float(np.median(e2e_arr))
        params = backbone_params + count_params(heads[name])

        rows.append({
            "model": name,
            "params_total": params,
            "params_head": count_params(heads[name]),
            "head_ms": float(np.median(head_arr)),
            "head_std": float(head_arr.std(ddof=1)),
            "forward_ms": e2e_ms,
            "forward_std": float(e2e_arr.std(ddof=1)),
            "forward_p95": float(np.percentile(e2e_arr, 95)),
            "with_decode_ms": (e2e_ms + decode_ms) if decode_ms else None,
        })

    print("\n" + "=" * 96)
    print(f"{'Modelo':24s} {'Params':>10s} {'Cabeca(ms)':>12s} {'Forward(ms)':>14s} "
          f"{'p95(ms)':>10s} {'+decode(ms)':>13s} {'clipes/s':>10s}")
    print("-" * 96)
    for r in rows:
        total = r["with_decode_ms"] or r["forward_ms"]
        print(f"{r['model']:24s} {r['params_total']/1e6:9.2f}M "
              f"{r['head_ms']:11.2f} {r['forward_ms']:13.2f} {r['forward_p95']:10.2f} "
              f"{(r['with_decode_ms'] if r['with_decode_ms'] else float('nan')):13.2f} "
              f"{1000.0/total:10.1f}")
    print("=" * 96)
    print("Medicoes intercaladas entre os modelos; valores sao medianas.")
    print("Forward = backbone + cabeca, sobre tensor sintetico (sem I/O).")
    print("+decode = forward + decodificacao/pre-processamento de um clipe real.")
    print("Valores dependem do hardware acima; regenere na maquina alvo antes de citar.\n")

    payload = {
        "hardware": hw,
        "runs": args.runs,
        "decode_median_ms": decode_ms,
        "decode_p95_ms": decode_p95,
        "backbone_median_ms": backbone_ms,
        "backbone_p95_ms": backbone_p95,
        "models": rows,
    }
    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Resultado salvo em: {args.save_json}")


if __name__ == "__main__":
    main()
