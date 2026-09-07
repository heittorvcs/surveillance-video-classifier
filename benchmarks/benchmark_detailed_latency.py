import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# 1. Carregar Backbone
weights_base = MobileNet_V3_Small_Weights.DEFAULT
base = mobilenet_v3_small(weights=weights_base)
backbone = base.features.eval()
avgpool = nn.AdaptiveAvgPool2d((1, 1)).eval()

# 2. Carregar as 3 cabeças
class TriStream_Kinetic(nn.Module):
    def __init__(self, feature_dim=576, hidden_dim=64, num_classes=2):
        super().__init__()
        self.gru_app = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_vel = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_acc = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        total_feat_dim = (hidden_dim * 4) * 3
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(total_feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out_app, _ = self.gru_app(x)
        feat_app = torch.cat([out_app.mean(dim=1), out_app.max(dim=1).values], dim=1)
        vel = x[:, 1:, :] - x[:, :-1, :]
        out_vel, _ = self.gru_vel(vel)
        feat_vel = torch.cat([out_vel.mean(dim=1), out_vel.max(dim=1).values], dim=1)
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        out_acc, _ = self.gru_acc(acc)
        feat_acc = torch.cat([out_acc.mean(dim=1), out_acc.max(dim=1).values], dim=1)
        combined = torch.cat([feat_app, feat_vel, feat_acc], dim=1)
        return self.classifier(combined)

class DualStream_MeanMax(nn.Module):
    def __init__(self, feature_dim=576, hidden_dim=64, num_classes=2):
        super().__init__()
        self.gru_app = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_mot = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 8, 96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes)
        )
    def forward(self, x):
        out_app, _ = self.gru_app(x)
        app_feat = torch.cat([out_app.mean(dim=1), out_app.max(dim=1).values], dim=1)
        diff = x[:, 1:, :] - x[:, :-1, :]
        out_mot, _ = self.gru_mot(diff)
        mot_feat = torch.cat([out_mot.mean(dim=1), out_mot.max(dim=1).values], dim=1)
        combined = torch.cat([app_feat, mot_feat], dim=1)
        return self.classifier(combined)

class BiGRU_Baseline(nn.Module):
    def __init__(self, feature_dim=576, hidden_dim=64, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, num_classes)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.classifier(out.mean(dim=1))

def resolve_path(filename, subfolders):
    for sub in subfolders:
        full = os.path.join(sub, filename)
        if os.path.exists(full):
            return full
    raise FileNotFoundError(f"Pesos não encontrados para: {filename}")

ens_subs = ["models/ensemble", "experiments/target_85", "legacy_experiments/target_85"]

m1 = TriStream_Kinetic().eval()
m1.load_state_dict(torch.load(resolve_path("model_tristream_s7.pth", ens_subs), map_location="cpu"))

m2 = DualStream_MeanMax().eval()
m2.load_state_dict(torch.load(resolve_path("model_dualmeanmax_s5.pth", ens_subs), map_location="cpu"))

m3 = DualStream_MeanMax().eval()
m3.load_state_dict(torch.load(resolve_path("model_dualmeanmax_s10.pth", ens_subs), map_location="cpu"))

m_base = BiGRU_Baseline().eval()

dummy_video = torch.randn(1, 16, 3, 224, 224)

# Aquecimento
with torch.no_grad():
    for _ in range(5):
        b, t, c, h, w = dummy_video.shape
        x = dummy_video.view(b * t, c, h, w)
        feat = avgpool(backbone(x)).flatten(1).view(b, t, -1)
        _ = m_base(feat)
        _ = (m1(feat) + m2(feat) + m3(feat)) / 3.0

N = 30

# 1. Medir Backbone Isolado
backbone_lats = []
with torch.no_grad():
    for _ in range(N):
        t0 = time.perf_counter()
        b, t, c, h, w = dummy_video.shape
        x = dummy_video.view(b * t, c, h, w)
        feat = avgpool(backbone(x)).flatten(1).view(b, t, -1)
        backbone_lats.append((time.perf_counter() - t0) * 1000)

# 2. Medir Cabeca Baseline
base_lats = []
with torch.no_grad():
    for _ in range(N):
        t0 = time.perf_counter()
        _ = m_base(feat)
        base_lats.append((time.perf_counter() - t0) * 1000)

# 3. Medir Cabeças do Ensemble (as 3 juntas sobre a feature ja extraída)
heads_lats = []
with torch.no_grad():
    for _ in range(N):
        t0 = time.perf_counter()
        p1 = torch.softmax(m1(feat), dim=1)
        p2 = torch.softmax(m2(feat), dim=1)
        p3 = torch.softmax(m3(feat), dim=1)
        p_ens = (p1 + p2 + p3) / 3.0
        heads_lats.append((time.perf_counter() - t0) * 1000)

# 4. Medir End-to-End Total (Vídeo 16 frames -> Predição Final)
total_lats = []
with torch.no_grad():
    for _ in range(N):
        t0 = time.perf_counter()
        b, t, c, h, w = dummy_video.shape
        x = dummy_video.view(b * t, c, h, w)
        feat = avgpool(backbone(x)).flatten(1).view(b, t, -1)
        p1 = torch.softmax(m1(feat), dim=1)
        p2 = torch.softmax(m2(feat), dim=1)
        p3 = torch.softmax(m3(feat), dim=1)
        p_ens = (p1 + p2 + p3) / 3.0
        total_lats.append((time.perf_counter() - t0) * 1000)

print("="*65)
print("DECOMPOSICAO DETALHADA DE LATENCIA (CPU)")
print("="*65)
print(f"1. Backbone MobileNetV3-Small (16 frames 224x224): {np.mean(backbone_lats):.2f} ms")
print(f"2. Cabeça Baseline (1 Bi-GRU simples):            {np.mean(base_lats):.2f} ms")
print(f"3. Cabeças do Ensemble 85% (3 modelos juntos):     {np.mean(heads_lats):.2f} ms")
print(f"4. Pipeline End-to-End Baseline (75.14% Acc):       {np.mean(backbone_lats) + np.mean(base_lats):.2f} ms")
print(f"5. Pipeline End-to-End Ensemble 85% (85.41% Acc):   {np.mean(total_lats):.2f} ms")
print(f"   - Latência P95:                                  {np.percentile(total_lats, 95):.2f} ms")
print(f"   - Throughput:                                    {1000.0/np.mean(total_lats):.2f} vídeos/s ({16000.0/np.mean(total_lats):.1f} FPS equivalente)")
print("="*65)

