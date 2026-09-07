import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ==============================================================================
# MODELO 1 — BASELINE MINIMALISTA (75.14% Acc | 76.14% Rec | 21 FN)
# ==============================================================================
class VideoClassifier(nn.Module):
    """
    Arquitetura Espaço-Temporal Baseline para Detecção de Violência em Vídeo:
    - Espacial: MobileNetV3-Small pré-treinado (ImageNet) com GAP (576 dim).
    - Temporal: Bi-GRU única (hidden_size=64, bidirectional -> 128 dim).
    - Cabeça Classificadora: Dropout(0.4) + Linear(128, 2).
    - Parâmetros Totais: ~1.17M (~4.8 MB).
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base = mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        feature_dim = 576
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Time=16, Channels=3, Height=224, Width=224)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)  # (b * t, 576)
        feat = feat.view(b, t, -1)     # (b, t, 576)
        
        gru_out, _ = self.gru(feat)    # (b, t, 128)
        temporal_feat = gru_out.mean(dim=1)
        logits = self.classifier(temporal_feat)
        return logits

class BiGRU_Head(nn.Module):
    """Cabeça Bi-GRU isolada para inferência ultrarrápida sobre features pré-extraídas"""
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

BiGRU_Baseline = VideoClassifier


# ==============================================================================
# MODELO 2 — INOVAÇÃO DUAL-STREAM LATENTE (82.16% Acc | 88.64% Rec | 10 FN)
# ==============================================================================
class VideoClassifierDualStream(nn.Module):
    """
    Arquitetura Dual-Stream Latente:
    - Espacial: MobileNetV3-Small pré-treinado congelado (576 dim).
    - Stream 1 (Aparência): Bi-GRU temporal de 16 quadros -> 128 dim.
    - Stream 2 (Velocidade Diferencial): Bi-GRU sobre delta_f = f_t - f_{t-1} -> 128 dim.
    - Cabeça de Fusão: Concatenação (256 dim) -> Dropout(0.4) -> Linear(64) -> ReLU -> Dropout(0.3) -> Linear(2).
    - Parâmetros Totais: ~1.44M (~5.9 MB).
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base = mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
                
        feature_dim = 576
        hidden_dim = 64
        self.gru_app = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_mot = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = feat.view(b, t, -1)
        
        out_app, _ = self.gru_app(feat)
        app_feat = out_app.mean(dim=1)
        
        diff = feat[:, 1:, :] - feat[:, :-1, :]
        out_mot, _ = self.gru_mot(diff)
        mot_feat = out_mot.mean(dim=1)
        
        combined = torch.cat([app_feat, mot_feat], dim=1)
        return self.classifier(combined)

class DualStream_Head(nn.Module):
    """Cabeça Dual-Stream isolada para inferência ultrarrápida sobre features pré-extraídas"""
    def __init__(self, feature_dim=576, hidden_dim=64, num_classes=2):
        super().__init__()
        self.gru_app = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru_mot = nn.GRU(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out_app, _ = self.gru_app(x)
        app_feat = out_app.mean(dim=1)
        diff = x[:, 1:, :] - x[:, :-1, :]
        out_mot, _ = self.gru_mot(diff)
        mot_feat = out_mot.mean(dim=1)
        combined = torch.cat([app_feat, mot_feat], dim=1)
        return self.classifier(combined)



# ==============================================================================
# MODELO 3 — ENSEMBLE CINÉTICO TRI-STREAM (85.41% Acc | 92.05% Rec | APENAS 7 FN)
# ==============================================================================
class TriStream_Kinetic(nn.Module):
    """Sub-modelo 1 do Ensemble: Aparência + Velocidade (Δf) + Aceleração de Impacto (Δ²f)"""
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
    """Sub-modelos 2 e 3 do Ensemble: Dual-Stream com Agregação Mean + Max Pooling"""
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

class KineticEnsembleClassifier(nn.Module):
    """
    Ensemble Cinético Completo (End-to-End):
    Executa o backbone MobileNetV3 UMA ÚNICA VEZ para extrair (B, 16, 576),
    distribui para as 3 cabeças leves e calcula a média das probabilidades de softmax.
    """
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base = mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for p in self.features.parameters():
            p.requires_grad = False
            
        self.m1 = TriStream_Kinetic()
        self.m2 = DualStream_MeanMax()
        self.m3 = DualStream_MeanMax()

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1).view(b, t, -1)
        
        logits1 = self.m1(feat)
        logits2 = self.m2(feat)
        logits3 = self.m3(feat)
        
        prob1 = torch.softmax(logits1, dim=1)
        prob2 = torch.softmax(logits2, dim=1)
        prob3 = torch.softmax(logits3, dim=1)
        
        # Média das probabilidades do ensemble
        return (prob1 + prob2 + prob3) / 3.0


# ==============================================================================
# FACTORY HELPER PARA CARREGAMENTO AUTOMÁTICO
# ==============================================================================
def load_classifier(model_name="ensemble", device="cpu", custom_path=None):
    """
    Carrega o classificador solicitado com seus respectivos pesos pré-treinados:
    - 'baseline'   -> Modelo 1 (Bi-GRU Simples, 75.14% Acc, th=0.50)
    - 'dualstream' -> Modelo 2 (Dual-Stream Latente, 82.16% Acc, th=0.50)
    - 'ensemble'   -> Modelo 3 (Ensemble Cinético Tri-Stream, 85.41% Acc, th=0.52)
    """
    model_name = model_name.lower()
    
    if model_name == "baseline":
        model = VideoClassifier().to(device)
        path = custom_path or "models/best_model.pth"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pesos do Baseline não encontrados em: {path}")
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        default_th = 0.50
        return model, default_th, "Baseline Minimalista (Bi-GRU)"
        
    elif model_name == "dualstream":
        model = VideoClassifierDualStream().to(device)
        candidate_paths = [
            custom_path,
            "models/best_model_dualstream_82acc.pth",
            "experiments/best_model_dualstream_82acc.pth",
            "legacy_experiments/best_model_dualstream_82acc.pth"
        ]
        path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
        if not path:
            raise FileNotFoundError("Pesos do DualStream não encontrados em models/ ou experiments/")
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        default_th = 0.50
        return model, default_th, "Dual-Stream Latente (Aparência + Velocidade)"
        
    elif model_name in ["ensemble", "tristream", "kinetic"]:
        model = KineticEnsembleClassifier().to(device)
        def resolve_path(filename, subfolders):
            for sub in subfolders:
                full = os.path.join(sub, filename)
                if os.path.exists(full):
                    return full
            raise FileNotFoundError(f"Pesos do Ensemble não encontrados para: {filename}")
            
        subfolders = ["models/ensemble", "experiments/target_85", "legacy_experiments/target_85"]
        p1 = resolve_path("model_tristream_s7.pth", subfolders)
        p2 = resolve_path("model_dualmeanmax_s5.pth", subfolders)
        p3 = resolve_path("model_dualmeanmax_s10.pth", subfolders)
        model.m1.load_state_dict(torch.load(p1, map_location=device))
        model.m2.load_state_dict(torch.load(p2, map_location=device))
        model.m3.load_state_dict(torch.load(p3, map_location=device))
        model.eval()
        default_th = 0.52
        return model, default_th, "Ensemble Cinético Tri-Stream (SOTA 85.41%)"
    else:
        raise ValueError(f"Modelo desconhecido: '{model_name}'. Escolha entre: 'baseline', 'dualstream', 'ensemble'.")
