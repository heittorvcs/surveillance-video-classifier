import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ==============================================================================
# MODELO 1 — BASELINE MINIMALISTA (teste cego: 75.14% Acc | 76.14% Rec | 21 FN)
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
# MODELO 2 — DUAL-STREAM LATENTE (teste cego: 82.16% Acc | 88.64% Rec | 10 FN)
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
# MODELO 3 — ENSEMBLE CINÉTICO TRI-STREAM
# Melhor checkpoint no teste: 85.41% Acc | 92.05% Rec | 7 FN.
# Média de 20 comitês formados sem seleção: 81.11% ± 1.21% (ver README, seção 5).
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
# REGISTRO DE CABEÇAS (fonte única para treino, benchmarks e avaliação)
# ==============================================================================
# As cabeças abaixo operam sobre features já extraídas do backbone congelado,
# no formato (B, T=16, D=576). Treino, benchmarks e avaliação importam daqui —
# nenhuma arquitetura é redefinida em outro arquivo.

HEAD_REGISTRY = {
    "baseline":   (BiGRU_Head,          "Modelo 1: Baseline Bi-GRU"),
    "dualstream": (DualStream_Head,     "Modelo 2: Dual-Stream Latente"),
    "tristream":  (TriStream_Kinetic,   "Modelo 3: Tri-Stream Cinético"),
    "dualmeanmax": (DualStream_MeanMax, "Dual-Stream Mean+Max (membro do ensemble)"),
}


def build_head(arch):
    """Instancia uma cabeça pelo nome. Retorna (modulo, descricao)."""
    key = arch.lower()
    if key not in HEAD_REGISTRY:
        raise ValueError(
            f"Arquitetura desconhecida: '{arch}'. Disponíveis: {sorted(HEAD_REGISTRY)}"
        )
    cls, desc = HEAD_REGISTRY[key]
    return cls(), desc


class EnsembleHead(nn.Module):
    """
    Comitê das 3 cabeças do Modelo 3 sobre features pré-extraídas.

    Retorna a média das probabilidades de softmax dos membros. Usado por
    src/evaluate.py, src/calibrate_threshold.py e benchmarks/, para que a
    lógica de fusão exista em um único lugar.
    """

    def __init__(self, members=None):
        super().__init__()
        if members is None:
            members = [TriStream_Kinetic(), DualStream_MeanMax(), DualStream_MeanMax()]
        self.members = nn.ModuleList(members)

    def forward(self, x):
        probs = [torch.softmax(m(x), dim=1) for m in self.members]
        return torch.stack(probs, dim=0).mean(dim=0)


ENSEMBLE_MEMBER_FILES = [
    ("member_tristream.pth", TriStream_Kinetic),
    ("member_dualmeanmax_a.pth", DualStream_MeanMax),
    ("member_dualmeanmax_b.pth", DualStream_MeanMax),
]

ENSEMBLE_SEARCH_DIRS = ["models/ensemble", "experiments/target_85", "legacy_experiments/target_85"]


def resolve_weights(filename, subfolders):
    """Primeiro caminho existente para `filename` dentre `subfolders`."""
    for sub in subfolders:
        full = os.path.join(sub, filename)
        if os.path.exists(full):
            return full
    raise FileNotFoundError(
        f"Pesos não encontrados para '{filename}' em: {', '.join(subfolders)}"
    )


def load_into_full_model(model, path, device="cpu"):
    """
    Carrega pesos em um modelo ponta a ponta (backbone + cabeça).

    Aceita duas formas de checkpoint: o modelo completo, ou apenas a cabeça — que
    é o que src/train.py salva, já que o backbone fica congelado e é sempre o
    MobileNetV3-Small do ImageNet. No segundo caso as chaves do backbone que já
    vêm do torchvision são mantidas, e exige-se que todas as chaves da cabeça
    estejam presentes, para que um checkpoint incompatível falhe em vez de
    carregar pela metade.
    """
    state = torch.load(path, map_location=device)
    has_backbone = any(k.startswith("features.") for k in state)

    if has_backbone:
        model.load_state_dict(state)
        return path

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Chaves inesperadas em '{path}': {sorted(unexpected)[:5]}")
    non_backbone_missing = [k for k in missing if not k.startswith("features.")]
    if non_backbone_missing:
        raise RuntimeError(
            f"Checkpoint '{path}' não contém a cabeça completa. "
            f"Faltando: {sorted(non_backbone_missing)[:5]}"
        )
    return path


def load_ensemble_head(device="cpu", subfolders=None):
    """Carrega o EnsembleHead com os checkpoints dos 3 membros."""
    subfolders = subfolders or ENSEMBLE_SEARCH_DIRS
    members = []
    for filename, cls in ENSEMBLE_MEMBER_FILES:
        member = cls()
        member.load_state_dict(torch.load(resolve_weights(filename, subfolders), map_location=device))
        members.append(member)
    model = EnsembleHead(members).to(device)
    model.eval()
    return model


# ==============================================================================
# FACTORY HELPER PARA CARREGAMENTO AUTOMÁTICO
# ==============================================================================
# Limiares de decisão padrão por modelo.
#
# Os três ficam em 0.50 de propósito. A seleção de semente/trio é feita sobre o
# split de VALIDAÇÃO com o limiar FIXO (src/select_single.py e src/select_ensemble.py):
# escolher semente e limiar ao mesmo tempo em 215 vídeos superajusta a validação —
# uma varredura de 91 limiares chegou a eleger θ = 0.15 para o dual-stream.
#
# Com o limiar fixo, a seleção mede a arquitetura, não o ponto de operação. A
# escolha do ponto de operação é uma decisão separada e explícita, feita por
# src/calibrate_threshold.py e passada em --threshold.
#
# O valor 0.52 usado na primeira versão veio de uma busca que maximizava acurácia
# no próprio teste — ver README, seção 5.
DEFAULT_THRESHOLDS = {
    "baseline": 0.50,
    "dualstream": 0.50,
    "ensemble": 0.50,
}


def load_classifier(model_name="ensemble", device="cpu", custom_path=None):
    """
    Carrega o classificador solicitado com seus respectivos pesos pré-treinados:
    - 'baseline'   -> Modelo 1 (Bi-GRU simples)
    - 'dualstream' -> Modelo 2 (Dual-Stream Latente)
    - 'ensemble'   -> Modelo 3 (Ensemble Cinético Tri-Stream)
    """
    model_name = model_name.lower()
    
    if model_name == "baseline":
        model = VideoClassifier().to(device)
        candidate_paths = [custom_path, "models/best_model.pth", "models/baseline_seed42.pth"]
        path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
        if not path:
            raise FileNotFoundError(f"Pesos do Baseline não encontrados em: {candidate_paths[1:]}")
        load_into_full_model(model, path, device)
        model.eval()
        default_th = DEFAULT_THRESHOLDS["baseline"]
        return model, default_th, "Baseline Minimalista (Bi-GRU)"

    elif model_name == "dualstream":
        model = VideoClassifierDualStream().to(device)
        candidate_paths = [
            custom_path,
            "models/model_dualstream_best.pth",
            "models/dualstream_seed42.pth",
            "experiments/model_dualstream_best.pth",
            "legacy_experiments/model_dualstream_best.pth",
        ]
        path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
        if not path:
            raise FileNotFoundError("Pesos do DualStream não encontrados em models/ ou experiments/")
        load_into_full_model(model, path, device)
        model.eval()
        default_th = DEFAULT_THRESHOLDS["dualstream"]
        return model, default_th, "Dual-Stream Latente (Aparência + Velocidade)"
        
    elif model_name in ["ensemble", "tristream", "kinetic"]:
        model = KineticEnsembleClassifier().to(device)
        p1 = resolve_weights(ENSEMBLE_MEMBER_FILES[0][0], ENSEMBLE_SEARCH_DIRS)
        p2 = resolve_weights(ENSEMBLE_MEMBER_FILES[1][0], ENSEMBLE_SEARCH_DIRS)
        p3 = resolve_weights(ENSEMBLE_MEMBER_FILES[2][0], ENSEMBLE_SEARCH_DIRS)
        model.m1.load_state_dict(torch.load(p1, map_location=device))
        model.m2.load_state_dict(torch.load(p2, map_location=device))
        model.m3.load_state_dict(torch.load(p3, map_location=device))
        model.eval()
        default_th = DEFAULT_THRESHOLDS["ensemble"]
        return model, default_th, "Ensemble Cinético Tri-Stream (melhor checkpoint)"
    else:
        raise ValueError(f"Modelo desconhecido: '{model_name}'. Escolha entre: 'baseline', 'dualstream', 'ensemble'.")
