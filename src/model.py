import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class VideoClassifier(nn.Module):
    """
    Arquitetura Espaco-Temporal para Classificacao de Violencia em Video.
    - Espacial: MobileNetV3-Small pre-treinado (ImageNet) com GAP (576 dim).
    - Temporal: Bi-GRU (hidden_size=64, bidirectional -> 128 dim).
    - Cabeca de Classificacao: Dropout + Linear -> 2 classes (NonFight=0, Fight=1).
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        
        # 1. Extrator de caracteristicas espaciais (Backbone 2D)
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base = mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        feature_dim = 576  # Dimensao do GAP do MobileNetV3-Small
        
        # 2. Modelador temporal (Bi-GRU)
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Classificador
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Time=16, Channels=3, Height=224, Width=224)
        b, t, c, h, w = x.shape
        
        # Achata Batch e Time para processar espaco em lote unico
        x = x.view(b * t, c, h, w)
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)  # (b * t, 576)
        
        # Restaura sequencia temporal
        feat = feat.view(b, t, -1)     # (b, t, 576)
        
        # Modelagem temporal e agregacao
        gru_out, _ = self.gru(feat)    # (b, t, 128)
        temporal_feat = gru_out.mean(dim=1)  # Temporal Average Pooling -> (b, 128)
        
        logits = self.classifier(temporal_feat)  # (b, 2)
        return logits
