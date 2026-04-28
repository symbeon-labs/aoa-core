# -*- coding: utf-8 -*-
"""
AOA-Core — OFP Model v0.1
Backbone EfficientNet-B0 com projection head para metric learning.
Produz embeddings L2-normalizados de 256 dimensões (Optical Fingerprint).
"""
import torch
import torch.nn as nn
import torchvision.models as models


class OFPModel(nn.Module):
    """
    Extrator de Optical Fingerprint (OFP).
    Arquitetura: EfficientNet-B0 (features) → AdaptivePool → Projector → L2-Norm

    O embedding resultante representa a 'identidade física' do selo,
    comparável via cosine similarity.
    """

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()

        # Backbone pré-treinado (feature extractor)
        base = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        self.backbone = base.features  # (B, 1280, 7, 7) para input 224x224

        # Pooling global
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Projection head (1280 -> 512 -> embedding_dim)
        self.projector = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (B, 3, 224, 224)
        Returns:
            embedding: tensor (B, embedding_dim) L2-normalizado
        """
        features = self.backbone(x)                     # (B, 1280, 7, 7)
        pooled = self.pool(features).view(x.size(0), -1)  # (B, 1280)
        embedding = self.projector(pooled)              # (B, embedding_dim)

        # L2 normalization — garante comparação via Inner Product = Cosine
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
