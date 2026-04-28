import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class OFPExtractor(nn.Module):
    def __init__(self, embedding_dim=256):
        super(OFPExtractor, self).__init__()
        # Load pre-trained backbone
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Remove original classifier
        num_ftrs = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Add custom embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        # L2 Normalization for Cosine Similarity
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
