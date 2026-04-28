# -*- coding: utf-8 -*-
"""
AOA-Core — Triplet Dataset Loader
Gera triplets (anchor, positive, negative) para treino com Triplet Loss.
Cada triplet garante que o modelo aprenda a separar selos diferentes
no espaço de embeddings.
"""
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple


class TripletDataset(Dataset):
    """
    Dataset para metric learning via Triplet Loss.

    Estrutura esperada:
        dataset/
            seal_001/
                img_01.jpg
                img_02.jpg
                ...
            seal_002/
                ...

    Cada pasta = um selo único (classe).
    """

    def __init__(self, root: str, transform=None, length: int = 10000):
        """
        Args:
            root: caminho raiz do dataset
            transform: transformações torchvision
            length: número virtual de samples por epoch
        """
        self.root = root
        self.transform = transform
        self.length = length

        # Mapear classes -> imagens
        self.classes = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]

        if len(self.classes) < 2:
            raise ValueError(f"Dataset precisa de pelo menos 2 classes. Encontrado: {len(self.classes)}")

        self.class_to_imgs = {
            cls: [
                os.path.join(root, cls, f)
                for f in os.listdir(os.path.join(root, cls))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            for cls in self.classes
        }

        # Filtrar classes com menos de 2 imagens (impossível gerar triplet)
        self.classes = [c for c in self.classes if len(self.class_to_imgs[c]) >= 2]
        print(f"[TripletDataset] {len(self.classes)} classes | {self.length} samples/epoch")

    def _load(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> Tuple:
        # Anchor + Positive: mesma classe (mesmo selo)
        anchor_class = random.choice(self.classes)
        anchor_path, positive_path = random.sample(self.class_to_imgs[anchor_class], 2)

        # Negative: classe diferente (selo diferente)
        negative_class = random.choice([c for c in self.classes if c != anchor_class])
        negative_path = random.choice(self.class_to_imgs[negative_class])

        anchor = self._load(anchor_path)
        positive = self._load(positive_path)
        negative = self._load(negative_path)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self) -> int:
        return self.length
