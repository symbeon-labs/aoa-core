# -*- coding: utf-8 -*-
"""
AOA-Core — Trainable OFP Pipeline v0.1
Augmentation transforms para metric learning.
Simula condições do mundo real para garantir robustez do embedding.
"""
import torchvision.transforms as T


def get_train_transforms():
    """
    Transforms agressivos para treino.
    Força o modelo a aprender features invariantes à iluminação,
    rotação e distorção de perspectiva.
    """
    return T.Compose([
        T.Resize((224, 224)),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        T.RandomRotation(degrees=15),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.RandomPerspective(distortion_scale=0.3, p=0.5),
        T.RandomHorizontalFlip(p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_eval_transforms():
    """
    Transforms mínimos para inferência e avaliação.
    Sem augmentation — apenas normalização padrão ImageNet.
    """
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
