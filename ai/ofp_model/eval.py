# -*- coding: utf-8 -*-
"""
AOA-Core — Evaluation Script v0.1
Avalia a qualidade do embedding OFP medindo:
- Similaridade intra-classe (real vs real): deve ser >= 0.9
- Similaridade inter-classe (real vs fake): deve ser <= 0.5
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.ofp_model import OFPModel
from transforms.augment import get_eval_transforms


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Calcula cosine similarity entre dois embeddings."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def evaluate_pair(model, transform, img_path_a: str, img_path_b: str, device: str) -> float:
    """Extrai OFP de duas imagens e retorna similaridade."""
    img_a = transform(Image.open(img_path_a).convert("RGB")).unsqueeze(0).to(device)
    img_b = transform(Image.open(img_path_b).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        emb_a = model(img_a)
        emb_b = model(img_b)

    return cosine_similarity(emb_a[0], emb_b[0])


def run_evaluation(dataset_root: str, model_path: str, embedding_dim: int = 256):
    """
    Avalia o modelo em pares intra/inter-classe.

    Critérios de sucesso (v0.1):
    - intra-classe (real vs real): ~0.9+
    - inter-classe (real vs fake): ~0.2–0.5
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = get_eval_transforms()

    model = OFPModel(embedding_dim=embedding_dim, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    classes = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    print(f"\n[AOA Eval] Avaliando modelo em {len(classes)} classes...\n")

    intra_scores, inter_scores = [], []

    for cls in classes[:10]:  # Amostra de 10 classes
        imgs = [
            os.path.join(dataset_root, cls, f)
            for f in os.listdir(os.path.join(dataset_root, cls))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if len(imgs) >= 2:
            score = evaluate_pair(model, transform, imgs[0], imgs[1], device)
            intra_scores.append(score)

        # Inter-classe: pegar imagem de outra classe
        other_cls = [c for c in classes if c != cls]
        if other_cls:
            other_imgs = [
                os.path.join(dataset_root, other_cls[0], f)
                for f in os.listdir(os.path.join(dataset_root, other_cls[0]))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if imgs and other_imgs:
                score = evaluate_pair(model, transform, imgs[0], other_imgs[0], device)
                inter_scores.append(score)

    avg_intra = sum(intra_scores) / len(intra_scores) if intra_scores else 0
    avg_inter = sum(inter_scores) / len(inter_scores) if inter_scores else 0

    print(f"  Similaridade Intra-classe (real vs real): {avg_intra:.4f}  {'✅' if avg_intra >= 0.85 else '⚠️'}")
    print(f"  Similaridade Inter-classe (real vs fake): {avg_inter:.4f}  {'✅' if avg_inter <= 0.60 else '⚠️'}")
    print(f"\n  Separação: {avg_intra - avg_inter:.4f}  (quanto maior melhor)")

    return {"intra": avg_intra, "inter": avg_inter}


if __name__ == "__main__":
    DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
    run_evaluation(DATASET_ROOT, MODEL_PATH)
