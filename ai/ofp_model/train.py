# -*- coding: utf-8 -*-
"""
AOA-Core — Training Pipeline v0.1
Treina o modelo OFP com Triplet Margin Loss para metric learning.
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.ofp_model import OFPModel
from dataset.triplet_loader import TripletDataset
from transforms.augment import get_train_transforms


def triplet_loss(anchor, positive, negative, margin: float = 0.3):
    """
    Triplet Margin Loss.
    Força: dist(anchor, positive) + margin < dist(anchor, negative)
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def train(
    dataset_root: str,
    output_path: str = "model.pth",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    embedding_dim: int = 256,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AOA Train] Device: {device}")

    # Modelo
    model = OFPModel(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Dataset
    dataset = TripletDataset(dataset_root, transform=get_train_transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Training loop
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (a, p, n) in enumerate(loader):
            a, p, n = a.to(device), p.to(device), n.to(device)

            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = triplet_loss(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        # Salvar melhor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path)
            print(f"  → Melhor modelo salvo: {output_path}")

    print(f"\n[AOA Train] Concluído. Melhor Loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset")
    train(dataset_root=DATASET_ROOT, epochs=10, batch_size=32)
