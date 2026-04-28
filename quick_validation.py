# -*- coding: utf-8 -*-
"""
AOA-Core — Quick Validation Script (Zero-Shot)
============================================================
Uso IMEDIATO — sem treino necessário.

Usa EfficientNet pré-treinado como feature extractor (zero-shot)
para medir separabilidade real vs fake AGORA.

COMO USAR:
----------
1. Tire 10-30 fotos do seu selo REAL (variações de luz, ângulo)
   → salve em: validation_data/real/

2. Tire 5-10 fotos de uma CÓPIA (print, screenshot na tela)
   → salve em: validation_data/fake/

3. Rode:
   python quick_validation.py

CRITÉRIOS DE SUCESSO (v0.1):
-----------------------------
real vs real → >= 0.85 ✅
real vs fake → <= 0.70 ✅
separação    → >= 0.15 ✅

Se atingir: você tem tecnologia real, não conceito.
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


# ─── CONFIG ────────────────────────────────────────────────────────────────────

REAL_DIR  = "validation_data/real"
FAKE_DIR  = "validation_data/fake"
RESULTS   = "validation_results.json"
NUM_PAIRS = 20   # pares a amostrar por categoria

# ─── TRANSFORMS ────────────────────────────────────────────────────────────────

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ─── MODEL (zero-shot feature extractor) ───────────────────────────────────────

print("[AOA] Carregando EfficientNet-B0 (zero-shot)...")
base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
base.classifier = torch.nn.Identity()
base.eval()
print("[AOA] Modelo pronto.\n")


# ─── HELPERS ────────────────────────────────────────────────────────────────────

def load_images(folder: str):
    """Carrega todas as imagens de um diretório."""
    folder = Path(folder)
    if not folder.exists():
        print(f"[ERRO] Pasta não encontrada: {folder}")
        print(f"       Crie a pasta e adicione fotos do selo.")
        sys.exit(1)

    paths = list(folder.glob("*.jpg")) + \
            list(folder.glob("*.jpeg")) + \
            list(folder.glob("*.png"))

    if len(paths) < 2:
        print(f"[ERRO] Precisa de pelo menos 2 imagens em {folder}")
        sys.exit(1)

    return paths


def extract_embedding(img_path: Path) -> torch.Tensor:
    """Extrai embedding L2-normalizado de uma imagem."""
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        emb = base(tensor)

    return F.normalize(emb, p=2, dim=1).squeeze(0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def measure_pairs(paths_a, paths_b, label: str, n: int = NUM_PAIRS):
    """Amostra N pares e mede similaridade média."""
    scores = []
    samples = min(n, len(paths_a) * len(paths_b))

    for _ in range(samples):
        p_a = random.choice(paths_a)
        p_b = random.choice(paths_b)
        if p_a == p_b:
            continue

        emb_a = extract_embedding(p_a)
        emb_b = extract_embedding(p_b)
        s = cosine_sim(emb_a, emb_b)
        scores.append(s)

    avg = sum(scores) / len(scores) if scores else 0.0
    mn  = min(scores) if scores else 0.0
    mx  = max(scores) if scores else 0.0

    print(f"  [{label}] Pares: {len(scores)} | Avg: {avg:.4f} | Min: {mn:.4f} | Max: {mx:.4f}")
    return {"label": label, "avg": avg, "min": mn, "max": mx, "n_pairs": len(scores)}


# ─── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AOA-Core — Validação Empírica v0.1")
    print("  GuardTag Optical Fingerprint (Zero-Shot)")
    print("=" * 60)

    real_paths = load_images(REAL_DIR)
    fake_paths = load_images(FAKE_DIR)

    print(f"\n[Dataset] Real: {len(real_paths)} imagens | Fake: {len(fake_paths)} imagens\n")

    print("[Medindo] Similaridade INTRA-CLASSE (real vs real)...")
    intra = measure_pairs(real_paths, real_paths, "real_vs_real")

    print("[Medindo] Similaridade INTER-CLASSE (real vs fake)...")
    inter = measure_pairs(real_paths, fake_paths, "real_vs_fake")

    separation = intra["avg"] - inter["avg"]

    # ─── RESULTADO FINAL ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTADO FINAL")
    print("=" * 60)

    r2r_ok  = intra["avg"] >= 0.85
    r2f_ok  = inter["avg"] <= 0.70
    sep_ok  = separation >= 0.15

    print(f"\n  Real vs Real:  {intra['avg']:.4f}  {'✅ OK' if r2r_ok else '⚠️  Abaixo do threshold (0.85)'}")
    print(f"  Real vs Fake:  {inter['avg']:.4f}  {'✅ OK' if r2f_ok else '⚠️  Acima do threshold (0.70)'}")
    print(f"  Separação:     {separation:.4f}  {'✅ OK' if sep_ok else '⚠️  Baixa separação (<0.15)'}")

    if r2r_ok and r2f_ok and sep_ok:
        verdict = "VALIDATED"
        print("\n  🔥 SISTEMA VALIDADO — Tecnologia Real, não conceito.")
    elif sep_ok:
        verdict = "PROMISING"
        print("\n  ⚡ PROMISSOR — Treino com dados reais vai elevar os números.")
    else:
        verdict = "NEEDS_DATA"
        print("\n  📷 Colete mais fotos variadas do selo real e repita.")

    # ─── SALVAR RESULTADOS ─────────────────────────────────────────────────────
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": "EfficientNet-B0 (zero-shot)",
        "real_images": len(real_paths),
        "fake_images": len(fake_paths),
        "intra_class": intra,
        "inter_class": inter,
        "separation": separation,
        "verdict": verdict,
        "thresholds": {"real_vs_real": 0.85, "real_vs_fake": 0.70, "separation": 0.15}
    }

    with open(RESULTS, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Resultados salvos: {RESULTS}")
    print("=" * 60)
    return result


if __name__ == "__main__":
    main()
