# -*- coding: utf-8 -*-
"""
AOA-Core — Inference API v0.1
API para receber imagens de captura e retornar o Optical Fingerprint (OFP)
e o Trust Score de autenticidade.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from PIL import Image
import io
import os
import sys
from datetime import datetime

# Adicionar caminhos para os modelos
sys.path.append(os.path.join(os.path.dirname(__file__), "ofp_model"))

from ofp_model.model.ofp_model import OFPModel
from ofp_model.transforms.augment import get_eval_transforms

app = FastAPI(title="AOA-Core Perception Engine", version="0.1")

# Configurações
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ofp_model", "model.pth")
EMBEDDING_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inicializar modelo
model = None
transform = get_eval_transforms()

@app.on_event("startup")
def load_model():
    global model
    model = OFPModel(embedding_dim=EMBEDDING_DIM, pretrained=False).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"[AOA] Modelo carregado de {MODEL_PATH}")
    else:
        print("[AOA] ATENÇÃO: model.pth não encontrado. Usando pesos aleatórios (apenas para teste)")
    model.eval()

@app.post("/perception/extract")
async def extract_ofp(file: UploadFile = File(...)):
    """
    Recebe uma imagem e retorna o embedding OFP.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Transform e Inferência
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model(tensor).squeeze(0).cpu().tolist()
            
        return {
            "gtid_candidate": None, # Futura busca no FAISS
            "ofp_embedding": embedding,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/perception/validate")
async def validate_seal(file: UploadFile = File(...), expected_gtid: str = None):
    """
    Valida um selo comparando o OFP extraído com o esperado (mock por enquanto).
    """
    # TODO: Integrar com FAISS Vector DB
    return {
        "trust_score": 0.0,
        "verdict": "PENDING_IMPLEMENTATION",
        "details": "Integração com FAISS Vector DB pendente."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
