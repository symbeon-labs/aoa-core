from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

# Mock imports for API
from inference.scoring import compute_trust_score

app = FastAPI(title='AOA Core - Perception API')

class ValidationRequest(BaseModel):
    gtid: str
    image_base64: str

@app.post('/validate/full')
async def validate_full(req: ValidationRequest):
    # Fluxo Exato do Projeto: QR -> IA -> SCORE -> Decisão
    # 1. extrair OFP (mocked para setup)
    optical_similarity = 0.95
    
    # 2. gerar Trust Score
    score = compute_trust_score(optical=optical_similarity, rf=1.0, history=1.0, tamper=1.0)
    
    # 3. retornar resultado
    return {
        "gtid": req.gtid,
        "trust_score": score,
        "optical_confidence": optical_similarity,
        "decision": "VALID" if score >= 85 else "SUSPECT"
    }
