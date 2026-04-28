import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from synthetic.generator import OpticalNoiseGenerator
from inference.matcher import OFPMatcher
from inference.scoring import compute_trust_score

def run_tests():
    print('--- ?? INICIANDO LOOP AOA CORE ---')
    generator = OpticalNoiseGenerator()
    matcher = OFPMatcher()
    
    # 1. Gerar 'Selo Físico' com GTID
    print('[+] Gerando Selo Real (GTID: TAG-001)')
    gtid = 'TAG-001'
    real_images = generator.generate_batch(gtid, count=2)
    
    # 2. Mocking Tensor Transform (N, C, H, W)
    t_img_real = torch.ones((1, 3, 224, 224)) # Dummy tensor representation
    t_img_fake = torch.zeros((1, 3, 224, 224))
    
    # 3. Registrar no FAISS
    print('[+] Registrando OFP no Banco FAISS...')
    emb_real = matcher.extract_ofp(t_img_real)
    matcher.db.add_embeddings(emb_real, gtid)
    
    # 4. Validar Cópia vs Real
    print('[??] Testando Match: Real vs Real')
    res_real = matcher.match(emb_real)
    score_real = res_real[0]['score'] if res_real else 0.0
    
    print('[??] Testando Match: Real vs Fake (Anomalia)')
    emb_fake = matcher.extract_ofp(t_img_fake)
    res_fake = matcher.match(emb_fake)
    score_fake = res_fake[0]['score'] if res_fake else 0.0
    
    # 5. Output
    print(f'\\n[RESULTADO] Similaridade Real:  {score_real:.4f} (> 0.90 esperado)')
    print(f'[RESULTADO] Similaridade Fake:  {score_fake:.4f} (< 0.75 esperado)')
    
    t_score = compute_trust_score(optical=score_real)
    print(f'\\n? TRUST SCORE FINAL: {t_score:.2f}/100')

if __name__ == '__main__':
    run_tests()
