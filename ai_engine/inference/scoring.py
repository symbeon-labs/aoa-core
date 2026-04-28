def compute_trust_score(optical: float, rf: float = 1.0, history: float = 1.0, tamper: float = 1.0) -> float:
    \"\"\"
    Fórmula Híbrida do Trust Score (GuardDrive)
    \"\"\"
    base_score = (
        optical * 0.45 +
        rf * 0.25 +
        history * 0.20 +
        tamper * 0.10
    ) * 100
    
    return max(0.0, min(100.0, base_score))
