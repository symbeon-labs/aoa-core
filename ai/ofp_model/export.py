# -*- coding: utf-8 -*-
"""
AOA-Core — Export Pipeline v0.1
Exporta o modelo OFP treinado para ONNX (deployment multi-plataforma).

ONNX permite:
- Inferência em CPU sem PyTorch instalado
- Deploy em Android/iOS (via ONNX Runtime Mobile)
- Conversão para TensorFlow Lite (opcional)
- Quantização INT8 para edge devices (Roadmap v0.4)
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.ofp_model import OFPModel


def export_to_onnx(
    model_path: str,
    output_path: str = "ofp_model.onnx",
    embedding_dim: int = 256,
    opset_version: int = 17,
):
    """
    Exporta o modelo OFP treinado para formato ONNX.

    Args:
        model_path: caminho para o .pth treinado
        output_path: caminho de saída do .onnx
        embedding_dim: dimensão do embedding (deve coincidir com o treino)
        opset_version: versão do ONNX opset (17 recomendado)
    """
    device = "cpu"  # Export sempre em CPU para compatibilidade

    # Carregar modelo treinado
    model = OFPModel(embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Input dummy (batch=1, RGB, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[AOA Export] Exportando para ONNX: {output_path}")
    print(f"  Opset: {opset_version} | Embedding dim: {embedding_dim}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "embedding": {0: "batch_size"}
        }
    )

    # Verificação de integridade
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ✅ Modelo ONNX validado com sucesso!")
    except ImportError:
        print("  ℹ️  onnx não instalado — validação pulada. (pip install onnx)")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Arquivo: {output_path} ({size_mb:.1f} MB)")

    return output_path


if __name__ == "__main__":
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "ofp_model.onnx")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERRO] Modelo não encontrado: {MODEL_PATH}")
        print("Execute train.py primeiro.")
    else:
        export_to_onnx(MODEL_PATH, OUTPUT_PATH)
