import os
import json

def prepare_knowledge_substrate():
    """
    Varre a base de conhecimento e prepara o contexto para o RAG.
    """
    kb_path = "c:/Users/João/Desktop/PROJETOS/01_CORE_SYSTEMS/guarddrive/aoa-core/substrate/knowledge_base"
    substrate = []
    
    for root, dirs, files in os.walk(kb_path):
        for file in files:
            if file.endswith(".md"):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    substrate.append({
                        "source": file,
                        "category": os.path.basename(root),
                        "content": content
                    })
    
    output_path = os.path.join(kb_path, "knowledge_substrate.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(substrate, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Substrato de Conhecimento gerado em: {output_path}")

if __name__ == "__main__":
    prepare_knowledge_substrate()
