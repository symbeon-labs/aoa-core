# 🧬 Substrato de Treinamento: AOA-Core & Magistrado Themis

Este manifesto define a estrutura de dados e conhecimento necessária para o
treinamento (fine-tuning) do ecossistema GuardDrive. O substrato é dividido em
duas camadas: **Visão Óptica** (Dataset) e **Inteligência Jurídica** (RAG).

---

## 1. Camada de Visão (Dataset OFP)

O objetivo é treinar o YOLOv11 e o EfficientNet-B0 para distinguir a "Verdade
Física" da "Fraude de Imagem".

### Classes de Treinamento:
| Classe | Descrição | Importância |
|---|---|---|
| `REAL_TAG` | GuardTag original com resina e DTM correta. | Referência |
| `FAKE_XEROX` | Fotocópia colorida do selo (plana). | Detecção de Relevo |
| `FAKE_SCREEN` | Foto do selo em uma tela digital. | Detecção de Moiré |
| `FAKE_REPRINT` | Impressão térmica ou jato de tinta sem OVI. | Detecção de Cor |
| `DAMAGED_TAG` | Selo real mas violado/rasgado (Tamper). | Segurança Jurídica |

### Estrutura de Metadados (`metadata.json`):
```json
{
  "entry_id": "SUB-0001",
  "source": "REAL_CAPTURE",
  "hardware_version": "v2.1",
  "lighting_condition": "DIRECT_SUN",
  "optical_signature": "VECTOR_L2_256",
  "label": "AUTHENTIC"
}
```

---

## 2. Camada de Conhecimento (RAG Knowledge Base)

O Magistrado Themis utiliza este substrato para fundamentar seus laudos.

### Fontes de Verdade Jurídica:
- **Artigo 159 (CPP):** Regras de admissibilidade de prova pericial.
- **Lei 12.030/09:** Disposições sobre perícias oficiais.
- **Protocolo UEAP:** Especificação técnica dos eventos de atestação.
- **Patente GD-IP-2026-0002:** Descrição dos métodos proprietários.

---

## 3. Metas de Performance (GATE 1)

1.  **AOA-Score Accuracy:** > 98% em ambiente controlado.
2.  **RAG Latency:** < 500ms para recuperação de contexto jurídico.
3.  **Cross-Validation:** O Trust Score deve ser derivado da consistência entre
    o vetor óptico e o histórico recuperado pelo RAG.

---

*Documento proprietário da Symbeon Labs. Uso restrito para treinamento do 
AOA-Core.*
