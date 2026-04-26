# 📐 AOA Geometry Specification v1.0

## 1. The Concentric Topology
The AOA marker is defined by a series of concentric rings where the radius ($R_n$) and thickness ($T_n$) follow a deterministic sequence derived from a cryptographic seed (The Vehicle Identity).

### 1.1 Atomic Radius Sequence
The sequence is defined by the following recurrence:
$R_{n+1} = R_n \cdot \phi + S_{byte}$

Where:
- $\phi$: The Golden Ratio (1.618033...)
- $S_{byte}$: A segment of the SHA-256 hash of the Hardware ID.

## 2. Spectral Overlap (Multiplexing)
Each ring possesses a spectral filter property:
- **Inner Ring (ID):** Reflective at 940nm (IR-Active).
- **Middle Ring (Integrity):** Absorbent at 905nm.
- **Outer Ring (Global Anchor):** Visible (DPI-High Contrast Black).

## 3. DTM Micro-Geometry
The pattern surface must be treated with a linear grating of **450 lines/mm**. This ensures that the specular reflection at a 45-degree angle generates a predictable "Ghost Pattern" identifiable by the AI model, effectively killing any attempt at 2D photocopy fraud.

---
*Technical document for industrial printer calibration and AI training.*
