/**
 * SYMBEON AOA CORE SDK - Public API
 * (c) 2026 Symbeon Labs
 * 
 * SOBERANIA ÓPTICA E ATESTAÇÃO DE REALIDADE FÍSICA.
 */

export * from './protocol';
export * from './Generator';
export * from './InferenceEngine';

const VERSION = "1.0.0-aoa-v2.1";

export function getSDKMetadata() {
  return {
    protocol: "Atomic Optical Signature",
    version: VERSION,
    licensing: "Proprietary (GuardDrive Tech / Symbeon Labs)",
    popeEnabled: true
  };
}
