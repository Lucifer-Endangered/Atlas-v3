#!/usr/bin/env python3
"""
ATLAS — Full Pipeline Orchestrator
====================================
Runs Phase 2 → 3 → 4 sequentially:
  Phase 2: Geometry Hashing (parts + assembly mapping)
  Phase 3: Graph Dataset Construction (PyG Data objects)
  Phase 4: GNN Training (GraphSAGE link prediction)

After training, use atlas_inference.py for Phase 5 (BOM → predicted assembly).

USAGE:
    python3 run_pipeline.py
"""

import os
import sys
import time
import json
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
PARTS_DIR     = os.path.join(BASE, "Output Parts")
ASM_DIR       = os.path.join(BASE, "Output Assemblies")
HASHED_DIR    = os.path.join(BASE, "hashed_output")
DATASET_DIR   = os.path.join(BASE, "dataset")
CHECKPOINT_DIR= os.path.join(BASE, "checkpoints")
VENV_PYTHON   = "python3"

def run_step(name, cmd):
    print(f"\n{'='*70}")
    print(f"  PHASE: {name}")
    print(f"{'='*70}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=BASE)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌ {name} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        sys.exit(1)
    print(f"\n✅ {name} completed in {elapsed:.1f}s")

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           ATLAS — Full ML Pipeline Orchestrator            ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Parts:      {PARTS_DIR}")
    print(f"║  Assemblies: {ASM_DIR}")
    print(f"║  Output:     {HASHED_DIR}")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Phase 2: Geometry Hashing ─────────────────────────────────────
    run_step("Phase 2 — Geometry Hashing + Assembly Mapping", [
        VENV_PYTHON, "geometry_hasher.py",
        "--parts-dir", PARTS_DIR,
        "--assemblies-dir", ASM_DIR,
        "--output", HASHED_DIR,
        "--precision", "4",
    ])

    # ── Phase 3: Graph Dataset Construction ───────────────────────────
    run_step("Phase 3 — PyG Graph Dataset Construction", [
        VENV_PYTHON, "atlas_dataset.py",
        "--parts-dir", os.path.join(HASHED_DIR, "parts"),
        "--assemblies-dir", os.path.join(HASHED_DIR, "assemblies"),
        "--output", DATASET_DIR,
        "--neg-ratio", "5",
    ])

    # ── Phase 4: GNN Training ─────────────────────────────────────────
    run_step("Phase 4 — GraphSAGE GNN Training", [
        VENV_PYTHON, "atlas_train.py",
        "--dataset", os.path.join(DATASET_DIR, "atlas_dataset.pt"),
        "--output", CHECKPOINT_DIR,
        "--epochs", "150",
        "--lr", "0.001",
        "--hidden", "128",
        "--dropout", "0.3",
    ])

    print(f"\n{'='*70}")
    print("🎉 ATLAS PIPELINE COMPLETE!")
    print(f"   Model checkpoint: {os.path.join(CHECKPOINT_DIR, 'atlas_best.pt')}")
    print(f"\n   To run inference:")
    print(f"   python3 atlas_inference.py --bom bom.json \\")
    print(f"       --parts-dir {os.path.join(HASHED_DIR, 'parts')} \\")
    print(f"       --model {os.path.join(CHECKPOINT_DIR, 'atlas_best.pt')} \\")
    print(f"       --output predicted_assembly.json")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
