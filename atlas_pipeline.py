"""
ATLAS Pipeline — Master Orchestrator
=======================================

Single entry point to run the complete ATLAS pipeline (Phases 2→5).
Phases 1 & 6 are C# scripts that run inside Autodesk Inventor.

USAGE:
    # Full pipeline (hash → build dataset → train → infer):
    python atlas_pipeline.py full \
        --parts-dir ./partsexport-new \
        --assemblies-dir ./assembliesexport-new \
        --bom ./bom.json \
        --output ./atlas_output

    # Individual phases:
    python atlas_pipeline.py hash --parts-dir ./partsexport-new --assemblies-dir ./assembliesexport-new
    python atlas_pipeline.py build --parts-dir ./hashed/parts --assemblies-dir ./hashed/assemblies
    python atlas_pipeline.py train --dataset ./dataset/atlas_dataset.pt
    python atlas_pipeline.py infer --bom ./bom.json --parts-dir ./hashed/parts --model ./checkpoints/atlas_best.pt
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path


def run_phase2_hashing(args):
    """Phase 2: Geometry Hashing."""
    print("\n" + "="*70)
    print("PHASE 2: GEOMETRY HASHING")
    print("="*70)
    from geometry_hasher import generate_training_data
    generate_training_data(args.parts_dir, args.assemblies_dir, args.hash_output)
    print("✅ Phase 2 complete")


def run_phase3_dataset(args):
    """Phase 3: Graph Construction."""
    print("\n" + "="*70)
    print("PHASE 3: GRAPH CONSTRUCTION")
    print("="*70)
    from atlas_dataset import AtlasGraphBuilder
    import torch

    builder = AtlasGraphBuilder(
        parts_dir=args.hashed_parts_dir,
        assemblies_dir=args.hashed_assemblies_dir,
        neg_ratio=args.neg_ratio,
    )
    dataset = builder.build_dataset()

    os.makedirs(args.dataset_output, exist_ok=True)
    save_path = os.path.join(args.dataset_output, "atlas_dataset.pt")
    torch.save(dataset, save_path)
    print(f"💾 Dataset saved: {save_path}")
    print("✅ Phase 3 complete")


def run_phase4_training(args):
    """Phase 4: GNN Training."""
    print("\n" + "="*70)
    print("PHASE 4: GNN TRAINING")
    print("="*70)
    from atlas_train import train_pipeline
    train_pipeline(args)
    print("✅ Phase 4 complete")


def run_phase5_inference(args):
    """Phase 5: Generative Inference."""
    print("\n" + "="*70)
    print("PHASE 5: GENERATIVE INFERENCE")
    print("="*70)
    from atlas_inference import run_inference
    run_inference(args)
    print("✅ Phase 5 complete")


def run_full_pipeline(args):
    """Run the complete pipeline (Phases 2-5)."""
    start = time.time()
    print("🚀 ATLAS Full Pipeline Starting...")
    print(f"   Parts:      {args.parts_dir}")
    print(f"   Assemblies: {args.assemblies_dir}")
    print(f"   Output:     {args.output}")

    base_output = args.output
    os.makedirs(base_output, exist_ok=True)

    # Phase 2
    args.hash_output = os.path.join(base_output, "hashed")
    run_phase2_hashing(args)

    # Phase 3
    args.hashed_parts_dir = os.path.join(base_output, "hashed", "parts")
    args.hashed_assemblies_dir = os.path.join(base_output, "hashed", "assemblies")
    args.dataset_output = os.path.join(base_output, "dataset")
    run_phase3_dataset(args)

    # Phase 4
    args.dataset = os.path.join(base_output, "dataset", "atlas_dataset.pt")
    args.output_dir = os.path.join(base_output, "checkpoints")
    # Use 'output' attr expected by train_pipeline
    orig_output = args.output
    args.output = args.output_dir
    run_phase4_training(args)
    args.output = orig_output

    # Phase 5
    args.model = os.path.join(base_output, "checkpoints", "atlas_best.pt")
    args.parts_dir_inf = os.path.join(base_output, "hashed", "parts")
    # Temporarily swap for inference
    orig_parts = args.parts_dir
    args.parts_dir = args.parts_dir_inf
    args.output = os.path.join(base_output, "predicted_assembly.json")
    run_phase5_inference(args)
    args.parts_dir = orig_parts

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"🏁 ATLAS Pipeline Complete in {elapsed:.1f}s")
    print(f"   Recipe: {os.path.join(base_output, 'predicted_assembly.json')}")
    print(f"   → Use reconstruction.cs in Inventor to build the assembly")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="ATLAS CAD Automation — Master Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline phase to run")

    # ── Full pipeline ─────────────────────────────────────────────────────────
    full_p = subparsers.add_parser("full", help="Run complete pipeline (Phases 2→5)")
    full_p.add_argument("--parts-dir", required=True, help="Raw part export directory")
    full_p.add_argument("--assemblies-dir", required=True, help="Raw assembly export directory")
    full_p.add_argument("--bom", required=True, help="BOM JSON for inference")
    full_p.add_argument("--output", default="./atlas_output", help="Base output directory")
    full_p.add_argument("--epochs", type=int, default=200)
    full_p.add_argument("--lr", type=float, default=0.001)
    full_p.add_argument("--hidden", type=int, default=128)
    full_p.add_argument("--dropout", type=float, default=0.3)
    full_p.add_argument("--neg-ratio", type=int, default=5)
    full_p.add_argument("--chunk-size", type=int, default=50000)
    full_p.add_argument("--llm-api-key", type=str, default=None)

    # ── Hash only ─────────────────────────────────────────────────────────────
    hash_p = subparsers.add_parser("hash", help="Phase 2: Geometry Hashing")
    hash_p.add_argument("--parts-dir", required=True)
    hash_p.add_argument("--assemblies-dir", default=None)
    hash_p.add_argument("--hash-output", default="./hashed")

    # ── Build dataset ─────────────────────────────────────────────────────────
    build_p = subparsers.add_parser("build", help="Phase 3: Graph Dataset Construction")
    build_p.add_argument("--hashed-parts-dir", required=True)
    build_p.add_argument("--hashed-assemblies-dir", required=True)
    build_p.add_argument("--dataset-output", default="./dataset")
    build_p.add_argument("--neg-ratio", type=int, default=5)

    # ── Train ─────────────────────────────────────────────────────────────────
    train_p = subparsers.add_parser("train", help="Phase 4: GNN Training")
    train_p.add_argument("--dataset", required=True)
    train_p.add_argument("--output", default="./checkpoints")
    train_p.add_argument("--epochs", type=int, default=200)
    train_p.add_argument("--lr", type=float, default=0.001)
    train_p.add_argument("--hidden", type=int, default=128)
    train_p.add_argument("--dropout", type=float, default=0.3)
    train_p.add_argument("--llm-api-key", type=str, default=None)

    # ── Infer ─────────────────────────────────────────────────────────────────
    infer_p = subparsers.add_parser("infer", help="Phase 5: Generative Inference")
    infer_p.add_argument("--bom", required=True)
    infer_p.add_argument("--parts-dir", required=True)
    infer_p.add_argument("--model", required=True)
    infer_p.add_argument("--output", default="./predicted_assembly.json")
    infer_p.add_argument("--chunk-size", type=int, default=50000)
    infer_p.add_argument("--confidence", type=float, default=0.65)
    infer_p.add_argument("--llm-api-key", type=str, default=None)

    args = parser.parse_args()

    if args.command == "full":
        run_full_pipeline(args)
    elif args.command == "hash":
        run_phase2_hashing(args)
    elif args.command == "build":
        run_phase3_dataset(args)
    elif args.command == "train":
        run_phase4_training(args)
    elif args.command == "infer":
        run_phase5_inference(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
