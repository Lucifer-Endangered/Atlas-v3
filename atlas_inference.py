"""
ATLAS Phase 5 — Generative Assembly Orchestration (Inference Engine)
=====================================================================

Given a BOM (list of part files), this engine:
1. Loads part geometries into a unified disconnected graph
2. Generates all cross-part face/edge pairs
3. Runs GNN inference in memory-safe chunks (prevents OOM)
4. Applies kinematic heuristics to sequence constraints
5. Outputs a predicted_assembly.json recipe for Phase 6 reconstruction

KEY FEATURES:
    - Chunked batch processing (configurable, default 50K pairs)
    - Kinematic sequencing: Insert (Primary) → Flush/Mate (Secondary)
    - Confidence thresholding with adaptive cutoff
    - Duplicate/redundant constraint pruning
    - Optional LLM refinement of constraint ordering

USAGE:
    python atlas_inference.py --bom bom.json \
                              --parts-dir ./hashed/parts \
                              --model ./checkpoints/atlas_best.pt \
                              --output ./predicted_assembly.json \
                              --chunk-size 50000

BOM FORMAT (bom.json):
    {
        "parts": [
            {"file": "Shaft.ipt", "hashed_json": "Shaft_hashed.json", "count": 1},
            {"file": "Bearing.ipt", "hashed_json": "Bearing_hashed.json", "count": 2},
            {"file": "Housing.ipt", "hashed_json": "Housing_hashed.json", "count": 1}
        ]
    }
"""

import json
import os
import time
import argparse
import itertools
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F

from atlas_model import AtlasGNN
from atlas_dataset import (
    build_face_features, build_edge_features, build_workplane_features,
    FEATURE_DIM, CONSTRAINT_TYPE_MAP,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONSTRAINT_NAMES = ["NoConstraint", "Mate", "Flush", "Insert", "Angle"]
DEFAULT_CHUNK_SIZE = 50000
CONFIDENCE_THRESHOLD = 0.65  # Minimum softmax probability to accept


# =============================================================================
# DEVICE SELECTION
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# BOM LOADER
# =============================================================================

def load_bom(bom_path: str) -> list:
    """Load Bill of Materials from JSON."""
    with open(bom_path, "r") as f:
        bom = json.load(f)
    return bom.get("parts", bom if isinstance(bom, list) else [])


# =============================================================================
# PART GRAPH LOADER
# =============================================================================

def load_part_into_graph(
    hashed_json_path: str,
    instance_id: str,
    part_meta: dict = None,
) -> dict:
    """
    Load a single hashed part JSON and extract nodes with features.

    Returns:
        {
            "instance_id": str,
            "file": str,
            "nodes": [
                {
                    "local_idx": int,
                    "geometry_hash": str,
                    "entity_type": str,
                    "features": List[float],  # 21-dim
                    "surface_type": str,
                    "area_cm2": float,
                    "center": list,
                    "normal": list,
                    "radius_cm": float,
                }
            ]
        }
    """
    with open(hashed_json_path, "r") as f:
        data = json.load(f)

    meta = part_meta or {"material": "", "mass_kg": 0.0}
    nodes = []

    # Faces
    for i, face in enumerate(data.get("face_hashes", [])):
        features = build_face_features(face, meta)
        nodes.append({
            "local_idx": len(nodes),
            "geometry_hash": face.get("geometry_hash"),
            "entity_type": "Face",
            "features": features,
            "surface_type": face.get("surface_type", ""),
            "area_cm2": face.get("area_cm2", 0),
            "center": face.get("center"),
            "normal": face.get("normal"),
            "radius_cm": face.get("radius_cm"),
            "reference_key_string": face.get("reference_key_string"),
        })

    # Edges
    for i, edge in enumerate(data.get("edge_hashes", [])):
        features = build_edge_features(edge, meta)
        nodes.append({
            "local_idx": len(nodes),
            "geometry_hash": edge.get("geometry_hash"),
            "entity_type": "Edge",
            "features": features,
            "curve_type": edge.get("curve_type", ""),
            "length_cm": edge.get("length_cm", 0),
            "midpoint": edge.get("midpoint"),
            "reference_key_string": edge.get("reference_key_string"),
        })

    return {
        "instance_id": instance_id,
        "file": data.get("source_file", ""),
        "part_name": data.get("part_name", ""),
        "nodes": nodes,
    }


# =============================================================================
# UNIFIED GRAPH BUILDER
# =============================================================================

def build_unified_graph(
    bom: list,
    parts_dir: str,
) -> Tuple[torch.Tensor, list, list]:
    """
    Build a unified disconnected graph from all BOM parts.

    Returns:
        features:       Tensor [N, 21]
        node_registry:  List of node metadata dicts (with global_idx)
        part_instances:  List of (instance_id, start_idx, end_idx) for cross-part pairing
    """
    all_features = []
    node_registry = []
    part_instances = []

    for entry in bom:
        count = entry.get("count", 1)
        hashed_json = entry.get("hashed_json", "")
        part_file = entry.get("file", "")

        hashed_path = os.path.join(parts_dir, hashed_json)
        if not os.path.exists(hashed_path):
            # Try finding by part name
            stem = Path(part_file).stem
            hashed_path = os.path.join(parts_dir, f"{stem}_hashed.json")
            if not os.path.exists(hashed_path):
                print(f"  ⚠️  Part not found: {hashed_json}")
                continue

        for inst in range(count):
            instance_id = f"{Path(part_file).stem}:{inst+1}"
            start_idx = len(all_features)

            part_graph = load_part_into_graph(hashed_path, instance_id)

            for node in part_graph["nodes"]:
                node["global_idx"] = len(all_features)
                node["instance_id"] = instance_id
                node["part_file"] = part_file
                all_features.append(node["features"])
                node_registry.append(node)

            end_idx = len(all_features)
            part_instances.append((instance_id, start_idx, end_idx))

            print(f"  📦 Loaded {instance_id}: {end_idx - start_idx} nodes")

    features = torch.tensor(all_features, dtype=torch.float)
    return features, node_registry, part_instances


# =============================================================================
# CROSS-PART PAIR GENERATION
# =============================================================================

def generate_cross_part_pairs(
    part_instances: list,
    node_registry: list,
    face_only: bool = True,
) -> List[Tuple[int, int]]:
    """
    Generate all cross-part node pairs for inference.
    Only pairs faces from different part instances (not same-instance pairs).

    Args:
        face_only: If True, only generate face-face pairs (most constraint-relevant)
    """
    pairs = []

    for (id_a, start_a, end_a), (id_b, start_b, end_b) in itertools.combinations(part_instances, 2):
        for i in range(start_a, end_a):
            if face_only and node_registry[i].get("entity_type") != "Face":
                continue
            for j in range(start_b, end_b):
                if face_only and node_registry[j].get("entity_type") != "Face":
                    continue
                pairs.append((i, j))

    return pairs


# =============================================================================
# CHUNKED INFERENCE
# =============================================================================

@torch.no_grad()
def chunked_inference(
    model: AtlasGNN,
    features: torch.Tensor,
    pairs: List[Tuple[int, int]],
    device: torch.device,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> List[dict]:
    """
    Run GNN inference in memory-safe chunks.

    Returns list of predictions with confidence scores.
    """
    model.eval()
    features = features.to(device)

    # Empty edge_index (disconnected graph — no structural edges between parts)
    edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    # Encode all nodes once
    embeddings = model.encode(features, edge_index)

    predictions = []
    total_pairs = len(pairs)
    num_chunks = (total_pairs + chunk_size - 1) // chunk_size

    print(f"\n🔮 Running inference on {total_pairs:,} pairs in {num_chunks} chunks...")

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_pairs)
        chunk_pairs = pairs[start:end]

        pair_tensor = torch.tensor(chunk_pairs, dtype=torch.long, device=device).t()  # [2, K]
        logits = model.predict_links(embeddings, pair_tensor)  # [K, 5]
        probs = F.softmax(logits, dim=-1)  # [K, 5]

        for k, (i, j) in enumerate(chunk_pairs):
            pred_class = probs[k].argmax().item()
            confidence = probs[k].max().item()

            if pred_class > 0 and confidence >= CONFIDENCE_THRESHOLD:
                predictions.append({
                    "src_idx": i,
                    "dst_idx": j,
                    "predicted_class": pred_class,
                    "constraint_type": CONSTRAINT_NAMES[pred_class],
                    "confidence": round(confidence, 4),
                    "class_probs": probs[k].cpu().tolist(),
                })

        if (chunk_idx + 1) % max(1, num_chunks // 10) == 0:
            print(f"    Chunk {chunk_idx+1}/{num_chunks} processed "
                  f"({len(predictions)} constraints found so far)")

    print(f"  ✅ {len(predictions)} candidate constraints above threshold")
    return predictions


# =============================================================================
# KINEMATIC SEQUENCING
# =============================================================================

def kinematic_sequencing(
    predictions: list,
    node_registry: list,
) -> list:
    """
    Apply kinematic heuristics to order and filter constraints.

    Strategy:
        1. INSERT constraints (Primary Anchors): Cylindrical alignments first
           - These lock 4 DOF (2 translational + 2 rotational)
        2. MATE/FLUSH constraints (Secondary Locks): Planar alignments
           - These lock remaining DOF
        3. ANGLE constraints (Tertiary): Angular relationships
        4. Remove redundant constraints (same part-pair, same type)
        5. Sort by confidence within each priority tier
    """
    # Annotate with node metadata
    for pred in predictions:
        src = node_registry[pred["src_idx"]]
        dst = node_registry[pred["dst_idx"]]
        pred["src_instance"] = src.get("instance_id", "")
        pred["dst_instance"] = dst.get("instance_id", "")
        pred["src_surface_type"] = src.get("surface_type", "")
        pred["dst_surface_type"] = dst.get("surface_type", "")
        pred["src_hash"] = src.get("geometry_hash", "")
        pred["dst_hash"] = dst.get("geometry_hash", "")
        pred["src_ref_key"] = src.get("reference_key_string", "")
        pred["dst_ref_key"] = dst.get("reference_key_string", "")
        pred["src_file"] = src.get("part_file", "")
        pred["dst_file"] = dst.get("part_file", "")

    # ── Priority tiers ────────────────────────────────────────────────────────
    inserts = [p for p in predictions if p["constraint_type"] == "Insert"]
    mates = [p for p in predictions if p["constraint_type"] == "Mate"]
    flushes = [p for p in predictions if p["constraint_type"] == "Flush"]
    angles = [p for p in predictions if p["constraint_type"] == "Angle"]

    # Sort each tier by confidence (descending)
    inserts.sort(key=lambda x: -x["confidence"])
    mates.sort(key=lambda x: -x["confidence"])
    flushes.sort(key=lambda x: -x["confidence"])
    angles.sort(key=lambda x: -x["confidence"])

    # ── Deduplication ─────────────────────────────────────────────────────────
    # For each part pair, keep at most:
    #   1 Insert, 2 Mates/Flushes, 1 Angle
    seen_pairs = defaultdict(lambda: {"Insert": 0, "Mate": 0, "Flush": 0, "Angle": 0})
    max_per_type = {"Insert": 1, "Mate": 2, "Flush": 2, "Angle": 1}

    sequenced = []

    for tier in [inserts, mates, flushes, angles]:
        for pred in tier:
            pair_key = tuple(sorted([pred["src_instance"], pred["dst_instance"]]))
            ctype = pred["constraint_type"]

            if seen_pairs[pair_key][ctype] < max_per_type.get(ctype, 1):
                sequenced.append(pred)
                seen_pairs[pair_key][ctype] += 1

    # ── Validation: Insert must have cylindrical faces ────────────────────────
    validated = []
    for pred in sequenced:
        if pred["constraint_type"] == "Insert":
            has_cylinder = (
                "Cylinder" in pred.get("src_surface_type", "") or
                "Cylinder" in pred.get("dst_surface_type", "")
            )
            if not has_cylinder:
                # Downgrade Insert to Mate if no cylindrical face
                pred["constraint_type"] = "Mate"
                pred["original_type"] = "Insert (downgraded)"
        validated.append(pred)

    return validated


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def generate_assembly_recipe(
    sequenced_constraints: list,
    bom: list,
    output_path: str,
):
    """
    Generate the predicted_assembly.json recipe for Phase 6 reconstruction.
    """
    recipe = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": "ATLAS v2.0 — GNN Inference Engine",
        "bom": bom,
        "total_constraints": len(sequenced_constraints),
        "constraints": [],
    }

    for i, pred in enumerate(sequenced_constraints):
        constraint = {
            "sequence_order": i + 1,
            "constraint_name": f"AI_Constraint_{i+1}",
            "constraint_type": f"k{pred['constraint_type']}Constraint",
            "confidence": pred["confidence"],
            "occurrence_one": pred["src_instance"],
            "occurrence_two": pred["dst_instance"],
            "entity_one": {
                "geometry_hash": pred.get("src_hash", ""),
                "reference_key_string": pred.get("src_ref_key", ""),
                "surface_type": pred.get("src_surface_type", ""),
                "entity_type": "Face",
                "owner_document": pred.get("src_file", ""),
            },
            "entity_two": {
                "geometry_hash": pred.get("dst_hash", ""),
                "reference_key_string": pred.get("dst_ref_key", ""),
                "surface_type": pred.get("dst_surface_type", ""),
                "entity_type": "Face",
                "owner_document": pred.get("dst_file", ""),
            },
            "offset_cm": 0.0,
            "angle_rad": None,
        }

        if pred.get("original_type"):
            constraint["note"] = pred["original_type"]

        recipe["constraints"].append(constraint)

    with open(output_path, "w") as f:
        json.dump(recipe, f, indent=2)

    print(f"\n✅ Assembly recipe saved: {output_path}")
    print(f"   Total constraints: {len(sequenced_constraints)}")

    # Summary by type
    type_counts = defaultdict(int)
    for c in sequenced_constraints:
        type_counts[c["constraint_type"]] += 1
    for ctype, count in sorted(type_counts.items()):
        print(f"   {ctype:>10s}: {count}")


# =============================================================================
# LLM REFINEMENT (OPTIONAL)
# =============================================================================

def refine_with_llm(
    constraints: list,
    bom: list,
    api_key: str = None,
    model_name: str = "gpt-4o-mini",
) -> list:
    """
    Optionally use an LLM to validate and reorder the predicted constraints.
    This catches mechanical impossibilities the GNN might miss.
    """
    if not api_key:
        return constraints

    try:
        import requests

        parts_desc = ", ".join(p.get("file", "") for p in bom)
        constraints_desc = json.dumps([{
            "type": c["constraint_type"],
            "src": c["src_instance"],
            "dst": c["dst_instance"],
            "confidence": c["confidence"],
            "src_surface": c.get("src_surface_type", ""),
            "dst_surface": c.get("dst_surface_type", ""),
        } for c in constraints[:20]], indent=2)  # Limit to top 20 for API

        prompt = f"""You are a mechanical assembly expert. Given these parts: {parts_desc}

And these AI-predicted assembly constraints (ordered by sequence):
{constraints_desc}

Please validate:
1. Are any constraints mechanically impossible?
2. Should any constraint types be changed? (e.g., Insert needs cylindrical faces)
3. Is the ordering optimal for assembly stability?

Respond with ONLY a JSON array of constraint indices to REMOVE (0-indexed), 
or an empty array [] if all look correct. No explanation."""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200,
        }

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers, json=body, timeout=20,
        )

        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            import re
            match = re.search(r'\[[\d,\s]*\]', content)
            if match:
                remove_indices = json.loads(match.group())
                if remove_indices:
                    print(f"🤖 LLM flagged {len(remove_indices)} constraints for removal")
                    constraints = [c for i, c in enumerate(constraints) if i not in remove_indices]
                else:
                    print("🤖 LLM validated all constraints ✅")
    except Exception as e:
        print(f"  [LLM] Refinement failed: {e}")

    return constraints


# =============================================================================
# MAIN INFERENCE PIPELINE
# =============================================================================

def run_inference(args):
    """Full inference pipeline."""
    device = get_device()

    # ── Load BOM ──────────────────────────────────────────────────────────────
    print(f"\n📋 Loading BOM: {args.bom}")
    bom = load_bom(args.bom)
    print(f"   {len(bom)} part types in BOM")

    # ── Build unified graph ───────────────────────────────────────────────────
    print(f"\n📦 Building unified graph...")
    features, node_registry, part_instances = build_unified_graph(bom, args.parts_dir)
    print(f"   Total nodes: {features.shape[0]}")
    print(f"   Part instances: {len(part_instances)}")

    # ── Generate cross-part pairs ─────────────────────────────────────────────
    pairs = generate_cross_part_pairs(part_instances, node_registry, face_only=True)
    print(f"   Cross-part face pairs: {len(pairs):,}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n🧠 Loading model: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    hidden = checkpoint.get("hidden_channels", 128)

    model = AtlasGNN(in_channels=21, hidden_channels=hidden, num_classes=5).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"   Loaded epoch {checkpoint.get('epoch', '?')}, "
          f"val_f1={checkpoint.get('val_f1', 0):.4f}")

    # ── Inference ─────────────────────────────────────────────────────────────
    predictions = chunked_inference(
        model, features, pairs, device,
        chunk_size=args.chunk_size,
    )

    # ── Kinematic sequencing ──────────────────────────────────────────────────
    print(f"\n⚙️  Applying kinematic sequencing...")
    sequenced = kinematic_sequencing(predictions, node_registry)
    print(f"   {len(sequenced)} constraints after sequencing")

    # ── Optional LLM refinement ───────────────────────────────────────────────
    if hasattr(args, 'llm_api_key') and args.llm_api_key:
        sequenced = refine_with_llm(
            sequenced, bom,
            api_key=args.llm_api_key,
            model_name=getattr(args, 'llm_model', 'gpt-4o-mini'),
        )

    # ── Generate recipe ───────────────────────────────────────────────────────
    generate_assembly_recipe(sequenced, bom, args.output)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ATLAS Phase 5 — Generative Inference")
    parser.add_argument("--bom", required=True, help="BOM JSON file path")
    parser.add_argument("--parts-dir", required=True, help="Hashed parts directory")
    parser.add_argument("--model", required=True, help="Trained model checkpoint (.pt)")
    parser.add_argument("--output", default="./predicted_assembly.json", help="Output recipe path")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Pairs per inference chunk")
    parser.add_argument("--confidence", type=float, default=0.65, help="Min confidence threshold")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key for refinement")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM model name")
    args = parser.parse_args()

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence

    run_inference(args)


if __name__ == "__main__":
    main()
