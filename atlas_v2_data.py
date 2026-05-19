"""
ATLAS V2 — Direct Geometry Training Data Builder
==================================================
Extracts training pairs directly from assembly constraint entities.
No reference key mapping needed — each constraint already contains
the full geometry of both constrained entities.

Training pair format:
  [entity_one_features(15d), entity_two_features(15d)] → constraint_type
"""

import json, os, math, hashlib, argparse, random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

SURFACE_TYPE_MAP = {
    "kPlaneSurface": 0, "kCylinderSurface": 1, "kConeSurface": 2,
    "kSphereSurface": 3, "kTorusSurface": 4, "kBSplineSurface": 5,
}
CURVE_TYPE_MAP = {
    "kLineCurve": 0, "kCircleCurve": 1, "kArcCurve": 2,
    "kEllipseCurve": 3, "kBSplineCurve": 4,
}
CONSTRAINT_MAP = {
    "kMateConstraint": 1, "kFlushConstraint": 2,
    "kInsertConstraint": 3, "kAngleConstraint": 4,
    "kTangentConstraint": 5,
}

def safe_float(v, default=0.0):
    if v is None: return default
    try: return float(v)
    except: return default

def safe_vec3(v):
    if v is None: return (0.0, 0.0, 0.0)
    if isinstance(v, dict):
        return (safe_float(v.get("x")), safe_float(v.get("y")), safe_float(v.get("z")))
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return (safe_float(v[0]), safe_float(v[1]), safe_float(v[2]))
    return (0.0, 0.0, 0.0)

def log_s(val):
    return math.log1p(abs(val) + 1e-8)

def entity_to_features(entity: dict) -> List[float]:
    """Convert a constraint entity to a 15-dim feature vector.
    
    [0]     is_face (1) or is_edge (0)
    [1]     surface/curve type encoded
    [2]     area_cm2 or length_cm (log-scaled)
    [3-5]   normal/tangent vector
    [6-8]   center/midpoint
    [9]     radius (0 if N/A)
    [10]    is_work_feature
    [11-13] start_vertex or bbox info
    [14]    half_angle / minor_radius / extra
    """
    f = [0.0] * 15
    
    st = entity.get("surface_type")
    ct = entity.get("curve_type")
    wf = entity.get("work_feature_name")
    
    if st:  # Face entity
        f[0] = 1.0
        f[1] = float(SURFACE_TYPE_MAP.get(st, 6))
        f[2] = log_s(safe_float(entity.get("area_cm2")))
        n = safe_vec3(entity.get("face_normal_at_center") or entity.get("normal"))
        f[3], f[4], f[5] = n
        c = safe_vec3(entity.get("face_point_at_center") or entity.get("center"))
        f[6], f[7], f[8] = c
        f[9] = safe_float(entity.get("radius_cm"))
        f[14] = safe_float(entity.get("half_angle_rad"))
    elif ct:  # Edge entity
        f[0] = 0.0
        f[1] = float(CURVE_TYPE_MAP.get(ct, 5))
        f[2] = log_s(safe_float(entity.get("length_cm")))
        t = safe_vec3(entity.get("edge_tangent_at_mid") or entity.get("tangent"))
        f[3], f[4], f[5] = t
        m = safe_vec3(entity.get("edge_midpoint") or entity.get("midpoint"))
        f[6], f[7], f[8] = m
        f[9] = safe_float(entity.get("radius_cm"))
        sv = safe_vec3(entity.get("edge_start_vertex") or entity.get("start_vertex"))
        f[11], f[12], f[13] = sv
    elif wf:  # Work feature
        f[0] = 0.5  # Neither face nor edge
        f[10] = 1.0
        n = safe_vec3(entity.get("face_normal_at_center"))
        f[3], f[4], f[5] = n
        c = safe_vec3(entity.get("face_point_at_center"))
        f[6], f[7], f[8] = c
    
    return f

ENTITY_DIM = 15

# =============================================================================
# PART FEATURE EXTRACTION (for BOM-based inference)
# =============================================================================

def part_face_to_features(face: dict) -> List[float]:
    """Convert a part face to the same 15-dim feature space."""
    f = [0.0] * ENTITY_DIM
    f[0] = 1.0
    f[1] = float(SURFACE_TYPE_MAP.get(face.get("surface_type"), 6))
    f[2] = log_s(safe_float(face.get("area_cm2")))
    n = safe_vec3(face.get("normal"))
    f[3], f[4], f[5] = n
    c = safe_vec3(face.get("center"))
    f[6], f[7], f[8] = c
    f[9] = safe_float(face.get("radius_cm"))
    f[14] = safe_float(face.get("half_angle_rad"))
    return f

def part_edge_to_features(edge: dict) -> List[float]:
    """Convert a part edge to the same 15-dim feature space."""
    f = [0.0] * ENTITY_DIM
    f[0] = 0.0
    f[1] = float(CURVE_TYPE_MAP.get(edge.get("curve_type"), 5))
    f[2] = log_s(safe_float(edge.get("length_cm")))
    t = safe_vec3(edge.get("tangent"))
    f[3], f[4], f[5] = t
    m = safe_vec3(edge.get("midpoint"))
    f[6], f[7], f[8] = m
    f[9] = safe_float(edge.get("radius_cm"))
    sv = safe_vec3(edge.get("start_vertex"))
    f[11], f[12], f[13] = sv
    return f

# =============================================================================
# TRAINING DATA EXTRACTION
# =============================================================================

def extract_training_pairs(asm_dir: str, parts_dir: str) -> dict:
    """Extract all training pairs from assembly constraints.
    
    Returns dict with:
        positive_pairs: List of (feat_one[15], feat_two[15], constraint_label, metadata)
        part_features:  Dict[part_name -> List of entity feature vectors]
    """
    positives = []
    part_features_cache = {}
    skipped = 0
    
    asm_files = sorted(Path(asm_dir).glob("*.json"))
    print(f"\n{'='*60}")
    print(f"Extracting training pairs from {len(asm_files)} assemblies")
    print(f"{'='*60}")
    
    for asm_path in asm_files:
        if asm_path.name.startswith("_"):
            continue
        try:
            with open(asm_path, "r", encoding="utf-8-sig") as f:
                asm = json.load(f)
            
            constraints = asm.get("assembly_graph", {}).get("constraint_edges", [])
            asm_name = asm.get("assembly_metadata", {}).get("assembly_name", asm_path.stem)
            count = 0
            
            for c in constraints:
                ctype = c.get("constraint_type", "")
                label = CONSTRAINT_MAP.get(ctype, 0)
                if label == 0:
                    continue
                
                e1 = c.get("entity_one", {})
                e2 = c.get("entity_two", {})
                
                # Skip if either entity is empty/missing geometry
                if len(e1) <= 1 or len(e2) <= 1:
                    skipped += 1
                    continue
                if not (e1.get("surface_type") or e1.get("curve_type") or e1.get("work_feature_name")):
                    skipped += 1
                    continue
                if not (e2.get("surface_type") or e2.get("curve_type") or e2.get("work_feature_name")):
                    skipped += 1
                    continue
                
                f1 = entity_to_features(e1)
                f2 = entity_to_features(e2)
                
                meta = {
                    "assembly": asm_name,
                    "constraint": c.get("constraint_name", ""),
                    "type": ctype,
                    "part_one": c.get("node_one_id", ""),
                    "part_two": c.get("node_two_id", ""),
                    "e1_type": e1.get("surface_type") or e1.get("curve_type") or "work",
                    "e2_type": e2.get("surface_type") or e2.get("curve_type") or "work",
                }
                
                positives.append((f1, f2, label, meta))
                count += 1
            
            if count > 0:
                print(f"  ✅ {asm_path.name}: {count} training pairs")
            
            # Cache part features for this assembly's parts
            nodes = asm.get("assembly_graph", {}).get("nodes", [])
            for node in nodes:
                pname = node.get("file_name", "")
                if not pname:
                    continue
                stem = Path(pname).stem
                if stem in part_features_cache:
                    continue
                # Try to load part JSON
                part_path = os.path.join(parts_dir, f"{stem}.json")
                if not os.path.exists(part_path):
                    continue
                try:
                    with open(part_path, "r", encoding="utf-8-sig") as pf:
                        pdata = json.load(pf)
                    feats = []
                    for face in pdata.get("faces", []):
                        feats.append({
                            "features": part_face_to_features(face),
                            "type": "face",
                            "surface_type": face.get("surface_type", ""),
                            "area_cm2": face.get("area_cm2", 0),
                        })
                    for edge in pdata.get("edges", []):
                        feats.append({
                            "features": part_edge_to_features(edge),
                            "type": "edge",
                            "curve_type": edge.get("curve_type", ""),
                        })
                    part_features_cache[stem] = feats
                except:
                    pass
                    
        except Exception as e:
            print(f"  ❌ {asm_path.name}: {e}")
    
    print(f"\n📊 Total positive pairs: {len(positives)}, Skipped: {skipped}")
    
    # Label distribution
    labels = Counter(p[2] for p in positives)
    inv_map = {v: k for k, v in CONSTRAINT_MAP.items()}
    for lbl, cnt in sorted(labels.items()):
        print(f"   {inv_map.get(lbl, '?'):25s}: {cnt}")
    
    return {
        "positives": positives,
        "part_features": part_features_cache,
    }


def build_dataset(positives: list, neg_ratio: int = 3) -> dict:
    """Build final training dataset with negative sampling.
    
    For negatives: randomly pair entities from different parts in the
    same assembly that are NOT constrained together.
    """
    X = []
    y = []
    
    # Positives: concatenate [f1, f2] → 30-dim input
    for f1, f2, label, meta in positives:
        X.append(f1 + f2)  # 30-dim
        y.append(label)
    
    num_pos = len(X)
    
    # Simple negatives: shuffle one side of the pairs
    indices = list(range(num_pos))
    for _ in range(neg_ratio):
        shuffled = indices.copy()
        random.shuffle(shuffled)
        for i, j in zip(indices, shuffled):
            if i == j:
                continue
            f1 = positives[i][0]
            f2 = positives[j][1]
            X.append(f1 + f2)
            y.append(0)  # No constraint
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    print(f"\n📊 Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Positives: {num_pos}, Negatives: {len(y) - num_pos}")
    
    return {"X": X, "y": y, "num_pos": num_pos}


def main():
    parser = argparse.ArgumentParser(description="ATLAS V2 — Build training data")
    parser.add_argument("--asm-dir", default="./Output Assemblies")
    parser.add_argument("--parts-dir", default="./Output Parts")
    parser.add_argument("--output", default="./v2_dataset")
    parser.add_argument("--neg-ratio", type=int, default=3)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    result = extract_training_pairs(args.asm_dir, args.parts_dir)
    dataset = build_dataset(result["positives"], args.neg_ratio)
    
    # Save
    np.save(os.path.join(args.output, "X.npy"), dataset["X"])
    np.save(os.path.join(args.output, "y.npy"), dataset["y"])
    
    # Save part features cache for inference
    pf_serializable = {}
    for k, v in result["part_features"].items():
        pf_serializable[k] = [{"features": e["features"], "type": e["type"],
                                "surface_type": e.get("surface_type", ""),
                                "curve_type": e.get("curve_type", ""),
                                "area_cm2": e.get("area_cm2", 0)} for e in v]
    with open(os.path.join(args.output, "part_features.json"), "w") as f:
        json.dump(pf_serializable, f)
    
    # Save metadata
    meta = {
        "num_samples": len(dataset["y"]),
        "num_positives": dataset["num_pos"],
        "feature_dim": 30,
        "num_classes": 6,  # 0=None,1=Mate,2=Flush,3=Insert,4=Angle,5=Tangent
        "entity_dim": ENTITY_DIM,
    }
    with open(os.path.join(args.output, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n💾 Saved to {args.output}/")
    print(f"   X.npy: {dataset['X'].shape}")
    print(f"   y.npy: {dataset['y'].shape}")


if __name__ == "__main__":
    main()
