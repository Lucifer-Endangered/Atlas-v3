"""
ATLAS Phase 3 — Graph Construction for PyTorch Geometric
==========================================================

Converts hashed part JSONs + mapped assembly JSONs into PyG Data objects.

NODE FEATURES (21-dim):
    [0-2]   Entity type one-hot (Face=1,0,0 | Edge=0,1,0 | WorkPlane=0,0,1)
    [3]     Area (cm²) or Length (cm) — normalized log-scale
    [4-6]   Center / Midpoint (x, y, z) — global coordinates
    [7-9]   Normal / Tangent vector (nx, ny, nz)
    [10-12] Bounding box dimensions (dx, dy, dz)
    [13]    Surface type encoding (0=Plane,1=Cyl,2=Cone,3=Sphere,4=Torus,5=Spline,6=Other)
    [14]    Radius (cm, 0 if N/A)
    [15]    Material hash (first 8 hex chars → float)
    [16]    Mass (kg, log-scale)
    [17-18] Kinematic DOF (translation_dof, rotation_dof) from assembly context
    [19-20] Loop topology (outer_loop_count, total_edge_count_in_loops)

EDGES:
    - Structural: B-Rep adjacency (face↔edge from loop data)
    - Target:     Assembly constraints (the labels the GNN must learn)

USAGE:
    python atlas_dataset.py --parts-dir ./hashed/parts \
                            --assemblies-dir ./hashed/assemblies \
                            --output ./dataset \
                            --neg-ratio 5
"""

import json
import os
import math
import random
import hashlib
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# ── Optional: PyTorch Geometric ──────────────────────────────────────────────
try:
    import torch
    from torch_geometric.data import Data, InMemoryDataset
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[WARNING] torch_geometric not installed. Will export raw numpy arrays.")


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURE_DIM = 21
NUM_CONSTRAINT_CLASSES = 5  # 0=None, 1=Mate, 2=Flush, 3=Insert, 4=Angle

SURFACE_TYPE_MAP = {
    "kPlaneSurface": 0, "kCylinderSurface": 1, "kConeSurface": 2,
    "kSphereSurface": 3, "kTorusSurface": 4, "kBSplineSurface": 5,
}

CONSTRAINT_TYPE_MAP = {
    "kMateConstraint": 1, "kMateConstraintObject": 1,
    "kFlushConstraint": 2, "kFlushConstraintObject": 2,
    "kInsertConstraint": 3, "kInsertConstraintObject": 3,
    "kAngleConstraint": 4, "kAngleConstraintObject": 4,
}


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def safe_float(val, default=0.0):
    """Safely convert value to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_vec3(vec, default=(0.0, 0.0, 0.0)):
    """Extract 3D vector from various formats."""
    if vec is None:
        return default
    if isinstance(vec, dict):
        return (safe_float(vec.get("x")), safe_float(vec.get("y")), safe_float(vec.get("z")))
    if isinstance(vec, (list, tuple)) and len(vec) >= 3:
        return (safe_float(vec[0]), safe_float(vec[1]), safe_float(vec[2]))
    return default


def log_scale(val, eps=1e-8):
    """Log-scale normalization for areas/lengths."""
    return math.log1p(abs(val) + eps)


def material_to_float(material_str):
    """Convert material string to a float via hash."""
    if not material_str:
        return 0.0
    h = hashlib.md5(material_str.encode()).hexdigest()[:8]
    return int(h, 16) / (16**8)  # Normalize to [0, 1]


def encode_surface_type(st):
    """Encode surface type as integer."""
    if st is None:
        return 6
    return SURFACE_TYPE_MAP.get(st, 6)


def build_face_features(face: dict, part_meta: dict) -> List[float]:
    """Build 21-dim feature vector for a face node."""
    features = [0.0] * FEATURE_DIM

    # [0-2] Entity type one-hot: Face
    features[0] = 1.0

    # [3] Area (log-scaled)
    area = safe_float(face.get("area_cm2"))
    features[3] = log_scale(area)

    # [4-6] Center point
    center = safe_vec3(face.get("center") or face.get("face_point_at_center"))
    features[4], features[5], features[6] = center

    # [7-9] Normal vector
    normal = safe_vec3(face.get("normal") or face.get("face_normal_at_center"))
    features[7], features[8], features[9] = normal

    # [10-12] Bounding box dimensions
    bbox_min = face.get("bbox_min") or face.get("face_bbox_min")
    bbox_max = face.get("bbox_max") or face.get("face_bbox_max")
    if bbox_min and bbox_max:
        bmin = safe_vec3(bbox_min)
        bmax = safe_vec3(bbox_max)
        features[10] = abs(bmax[0] - bmin[0])
        features[11] = abs(bmax[1] - bmin[1])
        features[12] = abs(bmax[2] - bmin[2])

    # [13] Surface type encoding
    features[13] = float(encode_surface_type(face.get("surface_type")))

    # [14] Radius
    features[14] = safe_float(face.get("radius_cm"))

    # [15] Material hash
    features[15] = material_to_float(part_meta.get("material", ""))

    # [16] Mass (log-scaled)
    features[16] = log_scale(safe_float(part_meta.get("mass_kg")))

    # [17-18] DOF (filled later from assembly context)
    features[17] = 0.0
    features[18] = 0.0

    # [19-20] Loop topology
    loops = face.get("loops", [])
    outer_count = sum(1 for lp in loops if lp.get("is_outer", False))
    total_edges_in_loops = sum(len(lp.get("edge_reference_keys", [])) for lp in loops)
    features[19] = float(outer_count)
    features[20] = float(total_edges_in_loops)

    return features


def build_edge_features(edge: dict, part_meta: dict) -> List[float]:
    """Build 21-dim feature vector for an edge node."""
    features = [0.0] * FEATURE_DIM

    # [0-2] Entity type one-hot: Edge
    features[1] = 1.0

    # [3] Length (log-scaled)
    length = safe_float(edge.get("length_cm"))
    features[3] = log_scale(length)

    # [4-6] Midpoint
    mid = safe_vec3(edge.get("midpoint") or edge.get("edge_midpoint"))
    features[4], features[5], features[6] = mid

    # [7-9] Tangent
    tan = safe_vec3(edge.get("tangent") or edge.get("edge_tangent_at_mid"))
    features[7], features[8], features[9] = tan

    # [10-12] Bounding box from vertices
    sv = safe_vec3(edge.get("start_vertex"))
    ev = safe_vec3(edge.get("end_vertex"))
    features[10] = abs(ev[0] - sv[0])
    features[11] = abs(ev[1] - sv[1])
    features[12] = abs(ev[2] - sv[2])

    # [13] Curve type encoding (reuse surface_type slot)
    ct = edge.get("curve_type", "")
    if "Line" in ct:
        features[13] = 0.0
    elif "Circle" in ct:
        features[13] = 1.0
    elif "Arc" in ct:
        features[13] = 2.0
    elif "Spline" in ct or "BSpline" in ct:
        features[13] = 5.0
    else:
        features[13] = 6.0

    # [15] Material hash
    features[15] = material_to_float(part_meta.get("material", ""))

    # [16] Mass
    features[16] = log_scale(safe_float(part_meta.get("mass_kg")))

    return features


def build_workplane_features(wp: dict, part_meta: dict) -> List[float]:
    """Build 21-dim feature vector for a workplane node."""
    features = [0.0] * FEATURE_DIM

    # [0-2] Entity type one-hot: WorkPlane
    features[2] = 1.0

    # [3] Area (virtual — tiny)
    features[3] = log_scale(safe_float(wp.get("area_cm2", 0.0001)))

    # [4-6] Center
    center = safe_vec3(wp.get("center"))
    features[4], features[5], features[6] = center

    # [7-9] Normal
    normal = safe_vec3(wp.get("normal"))
    features[7], features[8], features[9] = normal

    # [13] Surface type = Plane
    features[13] = 0.0

    # [15] Material
    features[15] = material_to_float(part_meta.get("material", ""))

    # [16] Mass
    features[16] = log_scale(safe_float(part_meta.get("mass_kg")))

    return features


# =============================================================================
# GRAPH BUILDER
# =============================================================================

class AtlasGraphBuilder:
    """
    Builds PyG-compatible graph data from hashed part + mapped assembly JSONs.

    Each assembly becomes one training graph containing all its parts'
    face/edge nodes, with B-Rep structural edges and constraint target labels.
    """

    def __init__(self, parts_dir: str, assemblies_dir: str, neg_ratio: int = 5):
        self.parts_dir = parts_dir
        self.assemblies_dir = assemblies_dir
        self.neg_ratio = neg_ratio  # Negative samples per positive constraint edge

        # Cache: part_name → {hash → node_index, features, ...}
        self.part_cache = {}

    def load_part(self, part_path: str) -> dict:
        """Load and cache a hashed part JSON."""
        with open(part_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    def build_part_graph(self, part_data: dict) -> dict:
        """
        Build graph nodes and structural edges for a single part.

        Returns:
            {
                "features": List[List[float]],  # N x 21
                "hash_to_idx": Dict[str, int],   # geometry_hash → node index
                "structural_edges": List[Tuple[int, int]],  # B-Rep adjacency
                "part_name": str,
                "ref_key_to_hash": Dict[str, str]
            }
        """
        features = []
        hash_to_idx = {}
        ref_key_to_idx = {}
        structural_edges = []

        part_name = part_data.get("part_name", "")

        # Extract part metadata for feature building
        # Try to load original part JSON for full metadata
        part_meta = {"material": "", "mass_kg": 0.0}

        # ── Face nodes ────────────────────────────────────────────────────────
        for face_entry in part_data.get("face_hashes", []):
            geo_hash = face_entry.get("geometry_hash")
            if not geo_hash or geo_hash in hash_to_idx:
                continue

            idx = len(features)
            hash_to_idx[geo_hash] = idx
            features.append(build_face_features(face_entry, part_meta))

            ref_key = face_entry.get("reference_key_string")
            if ref_key:
                ref_key_to_idx[ref_key] = idx

        # ── Edge nodes ────────────────────────────────────────────────────────
        for edge_entry in part_data.get("edge_hashes", []):
            geo_hash = edge_entry.get("geometry_hash")
            if not geo_hash or geo_hash in hash_to_idx:
                continue

            idx = len(features)
            hash_to_idx[geo_hash] = idx
            features.append(build_edge_features(edge_entry, part_meta))

            ref_key = edge_entry.get("reference_key_string")
            if ref_key:
                ref_key_to_idx[ref_key] = idx

        # ── Structural edges (B-Rep adjacency from hash_to_geometry) ─────────
        # We infer adjacency: edges are adjacent to the faces they bound.
        # This info comes from the part's face loops → edge reference keys.
        hash_to_geo = part_data.get("hash_to_geometry", {})
        ref_to_hash = part_data.get("ref_key_to_hash", {})

        # Build reverse map: ref_key → geometry_hash
        for face_entry in part_data.get("face_hashes", []):
            face_hash = face_entry.get("geometry_hash")
            if not face_hash or face_hash not in hash_to_idx:
                continue
            face_idx = hash_to_idx[face_hash]

            # We don't have direct loop→edge hash mapping here,
            # so we use ref_key cross-referencing
            ref_key = face_entry.get("reference_key_string")
            if ref_key and ref_key in ref_key_to_idx:
                # Look for edges adjacent to this face via the original part data
                pass  # Will be linked through assembly constraint data

        return {
            "features": features,
            "hash_to_idx": hash_to_idx,
            "structural_edges": structural_edges,
            "part_name": part_name,
            "ref_key_to_hash": part_data.get("ref_key_to_hash", {})
        }

    def build_assembly_graph(self, asm_data: dict) -> Optional[dict]:
        """
        Build a full assembly graph from a mapped assembly JSON.

        Combines all part sub-graphs into one mega-graph with:
        - Part-internal B-Rep structural edges
        - Cross-part constraint target edges (positive labels)
        - Negative-sampled non-constraint edges (negative labels)
        """
        constraints = asm_data.get("mapped_constraints", [])
        if not constraints:
            return None

        # ── Collect all unique part files referenced ──────────────────────────
        part_hashes_needed = set()
        for con in constraints:
            for side in ["one", "two"]:
                h = con.get(f"entity_{side}_hash")
                if h:
                    part_hashes_needed.add(h)

        # ── Load all parts and merge into single graph ────────────────────────
        all_features = []
        global_hash_to_idx = {}
        all_structural_edges = []
        part_membership = []  # Which part each node belongs to (for cross-part filtering)

        # Load all available hashed part files
        part_files = list(Path(self.parts_dir).glob("*_hashed.json"))
        part_idx_counter = 0

        for pf in part_files:
            try:
                pdata = self.load_part(str(pf))
                pg = self.build_part_graph(pdata)

                offset = len(all_features)
                for feat in pg["features"]:
                    all_features.append(feat)
                    part_membership.append(part_idx_counter)

                # Remap hash→idx with global offset
                for h, local_idx in pg["hash_to_idx"].items():
                    global_hash_to_idx[h] = local_idx + offset

                # Remap structural edges
                for (src, dst) in pg["structural_edges"]:
                    all_structural_edges.append((src + offset, dst + offset))

                part_idx_counter += 1
            except Exception as e:
                print(f"  [WARN] Failed loading part {pf.name}: {e}")

        if len(all_features) == 0:
            return None

        # ── Build constraint target edges ─────────────────────────────────────
        positive_edges = []  # (src_idx, dst_idx)
        positive_labels = []  # constraint class

        for con in constraints:
            h1 = con.get("entity_one_hash")
            h2 = con.get("entity_two_hash")
            ctype = con.get("constraint_type", "")

            if h1 and h2 and h1 in global_hash_to_idx and h2 in global_hash_to_idx:
                idx1 = global_hash_to_idx[h1]
                idx2 = global_hash_to_idx[h2]
                label = CONSTRAINT_TYPE_MAP.get(ctype, 0)

                if label > 0:
                    positive_edges.append((idx1, idx2))
                    positive_labels.append(label)

        # ── Negative sampling ─────────────────────────────────────────────────
        negative_edges = []
        negative_labels = []
        num_nodes = len(all_features)
        positive_set = set(positive_edges) | set((b, a) for a, b in positive_edges)

        # Only sample cross-part negatives (same-part faces can't be constrained)
        num_neg_needed = len(positive_edges) * self.neg_ratio
        attempts = 0
        max_attempts = num_neg_needed * 10

        while len(negative_edges) < num_neg_needed and attempts < max_attempts:
            i = random.randint(0, num_nodes - 1)
            j = random.randint(0, num_nodes - 1)
            attempts += 1

            if i == j:
                continue
            if part_membership[i] == part_membership[j]:
                continue  # Same part — skip
            if (i, j) in positive_set:
                continue

            negative_edges.append((i, j))
            negative_labels.append(0)
            positive_set.add((i, j))  # Prevent duplicates

        # ── Combine ───────────────────────────────────────────────────────────
        all_target_edges = positive_edges + negative_edges
        all_target_labels = positive_labels + negative_labels

        return {
            "assembly_name": asm_data.get("assembly_name", ""),
            "node_features": all_features,
            "structural_edges": all_structural_edges,
            "target_edges": all_target_edges,
            "target_labels": all_target_labels,
            "num_nodes": num_nodes,
            "num_positive": len(positive_edges),
            "num_negative": len(negative_edges),
            "part_membership": part_membership,
        }

    def to_pyg_data(self, graph: dict) -> Any:
        """Convert graph dict to PyTorch Geometric Data object."""
        if not HAS_PYG:
            return graph  # Return raw dict if PyG not available

        x = torch.tensor(graph["node_features"], dtype=torch.float)

        # Structural edges (undirected)
        if graph["structural_edges"]:
            se = torch.tensor(graph["structural_edges"], dtype=torch.long).t()
            # Make undirected
            se = torch.cat([se, se.flip(0)], dim=1)
        else:
            se = torch.zeros((2, 0), dtype=torch.long)

        # Target edges for link prediction
        if graph["target_edges"]:
            te = torch.tensor(graph["target_edges"], dtype=torch.long).t()
        else:
            te = torch.zeros((2, 0), dtype=torch.long)

        tl = torch.tensor(graph["target_labels"], dtype=torch.long)
        pm = torch.tensor(graph["part_membership"], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=se,
            target_edge_index=te,
            target_labels=tl,
            part_membership=pm,
            num_nodes=graph["num_nodes"],
        )
        data.assembly_name = graph["assembly_name"]

        return data

    def build_dataset(self) -> List:
        """Build all assembly graphs into a list of PyG Data objects."""
        dataset = []

        asm_files = list(Path(self.assemblies_dir).glob("*_mapped.json"))
        print(f"\n{'='*60}")
        print(f"Building graph dataset from {len(asm_files)} assemblies")
        print(f"{'='*60}")

        for af in sorted(asm_files):
            try:
                with open(af, "r", encoding="utf-8-sig") as f:
                    asm_data = json.load(f)

                graph = self.build_assembly_graph(asm_data)
                if graph is None:
                    print(f"  ⚠️  {af.name}: No valid constraints found")
                    continue

                data = self.to_pyg_data(graph)
                dataset.append(data)

                print(f"  ✅ {af.name}: {graph['num_nodes']} nodes, "
                      f"+{graph['num_positive']} / -{graph['num_negative']} edges")

            except Exception as e:
                print(f"  ❌ {af.name}: {e}")

        print(f"\n📊 Dataset: {len(dataset)} assembly graphs built")
        return dataset


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Phase 3 — Build PyG graph dataset from hashed CAD data"
    )
    parser.add_argument("--parts-dir", required=True, help="Hashed parts directory")
    parser.add_argument("--assemblies-dir", required=True, help="Mapped assemblies directory")
    parser.add_argument("--output", default="./dataset", help="Output directory")
    parser.add_argument("--neg-ratio", type=int, default=5, help="Negative samples per positive")

    args = parser.parse_args()

    builder = AtlasGraphBuilder(args.parts_dir, args.assemblies_dir, args.neg_ratio)
    dataset = builder.build_dataset()

    os.makedirs(args.output, exist_ok=True)

    if HAS_PYG:
        save_path = os.path.join(args.output, "atlas_dataset.pt")
        torch.save(dataset, save_path)
        print(f"\n💾 Saved PyG dataset: {save_path}")
    else:
        save_path = os.path.join(args.output, "atlas_dataset.json")
        serializable = []
        for g in dataset:
            serializable.append({
                "assembly_name": g.get("assembly_name", ""),
                "num_nodes": g["num_nodes"],
                "num_positive": g["num_positive"],
                "num_negative": g["num_negative"],
                "node_features": g["node_features"],
                "target_edges": g["target_edges"],
                "target_labels": g["target_labels"],
            })
        with open(save_path, "w") as f:
            json.dump(serializable, f)
        print(f"\n💾 Saved JSON dataset: {save_path}")


if __name__ == "__main__":
    main()
