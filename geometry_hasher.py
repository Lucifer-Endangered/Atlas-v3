"""
ATLAS Phase 1 — Consistent Geometry Hashing for Part-Assembly Entity Mapping
=============================================================================

This script creates deterministic, position-invariant hashes for every face and
edge entity extracted from part JSONs (by PartExtractor.cs). These hashes are
then used to map assembly constraint entities (from AssemblyExtractor.cs) to
their part-level geometry, enabling ML model training on the
part_geometry → assembly_constraints pipeline.

HASH DESIGN:
    The hash is a SHA-256 digest of a canonical signature built from geometric
    properties that are:
      1. STABLE across sessions (no transient keys or random IDs)
      2. UNIQUE within a part (discriminate between similar faces/edges)
      3. REPRODUCIBLE from both part-export and assembly-export data

FACE HASH = sha256(
    surface_type | rounded(area) | rounded(normal) | rounded(center) |
    rounded(radius) | rounded(bbox) | loop_topology
)

EDGE HASH = sha256(
    curve_type | rounded(length) | rounded(midpoint) | rounded(tangent) |
    rounded(start_vertex) | rounded(end_vertex)
)

The rounding precision (PRECISION) is configurable; default = 6 decimal places.
This tolerates floating-point noise from Inventor while preserving uniqueness.

USAGE:
    # Hash all part JSONs and create lookup tables
    python geometry_hasher.py --parts-dir ./partsexport-new --output ./hashed

    # Map assembly entities to part hashes
    python geometry_hasher.py --assemblies-dir ./assembliesexport-new \
                              --parts-dir ./partsexport-new \
                              --output ./hashed

    # Validate hash consistency (checks for collisions and unmapped entities)
    python geometry_hasher.py --validate --parts-dir ./partsexport-new
"""

import json
import hashlib
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any


# =============================================================================
# CONFIGURATION
# =============================================================================

PRECISION = 6       # Decimal places for rounding geometric values
HASH_ALGO = "sha256"  # Hash algorithm (sha256 for consistency, md5 for speed)


# =============================================================================
# CORE HASHING FUNCTIONS
# =============================================================================

def round_val(value: float, precision: int = PRECISION) -> str:
    """Round and format a float for hashing."""
    if value is None:
        return "None"
    return f"{round(float(value), precision):.{precision}f}"


def round_vec(vec, precision: int = PRECISION) -> str:
    """Round and format a vector/point array for hashing."""
    if vec is None:
        return "None"
    if isinstance(vec, dict):
        # Handle {"x": ..., "y": ..., "z": ...} format
        return f"({round_val(vec.get('x', 0), precision)},{round_val(vec.get('y', 0), precision)},{round_val(vec.get('z', 0), precision)})"
    if isinstance(vec, (list, tuple)):
        return "(" + ",".join(round_val(v, precision) for v in vec) + ")"
    return str(vec)


def hash_string(s: str) -> str:
    """Create a hex digest from a canonical string."""
    if HASH_ALGO == "sha256":
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    elif HASH_ALGO == "md5":
        return hashlib.md5(s.encode("utf-8")).hexdigest()
    else:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical_loop_signature(loops: List[dict]) -> str:
    """Create a canonical signature for face edge-loops (topology)."""
    if not loops:
        return "no_loops"
    
    sigs = []
    for loop in loops:
        is_outer = loop.get("is_outer", False)
        edge_count = len(loop.get("edge_reference_keys", []))
        sigs.append(f"{'O' if is_outer else 'I'}:{edge_count}")
    
    # Sort for consistency (inner loops can appear in any order)
    sigs.sort()
    return "|".join(sigs)


# =============================================================================
# FACE HASHING
# =============================================================================

def compute_face_hash(face: dict) -> str:
    """
    Compute a deterministic hash for a face from its geometric properties.
    
    The hash signature includes:
    - surface_type (Planar, Cylindrical, Conical, Spherical, etc.)
    - area (rounded)
    - normal vector at center (rounded)
    - center point (rounded)
    - radius (for curved surfaces)
    - bounding box (rounded)
    - loop topology (count of outer/inner loops and their edge counts)
    """
    parts = []
    
    # 1. Surface type
    surface_type = face.get("surface_type", "Unknown")
    parts.append(f"ST:{surface_type}")
    
    # 2. Area
    area = face.get("area_cm2") or face.get("area_mm2")
    parts.append(f"A:{round_val(area)}")
    
    # 3. Normal
    normal = face.get("normal")
    if normal is None:
        normal = face.get("face_normal_at_center")
    parts.append(f"N:{round_vec(normal)}")
    
    # 4. Center point
    center = face.get("center")
    if center is None:
        center = face.get("center_mm")
        if center is None:
            center = face.get("face_point_at_center")
    parts.append(f"C:{round_vec(center)}")
    
    # 5. Radius (for curved surfaces)
    radius = face.get("radius_cm")
    if radius is not None:
        parts.append(f"R:{round_val(radius)}")
    
    # 6. Minor radius (for torus)
    minor_radius = face.get("minor_radius_cm")
    if minor_radius is not None:
        parts.append(f"MR:{round_val(minor_radius)}")
    
    # 7. Half angle (for cones)
    half_angle = face.get("half_angle_rad")
    if half_angle is not None:
        parts.append(f"HA:{round_val(half_angle)}")
    
    # 8. Bounding box
    bbox_min = face.get("bbox_min")
    bbox_max = face.get("bbox_max")
    if bbox_min and bbox_max:
        parts.append(f"BB:{round_vec(bbox_min)}-{round_vec(bbox_max)}")
    elif face.get("face_bbox_min") and face.get("face_bbox_max"):
        parts.append(f"BB:{round_vec(face['face_bbox_min'])}-{round_vec(face['face_bbox_max'])}")
    
    # 9. Loop topology
    loops = face.get("loops")
    if loops:
        parts.append(f"L:{canonical_loop_signature(loops)}")
    
    signature = "|".join(parts)
    return hash_string(signature)


def compute_face_hash_from_assembly_entity(entity: dict) -> str:
    """
    Compute the face hash from an assembly entity's extracted data.
    This uses the same algorithm as compute_face_hash but reads from
    the assembly entity's field names.
    """
    face_proxy = {}
    
    face_proxy["surface_type"] = entity.get("surface_type", "Unknown")
    face_proxy["area_cm2"] = entity.get("area_cm2")
    face_proxy["normal"] = entity.get("face_normal_at_center")
    face_proxy["center"] = entity.get("face_point_at_center")
    face_proxy["radius_cm"] = entity.get("radius_cm")
    face_proxy["half_angle_rad"] = entity.get("half_angle_rad")
    
    if entity.get("face_bbox_min") and entity.get("face_bbox_max"):
        face_proxy["bbox_min"] = entity["face_bbox_min"]
        face_proxy["bbox_max"] = entity["face_bbox_max"]
    
    face_proxy["loops"] = entity.get("loops")
    
    return compute_face_hash(face_proxy)


# =============================================================================
# EDGE HASHING
# =============================================================================

def compute_edge_hash(edge: dict) -> str:
    """
    Compute a deterministic hash for an edge from its geometric properties.
    
    The hash signature includes:
    - curve_type (Line, Circle, Arc, Spline, etc.)
    - length (rounded)
    - midpoint (rounded)
    - tangent at midpoint (rounded)
    - start vertex (rounded)
    - end vertex (rounded)
    - radius (for circular edges)
    """
    parts = []
    
    # 1. Curve type
    curve_type = edge.get("curve_type", "Unknown")
    parts.append(f"CT:{curve_type}")
    
    # 2. Length
    length = edge.get("length_cm")
    parts.append(f"L:{round_val(length)}")
    
    # 3. Midpoint
    midpoint = edge.get("midpoint")
    if midpoint is None:
        midpoint = edge.get("edge_midpoint")
    parts.append(f"M:{round_vec(midpoint)}")
    
    # 4. Tangent
    tangent = edge.get("tangent")
    if tangent is None:
        tangent = edge.get("edge_tangent_at_mid")
    parts.append(f"T:{round_vec(tangent)}")
    
    # 5. Start vertex
    start_v = edge.get("start_vertex")
    if start_v is None:
        start_v = edge.get("edge_start_vertex")
    parts.append(f"SV:{round_vec(start_v)}")
    
    # 6. End vertex
    end_v = edge.get("end_vertex")
    if end_v is None:
        end_v = edge.get("edge_end_vertex")
    parts.append(f"EV:{round_vec(end_v)}")
    
    # 7. Radius (for circular edges)
    radius = edge.get("radius_cm")
    if radius is not None:
        parts.append(f"R:{round_val(radius)}")
    
    signature = "|".join(parts)
    return hash_string(signature)


def compute_edge_hash_from_assembly_entity(entity: dict) -> str:
    """
    Compute the edge hash from an assembly entity's extracted data.
    """
    edge_proxy = {}
    
    edge_proxy["curve_type"] = entity.get("curve_type", "Unknown")
    edge_proxy["length_cm"] = entity.get("length_cm")
    edge_proxy["midpoint"] = entity.get("edge_midpoint")
    edge_proxy["tangent"] = entity.get("edge_tangent_at_mid")
    edge_proxy["start_vertex"] = entity.get("edge_start_vertex")
    edge_proxy["end_vertex"] = entity.get("edge_end_vertex")
    edge_proxy["radius_cm"] = entity.get("radius_cm")
    
    return compute_edge_hash(edge_proxy)


# =============================================================================
# REFERENCE KEY BASED MAPPING (Primary method)
# =============================================================================

def build_refkey_to_hash_map(part_data: dict) -> Dict[str, str]:
    """
    Build a mapping from reference_key_string → geometry_hash for a single part.
    This is the PRIMARY mapping method since reference keys are guaranteed unique
    within a key context.
    """
    mapping = {}
    
    for face in part_data.get("faces", []):
        ref_key = face.get("reference_key_string")
        if ref_key:
            geo_hash = compute_face_hash(face)
            mapping[ref_key] = geo_hash
    
    for edge in part_data.get("edges", []):
        ref_key = edge.get("reference_key_string")
        if ref_key:
            geo_hash = compute_edge_hash(edge)
            mapping[ref_key] = geo_hash
    
    return mapping


# =============================================================================
# PART PROCESSING
# =============================================================================

def process_part_json(json_path: str) -> dict:
    """
    Process a single part JSON file and produce a hashed version.
    
    Returns dict with:
        - All original data
        - geometry_hash added to each face and edge
        - A ref_key_to_hash lookup table
    """
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    
    result = {
        "source_file": json_path,
        "part_name": data.get("part_metadata", {}).get("file_name", ""),
        "context_key_string": data.get("context_key_string", ""),
        "face_hashes": [],
        "edge_hashes": [],
        "ref_key_to_hash": {},
        "hash_to_geometry": {}
    }
    
    # Process faces
    for face in data.get("faces", []):
        geo_hash = compute_face_hash(face)
        
        face_entry = {
            "geometry_hash": geo_hash,
            "entity_type": "Face",
            "reference_key_string": face.get("reference_key_string"),
            "surface_type": face.get("surface_type"),
            "area_cm2": face.get("area_cm2"),
            "normal": face.get("normal"),
            "center": face.get("center"),
            "radius_cm": face.get("radius_cm"),
            "transient_key": face.get("transient_key")
        }
        result["face_hashes"].append(face_entry)
        
        ref_key = face.get("reference_key_string")
        if ref_key:
            result["ref_key_to_hash"][ref_key] = geo_hash
        
        # Store hash → geometry for reverse lookup
        result["hash_to_geometry"][geo_hash] = {
            "entity_type": "Face",
            "surface_type": face.get("surface_type"),
            "area_cm2": face.get("area_cm2"),
            "normal": face.get("normal"),
            "center": face.get("center"),
            "radius_cm": face.get("radius_cm"),
            "bbox_min": face.get("bbox_min"),
            "bbox_max": face.get("bbox_max"),
            "created_by_feature": face.get("created_by_feature")
        }
    
    # Process edges
    for edge in data.get("edges", []):
        geo_hash = compute_edge_hash(edge)
        
        edge_entry = {
            "geometry_hash": geo_hash,
            "entity_type": "Edge",
            "reference_key_string": edge.get("reference_key_string"),
            "curve_type": edge.get("curve_type"),
            "length_cm": edge.get("length_cm"),
            "midpoint": edge.get("midpoint"),
            "transient_key": edge.get("transient_key")
        }
        result["edge_hashes"].append(edge_entry)
        
        ref_key = edge.get("reference_key_string")
        if ref_key:
            result["ref_key_to_hash"][ref_key] = geo_hash
        
        result["hash_to_geometry"][geo_hash] = {
            "entity_type": "Edge",
            "curve_type": edge.get("curve_type"),
            "length_cm": edge.get("length_cm"),
            "midpoint": edge.get("midpoint"),
            "tangent": edge.get("tangent"),
            "start_vertex": edge.get("start_vertex"),
            "end_vertex": edge.get("end_vertex")
        }
    
    return result


def process_all_parts(parts_dir: str, output_dir: str) -> Dict[str, dict]:
    """Process all part JSONs in a directory and save hashed versions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Master lookup: indexed by context_key_string → {ref_key → hash}
    master_lookup = {}
    
    json_files = list(Path(parts_dir).glob("*.json"))
    print(f"\n{'='*60}")
    print(f"Processing {len(json_files)} part files from: {parts_dir}")
    print(f"{'='*60}")
    
    for json_path in sorted(json_files):
        try:
            result = process_part_json(str(json_path))
            
            # Save individual hashed part
            out_path = os.path.join(output_dir, f"{json_path.stem}_hashed.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            
            # Add to master lookup
            ctx_key = result.get("context_key_string", "")
            if ctx_key:
                master_lookup[ctx_key] = result["ref_key_to_hash"]
            
            n_faces = len(result["face_hashes"])
            n_edges = len(result["edge_hashes"])
            print(f"  ✅ {json_path.name}: {n_faces} faces, {n_edges} edges hashed")
            
        except Exception as e:
            print(f"  ❌ {json_path.name}: {e}")
    
    # Save master lookup
    master_path = os.path.join(output_dir, "_master_lookup.json")
    with open(master_path, "w") as f:
        json.dump(master_lookup, f, indent=2)
    print(f"\n📁 Master lookup saved: {master_path}")
    
    return master_lookup


# =============================================================================
# ASSEMBLY MAPPING
# =============================================================================

def map_assembly_entities(assembly_path: str, master_lookup: Dict[str, dict]) -> dict:
    """
    Map assembly constraint entities to their part-level geometry hashes.
    
    Uses TWO methods:
    1. PRIMARY: reference_key_string + context_key_string → part hash lookup
    2. FALLBACK: compute geometry hash from assembly entity data directly
    
    The fallback works because the assembly extractor captures the NATIVE
    geometry (not proxy geometry) which is identical to part geometry.
    """
    with open(assembly_path, "r", encoding="utf-8-sig") as f:
        asm_data = json.load(f)
    
    mapped_constraints = []
    total_entities = 0
    mapped_by_refkey = 0
    mapped_by_geohash = 0
    unmapped = 0
    
    constraints = asm_data.get("assembly_graph", {}).get("constraint_edges", [])
    if not constraints:
        constraints = asm_data.get("constraints", [])
    
    for constraint in constraints:
        mapped_constraint = {
            "constraint_name": constraint.get("constraint_name"),
            "constraint_type": constraint.get("constraint_type"),
            "node_one_id": constraint.get("node_one_id") or constraint.get("occurrence_one"),
            "node_two_id": constraint.get("node_two_id") or constraint.get("occurrence_two"),
            "offset_cm": constraint.get("offset_cm"),
            "angle_rad": constraint.get("angle_rad"),
            "solution": constraint.get("solution"),
            "entity_one_hash": None,
            "entity_two_hash": None,
            "entity_one_type": None,
            "entity_two_type": None,
            "mapping_method_one": None,
            "mapping_method_two": None
        }
        
        for side in ["one", "two"]:
            entity = constraint.get(f"entity_{side}")
            if not entity:
                continue
            
            total_entities += 1
            ctx_key = entity.get("context_key_string", "")
            ref_key = entity.get("reference_key_string", "")
            
            geo_hash = None
            method = None
            
            # Method 1: Reference key lookup
            if ctx_key in master_lookup and ref_key in master_lookup[ctx_key]:
                geo_hash = master_lookup[ctx_key][ref_key]
                method = "refkey_lookup"
                mapped_by_refkey += 1
            
            # Method 2: Compute from geometry
            if geo_hash is None:
                surface_type = entity.get("surface_type")
                curve_type = entity.get("curve_type")
                
                if surface_type:
                    geo_hash = compute_face_hash_from_assembly_entity(entity)
                    method = "geometry_hash_face"
                    mapped_by_geohash += 1
                elif curve_type:
                    geo_hash = compute_edge_hash_from_assembly_entity(entity)
                    method = "geometry_hash_edge"
                    mapped_by_geohash += 1
                else:
                    # WorkPlane / WorkAxis / WorkPoint
                    work_name = entity.get("work_feature_name")
                    if work_name:
                        sig = f"WORK:{work_name}|{round_vec(entity.get('face_normal_at_center'))}|{round_vec(entity.get('face_point_at_center'))}"
                        geo_hash = hash_string(sig)
                        method = "work_feature_hash"
                        mapped_by_geohash += 1
                    else:
                        unmapped += 1
            
            entity_type = entity.get("entity_type", "Unknown")
            if entity.get("surface_type"):
                entity_type = f"Face({entity['surface_type']})"
            elif entity.get("curve_type"):
                entity_type = f"Edge({entity['curve_type']})"
            elif entity.get("work_feature_name"):
                entity_type = f"Work({entity['work_feature_name']})"
            
            mapped_constraint[f"entity_{side}_hash"] = geo_hash
            mapped_constraint[f"entity_{side}_type"] = entity_type
            mapped_constraint[f"mapping_method_{side}"] = method
        
        mapped_constraints.append(mapped_constraint)
    
    return {
        "assembly_name": asm_data.get("assembly_metadata", {}).get("assembly_name", ""),
        "total_entities": total_entities,
        "mapped_by_refkey": mapped_by_refkey,
        "mapped_by_geohash": mapped_by_geohash,
        "unmapped": unmapped,
        "mapped_constraints": mapped_constraints
    }


def process_all_assemblies(assemblies_dir: str, master_lookup: Dict[str, dict], output_dir: str):
    """Process all assembly JSONs and map entities to part hashes."""
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = list(Path(assemblies_dir).glob("*.json"))
    print(f"\n{'='*60}")
    print(f"Mapping {len(json_files)} assemblies from: {assemblies_dir}")
    print(f"{'='*60}")
    
    summary = {"assemblies": [], "total_entities": 0, "total_mapped": 0, "total_unmapped": 0}
    
    for json_path in sorted(json_files):
        try:
            result = map_assembly_entities(str(json_path), master_lookup)
            
            out_path = os.path.join(output_dir, f"{json_path.stem}_mapped.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            
            total = result["total_entities"]
            mapped = result["mapped_by_refkey"] + result["mapped_by_geohash"]
            unmapped = result["unmapped"]
            
            summary["total_entities"] += total
            summary["total_mapped"] += mapped
            summary["total_unmapped"] += unmapped
            summary["assemblies"].append({
                "name": result["assembly_name"],
                "entities": total,
                "mapped": mapped,
                "unmapped": unmapped
            })
            
            pct = (mapped / total * 100) if total > 0 else 0
            print(f"  ✅ {json_path.name}: {mapped}/{total} entities mapped ({pct:.1f}%)")
            
        except Exception as e:
            print(f"  ❌ {json_path.name}: {e}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "_mapping_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    total = summary["total_entities"]
    mapped = summary["total_mapped"]
    pct = (mapped / total * 100) if total > 0 else 0
    print(f"\n📊 Overall: {mapped}/{total} entities mapped ({pct:.1f}%)")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_hashes(parts_dir: str):
    """Validate hash consistency: check for collisions and uniqueness."""
    print(f"\n{'='*60}")
    print(f"Validating hash consistency in: {parts_dir}")
    print(f"{'='*60}")
    
    all_face_hashes = defaultdict(list)
    all_edge_hashes = defaultdict(list)
    
    for json_path in sorted(Path(parts_dir).glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            
            part_name = data.get("part_metadata", {}).get("file_name", json_path.stem)
            
            for i, face in enumerate(data.get("faces", [])):
                h = compute_face_hash(face)
                all_face_hashes[h].append(f"{part_name}:face_{i}")
            
            for i, edge in enumerate(data.get("edges", [])):
                h = compute_edge_hash(edge)
                all_edge_hashes[h].append(f"{part_name}:edge_{i}")
            
        except Exception as e:
            print(f"  ⚠️  {json_path.name}: {e}")
    
    # Check for collisions (same hash, different geometry)
    face_collisions = {h: sources for h, sources in all_face_hashes.items() if len(sources) > 1}
    edge_collisions = {h: sources for h, sources in all_edge_hashes.items() if len(sources) > 1}
    
    print(f"\n  Total unique face hashes: {len(all_face_hashes)}")
    print(f"  Total unique edge hashes: {len(all_edge_hashes)}")
    print(f"  Face hash collisions (same hash, diff entity): {len(face_collisions)}")
    print(f"  Edge hash collisions (same hash, diff entity): {len(edge_collisions)}")
    
    if face_collisions:
        print(f"\n  ⚠️  Top 5 face collisions:")
        for h, sources in sorted(face_collisions.items(), key=lambda x: -len(x[1]))[:5]:
            print(f"    Hash {h[:16]}... → {len(sources)} entities: {sources[:3]}...")
    
    if not face_collisions and not edge_collisions:
        print(f"\n  ✅ No hash collisions detected. All hashes are unique!")
    
    # Note: Some collisions are EXPECTED for geometrically identical entities
    # (e.g., two planar faces with identical normal, area, position = same bolt hole pattern)
    print(f"\n  💡 Note: Collisions for truly identical geometry are expected (e.g., patterned features)")


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def generate_training_data(parts_dir: str, assemblies_dir: str, output_dir: str):
    """
    Generate training-ready data combining part hashes and assembly mappings.
    
    Output format per assembly constraint:
    {
        "constraint_type": "kMateConstraint",
        "entity_one": {
            "geometry_hash": "abc123...",
            "entity_type": "Face(kPlaneSurface)",
            "part_file": "PartA.ipt",
            "occurrence": "PartA:1"
        },
        "entity_two": {...},
        "offset_cm": 0.0,
        "angle_rad": null
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Process all parts
    master_lookup = process_all_parts(parts_dir, os.path.join(output_dir, "parts"))
    
    # Step 2: Map assemblies
    if assemblies_dir and os.path.exists(assemblies_dir):
        process_all_assemblies(assemblies_dir, master_lookup, os.path.join(output_dir, "assemblies"))
    
    print(f"\n✅ Training data generation complete!")
    print(f"   Parts hashed: {os.path.join(output_dir, 'parts')}")
    if assemblies_dir:
        print(f"   Assemblies mapped: {os.path.join(output_dir, 'assemblies')}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Geometry Hashing - Consistent entity hashing for CAD ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hash part files only:
  python geometry_hasher.py --parts-dir ./partsexport-new --output ./hashed

  # Full pipeline (parts + assemblies):
  python geometry_hasher.py --parts-dir ./partsexport-new --assemblies-dir ./assembliesexport-new --output ./hashed

  # Validate hash uniqueness:
  python geometry_hasher.py --validate --parts-dir ./partsexport-new
        """
    )
    
    parser.add_argument("--parts-dir", type=str, help="Directory containing part JSON files")
    parser.add_argument("--assemblies-dir", type=str, help="Directory containing assembly JSON files")
    parser.add_argument("--output", type=str, default="./hashed_output", help="Output directory")
    parser.add_argument("--validate", action="store_true", help="Run hash validation checks")
    parser.add_argument("--precision", type=int, default=6, help="Rounding precision (decimal places)")
    
    args = parser.parse_args()
    
    global PRECISION
    PRECISION = args.precision
    
    if args.validate and args.parts_dir:
        validate_hashes(args.parts_dir)
        return
    
    if args.parts_dir:
        generate_training_data(args.parts_dir, args.assemblies_dir, args.output)
    else:
        parser.print_help()
        print("\n⚠️  Please specify --parts-dir at minimum.")


if __name__ == "__main__":
    main()
