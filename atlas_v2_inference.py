"""
ATLAS V2 — Inference Engine
=============================
Given a BOM (list of part JSONs), predicts assembly constraints.
Optionally uses an LLM API to refine and validate predictions.
"""

import json, os, itertools, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from atlas_v2_train import AtlasV2Model, CONSTRAINT_NAMES
from atlas_v2_data import part_face_to_features, part_edge_to_features, ENTITY_DIM

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_part_entities(part_json_path: str) -> list:
    """Load part JSON and extract entity features."""
    with open(part_json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    entities = []
    meta = data.get("part_metadata", {})
    for i, face in enumerate(data.get("faces", [])):
        entities.append({
            "features": part_face_to_features(face),
            "type": "face", "index": i,
            "surface_type": face.get("surface_type", ""),
            "area_cm2": face.get("area_cm2", 0),
            "normal": face.get("normal"), "center": face.get("center"),
            "radius_cm": face.get("radius_cm"),
            "reference_key_string": face.get("reference_key_string", ""),
        })
    for i, edge in enumerate(data.get("edges", [])):
        entities.append({
            "features": part_edge_to_features(edge),
            "type": "edge", "index": i,
            "curve_type": edge.get("curve_type", ""),
            "length_cm": edge.get("length_cm", 0),
            "midpoint": edge.get("midpoint"),
            "reference_key_string": edge.get("reference_key_string", ""),
        })
    return entities, meta

def predict_assembly(args):
    device = get_device()
    print(f"🖥️  Device: {device}")

    # Load BOM
    with open(args.bom, "r") as f:
        bom = json.load(f)
    parts_list = bom.get("parts", bom if isinstance(bom, list) else [])
    print(f"📋 BOM: {len(parts_list)} part types")

    # Load model
    ckpt = torch.load(os.path.join(args.model_dir, "atlas_v2_best.pt"),
                      map_location=device, weights_only=False)
    model = AtlasV2Model(
        entity_dim=ckpt["entity_dim"], hidden=ckpt["hidden"],
        num_classes=ckpt["num_classes"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"🧠 Model loaded (epoch {ckpt['epoch']}, F1={ckpt['val_f1']:.3f})")

    # Load normalization
    mean = np.load(os.path.join(args.model_dir, "norm_mean.npy"))
    std = np.load(os.path.join(args.model_dir, "norm_std.npy"))

    # Load all parts
    part_instances = []
    for entry in parts_list:
        count = entry.get("count", 1)
        part_file = entry.get("file", "")
        part_json = entry.get("json", "")
        path = os.path.join(args.parts_dir, part_json or part_file)
        if not os.path.exists(path):
            stem = Path(part_file).stem
            path = os.path.join(args.parts_dir, f"{stem}.json")
        if not os.path.exists(path):
            print(f"  ⚠️  Not found: {part_file}")
            continue
        entities, meta = load_part_entities(path)
        for inst in range(count):
            inst_id = f"{Path(part_file).stem}:{inst+1}"
            part_instances.append({
                "instance_id": inst_id, "file": part_file,
                "entities": entities, "meta": meta,
            })
            print(f"  📦 {inst_id}: {len(entities)} entities")

    # Generate cross-part pairs: face-face AND circle_edge-circle_edge
    print(f"\n🔮 Generating predictions...")
    predictions = []
    threshold = args.confidence

    for (pi_a, pi_b) in itertools.combinations(range(len(part_instances)), 2):
        pa = part_instances[pi_a]
        pb = part_instances[pi_b]

        # Face-face pairs (Mate, Flush, Angle)
        faces_a = [e for e in pa["entities"] if e["type"] == "face"]
        faces_b = [e for e in pb["entities"] if e["type"] == "face"]
        # Edge-edge pairs for Insert (circle edges only)
        circ_a = [e for e in pa["entities"] if e["type"] == "edge" and e.get("curve_type") == "kCircleCurve"]
        circ_b = [e for e in pb["entities"] if e["type"] == "edge" and e.get("curve_type") == "kCircleCurve"]

        all_pairs = []
        for fa in faces_a:
            for fb in faces_b:
                all_pairs.append((fa, fb))
        for ea in circ_a:
            for eb in circ_b:
                all_pairs.append((ea, eb))

        if not all_pairs:
            continue

        # Build pair matrix
        pairs_feat = []
        for (ea, eb) in all_pairs:
            combined = np.array(ea["features"] + eb["features"], dtype=np.float32)
            combined = (combined - mean) / std
            pairs_feat.append(combined)

        # Batch inference
        X = torch.tensor(np.array(pairs_feat), dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X)
            probs = F.softmax(logits, dim=-1)

        for k in range(len(all_pairs)):
            pred_class = probs[k].argmax().item()
            conf = probs[k].max().item()
            if pred_class > 0 and conf >= threshold:
                ea, eb = all_pairs[k]
                predictions.append({
                    "constraint_type": f"k{CONSTRAINT_NAMES[pred_class]}Constraint",
                    "confidence": round(conf, 4),
                    "src_instance": pa["instance_id"],
                    "dst_instance": pb["instance_id"],
                    "src_surface_type": ea.get("surface_type", ea.get("curve_type", "")),
                    "dst_surface_type": eb.get("surface_type", eb.get("curve_type", "")),
                    "src_ref_key": ea.get("reference_key_string", ""),
                    "dst_ref_key": eb.get("reference_key_string", ""),
                    "src_file": pa["file"],
                    "dst_file": pb["file"],
                })

    # Sort by confidence, deduplicate
    predictions.sort(key=lambda x: -x["confidence"])
    seen = defaultdict(lambda: defaultdict(int))
    max_per = {"Insert": 1, "Mate": 2, "Flush": 2, "Angle": 1, "Tangent": 1}
    filtered = []
    for p in predictions:
        pair = tuple(sorted([p["src_instance"], p["dst_instance"]]))
        cname = CONSTRAINT_NAMES[list(CONSTRAINT_NAMES).index(
            p["constraint_type"].replace("k","").replace("Constraint",""))]
        if seen[pair][cname] < max_per.get(cname, 1):
            filtered.append(p)
            seen[pair][cname] += 1

    # Kinematic ordering: Insert → Mate/Flush → Angle
    priority = {"kInsertConstraint": 0, "kMateConstraint": 1,
                "kFlushConstraint": 1, "kAngleConstraint": 2, "kTangentConstraint": 3}
    filtered.sort(key=lambda x: (priority.get(x["constraint_type"], 9), -x["confidence"]))

    # Optional LLM refinement
    if args.llm_api_key:
        filtered = refine_with_llm(filtered, parts_list, args.llm_api_key)

    # Output
    recipe = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": "ATLAS V2 — Direct Geometry Matching",
        "bom": parts_list,
        "total_constraints": len(filtered),
        "constraints": [{
            "sequence_order": i+1,
            "constraint_type": c["constraint_type"],
            "confidence": c["confidence"],
            "occurrence_one": c["src_instance"],
            "occurrence_two": c["dst_instance"],
            "entity_one_surface": c["src_surface_type"],
            "entity_two_surface": c["dst_surface_type"],
            "entity_one_ref_key": c["src_ref_key"],
            "entity_two_ref_key": c["dst_ref_key"],
        } for i, c in enumerate(filtered)],
    }

    with open(args.output, "w") as f:
        json.dump(recipe, f, indent=2)

    print(f"\n✅ Predicted {len(filtered)} constraints → {args.output}")
    types = defaultdict(int)
    for c in filtered: types[c["constraint_type"]] += 1
    for t, n in sorted(types.items()): print(f"   {t}: {n}")


def refine_with_llm(constraints, bom, api_key):
    """Use LLM to validate constraint predictions."""
    try:
        import requests
        parts_desc = ", ".join(p.get("file", "") for p in bom)
        cdesc = json.dumps([{
            "type": c["constraint_type"], "conf": c["confidence"],
            "src": c["src_instance"], "dst": c["dst_instance"],
            "src_surf": c["src_surface_type"], "dst_surf": c["dst_surface_type"],
        } for c in constraints[:15]], indent=2)

        prompt = f"""You are a CAD assembly expert. Parts: {parts_desc}
AI-predicted constraints: {cdesc}
Validate: are any mechanically impossible? Should types change?
Respond with JSON array of 0-indexed constraint indices to REMOVE, or [] if all ok."""

        resp = requests.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": 200}, timeout=20)
        if resp.status_code == 200:
            import re
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r'\[[\d,\s]*\]', content)
            if match:
                remove = json.loads(match.group())
                if remove:
                    print(f"🤖 LLM removed {len(remove)} constraints")
                    constraints = [c for i, c in enumerate(constraints) if i not in remove]
                else:
                    print("🤖 LLM validated all ✅")
    except Exception as e:
        print(f"  [LLM] {e}")
    return constraints


def main():
    p = argparse.ArgumentParser(description="ATLAS V2 — Predict assembly from BOM")
    p.add_argument("--bom", required=True, help="BOM JSON file")
    p.add_argument("--parts-dir", default="./Output Parts", help="Part JSONs directory")
    p.add_argument("--model-dir", default="./v2_checkpoints", help="Model checkpoint dir")
    p.add_argument("--output", default="./predicted_assembly.json")
    p.add_argument("--confidence", type=float, default=0.6)
    p.add_argument("--llm-api-key", default=None)
    predict_assembly(p.parse_args())

if __name__ == "__main__":
    main()
