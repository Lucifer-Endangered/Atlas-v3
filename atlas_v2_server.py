"""
ATLAS V2 — Python Inference Server
====================================
HTTP REST API for the Inventor C# plugin.
Receives part geometries, runs ML inference + optional LLM refinement,
returns predicted constraints.

USAGE:
    python3 atlas_v2_server.py --model-dir ./v2_checkpoints --port 5050
    python3 atlas_v2_server.py --model-dir ./v2_checkpoints --port 5050 --llm-api-key sk-...
"""

import json, os, sys, argparse, itertools, time
import numpy as np
import torch
import torch.nn.functional as F
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import defaultdict

from atlas_v2_train import AtlasV2Model, CONSTRAINT_NAMES
from atlas_v2_data import (
    ENTITY_DIM, SURFACE_TYPE_MAP, CURVE_TYPE_MAP,
    safe_float, safe_vec3, log_s,
    part_face_to_features, part_edge_to_features,
)

# Global model state
MODEL = None
DEVICE = None
MEAN = None
STD = None
LLM_API_KEY = None
CONFIDENCE = 0.6

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_model(model_dir):
    global MODEL, DEVICE, MEAN, STD
    DEVICE = get_device()
    ckpt = torch.load(os.path.join(model_dir, "atlas_v2_best.pt"),
                      map_location=DEVICE, weights_only=False)
    MODEL = AtlasV2Model(
        entity_dim=ckpt["entity_dim"], hidden=ckpt["hidden"],
        num_classes=ckpt["num_classes"]
    ).to(DEVICE)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.eval()
    MEAN = np.load(os.path.join(model_dir, "norm_mean.npy"))
    STD = np.load(os.path.join(model_dir, "norm_std.npy"))
    print(f"🧠 Model loaded on {DEVICE} (epoch {ckpt['epoch']}, F1={ckpt['val_f1']:.3f})")


def extract_entities_from_inline(part_geom):
    """Extract entities from inline geometry sent by the C# plugin."""
    entities = []
    for face in part_geom.get("faces", []):
        entities.append({
            "features": part_face_to_features(face),
            "type": "face",
            "surface_type": face.get("surface_type", ""),
            "area_cm2": face.get("area_cm2", 0),
            "reference_key_string": face.get("reference_key_string", ""),
        })
    for edge in part_geom.get("edges", []):
        entities.append({
            "features": part_edge_to_features(edge),
            "type": "edge",
            "curve_type": edge.get("curve_type", ""),
            "reference_key_string": edge.get("reference_key_string", ""),
        })
    return entities


def predict(parts_data, threshold):
    """Run inference on parts data."""
    # Build part instances
    instances = []
    for part in parts_data:
        count = part.get("count", 1)
        stem = part.get("stem", part.get("file", "unknown"))
        geom = part.get("geometry", {})
        if not geom:
            continue
        entities = extract_entities_from_inline(geom)
        for i in range(count):
            instances.append({
                "instance_id": f"{stem}:{i+1}",
                "file": part.get("file", ""),
                "entities": entities,
            })

    # Generate cross-part pairs
    predictions = []
    for pi_a, pi_b in itertools.combinations(range(len(instances)), 2):
        pa, pb = instances[pi_a], instances[pi_b]
        
        faces_a = [e for e in pa["entities"] if e["type"] == "face"]
        faces_b = [e for e in pb["entities"] if e["type"] == "face"]
        circ_a = [e for e in pa["entities"] if e["type"] == "edge" and e.get("curve_type") == "kCircleCurve"]
        circ_b = [e for e in pb["entities"] if e["type"] == "edge" and e.get("curve_type") == "kCircleCurve"]

        all_pairs = [(fa, fb) for fa in faces_a for fb in faces_b]
        all_pairs += [(ea, eb) for ea in circ_a for eb in circ_b]

        if not all_pairs:
            continue

        feats = []
        for ea, eb in all_pairs:
            c = np.array(ea["features"] + eb["features"], dtype=np.float32)
            feats.append((c - MEAN) / STD)

        X = torch.tensor(np.array(feats), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(MODEL(X), dim=-1)

        for k, (ea, eb) in enumerate(all_pairs):
            cls = probs[k].argmax().item()
            conf = probs[k].max().item()
            if cls > 0 and conf >= threshold:
                predictions.append({
                    "constraint_type": f"k{CONSTRAINT_NAMES[cls]}Constraint",
                    "confidence": round(conf, 4),
                    "occurrence_one": pa["instance_id"],
                    "occurrence_two": pb["instance_id"],
                    "entity_one_surface": ea.get("surface_type", ea.get("curve_type", "")),
                    "entity_two_surface": eb.get("surface_type", eb.get("curve_type", "")),
                    "entity_one_ref_key": ea.get("reference_key_string", ""),
                    "entity_two_ref_key": eb.get("reference_key_string", ""),
                })

    # Deduplicate + kinematic ordering
    predictions.sort(key=lambda x: -x["confidence"])
    seen = defaultdict(lambda: defaultdict(int))
    max_per = {"Insert": 1, "Mate": 2, "Flush": 2, "Angle": 1, "Tangent": 1}
    filtered = []
    for p in predictions:
        pair = tuple(sorted([p["occurrence_one"], p["occurrence_two"]]))
        cn = CONSTRAINT_NAMES[[f"k{n}Constraint" for n in CONSTRAINT_NAMES].index(p["constraint_type"])]
        if seen[pair][cn] < max_per.get(cn, 1):
            filtered.append(p)
            seen[pair][cn] += 1

    priority = {"kInsertConstraint": 0, "kMateConstraint": 1,
                "kFlushConstraint": 1, "kAngleConstraint": 2, "kTangentConstraint": 3}
    filtered.sort(key=lambda x: (priority.get(x["constraint_type"], 9), -x["confidence"]))

    # LLM refinement
    if LLM_API_KEY:
        filtered = llm_refine(filtered, parts_data)

    return filtered


def llm_refine(constraints, parts):
    """Use LLM to validate and refine predictions."""
    try:
        import requests
        parts_desc = ", ".join(p.get("file", p.get("stem", "?")) for p in parts)
        cdesc = json.dumps([{
            "type": c["constraint_type"], "conf": c["confidence"],
            "src": c["occurrence_one"], "dst": c["occurrence_two"],
            "src_surf": c["entity_one_surface"], "dst_surf": c["entity_two_surface"],
        } for c in constraints[:20]], indent=2)

        prompt = f"""You are a CAD mechanical assembly expert. Given these parts: {parts_desc}

AI-predicted constraints (ordered by kinematic priority):
{cdesc}

Validate each constraint:
1. Is the surface pairing mechanically valid? (e.g., Insert needs circular edges, Mate/Flush need planar faces)
2. Are there any physically impossible constraints?
3. Should any constraint types be changed?

Respond with a JSON object:
{{"remove": [indices to remove], "change": [{{"index": i, "new_type": "kXConstraint"}}]}}
Or {{"remove": [], "change": []}} if all OK."""

        resp = requests.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": 300}, timeout=25)
        
        if resp.status_code == 200:
            import re
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r'\{[^}]*"remove"[^}]*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                removes = set(result.get("remove", []))
                changes = {c["index"]: c["new_type"] for c in result.get("change", [])}
                
                refined = []
                for i, c in enumerate(constraints):
                    if i in removes:
                        continue
                    if i in changes:
                        c["constraint_type"] = changes[i]
                        c["llm_modified"] = True
                    refined.append(c)
                
                print(f"🤖 LLM: removed {len(removes)}, changed {len(changes)}")
                return refined
    except Exception as e:
        print(f"[LLM] {e}")
    return constraints


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            
            parts = body.get("parts", [])
            threshold = body.get("confidence_threshold", CONFIDENCE)
            
            t0 = time.time()
            constraints = predict(parts, threshold)
            elapsed = time.time() - t0
            
            result = {
                "constraints": constraints,
                "total": len(constraints),
                "inference_time_s": round(elapsed, 2),
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            print(f"  → {len(constraints)} constraints in {elapsed:.1f}s")
        
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"service": "ATLAS V2", "status": "ready"}).encode())

    def log_message(self, format, *args):
        pass  # Suppress default logs


def main():
    global LLM_API_KEY, CONFIDENCE
    p = argparse.ArgumentParser(description="ATLAS V2 — Inference Server")
    p.add_argument("--model-dir", default="./v2_checkpoints")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--confidence", type=float, default=0.6)
    args = p.parse_args()

    LLM_API_KEY = args.llm_api_key
    CONFIDENCE = args.confidence

    load_model(args.model_dir)
    
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\n🚀 ATLAS V2 Server running on http://localhost:{args.port}")
    print(f"   POST /predict → Run inference")
    print(f"   LLM refinement: {'Enabled' if LLM_API_KEY else 'Disabled'}")
    print(f"   Confidence threshold: {CONFIDENCE}")
    print(f"   Press Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped")


if __name__ == "__main__":
    main()
