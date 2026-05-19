# PRANA CAD Automation Project

## 1. Project Overview

The PRANA project represents an end-to-end pipeline designed to bridge the gap between static mechanical CAD files and generative artificial intelligence. By extracting geometric boundary representation (B-Rep) data and assembly constraints from Autodesk Inventor, the system trains a Graph Neural Network (GNN) to understand how mechanical parts fit together. The trained AI can then take a simple Bill of Materials (BOM) and automatically generate the necessary mechanical constraints (Mates, Flushes, Inserts) to assemble the parts, effectively automating the CAD assembly process.

The pipeline is designed to be highly robust, handling the complexities of the Inventor COM API, resolving volatile reference keys through stable geometric hashing, and scaling to massive assemblies using graph-based machine learning.

### Core Architecture
The system operates in a cohesive, six-phase pipeline:
1.  **Phase 1: Data Extraction** (C# / Inventor API)
2.  **Phase 2: Hashing & Bridge** (Python / Geometric Fingerprinting)
3.  **Phase 3: Graph Construction** (Python / PyTorch Geometric)
4.  **Phase 4: AI Training** (Python / GraphSAGE GNN)
5.  **Phase 5: Generative Orchestration** (Python / ML Inference)
6.  **Phase 6: Assembly Execution** (C# / Inventor API)

![Atlas Pipeline Workflow](file:///Users/aryantembhurne/.gemini/antigravity/brain/4f3db23a-8de6-4f55-a117-7315813dfaf5/Pipeline.png)

---

## 2. Technical Workflow & Phase Breakdown

### Phase 1: Data Extraction (C#)
**Objective:** Translate proprietary `.ipt` (part) and `.iam` (assembly) files into structured, machine-readable JSON formats.

**Components:**
*   `PartExtractor.cs` (`19-p1.cs`): Opens individual part files and traverses their Boundary Representation (B-Rep) topology. It extracts highly detailed geometric features for every face (area, center, normal, surface type) and edge (length, midpoint, tangent, curve type). Crucially, it captures the internal topological graph (which faces connect to which edges).
*   `AssemblyExtractor.cs` (`19-a1.cs`): Parses assembly files to document the exact spatial transformations (rotation matrices, translation vectors) of every component. More importantly, it extracts the mechanical constraints (e.g., `kMateConstraint`, `kInsertConstraint`) and identifies the exact geometric entities (faces/edges) involved in each joint.

**Key Technical Solutions:**
*   **Safe Serialization:** The Inventor COM API frequently returns volatile proxy objects that crash standard JSON serializers. The extraction scripts implement strict `NullValueHandling` and `ReferenceLoopHandling` to prevent infinite loops and proxy leaks.
*   **Memory Management:** The scripts handle Inventor documents securely, opening them invisibly and closing them (`Close(false)`) without saving to prevent write-locks that disrupt batch processing.

### Phase 2: The Hashing Bridge (Python)
**Objective:** Solve the "Reference Key" mismatch problem between isolated parts and parts inside an assembly context.

**Components:**
*   `geometry_mapper.py`: In Autodesk Inventor, a face's internal ID (reference key) changes when it is placed inside an assembly (becoming a proxy). This script acts as the critical bridge. Instead of relying on broken COM keys, it uses a **Geometric Fingerprint**—matching entities based on `(surface_type, area, center_x, center_y, center_z, normal)`.
*   `hashing.py`: Once a geometric match is confirmed, this script assigns a stable, deterministic `SHA-256` hash to every entity. This guarantees that `Face A` in the part file has the exact same identifier as `Face A` in the assembly constraint data.

### Phase 3: Graph Construction (Python)
**Objective:** Convert the flat JSON geometries into PyTorch Geometric (PyG) graph structures for neural network processing.

**Components:**
*   `atlas_dataset.py`: This script builds a 21-dimensional feature vector for every node (face, edge, workplane).
    *   **Features:** Entity type (one-hot), area/length, global coordinates, normals, bounding box dimensions, surface type encodings, material hashes, and kinematic degrees of freedom.
    *   **Edges:** The internal B-Rep topology forms the structural edges of the graph. The assembly constraints (extracted in Phase 1) form the semantic target edges that the AI must learn to predict.
    *   **Negative Sampling:** The dataset generator heavily samples "No Constraint" edges (faces that don't connect) to teach the AI what *not* to assemble.

### Phase 4: AI Training (Python)
**Objective:** Train a model to understand mechanical compatibility.

**Components:**
*   `atlas_model.py`: Implements **AtlasGNN**, a 3-layer GraphSAGE (Graph Sample and Aggregate) neural network. GraphSAGE is highly effective here because it learns structural embeddings based on a node's local neighborhood (e.g., a cylindrical face surrounded by planar faces). The link-prediction head concatenates the embeddings of two faces, computes their difference and product, and predicts the likelihood of a constraint.
*   `atlas_train.py`: Handles the training loop. It utilizes Class-Weighted Cross-Entropy Loss to handle the extreme imbalance between the massive number of non-interacting faces ("No Constraint") and the rare, actual joints.

![Confusion Matrix](file:///Users/aryantembhurne/.gemini/antigravity/brain/4f3db23a-8de6-4f55-a117-7315813dfaf5/confusion_matrix.png)

### Phase 5: Generative Orchestration (Python)
**Objective:** The core inference engine. Given a loose list of parts (BOM), figure out how to put them together.

**Components:**
*   `atlas_inference.py`: This script simulates a virtual workbench.
    1.  **Instantiation:** It loads the requested parts from the BOM into a massive, disconnected graph.
    2.  **Exhaustive Pairing:** It generates every possible cross-part face combination.
    3.  **Inference:** The GNN predicts the constraint probability for millions of pairs. To prevent GPU Out-of-Memory (OOM) crashes, this is handled via **Chunked Batch Processing** (e.g., 50,000 pairs at a time).
    4.  **Kinematic Sequencing (Heuristics):** The highest confidence predictions are filtered logically. For example, it prioritizes `Insert` constraints for cylindrical alignments (Primary Anchors) and then applies `Flush/Mate` constraints (Secondary Locks) to secure the remaining rotational degrees of freedom, producing a strict mechanical recipe (`predicted_assembly.json`).

### Phase 6: Assembly Execution (C#)
**Objective:** Translate the AI's predicted instructions back into a physical 3D CAD model.

**Components:**
*   `reconstruction.cs`: This C# script connects to a live instance of Autodesk Inventor. It reads the AI-generated JSON recipe and programmatically drives the CAD engine.
    *   It imports the specified `.ipt` files into a new `.iam` assembly document.
    *   It locates the exact faces/edges using the stable reference keys or fallback geometric signatures.
    *   It executes the API calls (`AddMateConstraint`, `AddFlushConstraint`, `AddInsertConstraint2`) to physically snap the 3D models together according to the AI's blueprint.

---

## 4. Notable Engineering Milestones & Solutions

Throughout the development of ATLAS, several major technical hurdles were overcome:

1.  **The COM Proxy Memory Leak:** Initial extraction scripts crashed halfway through large datasets because Inventor COM proxies cannot be serialized natively by `Newtonsoft.Json`. This was solved by mapping deep API objects to primitive data classes (`EntityData`) before serialization.
2.  **The Reference Key Volatility Problem:** Discovered that Inventor dynamically alters topological reference keys when a part enters an assembly. The implementation of `geometry_mapper.py` (Phase 2) bypassed this flaw entirely, resulting in a nearly 100% geometric match rate.
3.  **Neural Network Feature Collapse:** Early models struggled to differentiate between identical parts placed in different orientations. The feature vector in `atlas_dataset.py` was expanded to 21 dimensions to include local coordinate bounding boxes and mass, allowing the GNN to contextualize the part's overall shape and scale.
4.  **OOM Inference Crashes:** Running inference on assemblies with 10+ parts resulted in millions of combinatorial edge predictions, crashing 8GB Apple Silicon/MPS architectures. The orchestration engine was refactored to use memory-safe, chunked tensor generation.
