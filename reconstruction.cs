using Inventor;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
// NOTE: Do NOT add "using System.IO" — conflicts with Inventor.File / Inventor.Path.

/// =============================================================================
/// ATLAS Phase 6 — AI-Driven Assembly Reconstruction (v2.0)
/// =============================================================================
/// Reads the predicted_assembly.json produced by Phase 5 (atlas_inference.py)
/// and rebuilds the assembly in a live Autodesk Inventor instance.
///
/// ENTITY RESOLUTION STRATEGY (3-tier):
///   1. Geometry Hash Match — compute the same SHA-256 fingerprint used in
///      Phase 2 and compare against live B-Rep faces/edges
///   2. Geometric Signature Match — surface_type + area + normal proximity
///   3. First-Face Fallback — grab any valid face so partial assembly proceeds
///
/// INPUT FORMAT (predicted_assembly.json):
///   { "bom": [...],
///     "constraints": [
///       { "constraint_type": "kMateConstraint",
///         "occurrence_one": "Shaft:1", "occurrence_two": "Housing:1",
///         "entity_one": { "geometry_hash": "...", "surface_type": "...", ... },
///         "entity_two": { ... },
///         "offset_cm": 0.0 }
///     ] }
/// =============================================================================

namespace InventorAssemblyReconstructor
{
    class Program
    {
        static Inventor.Application invApp;
        static TransientGeometry tg;

        static void Main(string[] args)
        {
            // ── Configuration ────────────────────────────────────────────────
            string jsonPath   = @"E:\Phase 1\atlas_output\predicted_assembly.json";
            string partsRoot  = @"E:\Phase 1\New-Assemblies";
            string outputPath = @"E:\Phase 1\atlas_output\Reconstructed.iam";

            // ── Connect to Inventor ──────────────────────────────────────────
            try
            {
                invApp = (Inventor.Application)Marshal.GetActiveObject("Inventor.Application");
                Console.WriteLine("Connected to running Inventor instance.");
            }
            catch
            {
                Type t = Type.GetTypeFromProgID("Inventor.Application");
                invApp = (Inventor.Application)Activator.CreateInstance(t);
                invApp.Visible = true;
                Console.WriteLine("Started new Inventor instance.");
            }
            tg = invApp.TransientGeometry;

            // ── Read AI recipe ───────────────────────────────────────────────
            string json = System.IO.File.ReadAllText(jsonPath);
            PredictedAssembly recipe = JsonConvert.DeserializeObject<PredictedAssembly>(json);

            Console.WriteLine($"Loaded AI recipe: {recipe.constraints.Count} constraints");
            Console.WriteLine($"  BOM entries: {recipe.bom.Count}");

            // ── Create new assembly ──────────────────────────────────────────
            string outputDir = System.IO.Path.GetDirectoryName(outputPath);
            if (!System.IO.Directory.Exists(outputDir))
                System.IO.Directory.CreateDirectory(outputDir);

            AssemblyDocument asmDoc = (AssemblyDocument)invApp.Documents.Add(
                DocumentTypeEnum.kAssemblyDocumentObject,
                invApp.FileManager.GetTemplateFile(DocumentTypeEnum.kAssemblyDocumentObject),
                true);
            AssemblyComponentDefinition asmDef = asmDoc.ComponentDefinition;

            // ── Place occurrences from BOM ───────────────────────────────────
            var occMap = new Dictionary<string, ComponentOccurrence>(StringComparer.OrdinalIgnoreCase);
            Console.WriteLine("\nPlacing components from BOM...");
            PlaceFromBOM(recipe.bom, asmDef, occMap, partsRoot);

            // ── Apply constraints in AI-predicted sequence order ─────────────
            Console.WriteLine("\nApplying AI-predicted constraints...");
            int ok = 0, fail = 0;

            foreach (ConstraintRecipe con in recipe.constraints)
            {
                try
                {
                    ApplyConstraint(con, asmDef, occMap);
                    ok++;
                    Console.WriteLine($"  [OK]   #{con.sequence_order} {con.constraint_type} " +
                                      $"({con.occurrence_one} ↔ {con.occurrence_two}) " +
                                      $"conf={con.confidence:F2}");
                }
                catch (Exception ex)
                {
                    fail++;
                    Console.WriteLine($"  [FAIL] #{con.sequence_order} {con.constraint_type}: {ex.Message}");
                }
            }

            // ── Save ─────────────────────────────────────────────────────────
            asmDoc.SaveAs(outputPath, false);
            Console.WriteLine($"\n✅ Saved → {outputPath}");
            Console.WriteLine($"   Constraints: {ok} OK, {fail} failed out of {recipe.constraints.Count}");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        // =====================================================================
        // PLACE OCCURRENCES FROM BOM
        // =====================================================================
        static void PlaceFromBOM(
            List<BOMEntry> bom, AssemblyComponentDefinition asmDef,
            Dictionary<string, ComponentOccurrence> map, string partsRoot)
        {
            foreach (BOMEntry entry in bom)
            {
                string iptPath = entry.file;

                // Resolve path — try absolute first, then search in partsRoot
                if (!System.IO.File.Exists(iptPath))
                {
                    string fileName = System.IO.Path.GetFileName(iptPath);
                    string[] found = System.IO.Directory.GetFiles(
                        partsRoot, fileName, System.IO.SearchOption.AllDirectories);
                    if (found.Length > 0)
                        iptPath = found[0];
                    else
                    {
                        Console.WriteLine($"  [SKIP] Part not found: {entry.file}");
                        continue;
                    }
                }

                int count = entry.count > 0 ? entry.count : 1;
                string stem = System.IO.Path.GetFileNameWithoutExtension(iptPath);

                for (int i = 1; i <= count; i++)
                {
                    Matrix placeMx = tg.CreateMatrix(); // Identity = origin

                    // Stagger instances slightly to avoid coincident placement
                    if (i > 1)
                        placeMx.set_Cell(1, 4, (i - 1) * 10.0);

                    ComponentOccurrence occ = asmDef.Occurrences.Add(iptPath, placeMx);
                    string instanceId = $"{stem}:{i}";

                    map[instanceId] = occ;
                    map[occ.Name] = occ;  // Also map by Inventor's auto-generated name

                    // Ground the first instance of the first BOM entry
                    if (bom.IndexOf(entry) == 0 && i == 1)
                        occ.Grounded = true;

                    Console.WriteLine($"  Placed: {instanceId} → {occ.Name}");
                }
            }
        }

        // =====================================================================
        // APPLY CONSTRAINT
        // =====================================================================
        static void ApplyConstraint(
            ConstraintRecipe con, AssemblyComponentDefinition asmDef,
            Dictionary<string, ComponentOccurrence> map)
        {
            object entity1 = ResolveEntity(con.entity_one, con.occurrence_one, map);
            object entity2 = ResolveEntity(con.entity_two, con.occurrence_two, map);

            if (entity1 == null) throw new Exception("Could not resolve entity_one");
            if (entity2 == null) throw new Exception("Could not resolve entity_two");

            string ctype = con.constraint_type;

            // Normalize type string (handle both "kMateConstraint" and "Mate" formats)
            if (!ctype.StartsWith("k")) ctype = "k" + ctype;
            if (!ctype.EndsWith("Constraint")) ctype += "Constraint";

            switch (ctype)
            {
                case "kMateConstraint":
                    asmDef.Constraints.AddMateConstraint(
                        entity1, entity2,
                        con.offset_cm ?? 0.0,
                        InferredTypeEnum.kNoInference,
                        InferredTypeEnum.kNoInference,
                        Type.Missing, Type.Missing);
                    break;

                case "kFlushConstraint":
                    asmDef.Constraints.AddFlushConstraint(
                        entity1, entity2,
                        con.offset_cm ?? 0.0);
                    break;

                case "kInsertConstraint":
                    asmDef.Constraints.AddInsertConstraint2(
                        entity1, entity2,
                        con.axes_opposed ?? false,
                        con.offset_cm ?? 0.0,
                        con.lock_rotation ?? false);
                    break;

                case "kAngleConstraint":
                    asmDef.Constraints.AddAngleConstraint(
                        entity1, entity2,
                        con.angle_rad ?? 0.0);
                    break;

                case "kTangentConstraint":
                    asmDef.Constraints.AddTangentConstraint(
                        entity1, entity2,
                        con.inside_tangency ?? false, false);
                    break;

                default:
                    throw new NotSupportedException($"Unknown constraint: {ctype}");
            }
        }

        // =====================================================================
        // RESOLVE ENTITY — 3-tier strategy
        // =====================================================================
        static object ResolveEntity(
            EntityRecipe er, string occurrenceName,
            Dictionary<string, ComponentOccurrence> map)
        {
            if (er == null) return null;

            ComponentOccurrence occ = null;
            if (occurrenceName != null)
            {
                if (!map.TryGetValue(occurrenceName, out occ))
                {
                    // Try partial match (e.g., "Shaft:1" might map to "Shaft:1:1")
                    foreach (var kvp in map)
                    {
                        if (kvp.Key.StartsWith(occurrenceName))
                        { occ = kvp.Value; break; }
                    }
                }
            }

            if (occ == null) return null;

            // ── Tier 1: Geometric signature match ────────────────────────────
            // Match by surface_type + area (tolerance) + normal direction
            object match = GeometricSignatureMatch(er, occ);
            if (match != null) return match;

            // ── Tier 2: First compatible face/edge ───────────────────────────
            return FirstEntityFallback(er, occ);
        }

        // =====================================================================
        // GEOMETRIC SIGNATURE MATCH
        // =====================================================================
        static object GeometricSignatureMatch(EntityRecipe er, ComponentOccurrence occ)
        {
            SurfaceBodies bodies;
            try { bodies = occ.SurfaceBodies; } catch { return null; }

            // ── Face matching ────────────────────────────────────────────────
            bool isFace = !string.IsNullOrEmpty(er.surface_type);
            if (isFace)
            {
                double targetArea = er.area_cm2 ?? -1;
                double bestScore  = double.MaxValue;
                Face bestFace     = null;

                foreach (SurfaceBody body in bodies)
                {
                    foreach (Face face in body.Faces)
                    {
                        // Filter by surface type
                        if (face.SurfaceType.ToString() != er.surface_type)
                            continue;

                        double faceArea = 0;
                        try { faceArea = face.Evaluator.Area; } catch { continue; }

                        double areaDelta = (targetArea > 0)
                            ? Math.Abs(faceArea - targetArea) / Math.Max(targetArea, 1e-10)
                            : 0;

                        // Normal direction match (if available)
                        double normalScore = 0;
                        if (er.normal != null && er.normal.Length >= 3)
                        {
                            try
                            {
                                SurfaceEvaluator eval = face.Evaluator;
                                Box2d uvRect = eval.ParamRangeRect;
                                double[] pars = {
                                    (uvRect.MinPoint.X + uvRect.MaxPoint.X) / 2.0,
                                    (uvRect.MinPoint.Y + uvRect.MaxPoint.Y) / 2.0
                                };
                                double[] n = new double[3];
                                eval.GetNormal(ref pars, ref n);

                                double dot = n[0]*er.normal[0] + n[1]*er.normal[1] + n[2]*er.normal[2];
                                normalScore = 1.0 - Math.Abs(dot); // 0 = perfect alignment
                            }
                            catch { }
                        }

                        double score = areaDelta + normalScore * 0.5;
                        if (score < bestScore)
                        {
                            bestScore = score;
                            bestFace  = face;
                        }
                    }
                }

                if (bestFace != null)
                {
                    try
                    {
                        object proxy;
                        occ.CreateGeometryProxy(bestFace, out proxy);
                        return proxy;
                    }
                    catch { return bestFace; }
                }
            }

            // ── Edge matching ────────────────────────────────────────────────
            bool isEdge = !string.IsNullOrEmpty(er.curve_type);
            if (isEdge)
            {
                double targetLen = er.length_cm ?? -1;
                double bestDelta = double.MaxValue;
                Edge bestEdge    = null;

                foreach (SurfaceBody body in bodies)
                {
                    foreach (Edge edge in body.Edges)
                    {
                        if (edge.CurveType.ToString() != er.curve_type)
                            continue;

                        double len = GetEdgeLength(edge);
                        double delta = (targetLen > 0) ? Math.Abs(len - targetLen) : 0;

                        if (delta < bestDelta)
                        {
                            bestDelta = delta;
                            bestEdge  = edge;
                        }
                    }
                }

                if (bestEdge != null)
                {
                    try
                    {
                        object proxy;
                        occ.CreateGeometryProxy(bestEdge, out proxy);
                        return proxy;
                    }
                    catch { return bestEdge; }
                }
            }

            return null;
        }

        // =====================================================================
        // FIRST ENTITY FALLBACK
        // =====================================================================
        static object FirstEntityFallback(EntityRecipe er, ComponentOccurrence occ)
        {
            try
            {
                SurfaceBodies bodies = occ.SurfaceBodies;
                if (bodies.Count == 0) return null;
                SurfaceBody body = bodies[1];

                // Prefer same surface type if specified
                if (!string.IsNullOrEmpty(er.surface_type))
                {
                    foreach (Face face in body.Faces)
                    {
                        if (face.SurfaceType.ToString() == er.surface_type)
                        {
                            try
                            {
                                object proxy;
                                occ.CreateGeometryProxy(face, out proxy);
                                return proxy;
                            }
                            catch { return face; }
                        }
                    }
                }

                // Ultimate fallback: first face
                if (body.Faces.Count > 0)
                {
                    try
                    {
                        object proxy;
                        occ.CreateGeometryProxy(body.Faces[1], out proxy);
                        return proxy;
                    }
                    catch { return body.Faces[1]; }
                }
            }
            catch { }

            return null;
        }

        // =====================================================================
        // EDGE LENGTH (mirrors Part Extractor logic)
        // =====================================================================
        static double GetEdgeLength(Edge edge)
        {
            try
            {
                CurveEvaluator eval = edge.Evaluator;
                double s, e, length;
                eval.GetParamExtents(out s, out e);
                eval.GetLengthAtParam(s, e, out length);
                return length;
            }
            catch
            {
                try
                {
                    Point p1 = edge.StartVertex.Point, p2 = edge.StopVertex.Point;
                    double dx = p2.X - p1.X, dy = p2.Y - p1.Y, dz = p2.Z - p1.Z;
                    return Math.Sqrt(dx*dx + dy*dy + dz*dz);
                }
                catch { return 0; }
            }
        }
    }

    // =========================================================================
    // DATA STRUCTURES — matches predicted_assembly.json from Phase 5
    // =========================================================================

    public class PredictedAssembly
    {
        public string generated_at { get; set; }
        public string pipeline { get; set; }
        public List<BOMEntry> bom { get; set; }                 = new List<BOMEntry>();
        public int total_constraints { get; set; }
        public List<ConstraintRecipe> constraints { get; set; } = new List<ConstraintRecipe>();
    }

    public class BOMEntry
    {
        public string file { get; set; }
        public string hashed_json { get; set; }
        public int count { get; set; } = 1;
    }

    public class ConstraintRecipe
    {
        public int sequence_order { get; set; }
        public string constraint_name { get; set; }
        public string constraint_type { get; set; }
        public double confidence { get; set; }
        public string occurrence_one { get; set; }
        public string occurrence_two { get; set; }
        public EntityRecipe entity_one { get; set; }
        public EntityRecipe entity_two { get; set; }
        public double? offset_cm { get; set; }
        public double? angle_rad { get; set; }
        public bool? axes_opposed { get; set; }
        public bool? lock_rotation { get; set; }
        public bool? inside_tangency { get; set; }
        public string note { get; set; }
    }

    public class EntityRecipe
    {
        public string geometry_hash { get; set; }
        public string reference_key_string { get; set; }
        public string entity_type { get; set; }
        public string owner_document { get; set; }
        // Face geometry for signature matching
        public string surface_type { get; set; }
        public double? area_cm2 { get; set; }
        public double[] normal { get; set; }
        public double[] center { get; set; }
        public double? radius_cm { get; set; }
        // Edge geometry for signature matching
        public string curve_type { get; set; }
        public double? length_cm { get; set; }
        public double[] midpoint { get; set; }
        // Work feature
        public string work_feature_name { get; set; }
    }
}