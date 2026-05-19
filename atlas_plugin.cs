/*
 * ATLAS V2 — Inventor Plugin: ML-Powered Assembly Automation
 * ===========================================================
 * 
 * This iLogic/C# script:
 *   1. Reads a BOM from the active assembly (or a JSON file)
 *   2. Extracts part geometry for each occurrence
 *   3. Calls the Python ML inference server (or runs inline)
 *   4. Applies predicted constraints to the assembly
 *
 * SETUP:
 *   1. Run the Python inference server:
 *      python3 atlas_v2_server.py --model-dir ./v2_checkpoints --port 5050
 *   2. Open/create an assembly in Inventor
 *   3. Run this script via iLogic or Add-In
 *
 * The script communicates with the Python server via HTTP REST API.
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Text;
using Inventor;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

// ===========================================================================
// CONFIG — Adjust these paths for your environment
// ===========================================================================
const string PYTHON_SERVER_URL = "http://localhost:5050";
const string PARTS_OUTPUT_DIR = @"C:\ATLAS\Output Parts";
const string LOG_FILE = @"C:\ATLAS\atlas_plugin.log";
const double CONFIDENCE_THRESHOLD = 0.65;

// ===========================================================================
// MAIN ENTRY POINT
// ===========================================================================
public void Main()
{
    var app = ThisApplication;
    Log("ATLAS V2 Plugin — Starting assembly automation...");

    // Validate active document
    if (app.ActiveDocument == null || app.ActiveDocumentType != DocumentTypeEnum.kAssemblyDocumentObject)
    {
        MsgBox.Show("Please open an Assembly document first.", "ATLAS Error");
        return;
    }

    var asmDoc = (AssemblyDocument)app.ActiveDocument;
    var asmDef = asmDoc.ComponentDefinition;
    Log($"Assembly: {asmDoc.DisplayName}, {asmDef.Occurrences.Count} occurrences");

    try
    {
        // Phase 1: Extract BOM + part geometry
        Log("Phase 1: Extracting BOM and part geometry...");
        var bom = ExtractBOM(asmDef);
        Log($"  BOM contains {bom.Count} unique parts");

        // Phase 2: Extract geometry for each part
        var partGeometries = new Dictionary<string, JObject>();
        foreach (var entry in bom)
        {
            var partDoc = entry.Value.PartDoc;
            if (partDoc != null)
            {
                var geom = ExtractPartGeometry(partDoc);
                partGeometries[entry.Key] = geom;
                
                // Save to disk for the Python server
                var outPath = System.IO.Path.Combine(PARTS_OUTPUT_DIR, entry.Key + ".json");
                System.IO.File.WriteAllText(outPath, geom.ToString(Formatting.Indented), Encoding.UTF8);
                Log($"  Extracted: {entry.Key} ({geom["faces"]?.Count() ?? 0} faces, {geom["edges"]?.Count() ?? 0} edges)");
            }
        }

        // Phase 3: Call ML inference server
        Log("Phase 2: Calling ML inference server...");
        var predictions = CallInferenceServer(bom, partGeometries);
        
        if (predictions == null || predictions.Count == 0)
        {
            Log("  No predictions returned. Check Python server.");
            MsgBox.Show("No constraint predictions received. Is the Python server running?", "ATLAS");
            return;
        }
        Log($"  Received {predictions.Count} constraint predictions");

        // Phase 4: Apply constraints
        Log("Phase 3: Applying constraints...");
        int applied = 0;
        int failed = 0;

        foreach (var pred in predictions)
        {
            try
            {
                bool success = ApplyConstraint(asmDef, pred);
                if (success)
                {
                    applied++;
                    Log($"  ✅ Applied: {pred["constraint_type"]} between {pred["occurrence_one"]} ↔ {pred["occurrence_two"]} (conf: {pred["confidence"]})");
                }
                else
                {
                    failed++;
                    Log($"  ⚠️ Skipped: {pred["constraint_type"]} - could not find matching entities");
                }
            }
            catch (Exception ex)
            {
                failed++;
                Log($"  ❌ Failed: {pred["constraint_type"]} - {ex.Message}");
            }
        }

        Log($"\nDone! Applied: {applied}, Failed: {failed}");
        MsgBox.Show($"ATLAS Assembly Automation Complete!\n\nApplied: {applied} constraints\nSkipped: {failed}", "ATLAS V2");
    }
    catch (Exception ex)
    {
        Log($"FATAL: {ex.Message}\n{ex.StackTrace}");
        MsgBox.Show($"Error: {ex.Message}", "ATLAS Error");
    }
}

// ===========================================================================
// BOM EXTRACTION
// ===========================================================================
class BOMEntry
{
    public string FileName;
    public string FullPath;
    public int Count;
    public PartDocument PartDoc;
    public List<string> OccurrenceNames = new List<string>();
}

Dictionary<string, BOMEntry> ExtractBOM(AssemblyComponentDefinition asmDef)
{
    var bom = new Dictionary<string, BOMEntry>();
    
    foreach (ComponentOccurrence occ in asmDef.Occurrences)
    {
        try
        {
            if (occ.DefinitionDocumentType != DocumentTypeEnum.kPartDocumentObject)
                continue;
            
            var partDoc = (PartDocument)occ.Definition.Document;
            string stem = System.IO.Path.GetFileNameWithoutExtension(partDoc.FullFileName);
            
            if (bom.ContainsKey(stem))
            {
                bom[stem].Count++;
                bom[stem].OccurrenceNames.Add(occ.Name);
            }
            else
            {
                bom[stem] = new BOMEntry
                {
                    FileName = System.IO.Path.GetFileName(partDoc.FullFileName),
                    FullPath = partDoc.FullFileName,
                    Count = 1,
                    PartDoc = partDoc,
                    OccurrenceNames = new List<string> { occ.Name },
                };
            }
        }
        catch (Exception ex)
        {
            Log($"  Warning: Could not process occurrence {occ.Name}: {ex.Message}");
        }
    }
    
    return bom;
}

// ===========================================================================
// PART GEOMETRY EXTRACTION
// ===========================================================================
JObject ExtractPartGeometry(PartDocument partDoc)
{
    var partDef = partDoc.ComponentDefinition;
    var result = new JObject();
    
    // Metadata
    result["part_metadata"] = new JObject
    {
        ["file_name"] = System.IO.Path.GetFileName(partDoc.FullFileName),
        ["full_path"] = partDoc.FullFileName,
        ["material"] = SafeGetMaterial(partDef),
        ["mass_kg"] = SafeGetMass(partDef),
    };
    
    // Faces
    var faces = new JArray();
    foreach (SurfaceBody body in partDef.SurfaceBodies)
    {
        foreach (Face face in body.Faces)
        {
            try
            {
                var fObj = ExtractFace(face);
                if (fObj != null) faces.Add(fObj);
            }
            catch { }
        }
    }
    result["faces"] = faces;
    
    // Edges
    var edges = new JArray();
    foreach (SurfaceBody body in partDef.SurfaceBodies)
    {
        foreach (Edge edge in body.Edges)
        {
            try
            {
                var eObj = ExtractEdge(edge);
                if (eObj != null) edges.Add(eObj);
            }
            catch { }
        }
    }
    result["edges"] = edges;
    
    return result;
}

JObject ExtractFace(Face face)
{
    var f = new JObject();
    f["transient_key"] = face.TransientKey;
    
    // Reference key
    try
    {
        byte[] refKey = new byte[0];
        face.GetReferenceKey(ref refKey, 0);
        f["reference_key_string"] = Convert.ToBase64String(refKey);
    }
    catch { f["reference_key_string"] = ""; }
    
    // Surface type
    f["surface_type"] = face.SurfaceType.ToString();
    
    // Area
    try { f["area_cm2"] = face.Evaluator.Area; }
    catch { f["area_cm2"] = 0; }
    
    // Normal and center
    try
    {
        double[] center = new double[3];
        double[] normal = new double[3];
        face.Evaluator.GetNormal(ref center, ref normal);
        f["normal"] = new JArray(normal[0], normal[1], normal[2]);
        f["center"] = new JArray(center[0], center[1], center[2]);
    }
    catch
    {
        f["normal"] = new JArray(0, 0, 0);
        f["center"] = new JArray(0, 0, 0);
    }
    
    // Radius (for cylinders/cones/spheres)
    try
    {
        if (face.SurfaceType == SurfaceTypeEnum.kCylinderSurface)
        {
            var cyl = (Cylinder)face.Geometry;
            f["radius_cm"] = cyl.Radius;
        }
        else if (face.SurfaceType == SurfaceTypeEnum.kConeSurface)
        {
            var cone = (Cone)face.Geometry;
            f["radius_cm"] = cone.Radius;
            f["half_angle_rad"] = cone.HalfAngle;
        }
        else if (face.SurfaceType == SurfaceTypeEnum.kSphereSurface)
        {
            var sphere = (Sphere)face.Geometry;
            f["radius_cm"] = sphere.Radius;
        }
    }
    catch { }
    
    return f;
}

JObject ExtractEdge(Edge edge)
{
    var e = new JObject();
    e["transient_key"] = edge.TransientKey;
    
    try
    {
        byte[] refKey = new byte[0];
        edge.GetReferenceKey(ref refKey, 0);
        e["reference_key_string"] = Convert.ToBase64String(refKey);
    }
    catch { e["reference_key_string"] = ""; }
    
    e["curve_type"] = edge.CurveType.ToString();
    
    try
    {
        double minP = 0, maxP = 0;
        edge.Evaluator.GetParamExtents(out minP, out maxP);
        double len = 0;
        edge.Evaluator.GetLengthAtParam(minP, maxP, out len);
        e["length_cm"] = len;
    }
    catch { e["length_cm"] = 0; }
    
    // Midpoint
    try
    {
        double minP = 0, maxP = 0;
        edge.Evaluator.GetParamExtents(out minP, out maxP);
        double midP = (minP + maxP) / 2.0;
        double[] midPt = new double[3];
        edge.Evaluator.GetPointAtParam(midP, ref midPt);
        e["midpoint"] = new JArray(midPt[0], midPt[1], midPt[2]);
    }
    catch { e["midpoint"] = new JArray(0, 0, 0); }
    
    // Tangent at mid
    try
    {
        double minP = 0, maxP = 0;
        edge.Evaluator.GetParamExtents(out minP, out maxP);
        double midP = (minP + maxP) / 2.0;
        double[] tangent = new double[3];
        edge.Evaluator.GetTangent(midP, ref tangent);
        e["tangent"] = new JArray(tangent[0], tangent[1], tangent[2]);
    }
    catch { e["tangent"] = new JArray(0, 0, 0); }
    
    // Start vertex
    try
    {
        if (edge.StartVertex != null)
        {
            var pt = edge.StartVertex.Point;
            e["start_vertex"] = new JArray(pt.X, pt.Y, pt.Z);
        }
    }
    catch { }
    
    // Radius for circles
    try
    {
        if (edge.CurveType == CurveTypeEnum.kCircleCurve)
        {
            var circle = (Circle)edge.Geometry;
            e["radius_cm"] = circle.Radius;
        }
    }
    catch { }
    
    return e;
}

// ===========================================================================
// ML INFERENCE SERVER CALL
// ===========================================================================
List<JObject> CallInferenceServer(Dictionary<string, BOMEntry> bom, Dictionary<string, JObject> geometries)
{
    using (var client = new HttpClient())
    {
        client.Timeout = TimeSpan.FromSeconds(120);
        
        var request = new JObject();
        var partsArr = new JArray();
        
        foreach (var kv in bom)
        {
            partsArr.Add(new JObject
            {
                ["file"] = kv.Value.FileName,
                ["stem"] = kv.Key,
                ["count"] = kv.Value.Count,
                ["geometry"] = geometries.ContainsKey(kv.Key) ? geometries[kv.Key] : null,
            });
        }
        
        request["parts"] = partsArr;
        request["confidence_threshold"] = CONFIDENCE_THRESHOLD;
        
        var content = new StringContent(request.ToString(), Encoding.UTF8, "application/json");
        
        try
        {
            var response = client.PostAsync($"{PYTHON_SERVER_URL}/predict", content).Result;
            var body = response.Content.ReadAsStringAsync().Result;
            var result = JObject.Parse(body);
            
            var constraints = result["constraints"]?.ToObject<List<JObject>>() ?? new List<JObject>();
            return constraints;
        }
        catch (Exception ex)
        {
            Log($"Server error: {ex.Message}");
            
            // Fallback: try to read from a pre-generated file
            var fallbackPath = System.IO.Path.Combine(PARTS_OUTPUT_DIR, "predicted_assembly.json");
            if (System.IO.File.Exists(fallbackPath))
            {
                Log("  Using fallback predicted_assembly.json");
                var data = JObject.Parse(System.IO.File.ReadAllText(fallbackPath));
                return data["constraints"]?.ToObject<List<JObject>>() ?? new List<JObject>();
            }
            
            return new List<JObject>();
        }
    }
}

// ===========================================================================
// CONSTRAINT APPLICATION
// ===========================================================================
bool ApplyConstraint(AssemblyComponentDefinition asmDef, JObject pred)
{
    string cType = pred["constraint_type"]?.ToString() ?? "";
    string occ1Name = pred["occurrence_one"]?.ToString() ?? "";
    string occ2Name = pred["occurrence_two"]?.ToString() ?? "";
    string refKey1 = pred["entity_one_ref_key"]?.ToString() ?? "";
    string refKey2 = pred["entity_two_ref_key"]?.ToString() ?? "";
    string surfType1 = pred["entity_one_surface"]?.ToString() ?? "";
    string surfType2 = pred["entity_two_surface"]?.ToString() ?? "";
    double confidence = pred["confidence"]?.Value<double>() ?? 0;
    
    if (confidence < CONFIDENCE_THRESHOLD)
        return false;
    
    // Find occurrences by name (stem:instance format)
    ComponentOccurrence occA = FindOccurrence(asmDef, occ1Name);
    ComponentOccurrence occB = FindOccurrence(asmDef, occ2Name);
    
    if (occA == null || occB == null)
    {
        Log($"    Could not find occurrences: {occ1Name}, {occ2Name}");
        return false;
    }
    
    // Find matching geometry entities using ref keys or geometry matching
    object entityA = FindEntity(occA, refKey1, surfType1);
    object entityB = FindEntity(occB, refKey2, surfType2);
    
    if (entityA == null || entityB == null)
    {
        Log($"    Could not find entities for constraint");
        return false;
    }
    
    // Apply the constraint
    switch (cType)
    {
        case "kMateConstraint":
            asmDef.Constraints.AddMateConstraint(entityA, entityB, 0);
            return true;
            
        case "kFlushConstraint":
            asmDef.Constraints.AddFlushConstraint(entityA, entityB, 0);
            return true;
            
        case "kInsertConstraint":
            asmDef.Constraints.AddInsertConstraint(entityA, entityB, true, 0);
            return true;
            
        case "kAngleConstraint":
            double angle = pred["angle_rad"]?.Value<double>() ?? 0;
            asmDef.Constraints.AddAngleConstraint(entityA, entityB, angle.ToString());
            return true;
            
        case "kTangentConstraint":
            asmDef.Constraints.AddTangentConstraint(entityA, entityB);
            return true;
            
        default:
            Log($"    Unknown constraint type: {cType}");
            return false;
    }
}

ComponentOccurrence FindOccurrence(AssemblyComponentDefinition asmDef, string name)
{
    // name format: "PartStem:1"
    string[] parts = name.Split(':');
    string stem = parts[0];
    int instance = parts.Length > 1 ? int.Parse(parts[1]) : 1;
    
    int count = 0;
    foreach (ComponentOccurrence occ in asmDef.Occurrences)
    {
        try
        {
            string occStem = System.IO.Path.GetFileNameWithoutExtension(
                ((Document)occ.Definition.Document).FullFileName);
            
            if (occStem == stem)
            {
                count++;
                if (count == instance)
                    return occ;
            }
        }
        catch { }
    }
    return null;
}

object FindEntity(ComponentOccurrence occ, string refKeyBase64, string surfType)
{
    // Try 1: Use reference key to find entity directly
    if (!string.IsNullOrEmpty(refKeyBase64))
    {
        try
        {
            byte[] refKey = Convert.FromBase64String(refKeyBase64);
            object entity = occ.Definition.Document.ReferenceKeyManager.BindKeyToObject(refKey, 0);
            if (entity != null)
                return CreateProxy(occ, entity);
        }
        catch { }
    }
    
    // Try 2: Find by surface/curve type (first match)
    try
    {
        var partDef = ((PartDocument)occ.Definition.Document).ComponentDefinition;
        
        foreach (SurfaceBody body in partDef.SurfaceBodies)
        {
            // Try faces
            if (surfType.Contains("Surface") || surfType.Contains("Plane"))
            {
                foreach (Face face in body.Faces)
                {
                    if (face.SurfaceType.ToString() == surfType ||
                        (surfType == "kPlaneSurface" && face.SurfaceType == SurfaceTypeEnum.kPlaneSurface) ||
                        (surfType == "kCylinderSurface" && face.SurfaceType == SurfaceTypeEnum.kCylinderSurface))
                    {
                        return CreateProxy(occ, face);
                    }
                }
            }
            
            // Try edges (for Insert)
            if (surfType.Contains("Curve") || surfType.Contains("Circle"))
            {
                foreach (Edge edge in body.Edges)
                {
                    if (edge.CurveType.ToString() == surfType ||
                        (surfType == "kCircleCurve" && edge.CurveType == CurveTypeEnum.kCircleCurve))
                    {
                        return CreateProxy(occ, edge);
                    }
                }
            }
        }
    }
    catch (Exception ex)
    {
        Log($"    FindEntity error: {ex.Message}");
    }
    
    return null;
}

object CreateProxy(ComponentOccurrence occ, object nativeEntity)
{
    try
    {
        if (nativeEntity is Face face)
            return occ.CreateForAssemblyContext(face);
        if (nativeEntity is Edge edge)
            return occ.CreateForAssemblyContext(edge);
        if (nativeEntity is WorkPlane wp)
            return occ.CreateForAssemblyContext(wp);
    }
    catch { }
    return nativeEntity;
}

// ===========================================================================
// HELPERS
// ===========================================================================
string SafeGetMaterial(PartComponentDefinition partDef)
{
    try { return partDef.Material.Name; }
    catch { return ""; }
}

double SafeGetMass(PartComponentDefinition partDef)
{
    try { return partDef.MassProperties.Mass; }
    catch { return 0; }
}

void Log(string msg)
{
    string line = $"[{DateTime.Now:HH:mm:ss}] {msg}";
    Console.WriteLine(line);
    try
    {
        System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(LOG_FILE));
        System.IO.File.AppendAllText(LOG_FILE, line + "\n");
    }
    catch { }
}
