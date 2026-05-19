using Inventor;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

/// =============================================================================
/// ATLAS Phase 1A — Part Geometry Extractor (v2.0)
/// =============================================================================
/// Extracts every geometric property the GNN pipeline needs from .ipt files:
///   - Full B-Rep topology (faces→edges via loops)
///   - 21-dim feature data: surface type, area, center, normal, bbox, radius,
///     cone half-angle, torus minor radii, cylinder axis
///   - WorkPlanes with virtual geometry for constraint targets
///   - Feature graph, hole connection points, sketch constraints
///   - Stable reference keys + context keys for Phase 2 hashing
///
/// OUTPUT: One JSON per part, consumed by geometry_hasher.py (Phase 2)
/// =============================================================================

namespace InventorPartExporter
{
    class Program
    {
        static Inventor.Application invApp;
        static int totalParts = 0, successParts = 0, failedParts = 0, partialParts = 0;
        static StreamWriter logWriter;

        static void Main(string[] args)
        {
            string baseInputRoot = @"C:\Users\hrish\Downloads\Assemblies for Training ML model";
            string baseOutputRoot = @"C:\Users\hrish\Downloads\Assemblies for Training ML model\Output Parts";

            // ── Connect / Launch Inventor ────────────────────────────────────
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

            if (!Directory.Exists(baseOutputRoot))
                Directory.CreateDirectory(baseOutputRoot);

            string logPath = System.IO.Path.Combine(baseOutputRoot, "_extraction_log.txt");
            logWriter = new StreamWriter(logPath, true, Encoding.UTF8) { AutoFlush = true };
            Log($"=== ATLAS Part Extraction started at {DateTime.Now} ===");
            Log($"Input:  {baseInputRoot}");
            Log($"Output: {baseOutputRoot}");

            string[] iptFiles = Directory.GetFiles(baseInputRoot, "*.ipt", SearchOption.AllDirectories);
            Log($"Found {iptFiles.Length} .ipt files");

            foreach (string iptPath in iptFiles)
            {
                totalParts++;
                ProcessPart(iptPath, baseOutputRoot);
            }

            Log($"\n=== Complete: {successParts} OK, {partialParts} partial, {failedParts} failed / {totalParts} total ===");
            logWriter.Close();
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        static void Log(string msg)
        {
            Console.WriteLine(msg);
            try { logWriter?.WriteLine(msg); } catch { }
        }

        // =====================================================================
        // PROCESS SINGLE PART
        // =====================================================================
        static void ProcessPart(string iptPath, string outputRoot)
        {
            PartDocument partDoc = null;
            PartExport export = new PartExport();
            bool hadErrors = false;
            string outputPath = System.IO.Path.Combine(outputRoot,
                System.IO.Path.GetFileNameWithoutExtension(iptPath) + ".json");

            try
            {
                Log($"\n── Opening Part: {iptPath}");
                Document openedDoc = invApp.Documents.Open(iptPath, false);
                partDoc = (PartDocument)openedDoc;
                PartComponentDefinition def = partDoc.ComponentDefinition;

                // ── Reference Key Context ────────────────────────────────────
                ReferenceKeyManager mgr = null;
                int keyContext = 0;
                try
                {
                    mgr = partDoc.ReferenceKeyManager;
                    keyContext = mgr.CreateKeyContext();
                    byte[] ctxArray = new byte[0];
                    mgr.SaveContextToArray(keyContext, ref ctxArray);
                    export.context_key_string = mgr.KeyToString(ctxArray);
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] KeyContext: {ex.Message}"); }

                // ── Part Metadata ────────────────────────────────────────────
                try
                {
                    export.part_metadata = new PartMetadata
                    {
                        file_name = partDoc.DisplayName,
                        full_path = partDoc.FullFileName,
                        internal_name = partDoc.InternalName,
                        units = partDoc.UnitsOfMeasure.LengthUnits.ToString(),
                        part_number = GetProp((Document)partDoc, "Design Tracking Properties", "Part Number"),
                        description = GetProp((Document)partDoc, "Design Tracking Properties", "Description"),
                        material = GetProp((Document)partDoc, "Design Tracking Properties", "Material"),
                        mass_kg = SafeMass(def)
                    };
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] Metadata: {ex.Message}"); }

                // ── Bounding Box ─────────────────────────────────────────────
                try
                {
                    Box rb = def.RangeBox;
                    export.bounding_box = new BoundingBox
                    {
                        min = ToPoint(rb.MinPoint),
                        max = ToPoint(rb.MaxPoint)
                    };
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] BBox: {ex.Message}"); }

                // ── 1. FACES ─────────────────────────────────────────────────
                Log("  [1/7] Extracting Faces + Edge Loops...");
                try
                {
                    foreach (SurfaceBody body in def.SurfaceBodies)
                        foreach (Face face in body.Faces)
                        {
                            try { export.faces.Add(ExtractFace(face, mgr, keyContext)); }
                            catch (Exception fex) { Log($"    [Face Skip] {fex.Message}"); }
                        }
                    Log($"    → {export.faces.Count} faces");
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] Faces: {ex.Message}"); }

                // ── 2. EDGES ─────────────────────────────────────────────────
                Log("  [2/7] Extracting Edges...");
                try
                {
                    foreach (SurfaceBody body in def.SurfaceBodies)
                        foreach (Edge edge in body.Edges)
                        {
                            try { export.edges.Add(ExtractEdge(edge, mgr, keyContext)); }
                            catch (Exception eex) { Log($"    [Edge Skip] {eex.Message}"); }
                        }
                    Log($"    → {export.edges.Count} edges");
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] Edges: {ex.Message}"); }

                // ── 3. WORK PLANES ───────────────────────────────────────────
                Log("  [3/7] Extracting WorkPlanes...");
                try { ExtractWorkPlanes(def, export, mgr, keyContext); }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] WorkPlanes: {ex.Message}"); }

                // ── 4. WORK AXES ─────────────────────────────────────────────
                Log("  [4/7] Extracting WorkAxes...");
                try { ExtractWorkAxes(def, export, mgr, keyContext); }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] WorkAxes: {ex.Message}"); }

                // ── 5. HOLES ─────────────────────────────────────────────────
                Log("  [5/7] Extracting Holes...");
                try
                {
                    foreach (HoleFeature hole in def.Features.HoleFeatures)
                    {
                        try
                        {
                            if (!hole.Suppressed)
                                export.connection_points.AddRange(ExtractHole(hole));
                        }
                        catch (Exception holeEx) { Log($"    [Hole Skip] {holeEx.Message}"); }
                    }
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] Holes collection: {ex.Message}"); }

                // ── 6. FEATURE GRAPH ─────────────────────────────────────────
                Log("  [6/7] Extracting Feature Graph...");
                try
                {
                    foreach (PartFeature feat in def.Features)
                    {
                        try { export.feature_graph.Add(ExtractFeatureNode(feat)); }
                        catch { }
                    }
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] Features: {ex.Message}"); }

                // ── 7. BODIES ────────────────────────────────────────────────
                Log("  [7/7] Extracting Body Volumes...");
                try
                {
                    foreach (SurfaceBody body in def.SurfaceBodies)
                    {
                        double vol = 0;
                        try
                        {
                            vol = body.Volume[0.0001];
                        }
                        catch { }
                        export.bodies.Add(new BodyData { body_name = body.Name, volume_cm3 = vol });
                    }
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] Bodies: {ex.Message}"); }

                if (hadErrors) partialParts++; else successParts++;
            }
            catch (Exception ex)
            {
                failedParts++;
                Log($"[FATAL] {iptPath}: {ex.Message}");
            }
            finally
            {
                // ALWAYS serialize whatever we collected
                SafeSerialize(export, outputPath);
                try { if (partDoc != null) partDoc.Close(false); } catch { }
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        // =====================================================================
        // FACE EXTRACTION — Complete geometry for 21-dim GNN features + hashing
        // =====================================================================
        static FaceData ExtractFace(Face face, ReferenceKeyManager mgr, int keyContext)
        {
            var data = new FaceData
            {
                transient_key = face.TransientKey,
                surface_type = face.SurfaceType.ToString(),
                area_cm2 = 0
            };

            // Area
            try { data.area_cm2 = face.Evaluator.Area; } catch { }

            // Reference Key (for Phase 2 bridge)
            try
            {
                byte[] key = new byte[0];
                face.GetReferenceKey(ref key, keyContext);
                data.reference_key_string = mgr.KeyToString(key);
            }
            catch { }

            // Normal + Center at UV midpoint
            try
            {
                SurfaceEvaluator eval = face.Evaluator;
                Box2d uv = eval.ParamRangeRect;
                double[] pars = {
                    (uv.MinPoint.X + uv.MaxPoint.X) / 2.0,
                    (uv.MinPoint.Y + uv.MaxPoint.Y) / 2.0
                };
                double[] normal = new double[3], pt = new double[3];
                eval.GetNormal(ref pars, ref normal);
                eval.GetPointAtParam(ref pars, ref pt);
                data.normal = normal;
                data.center = pt;
            }
            catch { }

            // Bounding Box
            try
            {
                Box box = face.Evaluator.RangeBox;
                data.bbox_min = ToPoint(box.MinPoint);
                data.bbox_max = ToPoint(box.MaxPoint);
            }
            catch { }

            // ── Surface-specific geometry (critical for GNN + hashing) ───────
            try
            {
                switch (face.SurfaceType)
                {
                    case SurfaceTypeEnum.kCylinderSurface:
                        Cylinder cyl = (Cylinder)face.Geometry;
                        data.radius_cm = cyl.Radius;
                        data.axis = new double[] {
                            cyl.AxisVector.X, cyl.AxisVector.Y, cyl.AxisVector.Z
                        };
                        break;

                    case SurfaceTypeEnum.kConeSurface:
                        Cone cone = (Cone)face.Geometry;
                        data.radius_cm = cone.Radius;
                        data.half_angle_rad = cone.HalfAngle;
                        data.axis = new double[] {
                            cone.AxisVector.X, cone.AxisVector.Y, cone.AxisVector.Z
                        };
                        break;

                    case SurfaceTypeEnum.kSphereSurface:
                        Sphere sph = (Sphere)face.Geometry;
                        data.radius_cm = sph.Radius;
                        break;

                    case SurfaceTypeEnum.kTorusSurface:
                        Torus tor = (Torus)face.Geometry;
                        data.radius_cm = tor.MajorRadius;
                        data.minor_radius_cm = tor.MinorRadius;
                        data.axis = new double[] {
                            tor.AxisVector.X, tor.AxisVector.Y, tor.AxisVector.Z
                        };
                        break;
                }
            }
            catch { }

            // Created-by feature (for generative context)
            try { if (face.CreatedByFeature != null) data.created_by_feature = face.CreatedByFeature.Name; } catch { }

            // ── Edge Loops (B-Rep topology → structural graph edges) ─────────
            data.loops = new List<LoopData>();
            try
            {
                foreach (EdgeLoop loop in face.EdgeLoops)
                {
                    var loopData = new LoopData { is_outer = loop.IsOuterEdgeLoop };
                    foreach (Edge loopEdge in loop.Edges)
                    {
                        try
                        {
                            byte[] eKey = new byte[0];
                            loopEdge.GetReferenceKey(ref eKey, keyContext);
                            loopData.edge_reference_keys.Add(mgr.KeyToString(eKey));
                        }
                        catch { }
                    }
                    data.loops.Add(loopData);
                }
            }
            catch { }

            return data;
        }

        // =====================================================================
        // EDGE EXTRACTION
        // =====================================================================
        static EdgeData ExtractEdge(Edge edge, ReferenceKeyManager mgr, int keyContext)
        {
            var data = new EdgeData
            {
                transient_key = edge.TransientKey,
                curve_type = edge.CurveType.ToString(),
                length_cm = GetEdgeLength(edge)
            };

            // Reference Key
            try
            {
                byte[] key = new byte[0];
                edge.GetReferenceKey(ref key, keyContext);
                data.reference_key_string = mgr.KeyToString(key);
            }
            catch { }

            // Midpoint, Tangent, Vertices
            try
            {
                CurveEvaluator eval = edge.Evaluator;

                if (eval != null)
                {
                    double s = 0, e = 0, length = 0;

                    eval.GetParamExtents(out s, out e);
                    eval.GetLengthAtParam(s, e, out length);

                    double midParam = 0;
                    eval.GetParamAtLength(s, length / 2.0, out midParam);

                    double[] arr = new double[] { midParam };
                    double[] pt = new double[3];
                    double[] tan = new double[3];

                    eval.GetPointAtParam(ref arr, ref pt);
                    eval.GetTangent(ref arr, ref tan);

                    data.midpoint = pt;
                    data.tangent = tan;
                }
            }
            catch
            {
            }

            try
            {
                data.start_vertex = new double[] {
                    edge.StartVertex.Point.X, edge.StartVertex.Point.Y, edge.StartVertex.Point.Z
                };
                data.end_vertex = new double[] {
                    edge.StopVertex.Point.X, edge.StopVertex.Point.Y, edge.StopVertex.Point.Z
                };
            }
            catch { }

            // Adjacent faces (B-Rep topology)
            data.adjacent_faces = new List<string>();
            foreach (Face adjFace in edge.Faces)
            {
                try
                {
                    byte[] refKey = new byte[0];
                    adjFace.GetReferenceKey(ref refKey, keyContext);
                    data.adjacent_faces.Add(mgr.KeyToString(refKey));
                }
                catch { }
            }

            // Edge radius (for circular edges)
            try
            {
                if (edge.CurveType == CurveTypeEnum.kCircleCurve)
                    data.radius_cm = ((Circle)edge.Geometry).Radius;
            }
            catch { }

            return data;
        }

        // =====================================================================
        // WORK PLANES
        // =====================================================================
        static void ExtractWorkPlanes(
            PartComponentDefinition def, PartExport export,
            ReferenceKeyManager mgr, int keyContext)
        {
            foreach (WorkPlane wp in def.WorkPlanes)
            {
                try
                {
                    Plane mathPlane = wp.Plane;
                    var wpData = new WorkPlaneData
                    {
                        work_feature_name = wp.Name,
                        center = new double[] {
                            mathPlane.RootPoint.X, mathPlane.RootPoint.Y, mathPlane.RootPoint.Z
                        },
                        normal = new double[] {
                            mathPlane.Normal.X, mathPlane.Normal.Y, mathPlane.Normal.Z
                        }
                    };

                    try
                    {
                        byte[] key = new byte[0];
                        wp.GetReferenceKey(ref key, keyContext);
                        wpData.reference_key_string = mgr.KeyToString(key);
                    }
                    catch { }

                    export.work_planes.Add(wpData);
                }
                catch { }
            }
        }

        // =====================================================================
        // WORK AXES
        // =====================================================================
        static void ExtractWorkAxes(
            PartComponentDefinition def, PartExport export,
            ReferenceKeyManager mgr, int keyContext)
        {
            foreach (WorkAxis wa in def.WorkAxes)
            {
                try
                {
                    Line axisLine = (Line)wa.Line;
                    var waData = new WorkAxisData
                    {
                        work_feature_name = wa.Name,
                        direction = new double[] {
                            axisLine.Direction.X, axisLine.Direction.Y, axisLine.Direction.Z
                        },
                        root_point = new double[] {
                            axisLine.RootPoint.X, axisLine.RootPoint.Y, axisLine.RootPoint.Z
                        }
                    };

                    try
                    {
                        byte[] key = new byte[0];
                        wa.GetReferenceKey(ref key, keyContext);
                        waData.reference_key_string = mgr.KeyToString(key);
                    }
                    catch { }

                    export.work_axes.Add(waData);
                }
                catch { }
            }
        }

        // =====================================================================
        // HOLES (connection point extraction for Insert constraints)
        // =====================================================================
        static List<ConnectionPoint> ExtractHole(HoleFeature hole)
        {
            var list = new List<ConnectionPoint>();
            string holeType = hole.Tapped ? "Tapped" : "Simple";
            double diamCm = 0;

            try
            {
                var diaParam = hole.HoleDiameter;

                if (diaParam != null)
                    diamCm = diaParam.Value;
            }
            catch
            {
                diamCm = 0;
            }

            list.Add(new ConnectionPoint
            {
                id = Guid.NewGuid().ToString(),
                feature_name = hole.Name,
                feature_type = "Hole",
                suppressed = hole.Suppressed,
                hole_properties = new HoleProperties
                {
                    hole_type = holeType,
                    diameter_cm = diamCm,
                    is_threaded = hole.Tapped
                }
            });
            return list;
        }

        // =====================================================================
        // FEATURE GRAPH
        // =====================================================================
        static FeatureNode ExtractFeatureNode(PartFeature feat)
        {
            var node = new FeatureNode
            {
                feature_name = feat.Name,
                feature_type = ClassifyFeature(feat),
                suppressed = feat.Suppressed
            };

            try
            {
                if (feat is ExtrudeFeature) node.operation = ((ExtrudeFeature)feat).Operation.ToString();
                if (feat is RevolveFeature) node.operation = ((RevolveFeature)feat).Operation.ToString();
                if (feat is SweepFeature) node.operation = ((SweepFeature)feat).Operation.ToString();
            }
            catch { }

            try
            {
                dynamic d = feat;
                foreach (object dep in d.DependentFeatures)
                { dynamic x = dep; node.child_features.Add((string)x.Name); }
            }
            catch { }

            try
            {
                dynamic d = feat;
                foreach (object dep in d.DependedOnFeatures)
                { dynamic x = dep; node.parent_features.Add((string)x.Name); }
            }
            catch { }

            return node;
        }

        static string ClassifyFeature(PartFeature f)
        {
            if (f is ExtrudeFeature) return "Extrude";
            if (f is RevolveFeature) return "Revolve";
            if (f is SweepFeature) return "Sweep";
            if (f is LoftFeature) return "Loft";
            if (f is HoleFeature) return "Hole";
            if (f is FilletFeature) return "Fillet";
            if (f is ChamferFeature) return "Chamfer";
            if (f is RectangularPatternFeature) return "RectangularPattern";
            if (f is CircularPatternFeature) return "CircularPattern";
            if (f is CombineFeature) return "BooleanCombine";
            return "Other";
        }

        // =====================================================================
        // UTILITIES
        // =====================================================================
        static double GetEdgeLength(Edge edge)
        {
            try
            {
                if (edge == null)
                    return 0;

                CurveEvaluator eval = edge.Evaluator;

                if (eval == null)
                    return 0;

                double s = 0, e = 0, length = 0;

                eval.GetParamExtents(out s, out e);
                eval.GetLengthAtParam(s, e, out length);

                return length;
            }
            catch
            {
                try
                {
                    if (edge.StartVertex == null || edge.StopVertex == null)
                        return 0;

                    Point p1 = edge.StartVertex.Point;
                    Point p2 = edge.StopVertex.Point;

                    return Math.Sqrt(
                        Math.Pow(p2.X - p1.X, 2) +
                        Math.Pow(p2.Y - p1.Y, 2) +
                        Math.Pow(p2.Z - p1.Z, 2));
                }
                catch
                {
                    return 0;
                }
            }
        }

        static PointData ToPoint(Point p) =>
            new PointData { x = p.X, y = p.Y, z = p.Z };

        static double SafeMass(PartComponentDefinition def)
        {
            try { return def.MassProperties.Mass; } catch { return 0; }
        }

        static string GetProp(Document doc, string setName, string propName)
        {
            try { return doc.PropertySets[setName][propName].Value?.ToString() ?? ""; }
            catch { return ""; }
        }

        static void SafeSerialize(object export, string outputPath)
        {
            string json = null;

            // Attempt 1: standard serialization
            try
            {
                var settings = new JsonSerializerSettings
                {
                    Formatting = Formatting.Indented,
                    NullValueHandling = NullValueHandling.Ignore,
                    ReferenceLoopHandling = ReferenceLoopHandling.Ignore,
                    Error = (sender, args) =>
                    {
                        Log($"  [Ser Warn] {args.ErrorContext.Path}: {args.ErrorContext.Error.Message}");
                        args.ErrorContext.Handled = true;
                    }
                };
                json = JsonConvert.SerializeObject(export, settings);
            }
            catch (Exception ex1)
            {
                Log($"  [Ser Error Attempt 1] {ex1.Message}");
            }

            // Attempt 2: JObject intermediary (strips COM proxies more aggressively)
            if (string.IsNullOrEmpty(json))
            {
                try
                {
                    var safeSettings = new JsonSerializerSettings
                    {
                        NullValueHandling = NullValueHandling.Ignore,
                        ReferenceLoopHandling = ReferenceLoopHandling.Ignore,
                        Error = (s, a) => { a.ErrorContext.Handled = true; }
                    };
                    var jObj = JObject.FromObject(export, JsonSerializer.Create(safeSettings));
                    json = jObj.ToString(Formatting.Indented);
                }
                catch (Exception ex2)
                {
                    Log($"  [Ser Error Attempt 2] {ex2.Message}");
                }
            }

            // Attempt 3: error marker so we know something went wrong
            if (string.IsNullOrEmpty(json))
            {
                json = "{\"error\": \"serialization_failed\", \"source\": \"" +
                       outputPath.Replace("\\", "/") + "\"}";
                Log($"  [CRITICAL] Writing error marker for {outputPath}");
            }

            // Write to disk
            try
            {
                string dir = System.IO.Path.GetDirectoryName(outputPath);
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                System.IO.File.WriteAllText(outputPath, json, Encoding.UTF8);
                Log($"  ✅ Saved → {outputPath} ({json.Length} chars)");
            }
            catch (Exception ioEx)
            {
                Log($"  [FILE WRITE FAILED] {outputPath}: {ioEx.Message}");
            }
        }
    }

    // =========================================================================
    // DATA STRUCTURES — aligned with geometry_hasher.py field expectations
    // =========================================================================

    public class PartExport
    {
        public PartMetadata part_metadata { get; set; }
        public string context_key_string { get; set; }
        public BoundingBox bounding_box { get; set; }
        public List<FaceData> faces { get; set; } = new List<FaceData>();
        public List<EdgeData> edges { get; set; } = new List<EdgeData>();
        public List<WorkPlaneData> work_planes { get; set; } = new List<WorkPlaneData>();
        public List<WorkAxisData> work_axes { get; set; } = new List<WorkAxisData>();
        public List<BodyData> bodies { get; set; } = new List<BodyData>();
        public List<ConnectionPoint> connection_points { get; set; } = new List<ConnectionPoint>();
        public List<FeatureNode> feature_graph { get; set; } = new List<FeatureNode>();
    }

    public class PartMetadata
    {
        public string file_name { get; set; }
        public string full_path { get; set; }
        public string internal_name { get; set; }
        public string units { get; set; }
        public string part_number { get; set; }
        public string description { get; set; }
        public string material { get; set; }
        public double mass_kg { get; set; }
    }

    public class BoundingBox
    {
        public PointData min { get; set; }
        public PointData max { get; set; }
    }

    public class PointData
    {
        public double x { get; set; }
        public double y { get; set; }
        public double z { get; set; }
    }

    public class FaceData
    {
        public int transient_key { get; set; }
        public string reference_key_string { get; set; }
        public string surface_type { get; set; }
        public double area_cm2 { get; set; }
        public double[] normal { get; set; }
        public double[] center { get; set; }
        public double? radius_cm { get; set; }
        public double? minor_radius_cm { get; set; }
        public double? half_angle_rad { get; set; }
        public double[] axis { get; set; }
        public PointData bbox_min { get; set; }
        public PointData bbox_max { get; set; }
        public string created_by_feature { get; set; }
        public List<LoopData> loops { get; set; } = new List<LoopData>();
    }

    public class LoopData
    {
        public bool is_outer { get; set; }
        public List<string> edge_reference_keys { get; set; } = new List<string>();
    }

    public class EdgeData
    {
        public int transient_key { get; set; }
        public string reference_key_string { get; set; }
        public string curve_type { get; set; }
        public double length_cm { get; set; }
        public double[] midpoint { get; set; }
        public double[] tangent { get; set; }
        public double[] start_vertex { get; set; }
        public double[] end_vertex { get; set; }
        public double? radius_cm { get; set; }
        public List<string> adjacent_faces { get; set; }
    }

    public class WorkPlaneData
    {
        public string entity_type { get; set; } = "WorkPlane";
        public string work_feature_name { get; set; }
        public string surface_type { get; set; } = "kPlaneSurface";
        public double area_cm2 { get; set; } = 0.0001;
        public double[] center { get; set; }
        public double[] normal { get; set; }
        public string reference_key_string { get; set; }
    }

    public class WorkAxisData
    {
        public string entity_type { get; set; } = "WorkAxis";
        public string work_feature_name { get; set; }
        public double[] direction { get; set; }
        public double[] root_point { get; set; }
        public string reference_key_string { get; set; }
    }

    public class BodyData
    {
        public string body_name { get; set; }
        public double volume_cm3 { get; set; }
    }

    public class ConnectionPoint
    {
        public string id { get; set; }
        public string feature_name { get; set; }
        public string feature_type { get; set; }
        public bool suppressed { get; set; }
        public HoleProperties hole_properties { get; set; }
    }

    public class HoleProperties
    {
        public string hole_type { get; set; }
        public double diameter_cm { get; set; }
        public bool is_threaded { get; set; }
    }

    public class FeatureNode
    {
        public string feature_name { get; set; }
        public string feature_type { get; set; }
        public string operation { get; set; }
        public bool suppressed { get; set; }
        public List<string> parent_features { get; set; } = new List<string>();
        public List<string> child_features { get; set; } = new List<string>();
    }
}