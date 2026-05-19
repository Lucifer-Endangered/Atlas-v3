using Inventor;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

/// =============================================================================
/// ATLAS Phase 1B — Assembly Constraint Extractor (v2.0)
/// =============================================================================
/// Extracts the assembly graph (occurrences + constraints + joints) from .iam files.
///
/// KEY IMPROVEMENTS over v1:
///   - Extracts constraint solution direction (Mate vs AntiMate)
///   - Extracts Insert axes_opposed + lock_rotation flags
///   - Captures native geometry (radius, axis, half_angle) from proxy faces
///   - Proper try/finally document cleanup (Close(false))
///   - O(1) reference key context caching per owner document
///   - WorkPlane/WorkAxis entity support for constraint targets
///   - All field names aligned with geometry_hasher.py expectations
///
/// OUTPUT: One JSON per assembly, consumed by geometry_hasher.py (Phase 2)
/// =============================================================================

namespace InventorAssemblyExporter
{
    class Program
    {
        static Inventor.Application invApp;
        static Dictionary<string, KeyContextData> contextCache =
            new Dictionary<string, KeyContextData>();
        static int totalAsm = 0, successAsm = 0, failedAsm = 0, partialAsm = 0;
        static StreamWriter logWriter;

        static void Main(string[] args)
        {
            string baseInputRoot  = @"E:\Phase 1\New-Assemblies";
            string baseOutputRoot = @"E:\Phase 1\assembliesexport-new";

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

            string logPath = Path.Combine(baseOutputRoot, "_extraction_log.txt");
            logWriter = new StreamWriter(logPath, true, Encoding.UTF8) { AutoFlush = true };
            Log($"=== ATLAS Assembly Extraction started at {DateTime.Now} ===");
            Log($"Input:  {baseInputRoot}");
            Log($"Output: {baseOutputRoot}");

            string[] iamFiles = Directory.GetFiles(baseInputRoot, "*.iam", SearchOption.AllDirectories);
            Log($"Found {iamFiles.Length} .iam files");

            foreach (string iamPath in iamFiles)
            {
                totalAsm++;
                ProcessAssembly(iamPath, baseOutputRoot);
            }

            Log($"\n=== Complete: {successAsm} OK, {partialAsm} partial, {failedAsm} failed / {totalAsm} total ===");
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
        // PROCESS SINGLE ASSEMBLY
        // =====================================================================
        static void ProcessAssembly(string iamPath, string outputRoot)
        {
            AssemblyDocument asmDoc = null;
            AssemblyExport export = new AssemblyExport();
            bool hadErrors = false;
            string outputPath = Path.Combine(outputRoot,
                Path.GetFileNameWithoutExtension(iamPath) + ".json");

            try
            {
                Log($"\n── Opening Assembly: {iamPath}");
                contextCache.Clear();

                Document openedDoc = invApp.Documents.Open(iamPath, false);
                asmDoc = (AssemblyDocument)openedDoc;
                AssemblyComponentDefinition def = asmDoc.ComponentDefinition;

                // ── Metadata ─────────────────────────────────────────────────
                try
                {
                    export.assembly_metadata = new AssemblyMetadata
                    {
                        assembly_name     = asmDoc.DisplayName,
                        full_file_name    = asmDoc.FullFileName,
                        internal_name     = asmDoc.InternalName,
                        total_occurrences = def.Occurrences.Count,
                        total_constraints = def.Constraints.Count
                    };
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] Metadata: {ex.Message}"); }

                // ── 1. PHYSICS ───────────────────────────────────────────────
                try
                {
                    MassProperties mp = def.MassProperties;
                    export.physics = new AssemblyPhysics
                    {
                        mass_kg        = mp.Mass,
                        center_of_mass = new double[] { mp.CenterOfMass.X, mp.CenterOfMass.Y, mp.CenterOfMass.Z }
                    };
                    double Ixx, Iyy, Izz, Ixy, Iyz, Ixz;
                    mp.XYZMomentsOfInertia(out Ixx, out Iyy, out Izz, out Ixy, out Iyz, out Ixz);
                    export.physics.inertia_tensor = new double[] { Ixx, Iyy, Izz, Ixy, Iyz, Ixz };
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] Physics: {ex.Message}"); }

                // ── 2. GRAPH NODES (Occurrences + DOF) ───────────────────────
                Log("  [1/3] Extracting Occurrences + DOF...");
                try
                {
                    ExtractOccurrences(def.Occurrences, export.assembly_graph.nodes, "");
                    Log($"    → {export.assembly_graph.nodes.Count} nodes");
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] Occurrences: {ex.Message}"); }

                // ── 3. CONSTRAINT EDGES ──────────────────────────────────────
                Log("  [2/3] Extracting Constraints...");
                try
                {
                    foreach (AssemblyConstraint constraint in def.Constraints)
                    {
                        try
                        {
                            export.assembly_graph.constraint_edges.Add(
                                ExtractConstraint(constraint));
                        }
                        catch (Exception cex)
                        {
                            Log($"    [Constraint Skip] {cex.Message}");
                        }
                    }
                    Log($"    → {export.assembly_graph.constraint_edges.Count} constraints");
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] Constraints: {ex.Message}"); }

                // ── 4. JOINT EDGES ───────────────────────────────────────────
                Log("  [3/3] Extracting Joints...");
                try
                {
                    foreach (AssemblyJoint joint in def.Joints)
                    {
                        try
                        {
                            export.assembly_graph.joint_edges.Add(ExtractJoint(joint));
                        }
                        catch (Exception jex)
                        {
                            Log($"    [Joint Skip] {jex.Message}");
                        }
                    }
                    Log($"    → {export.assembly_graph.joint_edges.Count} joints");
                }
                catch (Exception ex) { hadErrors = true; Log($"  [WARN] Joints: {ex.Message}"); }

                if (hadErrors) partialAsm++; else successAsm++;
            }
            catch (Exception ex)
            {
                failedAsm++;
                Log($"[FATAL] {iamPath}: {ex.Message}\n{ex.StackTrace}");
            }
            finally
            {
                // ALWAYS serialize whatever we collected
                SafeSerialize(export, outputPath);
                try { if (asmDoc != null) asmDoc.Close(false); } catch { }
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        // =====================================================================
        // OCCURRENCE EXTRACTION (recursive, with DOF)
        // =====================================================================
        static void ExtractOccurrences(
            ComponentOccurrences occurrences, List<ComponentNode> nodes, string parentPath)
        {
            foreach (ComponentOccurrence occ in occurrences)
            {
                string nodeId = string.IsNullOrEmpty(parentPath)
                    ? occ.Name
                    : parentPath + "/" + occ.Name;

                ComponentNode node = new ComponentNode
                {
                    node_id        = nodeId,
                    file_name      = "",
                    component_type = occ.DefinitionDocumentType.ToString(),
                    grounded       = occ.Grounded,
                    suppressed     = occ.Suppressed,
                    visible        = occ.Visible,
                    adaptive       = occ.Adaptive,
                    transform      = ExtractTransform(occ.Transformation)
                };

                try { node.file_name = occ.ReferencedDocumentDescriptor?.FullDocumentName ?? ""; } catch { }
                try { node.flexible = occ.Flexible; } catch { }

                // Degrees of Freedom
                try
                {
                    int transCount, rotCount;
                    ObjectsEnumerator transDOFs, rotDOFs;
                    Point dofCenter;
                    occ.GetDegreesOfFreedom(out transCount, out transDOFs,
                                           out rotCount, out rotDOFs, out dofCenter);
                    node.dof_translation   = transCount;
                    node.dof_rotation      = rotCount;
                    node.fully_constrained = (transCount == 0 && rotCount == 0 && !occ.Grounded);
                }
                catch { }

                nodes.Add(node);

                // Recurse into sub-assemblies
                if (occ.DefinitionDocumentType == DocumentTypeEnum.kAssemblyDocumentObject)
                {
                    try { ExtractOccurrences(occ.SubOccurrences, nodes, nodeId); } catch { }
                }
            }
        }

        // =====================================================================
        // CONSTRAINT EXTRACTION — with solution type + full entity geometry
        // =====================================================================
        static ConstraintEdge ExtractConstraint(AssemblyConstraint constraint)
        {
            var edge = new ConstraintEdge
            {
                constraint_name = constraint.Name,
                constraint_type = constraint.Type.ToString().Replace("Object", ""),
                suppressed      = constraint.Suppressed,
                health_status   = constraint.HealthStatus.ToString()
            };

            // Occurrence names
            try { edge.node_one_id = constraint.OccurrenceOne.Name; } catch { }
            try { edge.node_two_id = constraint.OccurrenceTwo.Name; } catch { }

            // Offset / Angle values
            dynamic dc = constraint;
            try { edge.offset_cm = (double)dc.Offset.Value; } catch { }
            try { edge.angle_rad = (double)dc.Angle.Value; } catch { }

            // ── Solution / direction info (critical for reconstruction) ──────
            try
            {
                if (constraint is MateConstraint)
                {
                    MateConstraint mc = (MateConstraint)constraint;
                    edge.solution = mc.SolutionType.ToString();
                }
            }
            catch { }

            try
            {
                if (constraint is InsertConstraint)
                {
                    InsertConstraint ic = (InsertConstraint)constraint;
                    edge.axes_opposed  = ic.AxesOpposed;
                    edge.lock_rotation = ic.LockRotation;
                }
            }
            catch { }

            try
            {
                if (constraint is AngleConstraint)
                {
                    AngleConstraint ac = (AngleConstraint)constraint;
                    edge.solution = ac.SolutionType.ToString();
                }
            }
            catch { }

            try
            {
                if (constraint is TangentConstraint)
                {
                    TangentConstraint tc = (TangentConstraint)constraint;
                    edge.inside_tangency = tc.InsideTangency;
                }
            }
            catch { }

            // ── Entity geometry (both sides) ─────────────────────────────────
            try { edge.entity_one = ExtractEntity(constraint.EntityOne); } catch { }
            try { edge.entity_two = ExtractEntity(constraint.EntityTwo); } catch { }

            return edge;
        }

        // =====================================================================
        // JOINT EXTRACTION
        // =====================================================================
        static JointEdge ExtractJoint(AssemblyJoint joint)
        {
            var jEdge = new JointEdge
            {
                joint_name        = joint.Name,
                joint_type        = joint.Definition.JointType.ToString(),
                health_status     = joint.HealthStatus.ToString(),
                suppressed        = joint.Suppressed,
                has_linear_limit  = joint.Definition.HasLinearPositionLimits,
                has_angular_limit = joint.Definition.HasAngularPositionLimits
            };

            try { jEdge.node_one_id = joint.OccurrenceOne?.Name; } catch { }
            try { jEdge.node_two_id = joint.OccurrenceTwo?.Name; } catch { }

            if (jEdge.has_linear_limit)
            {
                try
                {
                    jEdge.linear_start_cm = joint.Definition.LinearPositionStartLimit.Value;
                    jEdge.linear_end_cm   = joint.Definition.LinearPositionEndLimit.Value;
                }
                catch { }
            }
            if (jEdge.has_angular_limit)
            {
                try
                {
                    jEdge.angular_start_rad = joint.Definition.AngularPositionStartLimit.Value;
                    jEdge.angular_end_rad   = joint.Definition.AngularPositionEndLimit.Value;
                }
                catch { }
            }

            return jEdge;
        }

        // =====================================================================
        // ENTITY EXTRACTION — Native reach-through with full geometry
        // =====================================================================
        static EntityData ExtractEntity(object obj)
        {
            if (obj == null) return null;
            var data = new EntityData();

            try
            {
                dynamic proxy = obj;

                // Entity type
                try { data.entity_type = proxy.Type.ToString().Replace("k", "").Replace("Object", ""); }
                catch { data.entity_type = "Unknown"; }

                // Containing occurrence
                ComponentOccurrence occ = null;
                try { occ = proxy.ContainingOccurrence; } catch { return data; }
                if (occ == null) return data;
                data.proxy_context_occurrence = occ.Name;

                // Owner document
                Document doc = null;
                try { doc = (Document)occ.Definition.Document; } catch { return data; }
                if (doc == null) return data;
                data.owner_document = doc.FullFileName;

                // Reference key context (cached per document)
                KeyContextData ctxData;
                string docPath = doc.FullFileName;
                if (!contextCache.TryGetValue(docPath, out ctxData))
                {
                    ReferenceKeyManager mgr = doc.ReferenceKeyManager;
                    int ctxId = mgr.CreateKeyContext();
                    byte[] ctxBytes = new byte[0];
                    mgr.SaveContextToArray(ctxId, ref ctxBytes);
                    ctxData = new KeyContextData
                    {
                        ContextId     = ctxId,
                        ContextString = mgr.KeyToString(ctxBytes),
                        Manager       = mgr
                    };
                    contextCache[docPath] = ctxData;
                }
                data.context_key_string = ctxData.ContextString;

                // Native object (reach through proxy)
                object nativeObj = null;
                try { nativeObj = proxy.NativeObject; } catch { return data; }
                if (nativeObj == null) return data;

                // Reference key
                try
                {
                    dynamic nativeDyn = nativeObj;
                    byte[] refKey = new byte[0];
                    nativeDyn.GetReferenceKey(ref refKey, ctxData.ContextId);
                    data.reference_key_string = ctxData.Manager.KeyToString(refKey);
                }
                catch { }

                // ── FACE geometry ────────────────────────────────────────────
                if (nativeObj is Face)
                {
                    Face face = (Face)nativeObj;
                    data.surface_type = face.SurfaceType.ToString();
                    try { data.area_cm2 = face.Evaluator.Area; } catch { }

                    // Bounding box
                    try
                    {
                        Box box = face.Evaluator.RangeBox;
                        data.face_bbox_min = new double[] { box.MinPoint.X, box.MinPoint.Y, box.MinPoint.Z };
                        data.face_bbox_max = new double[] { box.MaxPoint.X, box.MaxPoint.Y, box.MaxPoint.Z };
                    }
                    catch { }

                    // Normal + center
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
                        data.face_normal_at_center = normal;
                        data.face_point_at_center  = pt;
                    }
                    catch { }

                    // Surface-specific: radius, axis, half-angle
                    try
                    {
                        switch (face.SurfaceType)
                        {
                            case SurfaceTypeEnum.kCylinderSurface:
                                Cylinder cyl = (Cylinder)face.Geometry;
                                data.radius_cm = cyl.Radius;
                                data.axis = new double[] { cyl.AxisVector.X, cyl.AxisVector.Y, cyl.AxisVector.Z };
                                break;
                            case SurfaceTypeEnum.kConeSurface:
                                Cone cone = (Cone)face.Geometry;
                                data.radius_cm     = cone.Radius;
                                data.half_angle_rad = cone.HalfAngle;
                                data.axis = new double[] { cone.AxisVector.X, cone.AxisVector.Y, cone.AxisVector.Z };
                                break;
                            case SurfaceTypeEnum.kSphereSurface:
                                Sphere sph = (Sphere)face.Geometry;
                                data.radius_cm = sph.Radius;
                                break;
                            case SurfaceTypeEnum.kTorusSurface:
                                Torus tor = (Torus)face.Geometry;
                                data.radius_cm       = tor.MajorRadius;
                                data.minor_radius_cm = tor.MinorRadius;
                                data.axis = new double[] { tor.AxisVector.X, tor.AxisVector.Y, tor.AxisVector.Z };
                                break;
                        }
                    }
                    catch { }

                    // Loops (B-Rep topology)
                    try
                    {
                        data.loops = new List<LoopData>();
                        foreach (EdgeLoop loop in face.EdgeLoops)
                        {
                            var loopData = new LoopData { is_outer = loop.IsOuterEdgeLoop };
                            foreach (Edge loopEdge in loop.Edges)
                            {
                                try
                                {
                                    byte[] eKey = new byte[0];
                                    loopEdge.GetReferenceKey(ref eKey, ctxData.ContextId);
                                    loopData.edge_reference_keys.Add(ctxData.Manager.KeyToString(eKey));
                                }
                                catch { }
                            }
                            data.loops.Add(loopData);
                        }
                    }
                    catch { }
                }
                // ── EDGE geometry ────────────────────────────────────────────
                else if (nativeObj is Edge)
                {
                    Edge edge = (Edge)nativeObj;
                    data.curve_type = edge.CurveType.ToString();

                    try
                    {
                        CurveEvaluator eval = edge.Evaluator;
                        double s, e;
                        eval.GetParamExtents(out s, out e);
                        double length;
                        eval.GetLengthAtParam(s, e, out length);
                        data.length_cm = length;

                        double midParam;
                        eval.GetParamAtLength(s, length / 2.0, out midParam);
                        double[] arr = { midParam }, pt = new double[3], tan = new double[3];
                        eval.GetPointAtParam(ref arr, ref pt);
                        eval.GetTangent(ref arr, ref tan);
                        data.edge_midpoint       = pt;
                        data.edge_tangent_at_mid = tan;
                    }
                    catch { }

                    try
                    {
                        data.edge_start_vertex = new double[] {
                            edge.StartVertex.Point.X, edge.StartVertex.Point.Y, edge.StartVertex.Point.Z };
                        data.edge_end_vertex = new double[] {
                            edge.StopVertex.Point.X, edge.StopVertex.Point.Y, edge.StopVertex.Point.Z };
                    }
                    catch { }
                }
                // ── WORKPLANE geometry ───────────────────────────────────────
                else if (nativeObj is WorkPlane)
                {
                    WorkPlane wp = (WorkPlane)nativeObj;
                    data.work_feature_name = wp.Name;
                    data.surface_type = "kPlaneSurface";
                    data.area_cm2 = 0.0001;
                    try
                    {
                        Plane pl = (Plane)wp.Plane;
                        data.face_normal_at_center = new double[] { pl.Normal.X, pl.Normal.Y, pl.Normal.Z };
                        data.face_point_at_center  = new double[] { pl.RootPoint.X, pl.RootPoint.Y, pl.RootPoint.Z };
                    }
                    catch { }
                }
                // ── WORKAXIS geometry ────────────────────────────────────────
                else if (nativeObj is WorkAxis)
                {
                    WorkAxis wa = (WorkAxis)nativeObj;
                    data.work_feature_name = wa.Name;
                    data.entity_type = "WorkAxis";
                    try
                    {
                        Line axisLine = (Line)wa.Line;
                        data.axis = new double[] { axisLine.Direction.X, axisLine.Direction.Y, axisLine.Direction.Z };
                        data.face_point_at_center = new double[] { axisLine.RootPoint.X, axisLine.RootPoint.Y, axisLine.RootPoint.Z };
                    }
                    catch { }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"    [Entity Warning] {ex.Message}");
            }

            return data;
        }

        // =====================================================================
        // TRANSFORM EXTRACTION
        // =====================================================================
        static TransformData ExtractTransform(Matrix matrix)
        {
            return new TransformData
            {
                rotation_matrix = new double[][]
                {
                    new double[] { matrix.get_Cell(1,1), matrix.get_Cell(1,2), matrix.get_Cell(1,3) },
                    new double[] { matrix.get_Cell(2,1), matrix.get_Cell(2,2), matrix.get_Cell(2,3) },
                    new double[] { matrix.get_Cell(3,1), matrix.get_Cell(3,2), matrix.get_Cell(3,3) }
                },
                translation_cm = new double[] { matrix.get_Cell(1,4), matrix.get_Cell(2,4), matrix.get_Cell(3,4) }
            };
        }

        // =====================================================================
        // SAFE SERIALIZATION
        // =====================================================================
        static void SafeSerialize(object export, string outputPath)
        {
            string json = null;

            // Attempt 1: standard serialization
            try
            {
                var settings = new JsonSerializerSettings
                {
                    Formatting            = Formatting.Indented,
                    NullValueHandling     = NullValueHandling.Ignore,
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
                        NullValueHandling     = NullValueHandling.Ignore,
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
                string dir = Path.GetDirectoryName(outputPath);
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                File.WriteAllText(outputPath, json, Encoding.UTF8);
                Log($"  ✅ Saved → {outputPath} ({json.Length} chars)");
            }
            catch (Exception ioEx)
            {
                Log($"  [FILE WRITE FAILED] {outputPath}: {ioEx.Message}");
            }
        }
    }

    // =========================================================================
    // DATA STRUCTURES — aligned with geometry_hasher.py + atlas_dataset.py
    // =========================================================================

    public class KeyContextData
    {
        public int ContextId;
        public string ContextString;
        public ReferenceKeyManager Manager;
    }

    public class AssemblyExport
    {
        public AssemblyMetadata assembly_metadata { get; set; }
        public AssemblyPhysics physics { get; set; }
        public AssemblyGraph assembly_graph { get; set; } = new AssemblyGraph();
    }

    public class AssemblyMetadata
    {
        public string assembly_name { get; set; }
        public string full_file_name { get; set; }
        public string internal_name { get; set; }
        public int total_occurrences { get; set; }
        public int total_constraints { get; set; }
    }

    public class AssemblyPhysics
    {
        public double mass_kg { get; set; }
        public double[] center_of_mass { get; set; }
        public double[] inertia_tensor { get; set; }
    }

    public class AssemblyGraph
    {
        public List<ComponentNode> nodes { get; set; }               = new List<ComponentNode>();
        public List<ConstraintEdge> constraint_edges { get; set; }   = new List<ConstraintEdge>();
        public List<JointEdge> joint_edges { get; set; }             = new List<JointEdge>();
    }

    public class ComponentNode
    {
        public string node_id { get; set; }
        public string file_name { get; set; }
        public string component_type { get; set; }
        public bool grounded { get; set; }
        public bool suppressed { get; set; }
        public bool visible { get; set; }
        public bool adaptive { get; set; }
        public bool flexible { get; set; }
        public int dof_translation { get; set; }
        public int dof_rotation { get; set; }
        public bool fully_constrained { get; set; }
        public TransformData transform { get; set; }
    }

    public class TransformData
    {
        public double[][] rotation_matrix { get; set; }
        public double[] translation_cm { get; set; }
    }

    public class ConstraintEdge
    {
        public string constraint_name { get; set; }
        public string constraint_type { get; set; }
        public bool suppressed { get; set; }
        public string health_status { get; set; }
        public string node_one_id { get; set; }
        public string node_two_id { get; set; }
        public double? offset_cm { get; set; }
        public double? angle_rad { get; set; }
        public string solution { get; set; }
        public bool? axes_opposed { get; set; }
        public bool? lock_rotation { get; set; }
        public bool? inside_tangency { get; set; }
        public EntityData entity_one { get; set; }
        public EntityData entity_two { get; set; }
    }

    public class JointEdge
    {
        public string joint_name { get; set; }
        public string joint_type { get; set; }
        public string health_status { get; set; }
        public bool suppressed { get; set; }
        public string node_one_id { get; set; }
        public string node_two_id { get; set; }
        public bool has_linear_limit { get; set; }
        public double? linear_start_cm { get; set; }
        public double? linear_end_cm { get; set; }
        public bool has_angular_limit { get; set; }
        public double? angular_start_rad { get; set; }
        public double? angular_end_rad { get; set; }
    }

    public class EntityData
    {
        public string entity_type { get; set; }
        public string proxy_context_occurrence { get; set; }
        public string owner_document { get; set; }
        public string reference_key_string { get; set; }
        public string context_key_string { get; set; }
        // Face geometry
        public string surface_type { get; set; }
        public double? area_cm2 { get; set; }
        public double[] face_normal_at_center { get; set; }
        public double[] face_point_at_center { get; set; }
        public double[] face_bbox_min { get; set; }
        public double[] face_bbox_max { get; set; }
        public double? radius_cm { get; set; }
        public double? minor_radius_cm { get; set; }
        public double? half_angle_rad { get; set; }
        public double[] axis { get; set; }
        public List<LoopData> loops { get; set; }
        // Edge geometry
        public string curve_type { get; set; }
        public double? length_cm { get; set; }
        public double[] edge_midpoint { get; set; }
        public double[] edge_tangent_at_mid { get; set; }
        public double[] edge_start_vertex { get; set; }
        public double[] edge_end_vertex { get; set; }
        // Work features
        public string work_feature_name { get; set; }
    }

    public class LoopData
    {
        public bool is_outer { get; set; }
        public List<string> edge_reference_keys { get; set; } = new List<string>();
    }
}