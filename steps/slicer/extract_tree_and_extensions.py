"""
Script da eseguire all'interno di 3D Slicer.
Uso: Slicer --python-script extract_tree_and_extensions.py --patient-dir /path/to/database/pz001 --flow-ext 25.0
"""

import argparse
import inspect
import os
import sys
import vtk
import slicer
import qt


def parse_args():
    parser = argparse.ArgumentParser(description="Slicer Centerline and Flow Extension Pipeline")
    parser.add_argument("--patient-dir", required=True, help="Path completo alla cartella del paziente (es. database/pz001)")
    parser.add_argument("--flow-ext", type=float, default=25.0, help="Lunghezza flow extension in mm")
    
    # Rimuove gli argomenti interni passati da Slicer all'avvio
    script_args = sys.argv[1:]
    if "--" in script_args:
        script_args = script_args[script_args.index("--") + 1:]
    
    print(f"[DEBUG] Slicer script argv: {sys.argv}", flush=True)
    print(f"[DEBUG] Slicer parsed args: {script_args}", flush=True)
    return parser.parse_args(script_args)


def _find_callable(obj, names):
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return method, name
    return None, None


def _call_compatible(method, **kwargs):
    try:
        sig = inspect.signature(method)
    except (ValueError, TypeError):
        return method(**kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return method(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return method(**filtered)


def _find_callable_by_substrings(obj, substrings):
    for name in dir(obj):
        lower_name = name.lower()
        if all(sub.lower() in lower_name for sub in substrings):
            method = getattr(obj, name, None)
            if callable(method):
                return method, name
    return None, None


def extract_boundary_endpoints(surface_polydata):
    """
    Rileva automaticamente i centri dei bordi aperti del modello VTK (cap/inlet/outlet).
    Ritorna una lista di punti 3D [(x,y,z), ...].
    """
    # 1. Estrazione dei bordi aperti del mesh
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(surface_polydata)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.Update()

    # 2. Separazione delle componenti connesse (ogni cap è un loop chiuso)
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(feature_edges.GetOutput())
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()

    n_regions = connectivity.GetNumberOfExtractedRegions()
    endpoints = []

    for i in range(n_regions):
        connectivity.InitializeSpecifiedRegionList()
        connectivity.AddSpecifiedRegion(i)
        connectivity.SetExtractionModeToSpecifiedRegions()
        connectivity.Update()

        # Calcolo del centro di gravità del loop del cap corrente
        cap_poly = connectivity.GetOutput()
        center = [0.0, 0.0, 0.0]
        n_points = cap_poly.GetNumberOfPoints()
        
        if n_points > 0:
            for j in range(n_points):
                pt = cap_poly.GetPoint(j)
                center[0] += pt[0]
                center[1] += pt[1]
                center[2] += pt[2]
            center[0] /= n_points
            center[1] /= n_points
            center[2] /= n_points
            endpoints.append(center)

    return endpoints


def infer_vtk_coordinate_system(filename):
    """Infer the coordinate system from a VTK file header comment.
    Returns either slicer.vtkMRMLModelStorageNode.CoordinateSystemLPS or
    slicer.vtkMRMLModelStorageNode.CoordinateSystemRAS.
    """
    if not os.path.exists(filename):
        return slicer.vtkMRMLModelStorageNode.CoordinateSystemRAS

    coordinate_system = None
    with open(filename, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("#"):
                break
            upper = line.upper().replace(" ", "")
            if "SPACE=RAS" in upper:
                coordinate_system = slicer.vtkMRMLModelStorageNode.CoordinateSystemRAS
                break
            if "SPACE=LPS" in upper:
                coordinate_system = slicer.vtkMRMLModelStorageNode.CoordinateSystemLPS
                break

    return coordinate_system or slicer.vtkMRMLModelStorageNode.CoordinateSystemRAS

def compute_inlet_from_top_slice(surface_polydata, offset=2.0):
    """
    Trova il centroide della sezione trasversale piu' alta dell'aorta (= inlet).

    Ipotesi anatomica: l'aorta prossimale (inlet) e' il punto piu' craniale del modello,
    quindi ha Z massima nel frame di Slicer (sia RAS che LPS hanno Superior come 3o asse).

    Robustezza (fail-loud, coerente con la pipeline):
      - verifica che il taglio non sia vuoto (altrimenti vtkCenterOfMass restituirebbe
        silenziosamente (0,0,0), cioe' un inlet completamente sbagliato);
      - se il piano interseca PIU' di una sezione (es. un ramo o l'arco che risale
        e ricade a quella quota), tiene SOLO la componente connessa piu' vicina
        all'apice, cosi' il centroide non cade a meta' strada tra due lumi;
      - compatta la sezione con vtkCleanPolyData PRIMA del centroide: il
        connectivity filter in output conserva TUTTI i punti in input, quindi senza
        clean vtkCenterOfMass medierebbe anche i punti delle altre sezioni.
    """
    bounds = surface_polydata.GetBounds()
    z_max = bounds[5]

    # Apice = vertice della superficie con Z massima (serve a disambiguare la sezione)
    apex = None
    n_pts = surface_polydata.GetNumberOfPoints()
    if n_pts == 0:
        raise RuntimeError("compute_inlet_from_top_slice: superficie senza punti.")
    for i in range(n_pts):
        p = surface_polydata.GetPoint(i)
        if apex is None or p[2] > apex[2]:
            apex = p

    # Piano di taglio orizzontale a 'offset' mm sotto l'apice
    plane = vtk.vtkPlane()
    plane.SetOrigin(0.0, 0.0, z_max - offset)
    plane.SetNormal(0.0, 0.0, 1.0)

    cutter = vtk.vtkCutter()
    cutter.SetInputData(surface_polydata)
    cutter.SetCutFunction(plane)
    cutter.Update()

    if cutter.GetOutput().GetNumberOfPoints() == 0:
        raise RuntimeError(
            f"compute_inlet_from_top_slice: taglio vuoto a z={z_max - offset:.3f} "
            f"(z_max={z_max:.3f}, offset={offset}). Regola --> offset."
        )

    # Componenti connesse del contorno di taglio
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputConnection(cutter.GetOutputPort())
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    n_regions = conn.GetNumberOfExtractedRegions()
    if n_regions == 0:
        raise RuntimeError("compute_inlet_from_top_slice: nessuna sezione dal taglio superiore.")
    if n_regions > 1:
        print(f"[WARN] compute_inlet_from_top_slice: {n_regions} sezioni alla quota superiore; "
              f"seleziono quella connessa all'apice.", flush=True)

    # Tieni SOLO la sezione connessa piu' vicina all'apice
    conn.SetExtractionModeToClosestPointRegion()
    conn.SetClosestPoint(apex[0], apex[1], apex[2])
    conn.Update()

    # Compatta: rimuove i punti non usati -> centroide calcolato SOLO su questa sezione
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(conn.GetOutputPort())
    cleaner.Update()
    section = cleaner.GetOutput()
    if section.GetNumberOfPoints() == 0:
        raise RuntimeError("compute_inlet_from_top_slice: sezione vuota dopo la connettivita'.")

    center_of_mass = vtk.vtkCenterOfMass()
    center_of_mass.SetInputData(section)
    center_of_mass.SetUseScalarsAsWeights(False)
    center_of_mass.Update()

    return list(center_of_mass.GetCenter())


def auto_detect_endpoints(logic, preprocessed_polydata, inlet_point, patient_id):
    """
    Replica il pulsante 'Auto-detect' della GUI Extract Centerline (onAutoDetectEndPoints),
    ma usando l'inlet calcolato come punto SORGENTE (seed) invece di un angolo arbitrario.

    Il flusso reale della GUI e' in DUE passi (verificato sul sorgente VMTK):
        network = logic.extractNetwork(preprocessed, endPointsNode)
        positions = logic.getEndPoints(network, startPointPosition=...)
    extractNetwork NON popola il nodo con gli endpoint: restituisce solo lo scheletro
    e legge dal nodo l'eventuale punto sorgente. Gli endpoint si ottengono con getEndPoints.

    Ritorna (source_position, outlet_positions):
      - source_position: endpoint della rete piu' vicino all'inlet (tip dell'aorta prossimale)
      - outlet_positions: tutti gli altri endpoint della rete (i cap di uscita)
    """
    # Nodo temporaneo con il solo inlet come SORGENTE.
    # Convenzione Slicer: control point DESELEZIONATO = start point / source.
    seed_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"__seed_{patient_id}")
    seed_node.CreateDefaultDisplayNodes()
    seed_node.AddControlPoint(vtk.vtkVector3d(inlet_point[0], inlet_point[1], inlet_point[2]))
    seed_node.SetNthControlPointSelected(0, False)

    # 1) Estrazione rete (come extractNetwork del bottone): apre un foro alla sorgente e skeletonizza
    network_polydata = logic.extractNetwork(preprocessed_polydata, seed_node)

    # 2) Recupera la posizione della sorgente dal nodo (come fa la GUI)
    start_index = logic.startPointIndexFromEndPointsMarkupsNode(seed_node)
    start_position = [0.0, 0.0, 0.0]
    seed_node.GetNthControlPointPosition(start_index, start_position)

    # 3) getEndPoints ordina la SORGENTE per prima (endpoint di rete piu' vicino a start_position)
    endpoint_positions = logic.getEndPoints(network_polydata, startPointPosition=start_position)

    slicer.mrmlScene.RemoveNode(seed_node)

    if len(endpoint_positions) < 2:
        raise RuntimeError(
            f"[{patient_id}] Auto-detect ha trovato solo {len(endpoint_positions)} endpoint "
            f"(servono inlet + almeno 1 outlet). Verifica preprocess/superficie."
        )

    source_position = list(endpoint_positions[0])           # tip vicino all'inlet -> lo scartiamo
    outlet_positions = [list(p) for p in endpoint_positions[1:]]  # gli altri = outlet

    return source_position, outlet_positions

def run_slicer_pipeline(patient_dir, flow_ext_length):
    patient_dir = os.path.abspath(patient_dir)
    patient_id = os.path.basename(os.path.normpath(patient_dir))
    input_vtk = os.path.join(patient_dir, "lumen_tree_cfd.vtk")
    endpoints_json = os.path.join(patient_dir, f"Endpoints_{patient_id}.mrk.json")
    output_tree_vtk = os.path.join(patient_dir, f"tree_model_{patient_id}.vtk")

    print(f"[{patient_id}] patient_dir resolved to: {patient_dir}", flush=True)
    print(f"[{patient_id}] input VTK resolved to: {input_vtk}", flush=True)

    if not os.path.exists(input_vtk):
        raise FileNotFoundError(f"File di input non trovato: {input_vtk}")

    # --- 1. CARICAMENTO MODELLO ---
    print(f"[{patient_id}] Caricamento modello superficie: {input_vtk}", flush=True)
    print(f"[{patient_id}] Esiste il file? {os.path.exists(input_vtk)}", flush=True)
    input_model_node = slicer.util.loadModel(input_vtk)
    if input_model_node is None:
        raise FileNotFoundError(f"[{patient_id}] Impossibile caricare il modello VTK: {input_vtk}")
    surface_polydata = input_model_node.GetPolyData()
    if surface_polydata is None:
        raise RuntimeError(f"[{patient_id}] Il modello caricato non contiene PolyData.")
    # Snapshot pristino della superficie APERTA, prima di qualsiasi azione GUI/preprocessing.
    # input_model_node viene modificato/cappato durante la sessione interattiva
    # (le due 'can't reconstruct new profile' nel log vengono da lì).
    original_surface = vtk.vtkPolyData()
    original_surface.DeepCopy(surface_polydata)
    n_open_original = len(extract_boundary_endpoints(original_surface))
    print(f"[{patient_id}] Bordi aperti nella superficie caricata: {n_open_original}", flush=True)
    

# --- 2. ESTRAZIONE ENDPOINTS (equivalente al bottone 'Auto-detect' della GUI) ---
    print(f"[{patient_id}] Calcolo automatico Source Point (Inlet) dalla sezione superiore...", flush=True)
    inlet_point = compute_inlet_from_top_slice(surface_polydata, offset=2.0)
    print(f"[{patient_id}] Inlet (centroide sezione piu' alta): "
          f"({inlet_point[0]:.2f}, {inlet_point[1]:.2f}, {inlet_point[2]:.2f})", flush=True)

    from ExtractCenterline import ExtractCenterlineLogic
    logic = ExtractCenterlineLogic()

    # Preprocess IDENTICO alla GUI (decimazione ~5000 pt, clean, triangolazione, normali).
    # extractNetwork/getEndPoints DEVONO girare sulla superficie preprocessata, non sulla raw:
    # sulla mesh piena la network extraction e' lentissima/instabile.
    print(f"[{patient_id}] Preprocess superficie per estrazione rete (5000 pt, aggr. 4.0)...", flush=True)
    preprocessed_polydata = logic.preprocess(surface_polydata, 5000, 4.0, False)

    # Auto-detect reale: extractNetwork + getEndPoints, con l'inlet come seed sorgente.
    print(f"[{patient_id}] Auto-detect endpoints (extractNetwork + getEndPoints)...", flush=True)
    source_position, outlet_positions = auto_detect_endpoints(
        logic, preprocessed_polydata, inlet_point, patient_id
    )

    # Sanity check: quanto dista il tip di rete abbinato dall'inlet calcolato?
    d_src = sum((a - b) ** 2 for a, b in zip(source_position, inlet_point)) ** 0.5
    print(f"[{patient_id}] Tip di rete piu' vicino all'inlet a {d_src:.2f} mm dal centroide.", flush=True)
    if d_src > 15.0:
        print(f"[WARN] [{patient_id}] Il tip inlet della rete e' lontano dal centroide "
              f"({d_src:.2f} mm): controlla che la sezione superiore sia davvero l'inlet.", flush=True)

    # Nodo finale: indice 0 = Inlet (centroide ESATTO, deselezionato = SORGENTE), poi gli Outlet.
    fiducial_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"Endpoints_{patient_id}")
    fiducial_node.CreateDefaultDisplayNodes()

    fiducial_node.AddControlPoint(vtk.vtkVector3d(inlet_point[0], inlet_point[1], inlet_point[2]))
    fiducial_node.SetNthControlPointLabel(0, "Inlet")
    fiducial_node.SetNthControlPointSelected(0, False)   # deselezionato => source per extractCenterline

    for k, pt in enumerate(outlet_positions, start=1):
        fiducial_node.AddControlPoint(vtk.vtkVector3d(pt[0], pt[1], pt[2]))
        idx = fiducial_node.GetNumberOfControlPoints() - 1
        fiducial_node.SetNthControlPointLabel(idx, f"Outlet_{k}")
        fiducial_node.SetNthControlPointSelected(idx, True)  # selezionato => target

    print(f"[{patient_id}] Completato: 1 Inlet + {len(outlet_positions)} Outlets identificati.", flush=True)
    # --- 3. SALVATAGGIO AUTOMATICO SENZA INTERVENTO UMANO ---
    slicer.util.saveNode(fiducial_node, endpoints_json)
    print(f"[{patient_id}] Endpoints generati e salvati automaticamente in: {endpoints_json}", flush=True)

    # --- 4. ESTRAZIONE CENTERLINES & FLOW EXTENSIONS ---
    print(f"[{patient_id}] Calcolo Centerlines e applicazione Flow Extensions ({flow_ext_length} mm)...")

    try:
        from ExtractCenterline import ExtractCenterlineLogic
        logic = ExtractCenterlineLogic()

        # Nodi di destinazione per VMTK / ExtractCenterline
        centerline_curve_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "CenterlineCurve")
        centerline_poly_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "CenterlinePoly")

        # Preprocessa la superficie se possibile
        if hasattr(logic, 'preprocess'):
            print(f"[{patient_id}] Usando ExtractCenterlineLogic.preprocess", flush=True)
            try:
                surface_polydata = logic.preprocess(surface_polydata, 5000, 4.0, False)
            except Exception as e:
                print(f"[{patient_id}] Errore preprocess: {e}", flush=True)

        # Estrazione centerline
        if not hasattr(logic, 'extractCenterline'):
            raise AttributeError(
                f"ExtractCenterlineLogic has no extractCenterline method. Available methods: {', '.join([name for name in dir(logic) if callable(getattr(logic, name))])}"
            )

        print(f"[{patient_id}] Usando ExtractCenterlineLogic.extractCenterline", flush=True)
        centerline_polydata, voronoi_diagram = logic.extractCenterline(
            surfacePolyData=surface_polydata,
            endPointsMarkupsNode=fiducial_node,
            curveSamplingDistance=1.0,
        )

        # Salvataggio del modello centerline come VTK con informazioni di sistema di coordinate
        centerline_model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"Centerline_{patient_id}")
        centerline_model_node.SetAndObserveMesh(centerline_polydata)
        parent_transform = None
        if hasattr(input_model_node, 'GetParentTransformNode'):
            parent_transform = input_model_node.GetParentTransformNode()
        if parent_transform is not None:
            centerline_model_node.SetAndObserveTransformNodeID(parent_transform.GetID())
        centerline_model_node.CreateDefaultDisplayNodes()
        centerline_model_node.GetDisplayNode().SetColor(1.0, 0.0, 0.0)

        coordinate_system = infer_vtk_coordinate_system(input_vtk)
        coordinate_system_name = slicer.vtkMRMLModelStorageNode().GetCoordinateSystemAsString(coordinate_system)
        print(f"[{patient_id}] Coordinate system inferred from input VTK: {coordinate_system_name}", flush=True)

        storage_node = slicer.vtkMRMLModelStorageNode()
        storage_node.SetFileName(output_tree_vtk)
        storage_node.SetCoordinateSystem(coordinate_system)
        success = storage_node.WriteData(centerline_model_node)
        if not success:
            raise RuntimeError(f"[{patient_id}] Impossibile scrivere il modello centerline su {output_tree_vtk}")

        print(f"[{patient_id}] Modello finale salvato in: {output_tree_vtk}")

        # --- 5. FLOW EXTENSIONS AND CAPPED MODEL ---
        output_cap_vtk = os.path.join(patient_dir, "lumen_tree_cfd_cap.vtk")
        print(f"[{patient_id}] Generazione modello cap+flow extension in: {output_cap_vtk}", flush=True)

        from ClipVessel import ClipVesselLogic
        clip_logic = ClipVesselLogic()

        # lumen_tree_cfd.vtk e' CHIUSA (da segmentazione): il clip APRE i cap agli endpoint,
        # poi estende, poi ri-cappa. Per questo serve clipVessel, non extendVessel diretto.
        cleaner = vtk.vtkCleanPolyData(); cleaner.SetInputData(original_surface); cleaner.Update()
        tri = vtk.vtkTriangleFilter(); tri.PassLinesOff(); tri.PassVertsOff()
        tri.SetInputConnection(cleaner.GetOutputPort()); tri.Update()
        surface_to_clip = tri.GetOutput()

        print(f"[{patient_id}] Clip + flow extensions ({flow_ext_length} mm, BOUNDARY_NORMAL)...", flush=True)
        cap_flow_polydata = clip_logic.clipVessel(
            surfacePolyData=surface_to_clip,
            centerlinesNode=centerline_model_node,
            clipPointsMarkupsNode=fiducial_node,
            cap=True,
            addFlowExtensions=True,
            extensionLength=flow_ext_length,
            extensionMode="BOUNDARY_NORMAL",   # MAIUSCOLO: "boundarynormal" veniva ignorato
        )

        # fail-loud 1: clipVessel disattiva SILENZIOSAMENTE cap+extension sui bordi non planari
        failures = getattr(clip_logic, "lastPlanarityFailures", None)
        if failures:
            labels = ", ".join(f["label"] for f in failures)
            raise RuntimeError(f"[{patient_id}] Bordi non planari: cap+flow ext SALTATI da ClipVessel: {labels}")

        # fail-loud 2: dopo il capping finale il modello deve essere chiuso
        n_open_after = len(extract_boundary_endpoints(cap_flow_polydata))
        if n_open_after != 0:
            raise RuntimeError(f"[{patient_id}] Modello finale con {n_open_after} bordi aperti: capping non riuscito.")
        
        cap_model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"CapSurface_{patient_id}")
        cap_model_node.SetAndObserveMesh(cap_flow_polydata)
        if parent_transform is not None:
            cap_model_node.SetAndObserveTransformNodeID(parent_transform.GetID())
        cap_model_node.CreateDefaultDisplayNodes()
        cap_model_node.GetDisplayNode().SetColor(0.0, 0.0, 1.0)

        cap_storage_node = slicer.vtkMRMLModelStorageNode()
        cap_storage_node.SetFileName(output_cap_vtk)
        cap_storage_node.SetCoordinateSystem(coordinate_system)
        success = cap_storage_node.WriteData(cap_model_node)
        if not success:
            raise RuntimeError(f"[{patient_id}] Impossibile scrivere il modello cap+flow extension su {output_cap_vtk}")

        print(f"[{patient_id}] Modello cap+flow extension salvato in: {output_cap_vtk}")

    except ImportError:
        print("ERRORE: Modulo SlicerExtension-VMTK non trovato in 3D Slicer.")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    run_slicer_pipeline(args.patient_dir, args.flow_ext)