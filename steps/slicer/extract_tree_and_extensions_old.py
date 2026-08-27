"""
Script da eseguire all'interno di 3D Slicer.
Uso: Slicer --python-script extract_tree_and_extensions.py --patient-dir /path/to/database/pz001 --flow-ext 25.0
"""

import argparse
import inspect
import math
import os
import sys
from collections import defaultdict, deque

import vtk
import slicer
import qt


def parse_args():
    """Parsa gli argomenti della riga di comando passati da Slicer al workflow."""
    parser = argparse.ArgumentParser(description="Slicer Centerline and Flow Extension Pipeline")
    parser.add_argument("--patient-dir", required=True, help="Path completo alla cartella del paziente (es. database/pz001)")
    parser.add_argument("--flow-ext", type=float, default=25.0, help="Lunghezza flow extension in mm")
    parser.add_argument("--clip-inset", type=float, default=2.0, help="Inset per il clipping (mm)")
    # --- Pruning post-biforcazione: frazione fissa della lunghezza del RAMO DISTALE ---
    parser.add_argument("--prune-frac", type=float, default=0.30,
                        help="Frazione della lunghezza del ramo distale (leaf->biforcazione) da rimuovere dal leaf (0-1)")
    parser.add_argument("--min-keep", type=float, default=4.0,
                        help="Margine minimo da NON avvicinare alla biforcazione: clamp dell'inset (mm)")
    parser.add_argument("--bif-tol", type=float, default=1.0,
                        help="Tolleranza di coincidenza per rilevare la biforcazione (tronco condiviso fra celle) (mm)")
    # --- Annullamento spikes: leaf con raggio locale troppo piccolo ---
    parser.add_argument("--spike-leaf-radius", type=float, default=0.6,
                        help="Sotto questo raggio locale al leaf, l'outlet e' scartato come spike (mm)")
    
    # Rimuove gli argomenti interni passati da Slicer all'avvio
    script_args = sys.argv[1:]
    if "--" in script_args:
        script_args = script_args[script_args.index("--") + 1:]
    
    print(f"[DEBUG] Slicer script argv: {sys.argv}", flush=True)
    print(f"[DEBUG] Slicer parsed args: {script_args}", flush=True)
    return parser.parse_args(script_args)


def extract_boundary_endpoints(surface_polydata):
    """
    Individua i bordi aperti della superficie e ne calcola i centroidi.

    Serve per capire se il modello finale e' chiuso o se sono rimasti fori.
    Ritorna una lista di punti 3D [(x, y, z), ...].
    """
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(surface_polydata)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.Update()

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
    """Deduce il sistema di coordinate leggendo i commenti iniziali del file VTK."""
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
    Trova il punto di ingresso prendendo la sezione piu' alta della superficie.

    L'idea e' tagliare il modello vicino all'estremo superiore e usare il
    centroide della sezione come inlet iniziale per la centerline.
    """
    bounds = surface_polydata.GetBounds()
    z_max = bounds[5]

    apex = None
    n_pts = surface_polydata.GetNumberOfPoints()
    if n_pts == 0:
        raise RuntimeError("compute_inlet_from_top_slice: superficie senza punti.")
    for i in range(n_pts):
        p = surface_polydata.GetPoint(i)
        if apex is None or p[2] > apex[2]:
            apex = p

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

    conn.SetExtractionModeToClosestPointRegion()
    conn.SetClosestPoint(apex[0], apex[1], apex[2])
    conn.Update()

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


def smooth_surface_polydata(surface_polydata, iterations=60, pass_band=0.003):
    """Smussa la superficie per ridurre rumore geometrico prima del clipping."""
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(surface_polydata)
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(120.0)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    return smoother.GetOutput()


def auto_detect_endpoints(logic, preprocessed_polydata, inlet_point, patient_id):
    """
    Ricava automaticamente inlet e outlet dalla rete di centerline.

    Usa un seed sull'inlet per estrarre la rete e poi chiede a Slicer i punti
    terminali della struttura.
    """
    seed_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"__seed_{patient_id}")
    seed_node.CreateDefaultDisplayNodes()
    seed_node.AddControlPoint(vtk.vtkVector3d(inlet_point[0], inlet_point[1], inlet_point[2]))
    seed_node.SetNthControlPointSelected(0, False)

    network_polydata = logic.extractNetwork(preprocessed_polydata, seed_node)

    start_index = logic.startPointIndexFromEndPointsMarkupsNode(seed_node)
    start_position = [0.0, 0.0, 0.0]
    seed_node.GetNthControlPointPosition(start_index, start_position)

    endpoint_positions = logic.getEndPoints(network_polydata, startPointPosition=start_position)

    slicer.mrmlScene.RemoveNode(seed_node)

    if len(endpoint_positions) < 2:
        raise RuntimeError(
            f"[{patient_id}] Auto-detect ha trovato solo {len(endpoint_positions)} endpoint "
            f"(servono inlet + almeno 1 outlet). Verifica preprocess/superficie."
        )

    source_position = list(endpoint_positions[0])
    outlet_positions = [list(p) for p in endpoint_positions[1:]]

    return source_position, outlet_positions



#  togliere "SPIKE" + PRUNING POST-BIFORCAZIONE  (metodo cell-based, stile ClipVessel)
#
#  La centerline di extractCenterline ha una CELLA per outlet: ogni cella e' un
#  path completo inlet(root) -> leaf(outlet), con array "Radius" per-punto.
#  Per prunare "piu' internamente" si cammina dal leaf verso l'interno finche'
#  non si copre una FRAZIONE FISSA della lunghezza della cella (insetFromEndpoint
#  adattato: targetDistance = prune_fraction * cellLength invece di
#  insetFactor * raggio). Il punto raggiunto diventa il nuovo clip point.


def fmt_pt(p):
    """Formatta un punto 3D per i log umani."""
    return f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})"


def fmt_r(r):
    """Formatta il raggio locale di un leaf per i log umani."""
    return "n/a" if r is None else f"{r:.2f}mm"


def get_radius_array(centerlines):
    """Recupera l'array dei raggi della centerline se disponibile."""
    for name in ("Radius", "MaximumInscribedSphereRadius"):
        arr = centerlines.GetPointData().GetArray(name)
        if arr is not None:
            return arr, name
    return None, None


def cell_arc_length(centerlines, cell):
    """Calcola la lunghezza d'arco di una cella della centerline."""
    n = cell.GetNumberOfPoints()
    total = 0.0
    prev = centerlines.GetPoint(cell.GetPointId(0))
    for i in range(1, n):
        cur = centerlines.GetPoint(cell.GetPointId(i))
        total += vtk.vtkMath.Distance2BetweenPoints(prev, cur) ** 0.5
        prev = cur
    return total


def inset_from_endpoint(centerlines, cell, endpoint_offset, step, target_distance):
    """
    Avanza lungo una cella della centerline dal leaf verso l'interno.

    Lo spostamento continua finche' non si raggiunge la distanza richiesta o
    finisce la cella. Ritorna posizione, indice raggiunto, point id e distanza
    effettivamente coperta.
    """
    n = cell.GetNumberOfPoints()
    endpoint_point_id = cell.GetPointId(endpoint_offset)
    accumulated = 0.0
    previous = centerlines.GetPoint(endpoint_point_id)
    reached_point_id = endpoint_point_id
    reached_index = endpoint_offset
    index = endpoint_offset
    while accumulated < target_distance:
        next_index = index + step
        if next_index < 0 or next_index >= n:
            break
        next_point_id = cell.GetPointId(next_index)
        next_position = centerlines.GetPoint(next_point_id)
        accumulated += vtk.vtkMath.Distance2BetweenPoints(previous, next_position) ** 0.5
        previous = next_position
        reached_point_id = next_point_id
        reached_index = next_index
        index = next_index
    return list(centerlines.GetPoint(reached_point_id)), reached_index, reached_point_id, accumulated


def compute_terminuses(centerlines, inlet_point, prune_fraction, radius_array,
                       spike_leaf_radius, min_keep_mm, patient_id, bif_tol=1.0):
    """
    Analizza ogni ramo della centerline e decide dove troncarlo.

    Per ogni cella individua il leaf piu' lontano dall'inlet, scarta gli spike
    troppo sottili, e calcola il nuovo clip point arretrando di una frazione
    della lunghezza del ramo distale. Ritorna la lista degli outlet validi e
    quella dei rami scartati.
    """
    n_cells = centerlines.GetNumberOfCells()
    limits = compute_branch_limits(centerlines, inlet_point, bif_tol)
    outlets = []
    dropped = []
    seen_leaves = []  # posizioni leaf gia' viste (dedup)

    def leaf_radius(pid):
        return None if radius_array is None else float(radius_array.GetValue(pid))

    for cell_index in range(n_cells):
        cell = centerlines.GetCell(cell_index)
        n = cell.GetNumberOfPoints()
        if n < 2:
            continue

        start_id = cell.GetPointId(0)
        end_id = cell.GetPointId(n - 1)
        p_start = centerlines.GetPoint(start_id)
        p_end = centerlines.GetPoint(end_id)
        d_start = vtk.vtkMath.Distance2BetweenPoints(p_start, inlet_point)
        d_end = vtk.vtkMath.Distance2BetweenPoints(p_end, inlet_point)

        # leaf = estremo piu' lontano dall'inlet; step verso l'interno
        if d_end >= d_start:
            leaf_offset, step, leaf_id = n - 1, -1, end_id
        else:
            leaf_offset, step, leaf_id = 0, +1, start_id
        leaf_pos = centerlines.GetPoint(leaf_id)

        # dedup leaf coincidenti (0.0025 mm^2 ~ 0.05 mm)
        if any(vtk.vtkMath.Distance2BetweenPoints(leaf_pos, s) < 0.0025 for s in seen_leaves):
            continue
        seen_leaves.append(leaf_pos)

        r_leaf = leaf_radius(leaf_id)
        cell_len = cell_arc_length(centerlines, cell)
        branch_len = limits[cell_index]["branch_len"]
        has_bif = limits[cell_index]["bif_index"] is not None

        entry = {
            "cell_index": cell_index, "leaf_offset": leaf_offset, "step": step,
            "leaf_pos": list(leaf_pos), "leaf_radius": r_leaf,
            "cell_len": cell_len, "branch_len": branch_len, "has_bif": has_bif,
        }

        # --- DESPIKE: leaf a raggio troppo piccolo = spike ---
        if r_leaf is not None and r_leaf < spike_leaf_radius:
            dropped.append(entry)
            continue

        # --- PRUNING: frazione della lunghezza del RAMO DISTALE, dal leaf ---
        # base = ramo distale se una biforcazione e' stata trovata, altrimenti l'intera cella
        base_len = branch_len if has_bif else cell_len
        target = prune_fraction * base_len
        # guardia biforcazione: non avvicinarsi a meno di min_keep_mm dalla biforcazione
        max_target = max(0.0, base_len - min_keep_mm)
        clamped = False
        if target > max_target:
            target = max_target
            clamped = True

        position, reached_index, reached_id, covered = inset_from_endpoint(
            centerlines, cell, leaf_offset, step, target
        )

        entry.update({
            "position": position, "reached_index": reached_index,
            "reached_id": reached_id, "covered": covered, "clamped": clamped,
        })
        outlets.append(entry)

    return outlets, dropped


def compute_branch_limits(centerlines, inlet_point, bif_tol):
    """
    Misura, per ogni cella, la lunghezza del solo ramo distale fino alla biforcazione.

    La biforcazione viene riconosciuta come il primo punto, camminando dal leaf
    verso l'interno, che coincide con un punto condiviso da un'altra cella.
    """
    n_cells = centerlines.GetNumberOfCells()

    merged = vtk.vtkPoints()
    cell_of_point = []
    for ci in range(n_cells):
        cell = centerlines.GetCell(ci)
        for j in range(cell.GetNumberOfPoints()):
            merged.InsertNextPoint(centerlines.GetPoint(cell.GetPointId(j)))
            cell_of_point.append(ci)

    pd = vtk.vtkPolyData()
    pd.SetPoints(merged)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(pd)
    locator.BuildLocator()
    found = vtk.vtkIdList()

    limits = {}
    for ci in range(n_cells):
        cell = centerlines.GetCell(ci)
        n = cell.GetNumberOfPoints()
        if n < 2:
            limits[ci] = {"branch_len": 0.0, "bif_index": None}
            continue

        p_start = centerlines.GetPoint(cell.GetPointId(0))
        p_end = centerlines.GetPoint(cell.GetPointId(n - 1))
        d_start = vtk.vtkMath.Distance2BetweenPoints(p_start, inlet_point)
        d_end = vtk.vtkMath.Distance2BetweenPoints(p_end, inlet_point)
        # ordine leaf -> root
        order = range(n - 1, -1, -1) if d_end >= d_start else range(0, n)

        branch_len = 0.0
        prev = None
        bif_index = None
        for idx in order:
            pos = centerlines.GetPoint(cell.GetPointId(idx))
            if prev is not None:
                branch_len += vtk.vtkMath.Distance2BetweenPoints(prev, pos) ** 0.5
            prev = pos
            locator.FindPointsWithinRadius(bif_tol, pos, found)
            shared = any(cell_of_point[found.GetId(m)] != ci for m in range(found.GetNumberOfIds()))
            if shared:
                bif_index = idx
                break

        limits[ci] = {"branch_len": branch_len, "bif_index": bif_index}
    return limits


def build_pruned_centerline(centerlines, radius_array, outlets):
    """
    Ricostruisce la centerline finale mantenendo solo i rami validi e troncati.

    Ogni outlet reale conserva il tratto dal lato root fino al clip point
    scelto nella fase di pruning, mentre gli spike vengono esclusi. Se esiste,
    l'array Radius viene preservato.
    """
    out_points = vtk.vtkPoints()
    out_lines = vtk.vtkCellArray()
    out_radius = None
    if radius_array is not None:
        out_radius = vtk.vtkDoubleArray()
        out_radius.SetName("Radius")
        out_radius.SetNumberOfComponents(1)

    for o in outlets:
        cell = centerlines.GetCell(o["cell_index"])
        n = cell.GetNumberOfPoints()
        leaf_offset = o["leaf_offset"]
        reached_index = o["reached_index"]

        # indici tenuti = dal lato root fino al clip point
        if leaf_offset == n - 1:      # leaf in coda, root in testa -> tieni [0 .. reached_index]
            kept_range = range(0, reached_index + 1)
        else:                         # leaf in testa, root in coda -> tieni [reached_index .. n-1]
            kept_range = range(reached_index, n)

        ids = []
        for i in kept_range:
            pid = cell.GetPointId(i)
            new_id = out_points.InsertNextPoint(centerlines.GetPoint(pid))
            ids.append(new_id)
            if out_radius is not None:
                out_radius.InsertNextValue(float(radius_array.GetValue(pid)))

        if len(ids) >= 2:
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(ids))
            for k, nid in enumerate(ids):
                polyline.GetPointIds().SetId(k, nid)
            out_lines.InsertNextCell(polyline)

    out = vtk.vtkPolyData()
    out.SetPoints(out_points)
    out.SetLines(out_lines)
    if out_radius is not None:
        out.GetPointData().AddArray(out_radius)
        out.GetPointData().SetActiveScalars("Radius")
    return out




def run_slicer_pipeline(patient_dir, flow_ext_length,
                        prune_fraction=0.30, spike_leaf_radius=0.6, min_keep_mm=4.0,
                        bif_tol=1.0):
    """Esegue l'intera pipeline Slicer: centerline, pruning, capping e flow extensions."""
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
    input_model_node = slicer.util.loadModel(input_vtk)
    if input_model_node is None:
        raise FileNotFoundError(f"[{patient_id}] Impossibile caricare il modello VTK: {input_vtk}")
    surface_polydata = input_model_node.GetPolyData()
    if surface_polydata is None:
        raise RuntimeError(f"[{patient_id}] Il modello caricato non contiene PolyData.")
    
    original_surface = vtk.vtkPolyData()
    original_surface.DeepCopy(surface_polydata)
    n_open_original = len(extract_boundary_endpoints(original_surface))
    print(f"[{patient_id}] Bordi aperti nella superficie caricata: {n_open_original}", flush=True)

    # --- 2. ESTRAZIONE ENDPOINTS ---
    print(f"[{patient_id}] Calcolo automatico Source Point (Inlet) dalla sezione superiore...", flush=True)
    inlet_point = compute_inlet_from_top_slice(surface_polydata, offset=2.0)
    print(f"[{patient_id}] Inlet: ({inlet_point[0]:.2f}, {inlet_point[1]:.2f}, {inlet_point[2]:.2f})", flush=True)

    from ExtractCenterline import ExtractCenterlineLogic
    logic = ExtractCenterlineLogic()

    print(f"[{patient_id}] Preprocess superficie per estrazione rete (5000 pt, aggr. 4.0)...", flush=True)
    preprocessed_polydata = logic.preprocess(surface_polydata, 5000, 4.0, False)

    print(f"[{patient_id}] Auto-detect endpoints...", flush=True)
    source_position, outlet_positions = auto_detect_endpoints(
        logic, preprocessed_polydata, inlet_point, patient_id
    )

    d_src = sum((a - b) ** 2 for a, b in zip(source_position, inlet_point)) ** 0.5
    print(f"[{patient_id}] Tip di rete piu' vicino all'inlet a {d_src:.2f} mm dal centroide.", flush=True)

    # fiducial GREZZI: usati solo per la PRIMA estrazione centerline.
    # Gli endpoint definitivi (de-spiked + accorciati) sono ricostruiti e salvati dopo l'analisi del grafo.
    fiducial_raw = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"EndpointsRaw_{patient_id}")
    fiducial_raw.CreateDefaultDisplayNodes()

    fiducial_raw.AddControlPoint(vtk.vtkVector3d(inlet_point[0], inlet_point[1], inlet_point[2]))
    fiducial_raw.SetNthControlPointLabel(0, "Inlet")
    fiducial_raw.SetNthControlPointSelected(0, False)

    for k, pt in enumerate(outlet_positions, start=1):
        fiducial_raw.AddControlPoint(vtk.vtkVector3d(pt[0], pt[1], pt[2]))
        idx = fiducial_raw.GetNumberOfControlPoints() - 1
        fiducial_raw.SetNthControlPointLabel(idx, f"Outlet_{k}")
        fiducial_raw.SetNthControlPointSelected(idx, True)

    print(f"[{patient_id}] Candidati grezzi: 1 Inlet + {len(outlet_positions)} outlet (pre despike/pruning).", flush=True)

    # --- 4. ESTRAZIONE CENTERLINES ---
    print(f"[{patient_id}] Calcolo Centerlines e applicazione Flow Extensions ({flow_ext_length} mm)...")

    try:
        from ExtractCenterline import ExtractCenterlineLogic
        logic = ExtractCenterlineLogic()

        if hasattr(logic, 'preprocess'):
            try:
                surface_polydata = logic.preprocess(surface_polydata, 5000, 4.0, False)
            except Exception as e:
                print(f"[{patient_id}] Errore preprocess: {e}", flush=True)

        print(f"[{patient_id}] Usando ExtractCenterlineLogic.extractCenterline", flush=True)
        centerline_polydata, voronoi_diagram = logic.extractCenterline(
            surfacePolyData=surface_polydata,
            endPointsMarkupsNode=fiducial_raw,
            curveSamplingDistance=1.0,
        )

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
        print(f"[{patient_id}] Coordinate system inferred: {coordinate_system_name}", flush=True)

        filler = vtk.vtkFillHolesFilter()

        print(f"[{patient_id}] Smoothing superficie prima del clipping...", flush=True)
        smoothed_surface = smooth_surface_polydata(filler.GetOutput(), iterations=20, pass_band=0.01)

        # --- 4b. DESPIKE + PRUNING POST-BIFORCAZIONE (metodo cell-based ClipVessel) ---
        radius_array, radius_name = get_radius_array(centerline_polydata)
        print(f"[{patient_id}] Analisi celle centerline: despike (raggio-leaf < {spike_leaf_radius}mm) + "
              f"pruning {prune_fraction*100:.0f}% del RAMO DISTALE dal leaf (margine biforcazione {min_keep_mm}mm, "
              f"radius array: {radius_name or 'ASSENTE -> despike disattivato'}).", flush=True)

        outlets, dropped = compute_terminuses(
            centerline_polydata,
            inlet_point=inlet_point,
            prune_fraction=prune_fraction,
            radius_array=radius_array,
            spike_leaf_radius=spike_leaf_radius,
            min_keep_mm=min_keep_mm,
            patient_id=patient_id,
            bif_tol=bif_tol,
        )

        print(f"[{patient_id}] Despike: {len(dropped)} spike scartati, {len(outlets)} outlet reali tenuti.", flush=True)
        for b in dropped:
            print(f"[{patient_id}]   - SPIKE scartato @ {fmt_pt(b['leaf_pos'])} "
                  f"(r_leaf={fmt_r(b['leaf_radius'])}, cell_len={b['cell_len']:.1f}mm)", flush=True)
        for k, b in enumerate(outlets, start=1):
            flag = " [CLAMP margine-bif]" if b.get("clamped") else ""
            bif = f"ramo_distale={b['branch_len']:.1f}mm" if b["has_bif"] else f"NO-bif cell_len={b['cell_len']:.1f}mm"
            print(f"[{patient_id}]   - Outlet_{k} leaf @ {fmt_pt(b['leaf_pos'])} -> clip @ {fmt_pt(b['position'])} "
                  f"(rimossi {b['covered']:.1f}mm; {bif}){flag}", flush=True)

        if len(outlets) < 1:
            raise RuntimeError(f"[{patient_id}] Dopo il despike non resta alcun outlet reale: rivedere spike_leaf_radius.")

        # fiducial DEFINITIVI: inlet invariato + outlet RIPOSIZIONATI al clip point insettato.
        # Clip point piu' interni = ClipVessel tronca il vaso li', cappa e aggancia la flow extension.
        fiducial_clean = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"Endpoints_{patient_id}")
        fiducial_clean.CreateDefaultDisplayNodes()
        fiducial_clean.AddControlPoint(vtk.vtkVector3d(inlet_point[0], inlet_point[1], inlet_point[2]))
        fiducial_clean.SetNthControlPointLabel(0, "Inlet")
        fiducial_clean.SetNthControlPointSelected(0, False)
        for k, b in enumerate(outlets, start=1):
            cp = b["position"]
            fiducial_clean.AddControlPoint(vtk.vtkVector3d(cp[0], cp[1], cp[2]))
            idx = fiducial_clean.GetNumberOfControlPoints() - 1
            fiducial_clean.SetNthControlPointLabel(idx, f"Outlet_{k}")
            fiducial_clean.SetNthControlPointSelected(idx, True)
        if parent_transform is not None:
            fiducial_clean.SetAndObserveTransformNodeID(parent_transform.GetID())

        slicer.util.saveNode(fiducial_clean, endpoints_json)
        print(f"[{patient_id}] Endpoints definitivi (1 Inlet + {len(outlets)} Outlet) salvati in: {endpoints_json}", flush=True)

        # --- tree_model = centerline POTATA (spike esclusi, celle troncate al clip point) ---
        pruned_centerline = build_pruned_centerline(centerline_polydata, radius_array, outlets)
        pruned_model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"CenterlinePruned_{patient_id}")
        pruned_model_node.SetAndObserveMesh(pruned_centerline)
        if parent_transform is not None:
            pruned_model_node.SetAndObserveTransformNodeID(parent_transform.GetID())
        pruned_model_node.CreateDefaultDisplayNodes()

        storage_node = slicer.vtkMRMLModelStorageNode()
        storage_node.SetFileName(output_tree_vtk)
        storage_node.SetCoordinateSystem(coordinate_system)
        success = storage_node.WriteData(pruned_model_node)
        if not success:
            raise RuntimeError(f"[{patient_id}] Impossibile scrivere la centerline potata su {output_tree_vtk}")

        print(f"[{patient_id}] Centerline potata salvata in: {output_tree_vtk}")

        # --- 5. RIPRISTINO, CHIUSURA E CAPPING/FLOW EXTENSION ---
        output_cap_vtk = os.path.join(patient_dir, "lumen_tree_cfd_cap.vtk")
        print(f"[{patient_id}] Generazione modello cap+flow extension in: {output_cap_vtk}", flush=True)

        from ClipVessel import ClipVesselLogic
        clip_logic = ClipVesselLogic()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(original_surface)
        cleaner.Update()

        tri = vtk.vtkTriangleFilter()
        tri.PassLinesOff()
        tri.PassVertsOff()
        tri.SetInputConnection(cleaner.GetOutputPort())
        tri.Update()

        # PASSAGGIO DI RIPRISTINO E CHIUSURA (FillHoles)
        print(f"[{patient_id}] Chiusura e ripristino mesh primordiale...", flush=True)
        #filler = vtk.vtkFillHolesFilter()
        filler.SetInputConnection(tri.GetOutputPort())
        filler.SetHoleSize(100.0)
        filler.Update()

        

        # Ricalcolo Normali
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(smoothed_surface)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.ConsistencyOn()
        normals.SplittingOff()
        normals.Update()

        surface_to_clip = normals.GetOutput()

        # FASE DI CLIPPING CON METODO PLANE ED INSET
        print(f"[{patient_id}] Clip (Method: PLANE, ((((Inset: 2.0x todo yet))))) + flow extensions ({flow_ext_length} mm)...", flush=True)
        
        # Recupero dell'Enum per PLANE da ClipVesselLogic
        clipping_method = getattr(clip_logic, "CLIPPING_METHOD_CENTERLINE_SECTION", 
                  getattr(clip_logic, "CENTERLINE_SECTION", "CenterlineSection"))
        ext_mode = getattr(clip_logic, "CENTERLINE_DIRECTION", getattr(clip_logic, "CENTERLINE_DIRECTION", "CENTERLINE_DIRECTION"))

        if hasattr(clip_logic, "clipInset"):
            clip_logic.clipInset = getattr(args, "clip_inset", 2.0)
        elif hasattr(clip_logic, "setClipInset"):
            clip_logic.setClipInset(getattr(args, "clip_inset", 2.0))

        # NB: centerline PIENA (arriva ai tip) + clip point INTERNI (spostati a valle della biforcazione).
        # E' l'uso nativo di ClipVessel: tronca ogni ramo al clip point, cappa, e aggancia la flow extension.
        clip_kwargs = dict(
            surfacePolyData=surface_to_clip,
            centerlinesNode=centerline_model_node,
            clipPointsMarkupsNode=fiducial_clean,
            clippingMethod=clipping_method,
            cap=True,
            addFlowExtensions=True,
            extensionLength=flow_ext_length,
            extensionMode=ext_mode,
        )
        try:
            cap_flow_polydata = clip_logic.clipVessel(
                **clip_kwargs,
                clipInset=getattr(args, "clip_inset", 2.0),
            )
        except TypeError:
            cap_flow_polydata = clip_logic.clipVessel(**clip_kwargs)

        # Check planarietà
        failures = getattr(clip_logic, "lastPlanarityFailures", None)
        if failures:
            labels = ", ".join(f["label"] for f in failures)
            raise RuntimeError(f"[{patient_id}] Bordi non planari: cap+flow ext SALTATI da ClipVessel: {labels}")

        # Check chiusura mesh finale
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
    run_slicer_pipeline(
        args.patient_dir, args.flow_ext,
        prune_fraction=args.prune_frac,
        spike_leaf_radius=args.spike_leaf_radius,
        min_keep_mm=args.min_keep,
        bif_tol=args.bif_tol,
    )