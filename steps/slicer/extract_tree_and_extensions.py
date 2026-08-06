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

    # --- 2. ESTRAZIONE AUTOMATICA ENDPOINTS ---
    print(f"[{patient_id}] Estrazione automatica degli endpoints...", flush=True)
    auto_endpoints = extract_boundary_endpoints(surface_polydata)

    # Creazione nodo Markups Fiducial per Slicer
    fiducial_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"Endpoints_{patient_id}")
    if hasattr(fiducial_node, "CreateDefaultDisplayNodes"):
        fiducial_node.CreateDefaultDisplayNodes()
    for pt in auto_endpoints:
        try:
            fiducial_node.AddControlPointWorld(pt)
        except TypeError:
            fiducial_node.AddControlPointWorld(pt[0], pt[1], pt[2])
        except AttributeError:
            fiducial_node.AddControlPoint(pt)

    # Assicura che il modulo ExtractCenterline sia selezionato per la verifica umana
    try:
        slicer.util.selectModule('ExtractCenterline')
        extract_widget = slicer.modules.extractcenterline.widgetRepresentation().self()
        if hasattr(extract_widget, 'show'):
            extract_widget.show()
        if hasattr(extract_widget, 'parameterNode'):
            parameter_node = extract_widget.parameterNode()
            if parameter_node is not None:
                try:
                    parameter_node.SetNodeReferenceID('InputSurface', input_model_node.GetID())
                except Exception:
                    pass
                try:
                    parameter_node.SetNodeReferenceID('EndPoints', fiducial_node.GetID())
                except Exception:
                    pass
    except Exception:
        pass

    # Configurazione vista 3D per l'operatore umano
    slicer.util.resetThreeDViews()
    
    # --- 3. FASE INTERATTIVA (CORREZIONE MANUALE) ---
    print("\n" + "=" * 60, flush=True)
    print(f" INTERVENTO RICHIESTO: Correggi la posizione degli endpoints nella GUI.", flush=True)
    print(f" Una volta terminato, clicca sul pulsante 'CONFERMA E CHIUDI GUI' nella barra in alto.", flush=True)
    print("=" * 60 + "\n", flush=True)

    # Aggiunta pulsante di conferma personalizzato nell'interfaccia Slicer
    dock_widget = qt.QDockWidget("CFD Pipeline - Revisione Endpoints")
    layout = qt.QVBoxLayout()
    button = qt.QPushButton("CONFERMA E CHIUDI GUI")
    button.setStyleSheet("font-weight: bold; background-color: #28a745; color: white; padding: 10px; font-size: 14px;")
    
    # Definizione flag di completamento
    user_finished = {"done": False}
    event_loop = qt.QEventLoop()

    def on_confirm():
        # Salva Markups JSON
        slicer.util.saveNode(fiducial_node, endpoints_json)
        print(f"[{patient_id}] Endpoints salvati in: {endpoints_json}", flush=True)
        user_finished["done"] = True
        event_loop.quit()

    try:
        button.connect("clicked()", on_confirm)
    except Exception:
        button.clicked.connect(on_confirm)

    layout.addWidget(button)
    widget = qt.QWidget()
    widget.setLayout(layout)
    dock_widget.setWidget(widget)
    slicer.util.mainWindow().addDockWidget(qt.Qt.TopDockWidgetArea, dock_widget)
    dock_widget.show()
    slicer.util.selectModule('ExtractCenterline')
    slicer.app.processEvents()

    # Blocco in ascolto finché l'utente non completa l'operazione interattiva
    event_loop.exec_()

    if not user_finished["done"]:
        print(f"[{patient_id}] Chiusura senza conferma. Interruzione pipeline.")
        sys.exit(1)

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

        cap_flow_polydata = clip_logic.clipVessel(
            surfacePolyData=surface_polydata,
            centerlinesNode=centerline_model_node,
            clipPointsMarkupsNode=fiducial_node,
            cap=True,
            addFlowExtensions=True,
            extensionLength=flow_ext_length,
            extensionMode="boundarynormal",
        )

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