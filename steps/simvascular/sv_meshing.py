"""sv_meshing.py  (World-2, gira dentro l'interprete SimVascular)

    input : --input   modello .vtp del progetto (Models/<name>.vtp).
            --out-dir cartella del paziente (contiene faces_named.json).
    output: <proj>/Meshes/<name>.vtu               mesh di volume (TetGen).
            <proj>/Meshes/<name>_surface.vtp       superficie esterna.
            <proj>/project_files.xml                 aggiornato per la GUI SimVascular.
            <out-dir>/mesh_info.json                 SSOT del meshing (checkpoint per-passaggio).
            <out-dir>/sv_meshing.log                 log persistente + output NATIVO TetGen/MMG.

    Note di robustezza:
      * L'output nativo di TetGen/MMG (C stdio) NON passa da sys.stdout: quando lo
        stdout e' rediretto su file diventa buffered-a-blocchi e un segfault nativo
        lo perde. Per questo attorno a generate_mesh usiamo _FdCapture (os.dup2) che
        rediroga i file descriptor 1/2 su un file line-buffered -> le tracce native
        sopravvivono anche a un crash.
      * Ogni passaggio scrive un checkpoint atomico (temp + os.replace) su disco,
        cosi' lo stato su disco riflette sempre l'ultimo passaggio riuscito.
"""

import argparse
import contextlib
import datetime
import json
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import vtk
import sv

FACES_NAMED = "faces_named.json"
VTK_TRIANGLE = 5

# ---------------------------------------------------------------------------
# Logging persistente (flush riga-per-riga -> sopravvive a crash Python)
# ---------------------------------------------------------------------------
_LOG_FH = None
_LOG_PATH = None


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def init_logging(out_dir):
    """Apre il file di log persistente. line-buffered per non perdere righe."""
    global _LOG_FH, _LOG_PATH
    _LOG_PATH = os.path.join(out_dir, "sv_meshing.log")
    _LOG_FH = open(_LOG_PATH, "a", buffering=1)  # buffering=1 -> line-buffered
    _log("=" * 78)
    _log("AVVIO sv_meshing  pid={}".format(os.getpid()))


def _log(msg, level="INFO"):
    """Scrive su stdout E sul file di log, flushando subito entrambi."""
    line = "[{}][{}] {}".format(_ts(), level, msg)
    try:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except Exception:
        pass
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()
        except Exception:
            pass


def _hr(title=""):
    _log("-" * 78)
    if title:
        _log(title)


@contextlib.contextmanager
def step(name):
    """Delimita un passaggio nel log con timer e traceback automatico su errore."""
    _log(">>> START : {}".format(name))
    t0 = time.time()
    try:
        yield
    except BaseException:
        _log("!!! FAIL  : {} dopo {:.2f}s".format(name, time.time() - t0), "ERROR")
        _log(traceback.format_exc(), "ERROR")
        raise
    else:
        _log("<<< DONE  : {} ({:.2f}s)".format(name, time.time() - t0))


class _FdCapture(object):
    """Redirige i file descriptor C 1/2 (stdout/stderr nativi) sul file di log.

    Serve a catturare l'output di TetGen/MMG, che scrivono in C e NON passano
    da sys.stdout. Il file e' line-buffered: cio' che viene stampato prima di
    un eventuale segfault resta su disco.
    """

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved_out = os.dup(1)
        self._saved_err = os.dup(2)
        self._fh = open(self.path, "a", buffering=1)
        self._fh.write("[{}] --- inizio cattura output NATIVO TetGen/MMG ---\n".format(_ts()))
        self._fh.flush()
        os.dup2(self._fh.fileno(), 1)
        os.dup2(self._fh.fileno(), 2)
        return self

    def __exit__(self, *exc):
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(self._saved_out, 1)
        os.dup2(self._saved_err, 2)
        os.close(self._saved_out)
        os.close(self._saved_err)
        try:
            self._fh.write("[{}] --- fine cattura output NATIVO ---\n".format(_ts()))
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
        except Exception:
            pass
        return False  # non sopprime eccezioni


# ---------------------------------------------------------------------------
# Helper generici
# ---------------------------------------------------------------------------
def _die(msg):
    _log("ERRORE FATALE: {}".format(msg), "ERROR")
    return 1


def _str2bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _atomic_write_json(path, obj):
    """Scrittura atomica (temp + os.replace) per non lasciare mai file monchi."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def checkpoint(out_dir, meta, stage):
    """Persiste mesh_info.json dopo un passaggio. Aggiorna 'stage' e timestamp."""
    snap = dict(meta)
    snap["stage"] = stage
    snap["ts"] = _ts()
    _atomic_write_json(os.path.join(out_dir, "mesh_info.json"), snap)
    _log("checkpoint -> mesh_info.json (stage='{}')".format(stage))


def parse_args():
    p = argparse.ArgumentParser(description="SimVascular: genera la mesh di volume del modello.")
    p.add_argument("--input", required=True, help="modello .vtp nel progetto (Models/<name>.vtp).")
    p.add_argument("--out-dir", required=True, help="cartella del paziente (contiene faces_named.json).")
    p.add_argument("--gmes", type=float, default=0.25, help="global max edge size (cm).")
    p.add_argument("--bl", default="true", help="boundary layer on/off (true/false).")
    p.add_argument("--portion-edge-size", type=float, default=0.5,
                   help="edge_size_fraction iniziale del boundary layer.")
    p.add_argument("--bl-num-layers", type=int, default=2, help="numero di layer del BL.")
    p.add_argument("--bl-decreasing-ratio", type=float, default=0.6,
                   help="rapporto di decrescita tra layer successivi.")
    p.add_argument("--step", type=float, default=0.10, help="decremento di portion a ogni retry.")
    p.add_argument("--floor", type=float, default=0.10, help="portion minima ammessa (inclusa).")
    p.add_argument("--mesh-name", default=None, help="nome mesh in Meshes/ (default = nome modello).")
    p.add_argument("--probe", action="store_true",
                   help="carica il modello, ispeziona superficie e facce, NON meshia.")

    # Remesh manuale della superficie del modello (MMG). DEFAULT OFF:
    # il remesh MMG standalone normalizza a unit-box su metrica degenere e
    # droppa i ModelFaceID -> lasciamo che TetGen faccia il remesh interno via gmes.
    p.add_argument("--remesh", default="false",
                   help="remesh MMG superficie PRIMA del volume (true/false). SCONSIGLIATO.")
    p.add_argument("--remesh-hmin", type=float, default=None, help="hmin remesh (default = gmes/5).")
    p.add_argument("--remesh-hmax", type=float, default=None, help="hmax remesh (default = gmes).")
    p.add_argument("--remesh-hausd", type=float, default=0.01, help="tolleranza hausdorff remesh (cm).")
    p.add_argument("--remesh-angle", type=float, default=50.0, help="angolo conservazione spigoli remesh.")

    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    return p.parse_args(argv)


def env_report(args, in_path, out_dir):
    """Dump ambiente: versioni, path, esistenza file. Prima traccia utile nel log."""
    _hr("AMBIENTE")
    _log("python   : {}".format(sys.version.split()[0]))
    try:
        _log("vtk      : {}".format(vtk.vtkVersion().GetVTKVersion()))
    except Exception as e:
        _log("vtk      : (versione non leggibile: {})".format(e))
    for attr in ("__version__", "version"):
        if hasattr(sv, attr):
            _log("sv       : {}".format(getattr(sv, attr)))
            break
    _log("cwd      : {}".format(os.getcwd()))
    _log("input    : {}  (exists={})".format(in_path, os.path.exists(in_path)))
    _log("out-dir  : {}  (exists={})".format(out_dir, os.path.isdir(out_dir)))
    _log("args     : {}".format(vars(args)))


def inspect_surface(path):
    """Pre-flight sulla superficie .vtp: la fotografa PRIMA del meshing cosi' il
    log contiene la diagnosi anche se generate_mesh muore in nativo.

    Fail-loud sui blocchi certi: NaN nei punti, celle non-triangolari,
    ModelFaceID assente. Warn (non fatale) su unit-box, celle degeneri,
    bordi aperti, regioni multiple.
    """
    _hr("PRE-FLIGHT SUPERFICIE: {}".format(path))
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    pd = reader.GetOutput()

    n_pts = pd.GetNumberOfPoints()
    n_cells = pd.GetNumberOfCells()
    b = pd.GetBounds()  # (xmin,xmax,ymin,ymax,zmin,zmax)
    dx, dy, dz = b[1] - b[0], b[3] - b[2], b[5] - b[4]
    max_axis = max(dx, dy, dz)
    _log("punti={}  celle={}".format(n_pts, n_cells))
    _log("bounds x[{:.4g},{:.4g}] y[{:.4g},{:.4g}] z[{:.4g},{:.4g}]".format(*b))
    _log("delta  dx={:.4g} dy={:.4g} dz={:.4g}  (asse max={:.4g})".format(dx, dy, dz, max_axis))

    info = {"n_points": int(n_pts), "n_cells": int(n_cells),
            "bounds": [float(x) for x in b], "max_axis": float(max_axis)}

    if n_pts == 0 or n_cells == 0:
        raise RuntimeError("Superficie vuota (punti={}, celle={}).".format(n_pts, n_cells))

    # --- firma unit-box (mesh MMG non un-scalata a monte) ---
    near0 = (abs(b[0]) < 1e-3 and abs(b[2]) < 1e-3 and abs(b[4]) < 1e-3)
    if near0 and 0.9 < max_axis < 1.1:
        _log("WARNING: firma UNIT-BOX (min~0, asse max~1). Probabile output MMG "
             "normalizzato NON un-scalato a monte -> unita' corrotte.", "WARN")
        info["unit_box_suspected"] = True

    # --- NaN / Inf nei punti ---
    nan_pts = 0
    try:
        import numpy as _np
        from vtk.util.numpy_support import vtk_to_numpy
        arr = vtk_to_numpy(pd.GetPoints().GetData())
        nan_pts = int(_np.count_nonzero(~_np.isfinite(arr)))
    except Exception:
        # fallback senza numpy: controlla solo i bounds
        if any(not (x == x) or x in (float("inf"), float("-inf")) for x in b):
            nan_pts = -1  # ignoto ma bounds gia' non-finiti
    _log("coordinate non-finite (NaN/Inf): {}".format(nan_pts))
    if nan_pts != 0:
        raise RuntimeError("Superficie contiene coordinate non-finite -> MMG dara' "
                           "'wrong metric'. Ripulire a monte.")

    # --- tipi di cella: tutte triangoli? ---
    ct = vtk.vtkCellTypes()
    pd.GetCellTypes(ct)
    types = [ct.GetCellType(i) for i in range(ct.GetNumberOfTypes())]
    _log("tipi di cella presenti: {}".format(types))
    non_tri = [t for t in types if t != VTK_TRIANGLE]
    if non_tri:
        raise RuntimeError("Celle non-triangolari presenti {} -> triangolare a monte "
                           "(TetGen/MMG richiedono triangoli).".format(non_tri))

    # --- ModelFaceID ---
    fid = pd.GetCellData().GetArray("ModelFaceID")
    if fid is None:
        raise RuntimeError("Array 'ModelFaceID' assente: modello non partizionato "
                           "(rieseguire sv_apply).")
    rng = fid.GetRange()
    ids = set(int(fid.GetValue(i)) for i in range(fid.GetNumberOfTuples()))
    _log("ModelFaceID: range=({:.0f},{:.0f})  id_distinti={}".format(rng[0], rng[1], sorted(ids)))
    info["face_ids"] = sorted(ids)

    # --- celle degeneri (area ~0): innesco tipico del 'wrong metric' ---
    try:
        sz = vtk.vtkCellSizeFilter()
        sz.SetInputData(pd)
        sz.ComputeAreaOn()
        sz.ComputeVolumeOff()
        sz.ComputeLengthOff()
        sz.ComputeVertexCountOff()
        sz.Update()
        areas = sz.GetOutput().GetCellData().GetArray("Area")
        if areas is not None:
            amin, amax = areas.GetRange()
            n_deg = sum(1 for i in range(areas.GetNumberOfTuples())
                        if areas.GetValue(i) <= 1e-9)
            _log("area celle: min={:.3e} max={:.3e}  degeneri(<=1e-9)={}".format(amin, amax, n_deg))
            info["n_degenerate_cells"] = int(n_deg)
            if n_deg > 0:
                _log("WARNING: {} celle degeneri -> sizing NaN/neg in MMG possibile "
                     "('wrong metric'). Filtrare micro-cap a monte.".format(n_deg), "WARN")
    except Exception as e:
        _log("area check saltato: {}".format(e), "WARN")

    # --- bordi aperti / non-manifold ---
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    n_open = fe.GetOutput().GetNumberOfCells()
    _log("bordi aperti/non-manifold: {}".format(n_open))
    info["n_open_edges"] = int(n_open)
    if n_open > 0:
        _log("WARNING: superficie non chiusa/non-manifold ({} spigoli).".format(n_open), "WARN")

    # --- numero di regioni connesse (atteso 1) ---
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(pd)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    n_reg = conn.GetNumberOfExtractedRegions()
    _log("regioni connesse: {}".format(n_reg))
    info["n_regions"] = int(n_reg)
    if n_reg != 1:
        _log("WARNING: {} regioni connesse (attesa 1).".format(n_reg), "WARN")

    _log("pre-flight OK (blocchi certi non rilevati).")
    return info


def load_wall_id(out_dir):
    """wall_id dalla SSOT scritta da sv_apply. Fail loud se assente/incoerente."""
    path = os.path.join(out_dir, FACES_NAMED)
    if not os.path.exists(path):
        raise FileNotFoundError("{} non trovato (esegui prima sv_apply).".format(path))
    data = json.loads(open(path).read())
    wall_id = data.get("wall_id")
    if wall_id is None:
        raise ValueError("{}: manca 'wall_id'.".format(path))
    return int(wall_id), data


def remesh_model_surface(model_vtp_path, hmin, hmax, hausd, angle):
    """[SCONSIGLIATO] Rimaglia la superficie con MMG. Puo' normalizzare a unit-box
    e droppare i ModelFaceID: usare solo con piena consapevolezza."""
    _log("WARNING: remesh MMG manuale ATTIVO -> rischio unit-box + perdita ModelFaceID.", "WARN")
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(model_vtp_path)
    reader.Update()
    surface = reader.GetOutput()

    if not surface.GetCellData().HasArray("ModelFaceID"):
        raise RuntimeError("remesh: array 'ModelFaceID' assente sul modello.")

    if hmin >= hmax:
        hmin = max(0.01, hmax / 5.0)
        _log("hmin corretto a {} per proteggere le giunzioni.".format(hmin))

    with _FdCapture(_LOG_PATH):
        remeshed = sv.mesh_utils.remesh(
            surface=surface, hmin=hmin, hmax=hmax,
            angle=angle, hgrad=1.1, hausd=hausd,
        )

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(remeshed)
    cleaner.PointMergingOn()
    cleaner.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(cleaner.GetOutput())
    tri.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(model_vtp_path)
    writer.SetInputData(tri.GetOutput())
    writer.Write()
    _log("remesh + pulizia completati (modello sovrascritto).")


def portion_sequence(start, step_, floor):
    """[start, start-step, ..., >= floor]. Almeno [floor] se start < floor."""
    seq = []
    p = round(float(start), 4)
    step_ = round(float(step_), 4)
    floor = round(float(floor), 4)
    while p >= floor - 1e-9:
        seq.append(round(p, 4))
        p = round(p - step_, 4)
    return seq if seq else [floor]


def build_mesher(model_vtp, wall_id, gmes, bl, portion, n_layers, ratio):
    """Costruisce un TetGen fresco: carica modello, verifica/marca wall, opzioni."""
    mesher = sv.meshing.TetGen()
    mesher.load_model(model_vtp)

    face_ids = [int(x) for x in mesher.get_model_face_ids()]
    _log("ModelFaceID nel mesher: {}".format(sorted(face_ids)))
    if wall_id not in face_ids:
        raise RuntimeError("wall_id {} assente dai ModelFaceID {}: modello non coerente "
                           "con faces_named.json.".format(wall_id, sorted(face_ids)))
    mesher.set_walls([wall_id])

    opt = sv.meshing.TetGenOptions(
        global_edge_size=gmes, surface_mesh_flag=True, volume_mesh_flag=True
    )
    if bl:
        mesher.set_boundary_layer_options(
            number_of_layers=n_layers,
            edge_size_fraction=portion,
            layer_decreasing_ratio=ratio,
            constant_thickness=False,
        )
    return mesher, opt, face_ids


def register_mesh_in_project_xml(project_dir, mesh_name):
    """Registra la mesh in project_files.xml per renderla visibile nella GUI SV."""
    proj_xml = os.path.join(project_dir, "project_files.xml")
    if not os.path.exists(proj_xml):
        _log("project_files.xml non trovato in {}: la GUI potrebbe non mostrare la mesh.".format(
            project_dir), "WARN")
        return
    mesh_rel = os.path.join("Meshes", "{}.vtu".format(mesh_name))
    try:
        tree = ET.parse(proj_xml)
        root = tree.getroot()
        mesh_tab = root.find("mesh_tab")
        if mesh_tab is None:
            mesh_tab = ET.SubElement(root, "mesh_tab")
        found = False
        for m in mesh_tab.findall("mesh"):
            if m.get("name") == mesh_name:
                m.set("file", mesh_rel)
                m.set("type", "TetGen")
                found = True
                break
        if not found:
            e = ET.SubElement(mesh_tab, "mesh")
            e.set("name", mesh_name)
            e.set("file", mesh_rel)
            e.set("type", "TetGen")
        tree.write(proj_xml, encoding="utf-8", xml_declaration=True)
        _log("project_files.xml aggiornato per la mesh '{}'.".format(mesh_name))
    except Exception as e:
        _log("errore aggiornamento project_files.xml: {}".format(e), "WARN")


def write_outputs(mesher, project_dir, mesh_name, out_dir, meta):
    """Scrive .vtu, _surface.vtp, mesh_info.json e aggiorna l'XML del progetto."""
    mesh_dir = os.path.join(project_dir, "Meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    vtu_out = os.path.join(mesh_dir, "{}.vtu".format(mesh_name))
    surf_out = os.path.join(mesh_dir, "{}_surface.vtp".format(mesh_name))

    mesher.write_mesh(vtu_out)  # vtkUnstructuredGrid -> .vtu
    _log("scritto .vtu: {}".format(vtu_out))

    surf = mesher.get_surface()
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(surf_out)
    w.SetInputData(surf)
    w.Write()
    _log("scritto _surface.vtp: {}".format(surf_out))

    ug = mesher.get_mesh()
    meta["n_cells"] = int(ug.GetNumberOfCells())
    meta["n_points"] = int(ug.GetNumberOfPoints())
    meta["volume_mesh"] = os.path.relpath(vtu_out, out_dir)
    _atomic_write_json(os.path.join(out_dir, "mesh_info.json"), meta)

    register_mesh_in_project_xml(project_dir, mesh_name)
    return vtu_out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    # 1) Parsing argomenti. Nessun side-effect, puo' fallire su argv malformato.
    args = parse_args()
    in_path = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out_dir)

    # 2) Log persistente: DEVE essere la prima cosa, cosi' ogni errore successivo
    #    lascia una traccia su <out-dir>/sv_meshing.log.
    os.makedirs(out_dir, exist_ok=True)
    init_logging(out_dir)

    # Da qui in poi TUTTO e' dentro un try globale: qualunque eccezione Python
    # finisce nel log con traceback completo (non piu' fallimenti "muti").
    try:
        # 3) Dump ambiente (versioni/path/esistenza file): prima diagnostica utile.
        env_report(args, in_path, out_dir)

        # 4) Guard di esistenza input. Fail-loud immediato se manca il modello.
        if not os.path.exists(in_path):
            return _die("modello non trovato: {} (esegui prima sv_apply).".format(in_path))

        # 5) Derivazione path di progetto dal path del modello:
        #    Models/<name>.vtp -> models_dir=Models, project_dir=<proj>.
        models_dir = os.path.dirname(in_path)
        project_dir = os.path.dirname(models_dir)
        mesh_name = args.mesh_name or os.path.splitext(os.path.basename(in_path))[0]
        _log("project_dir={}  mesh_name={}".format(project_dir, mesh_name))

        # 6) Carica wall_id dalla SSOT (faces_named.json di sv_apply). Fail-loud.
        with step("load_wall_id"):
            wall_id, _named = load_wall_id(out_dir)
        _log("wall_id = {}".format(wall_id))

        # 7) meta = stato del meshing. Lo aggiorniamo e lo persistiamo (checkpoint)
        #    dopo OGNI passaggio, cosi' su disco resta sempre l'ultimo stato buono.
        bl = _str2bool(args.bl)
        meta = {
            "units": "cm", "gmes": args.gmes, "boundary_layer": bl,
            "wall_id": wall_id, "mesh": mesh_name,
            "input": in_path, "attempts": [],
        }
        checkpoint(out_dir, meta, "init")  # <-- salvataggio dopo passaggio 6-7

        # 8) PRE-FLIGHT superficie: fotografa bbox/unit-box/degeneri/NaN/FaceID.
        #    Scrive tutto nel log PRIMA del meshing -> diagnosi anche se TetGen crasha.
        with step("inspect_surface"):
            surf_info = inspect_surface(in_path)
        meta["surface_preflight"] = surf_info
        checkpoint(out_dir, meta, "preflight")  # <-- salvataggio dopo passaggio 8

        # 9) Modalita' PROBE: ispezione soltanto, nessun meshing.
        if args.probe:
            _log("PROBE: ispezione completata, esco senza meshiare.")
            checkpoint(out_dir, meta, "probe_done")
            return 0

        # 10) [OPZIONALE, default OFF] Remesh MMG manuale della superficie.
        #     Sovrascrive Models/<name>.vtp. Sconsigliato (unit-box + drop FaceID).
        if _str2bool(args.remesh):
            hmax = args.remesh_hmax if args.remesh_hmax is not None else args.gmes
            hmin = args.remesh_hmin if args.remesh_hmin is not None else max(0.01, hmax / 5.0)
            with step("remesh_model_surface (MMG)"):
                remesh_model_surface(in_path, hmin, hmax, args.remesh_hausd, args.remesh_angle)
            # Ri-ispeziona dopo il remesh: verifica che NON sia comparsa la unit-box.
            with step("inspect_surface (post-remesh)"):
                meta["surface_postremesh"] = inspect_surface(in_path)
            checkpoint(out_dir, meta, "post_remesh")  # <-- salvataggio dopo passaggio 10
        else:
            _log("remesh manuale OFF: TetGen fara' il remesh interno via gmes.")

        # 11) Sequenza portion del boundary layer (retry decrescente). Se BL OFF -> [None].
        if bl:
            portions = portion_sequence(args.portion_edge_size, args.step, args.floor)
            _log("BL ON. gmes={} portion da provare={}".format(args.gmes, portions))
        else:
            portions = [None]
            _log("BL OFF. gmes={} (nessun retry).".format(args.gmes))

        # 12) Ciclo di tentativi. Ogni tentativo:
        #     a) build_mesher: TetGen fresco + wall + opzioni (verifica ModelFaceID).
        #     b) generate_mesh: chiamata NATIVA -> avvolta in _FdCapture per catturare
        #        l'output C di TetGen/MMG anche in caso di crash.
        #     c) write_outputs: .vtu + _surface.vtp + json + xml (salvataggio su disco).
        #     Il fallimento di un tentativo viene loggato E salvato (checkpoint) e si
        #     passa al portion successivo.
        for portion in portions:
            label = "no-BL" if portion is None else "portion={}".format(portion)
            _log("=" * 40)
            _log(">>> TENTATIVO: gmes={} {}".format(args.gmes, label))
            try:
                # 12a) costruzione mesher (fail-loud se wall_id non e' tra i FaceID)
                with step("build_mesher [{}]".format(label)):
                    mesher, opt, _ = build_mesher(
                        in_path, wall_id, args.gmes, bl, portion,
                        args.bl_num_layers, args.bl_decreasing_ratio,
                    )

                # 12b) generate_mesh NATIVO con cattura fd (qui muore di solito)
                _log("chiamo generate_mesh() [output nativo -> sv_meshing.log]...")
                with step("generate_mesh [{}]".format(label)):
                    with _FdCapture(_LOG_PATH):
                        mesher.generate_mesh(opt)
                _log("generate_mesh() completata senza crash nativo.")

                # 12c) scrittura output + registrazione progetto (salvataggio finale)
                meta["portion_edge_size_used"] = portion
                meta["bl_num_layers"] = args.bl_num_layers if bl else None
                meta["bl_decreasing_ratio"] = args.bl_decreasing_ratio if bl else None
                meta["attempts"].append({"portion": portion, "result": "ok"})
                with step("write_outputs [{}]".format(label)):
                    vtu_out = write_outputs(mesher, project_dir, mesh_name, out_dir, meta)
                checkpoint(out_dir, meta, "done")  # <-- salvataggio dopo successo

                _log("[OK] mesh generata ({}): {} [celle={}]".format(
                    label, vtu_out, meta["n_cells"]))
                return 0

            except BaseException as exc:
                # Cattura ANCHE errori non-Exception (es. alcuni wrapper nativi).
                msg = "{}".format(exc)
                _log("[FAIL] {} -> {}".format(label, msg), "ERROR")
                meta["attempts"].append({"portion": portion, "result": "fail", "error": msg})
                checkpoint(out_dir, meta, "attempt_failed")  # <-- salvataggio dopo fallimento

        # 13) Tutti i portion falliti: riepilogo e uscita con errore.
        _log("meshing fallito a tutti i portion.", "ERROR")
        for a in meta["attempts"]:
            _log("   - portion={} : {}".format(a.get("portion"), a.get("error")), "ERROR")
        return _die("meshing fallito (gmes={}, bl={}, portions={}).".format(
            args.gmes, bl, [a.get("portion") for a in meta["attempts"]]))

    except BaseException:
        # Rete di sicurezza globale: qualsiasi cosa non gestita finisce qui, con
        # traceback completo nel log. Fine dei fallimenti "senza tracce".
        _log("ECCEZIONE NON GESTITA in main:", "ERROR")
        _log(traceback.format_exc(), "ERROR")
        return 1
    finally:
        _log("uscita sv_meshing.")
        if _LOG_FH is not None:
            try:
                _LOG_FH.flush()
                os.fsync(_LOG_FH.fileno())
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())