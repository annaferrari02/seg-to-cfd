"""extract_faces.py  --  MONDO 2 (interprete Python di SimVascular).

Prima passata SimVascular della pipeline. Carica il modello geometrico,
identifica le facce per angolo di separazione, distingue la PARETE (wall) dai
TAPPI (cap) ed estrae la geometria dei cap. NON assegna ruoli: inlet vs outlet
e' una proprieta' topologica che viene da tag_endpoints e viene applicata solo
piu' tardi, a sv_match.

Uso (di norma lanciato dall'adapter, non a mano):
    simvascular --python -- extract_faces.py \
        --input <lumen_tree_cfd_clip_cm.vtk> \
        --out-dir <cartella del paziente> \
        --separation-angle 50

CONTRATTO
    input : --input   file .vtk (PolyData legacy, gia' in cm, triangolato e
                       pulito da ParaView).
            --out-dir cartella del paziente dove scrivere gli output.
            --separation-angle  angolo per compute_boundary_faces (default 50).
    output: <out-dir>/model.vtp     modello con l'array ModelFaceID (lo
                                     ricarichera' sv_apply senza ricalcolare).
            <out-dir>/cap_faces.json {units, separation_angle, wall_face_ids,
                                     cap_faces:[{face_id, centroid, radius,
                                     area, planarity, circularity}]}. La lista
                                     cap_faces e' cio' che consuma sv_match_core.
    exit  : != 0 a OGNI ambiguita' (fail loud): input illeggibile, nessun cap,
            meno di 2 cap, nessuna parete.
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import sv


# --- Soglie di classificazione cap vs wall -------------------------------------
# Un cap e' un DISCO: piatto, tondo, con un solo anello di bordo.
#   planarity  = rms fuori-piano / RAGGIO effettivo (NON la lunghezza): disco ~0,
#                lembo di parete curvo molto piu' alto. Adimensionale, scala-invariante.
#   circularity= 4*pi*area / perimetro^2: disco ~1, striscia allungata -> 0.
#   min_cap_area = floor per scartare le schegge di triangolazione (aspect ratio
#                  altissimo, area ~1e-4 cm^2) prima di classificarle.
DEFAULT_PLANARITY_TOL = 0.15
DEFAULT_MIN_CIRCULARITY = 0.60
DEFAULT_MIN_CAP_AREA = 0.01   # cm^2


def _die(msg):
    """Stampa un errore su stderr e restituisce 1 (fail loud)."""
    sys.stderr.write("ERRORE: {}\n".format(msg))
    sys.stderr.flush()
    return 1


def parse_args():
    parser = argparse.ArgumentParser(description="SimVascular: estrazione facce (wall/cap).")
    parser.add_argument("--input", required=True, help="modello .vtk di input (PolyData, cm).")
    parser.add_argument("--out-dir", required=True, help="cartella di output (work/ del pz).")
    parser.add_argument("--separation-angle", type=float, default=50.0,
                        help="angolo per compute_boundary_faces (default 50).")
    parser.add_argument("--cap-planarity-tol", type=float, default=DEFAULT_PLANARITY_TOL,
                        help="soglia planarita' (rms/raggio) per un cap (default 0.15).")
    parser.add_argument("--min-circularity", type=float, default=DEFAULT_MIN_CIRCULARITY,
                        help="circolarita' minima per un cap (default 0.60).")
    parser.add_argument("--min-cap-area", type=float, default=DEFAULT_MIN_CAP_AREA,
                        help="area minima (cm^2) sotto cui la faccia e' una scheggia -> wall (default 0.01).")
    parser.add_argument("--remesh", dest="remesh", action="store_true", default=True,
                        help="ri-triangola la superficie prima delle facce (default on).")
    parser.add_argument("--no-remesh", dest="remesh", action="store_false",
                        help="disattiva il remesh di pulizia.")
    parser.add_argument("--remesh-hmin", type=float, default=0.05, help="edge size min MMG (cm).")
    parser.add_argument("--remesh-hmax", type=float, default=0.05, help="edge size max MMG (cm).")
    parser.add_argument("--remesh-angle", type=float, default=60.0, help="feature angle (preserva i rim dei cap).")
    parser.add_argument("--remesh-hgrad", type=float, default=1.1, help="gradazione MMG.")
    parser.add_argument("--remesh-hausd", type=float, default=0.01, help="distanza di Hausdorff MMG (cm).")

    # SimVascular puo' anteporre argomenti propri: tieni solo cio' che segue "--".
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    print("[DEBUG] extract_faces argv: {}".format(sys.argv), flush=True)
    return parser.parse_args(argv)


def read_surface(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()

    # Estrai la superficie e triangola.
    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(reader.GetOutput())
    geom.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(geom.GetOutput())
    tri.Update()

    # Unisci i punti coincidenti. IMPORTANTE: NON convertire le celle degeneri
    # in linee/vertici. Il VTK di SimVascular, unendo punti quasi-coincidenti,
    # fa collassare qualche triangolo in uno "sliver" ad area nulla e, con
    # l'opzione di default attiva, lo trasforma in una LINEA. Una cella 1D nel
    # modello manda in stallo compute_boundary_faces.
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(tri.GetOutput())
    cleaner.PointMergingOn()
    cleaner.ConvertPolysToLinesOff()
    cleaner.ConvertLinesToPointsOff()
    cleaner.ConvertStripsToPolysOff()
    cleaner.Update()
    cleaned = cleaner.GetOutput()

    # DOPO il clean, tieni SOLO punti + poligoni. Cosi' sparisce qualunque cella
    # 1D residua: sia la polilinea della flow-extension gia' nel file, sia gli
    # eventuali triangoli collassati dal merge. Questo passo va fatto DOPO il
    # clean, non prima: se lo si fa prima, il clean rigenera le linee.
    pd = vtk.vtkPolyData()
    pd.SetPoints(cleaned.GetPoints())
    pd.SetPolys(cleaned.GetPolys())

    print("[DEBUG] read_surface finale: points={} polys={} lines={} verts={}".format(
        pd.GetNumberOfPoints(), pd.GetNumberOfPolys(),
        pd.GetNumberOfLines(), pd.GetNumberOfVerts()
    ))
    if pd.GetNumberOfPolys() == 0:
        raise ValueError(
            "letti {} punti ma 0 poligoni: file probabilmente in legacy VTK 5.1 "
            "non leggibile da questo SimVascular. Riscrivi il .vtk in 4.2 "
            "dallo step ParaView.".format(pd.GetNumberOfPoints())
        )

    # Normali coerenti PRIMA di darla a SimVascular. compute_boundary_faces
    # distingue le facce dall'angolo tra le normali di triangoli adiacenti: se
    # il winding in uscita da ParaView e' incoerente, triangoli complanari
    # sembrano a ~180 gradi, l'algoritmo frammenta e si impunta. ConsistencyOn
    # uniforma il verso; AutoOrientNormalsOn le manda verso l'esterno (la
    # superficie e' chiusa); SplittingOff preserva la topologia (niente vertici
    # duplicati). E' il pre-condizionamento che la GUI fa all'import e che il
    # percorso set_surface() su PolyData grezza salta.
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(pd)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.ComputePointNormalsOn()
    normals.NonManifoldTraversalOff()
    normals.Update()
    pd = normals.GetOutput()
    print("[DEBUG] normali coerenti: cell_normals={} point_normals={}".format(
        pd.GetCellData().GetNormals() is not None,
        pd.GetPointData().GetNormals() is not None), flush=True)

    return pd

def _surface_topology(pd):
    """(free_edges, non_manifold_edges). Superficie CFD valida = (0, 0)."""
    def _cnt(setter):
        fe = vtk.vtkFeatureEdges(); fe.SetInputData(pd)
        fe.BoundaryEdgesOff(); fe.NonManifoldEdgesOff()
        fe.FeatureEdgesOff(); fe.ManifoldEdgesOff()
        setter(fe); fe.Update(); return fe.GetOutput().GetNumberOfCells()
    return _cnt(lambda f: f.BoundaryEdgesOn()), _cnt(lambda f: f.NonManifoldEdgesOn())


def remesh_surface(pd, args):
    """Ri-triangola la superficie a taglia (quasi) uniforme: l'equivalente del
    Remesh nel modulo Model della GUI. Toglie i triangoli-ago degeneri che l'MMG
    INTERNO del mesher non riesce a coarsare. Deve girare PRIMA di
    compute_boundary_faces, cosi' i ModelFaceID nascono su superficie pulita."""
    before = pd.GetNumberOfCells()
    try:
        out = sv.mesh_utils.remesh(
            pd, hmin=args.remesh_hmin, hmax=args.remesh_hmax,
            angle=args.remesh_angle, hgrad=args.remesh_hgrad, hausd=args.remesh_hausd,
        )
    except Exception as exc:
        raise RuntimeError("sv.mesh_utils.remesh fallito (hmin={} hmax={} hausd={}): {}".format(
            args.remesh_hmin, args.remesh_hmax, args.remesh_hausd, exc))

    # Normali coerenti dopo la ri-triangolazione (stesso condizionamento di read_surface).
    n = vtk.vtkPolyDataNormals(); n.SetInputData(out)
    n.SplittingOff(); n.ConsistencyOn(); n.AutoOrientNormalsOn()
    n.ComputeCellNormalsOn(); n.ComputePointNormalsOn(); n.NonManifoldTraversalOff()
    n.Update(); out = n.GetOutput()

    # Guardia fail-loud: il remesh NON deve bucare ne' rendere non-manifold.
    free, nonman = _surface_topology(out)
    if free > 0 or nonman > 0:
        raise RuntimeError("remesh ha rotto la superficie: free={} non_manifold={} "
                           "(alza --remesh-hausd o --remesh-hmin/hmax).".format(free, nonman))
    print("[DEBUG] remesh: {} -> {} triangoli (free=0 non_manifold=0)".format(
        before, out.GetNumberOfCells()), flush=True)
    return out

def _used_points(face_pd):
    """Punti REALMENTE usati dalla faccia. get_face_polydata porta con se' il
    vtkPoints dell'intero modello: senza compattare, ogni fit di piano girerebbe
    sui punti di tutta la superficie, non della singola faccia."""
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(face_pd)
    clean.PointMergingOff()          # non fondere, solo rimuovere i punti orfani
    clean.Update()
    return vtk_to_numpy(clean.GetOutput().GetPoints().GetData())


def area_and_centroid(face_pd):
    """Area totale e centroide PESATO SULL'AREA (centro geometrico vero del cap;
    la media dei punti sarebbe sbilanciata dove la mesh e' piu' fitta)."""
    pts = vtk_to_numpy(face_pd.GetPoints().GetData())
    total_area = 0.0
    weighted = np.zeros(3)
    for c in range(face_pd.GetNumberOfCells()):
        cell = face_pd.GetCell(c)
        if cell.GetNumberOfPoints() != 3:
            continue
        i0, i1, i2 = cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)
        p0, p1, p2 = pts[i0], pts[i1], pts[i2]
        a = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        total_area += a
        weighted += a * (p0 + p1 + p2) / 3.0
    if total_area <= 0.0:
        return 0.0, None
    return total_area, weighted / total_area


def out_of_plane_rms(face_pd):
    """RMS ASSOLUTO (cm) della distanza dei punti dal piano ai minimi quadrati.
    La normale del piano e' la direzione a varianza minima (ultima riga di V^T
    della SVD). Da normalizzare sul RAGGIO effettivo dal chiamante, non qui."""
    pts = _used_points(face_pd)
    c = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return float(np.sqrt(((c @ vt[-1]) ** 2).mean()))


def face_boundary(face_pd):
    """Perimetro dei bordi aperti e NUMERO DI ANELLI di bordo della faccia.
    Un cap ha esattamente 1 anello (il rim); la parete ne ha molti (uno per
    ogni cap con cui confina). n_loops>1 => parete, senza ambiguita'."""
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(face_pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    edges = fe.GetOutput()
    if edges.GetNumberOfCells() == 0:
        return 0.0, 0

    pts = vtk_to_numpy(edges.GetPoints().GetData())
    perim = 0.0
    ids = vtk.vtkIdList()
    lines = edges.GetLines()
    lines.InitTraversal()
    while lines.GetNextCell(ids):
        for k in range(ids.GetNumberOfIds() - 1):
            perim += float(np.linalg.norm(pts[ids.GetId(k + 1)] - pts[ids.GetId(k)]))

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(edges)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    return perim, conn.GetNumberOfExtractedRegions()


def classify_face(fpd, area, args):
    """Ritorna (is_cap, metrics). Un cap deve essere: 1 anello di bordo, tondo
    (circ>=soglia) E piatto (planar<soglia). Il triplo AND rende robusti anche
    ai lembi di parete a loop singolo, che sono comunque allungati e/o curvi."""
    perim, n_loops = face_boundary(fpd)
    circ = 4.0 * np.pi * area / (perim * perim) if perim > 0 else 0.0
    r_eff = float(np.sqrt(area / np.pi))
    planar = out_of_plane_rms(fpd) / r_eff if r_eff > 0 else 1.0

    is_cap = (n_loops == 1) and (circ >= args.min_circularity) and (planar < args.cap_planarity_tol)
    ambiguous = n_loops == 1 and (circ >= args.min_circularity) != (planar < args.cap_planarity_tol)
    return is_cap, {"circ": circ, "planar": planar, "n_loops": n_loops, "ambiguous": ambiguous}


def main():
    args = parse_args()

    in_path = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out_dir)
    print("[DEBUG] extract_faces abs input: {}".format(in_path), flush=True)
    print("[DEBUG] extract_faces abs out-dir: {}".format(out_dir), flush=True)
    if not os.path.exists(in_path):
        return _die("input non trovato: {}".format(in_path))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 1) Carica e condiziona la superficie.
    try:
        surface = read_surface(str(in_path))
        print("[DEBUG] read_surface OK", flush=True)
    except Exception as exc:
        return _die(str(exc))

    if args.remesh:
        try:
            surface = remesh_surface(surface, args)
            print("[DEBUG] REMESH OK", flush=True)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return _die(str(exc))

    # Scrivi un .vtp temporaneo e fallo RILEGGERE a SimVascular col reader nativo:
    # e' cio' che fa la GUI (importa un FILE). Modeler.read inizializza il modello
    # meglio di set_surface() su PolyData grezza, che lasciava compute_boundary_faces
    # in stallo.
    sv_input = os.path.join(out_dir, "_sv_input.vtp")
    try:
        w = vtk.vtkXMLPolyDataWriter()
        w.SetFileName(sv_input)
        w.SetInputData(surface)
        w.Write()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return _die("errore scrittura input SV temporaneo: {}".format(exc))

    try:
        modeler = sv.modeling.Modeler(sv.modeling.Kernel.POLYDATA)
        model = modeler.read(sv_input)
        print("[DEBUG] Modeler.read OK", flush=True)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return _die("errore lettura modello SV (Modeler.read): {}".format(exc))

    # 2) Identifica le facce per angolo di separazione.
    try:
        face_ids = model.compute_boundary_faces(angle=args.separation_angle)
        print("[DEBUG] compute_boundary_faces: {} facce {}".format(len(face_ids), face_ids), flush=True)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return _die("compute_boundary_faces ha fallito: {}".format(exc))
    if not face_ids:
        return _die("compute_boundary_faces non ha prodotto facce.")

    # 3) Classifica per FORMA (non per sola planarita'): disco piatto e tondo = cap.
    caps = []
    wall_face_ids = []
    print("[DEBUG] {:>7} {:>11} {:>7} {:>8} {:>6}  classe".format(
        "face_id", "area", "circ", "planar", "loops"), flush=True)
    for fid in face_ids:
        fid = int(fid)
        fpd = model.get_face_polydata(fid)
        area, centroid = area_and_centroid(fpd)

        # Scheggia / faccia degenere: NON e' un errore, e' triangolazione sporca
        # alle giunzioni. Troppo piccola per essere un cap reale -> confluisce nel wall.
        if area < args.min_cap_area or centroid is None:
            wall_face_ids.append(fid)
            print("[DEBUG] {:>7} {:>11.5f} {:>7} {:>8} {:>6}  wall (sliver)".format(
                fid, area, "-", "-", "-"), flush=True)
            continue

        is_cap, m = classify_face(fpd, area, args)
        if m["ambiguous"]:
            print("[WARN] faccia {} ambigua: circ={:.2f} planar={:.3f} loops={} area={:.4f}".format(
                fid, m["circ"], m["planar"], m["n_loops"], area), flush=True)

        klass = "cap" if is_cap else "wall"
        print("[DEBUG] {:>7} {:>11.5f} {:>7.3f} {:>8.3f} {:>6d}  {}".format(
            fid, area, m["circ"], m["planar"], m["n_loops"], klass), flush=True)

        if is_cap:
            caps.append({
                "face_id": fid,
                "centroid": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                "radius": float(np.sqrt(area / np.pi)),
                "area": float(area),
                "planarity": float(m["planar"]),
                "circularity": float(m["circ"]),
            })
        else:
            wall_face_ids.append(fid)

    # 4) Controlli di sanita' (fail loud).
    if len(caps) < 2:
        return _die("trovati {} cap (servono >= 2: 1 inlet + >= 1 outlet). ".format(len(caps))
                    + "Abbassa --separation-angle (cap fusi con la parete) o "
                    + "controlla --min-circularity/--cap-planarity-tol.")
    if not wall_face_ids:
        return _die("nessuna faccia di parete identificata.")

    # 5) Scrivi il modello con ModelFaceID (per sv_apply) e il JSON dei cap.
    model_out = os.path.join(out_dir, "model.vtp")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(model_out)
    writer.SetInputData(model.get_polydata())
    writer.Write()

    cap_json = os.path.join(out_dir, "cap_faces.json")
    with open(cap_json, "w") as f:
        json.dump({
            "units": "cm",
            "separation_angle": args.separation_angle,
            "wall_face_ids": wall_face_ids,
            "cap_faces": caps,
        }, f, indent=2)

    print("[OK] {} cap, {} facce di parete.".format(len(caps), len(wall_face_ids)), flush=True)
    print("[OK] scritto {}".format(model_out), flush=True)
    print("[OK] scritto {}".format(cap_json), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())