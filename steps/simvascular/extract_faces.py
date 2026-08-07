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

CONTRATTO (come gli altri step)
    input : --input   file .vtk (PolyData legacy, gia' in cm, triangolato e
                       pulito da ParaView).
            --out-dir cartella del paziente dove scrivere gli output.
            --separation-angle  angolo per compute_boundary_faces (default 50).
    output: <out-dir>/model.vtp     modello con l'array ModelFaceID (lo
                                     ricarichera' sv_apply senza ricalcolare).
            <out-dir>/cap_faces.json {units, separation_angle, wall_face_ids,
                                     cap_faces:[{face_id, centroid, radius,
                                     area, planarity}]}. La lista cap_faces e'
                                     esattamente cio' che consuma sv_match_core.
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


# Soglia di planarita' (rms fuori-piano / diametro della faccia). Sotto = cap.
# Un disco piatto sta ~1e-3..1e-2; un pezzo di parete cilindrico ~0.1..0.3.
DEFAULT_PLANARITY_TOL = 0.05


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
                        help="soglia planarita' per classificare un cap (default 0.05).")

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


def area_and_centroid(face_pd: vtk.vtkPolyData):
    """Area totale e centroide PESATO SULL'AREA di una faccia triangolata.

    Il centroide pesato sull'area e' il centro geometrico vero del cap; la media
    dei punti sarebbe sbilanciata dove la mesh e' piu' fitta.
    """
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


def planarity(face_pd: vtk.vtkPolyData) -> float:
    """Quanto e' piatta una faccia: rms della distanza fuori-piano / diametro.

    Fit del piano ai minimi quadrati via SVD; la normale e' la direzione a
    varianza minima. Valore ~0 = disco piatto (cap); valori alti = superficie
    curva (parete). Scala-invariante.
    """
    pts = vtk_to_numpy(face_pd.GetPoints().GetData())
    centered = pts - pts.mean(axis=0)
    # Direzione a varianza minima = ultima riga di V^T della SVD.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    out_of_plane = np.abs(centered @ normal)
    rms = float(np.sqrt((out_of_plane ** 2).mean()))
    diameter = 2.0 * float(np.linalg.norm(centered, axis=1).max())
    return rms / diameter if diameter > 0 else 0.0


def main() -> int:
    args = parse_args()

    in_path = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out_dir)
    print("[DEBUG] extract_faces cwd: {}".format(os.getcwd()), flush=True)
    print("[DEBUG] extract_faces abs input: {}".format(in_path), flush=True)
    print("[DEBUG] extract_faces abs out-dir: {}".format(out_dir), flush=True)
    if not os.path.exists(in_path):
        return _die("input non trovato: {}".format(in_path))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 1) Carica la superficie e costruisci il modello PolyData.
    try:
        surface = read_surface(str(in_path))
        print("[DEBUG] read_surface OK", flush=True)
    except Exception as exc:
        return _die(str(exc))

    # Scrivi la superficie condizionata in un .vtp temporaneo e falla RILEGGERE
    # a SimVascular col suo reader nativo. E' esattamente cio' che fa la GUI:
    # importa un FILE, non un oggetto VTK. Il percorso di lettura da file
    # (Modeler.read) inizializza il modello in modo piu' completo di
    # set_surface()/costruttore, che finora lasciava compute_boundary_faces in
    # stallo.
    sv_input = os.path.join(out_dir, "_sv_input.vtp")
    try:
        w = vtk.vtkXMLPolyDataWriter()
        w.SetFileName(sv_input)
        w.SetInputData(surface)
        w.Write()
        print("[DEBUG] scritto input SV temporaneo: {}".format(sv_input), flush=True)
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
    #    NB: se lo script si ferma QUI senza errore ne' riga successiva:
    #      - lascialo girare qualche minuto: potrebbe essere LENTO per la
    #        qualita' dei triangoli (schegge ad aspect ratio altissimo);
    #      - controlla nel Task Manager se il processo consuma CPU (lento/loop)
    #        oppure e' sparito (crash C++).
    try:
        print("[DEBUG] chiamo compute_boundary_faces(angle={}) ...".format(
            args.separation_angle), flush=True)
        face_ids = model.compute_boundary_faces(angle=args.separation_angle)
        print("[DEBUG] compute_boundary_faces returned {} faces".format(len(face_ids)), flush=True)
        print("[DEBUG] facce trovate (angolo {}): {}".format(args.separation_angle, face_ids), flush=True)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return _die("compute_boundary_faces ha fallito: {}".format(exc))

    if not face_ids:
        return _die("compute_boundary_faces non ha prodotto facce.")

    # 3) Classifica ogni faccia per planarita': cap (piatta) vs wall (curva).
    caps = []
    wall_face_ids = []
    print("[DEBUG] {:>7} {:>12} {:>10}  classe".format('face_id', 'area', 'planarita'), flush=True)
    for fid in face_ids:
        fpd = model.get_face_polydata(int(fid))
        area, centroid = area_and_centroid(fpd)
        if area <= 0.0 or centroid is None:
            return _die("faccia {}: area nulla, geometria degenere.".format(fid))
        p = planarity(fpd)
        is_cap = p < args.cap_planarity_tol
        klass = "cap" if is_cap else "wall"
        print("[DEBUG] {fid:>7} {area:>12.5f} {p:>10.4f}  {klass}".format(fid=int(fid), area=area, p=p, klass=klass), flush=True)
        if is_cap:
            radius = float(np.sqrt(area / np.pi))
            caps.append({
                "face_id": int(fid),
                "centroid": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                "radius": radius,
                "area": float(area),
                "planarity": float(p),
            })
        else:
            wall_face_ids.append(int(fid))

    # 4) Controlli di sanita' (fail loud).
    if len(caps) < 2:
        return _die(
            "trovati {} cap (servono >= 2: 1 inlet + >= 1 outlet). ".format(len(caps))
            + "Prova un --separation-angle piu' basso: i cap potrebbero essersi "
            + "fusi con la parete."
        )
    if not wall_face_ids:
        return _die("nessuna faccia di parete identificata (tutte planari?).")

    # 5) Scrivi il modello con ModelFaceID (per sv_apply) e il JSON dei cap.
    model_out = os.path.join(out_dir, "model.vtp")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(model_out)
    writer.SetInputData(model.get_polydata())
    writer.Write()

    cap_json = os.path.join(out_dir, "cap_faces.json")
    payload = {
        "units": "cm",
        "separation_angle": args.separation_angle,
        "wall_face_ids": wall_face_ids,
        "cap_faces": caps,
    }
    with open(cap_json, "w") as f:
        f.write(json.dumps(payload, indent=2))

    print("[OK] {} cap, {} facce di parete.".format(len(caps), len(wall_face_ids)), flush=True)
    print("[OK] scritto {}".format(model_out), flush=True)
    print("[OK] scritto {}".format(cap_json), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())