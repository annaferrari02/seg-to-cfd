"""sv_meshing.py  

    input : --input   modello .vtp del progetto (ModelFaceID finali di sv_apply).
            --out-dir cartella del paziente (contiene faces_named.json).
    output: <proj>/Meshes/<name>/<name>.vtu          mesh di volume (TetGen).
            <proj>/Meshes/<name>/<name>_surface.vtp  superficie esterna.
            <out-dir>/mesh_info.json                  SSOT del meshing (parametri
                                                      usati, portion vincente,
                                                      conteggi, log tentativi).
    exit  : != 0 se il BL fallisce a TUTTI i portion, o per input incoerente
            (wall_id assente dal modello, faces_named.json mancante, ecc.).

"""

import argparse
import json
import os
import sys
import traceback

import vtk

import sv

FACES_NAMED = "faces_named.json"


def _die(msg):
    sys.stderr.write("ERRORE: {}\n".format(msg))
    sys.stderr.flush()
    return 1


def _str2bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


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
                   help="carica il modello, stampa facce e opzioni disponibili, NON meshia.")
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    return p.parse_args(argv)


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


def portion_sequence(start, step, floor):
    """[start, start-step, ..., >= floor]. Almeno [floor] se start < floor."""
    seq = []
    p = round(float(start), 4)
    step = round(float(step), 4)
    floor = round(float(floor), 4)
    while p >= floor - 1e-9:
        seq.append(round(p, 4))
        p = round(p - step, 4)
    return seq if seq else [floor]


def build_mesher(model_vtp, wall_id, gmes, bl, portion, n_layers, ratio):
    """Costruisce un TetGen fresco: carica modello, marca wall, prepara opzioni."""
    mesher = sv.meshing.TetGen()
    mesher.load_model(model_vtp)

    face_ids = [int(x) for x in mesher.get_model_face_ids()]
    if wall_id not in face_ids:
        raise RuntimeError(
            "wall_id {} assente dai ModelFaceID del modello {}: modello non "
            "coerente con faces_named.json.".format(wall_id, sorted(face_ids))
        )
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


def write_outputs(mesher, project_dir, mesh_name, out_dir, meta):
    """Scrive .vtu (volume) + _surface.vtp + mesh_info.json. Ritorna il path .vtu."""
    mesh_dir = os.path.join(project_dir, "Meshes", mesh_name)
    os.makedirs(mesh_dir, exist_ok=True)

    vtu_out = os.path.join(mesh_dir, "{}.vtu".format(mesh_name))
    mesher.write_mesh(vtu_out)  # TetGen -> vtkUnstructuredGrid .vtu

    surf = mesher.get_surface()
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(os.path.join(mesh_dir, "{}_surface.vtp".format(mesh_name)))
    w.SetInputData(surf)
    w.Write()

    ug = mesher.get_mesh()
    meta["n_cells"] = int(ug.GetNumberOfCells())
    meta["n_points"] = int(ug.GetNumberOfPoints())
    meta["volume_mesh"] = os.path.relpath(vtu_out, out_dir)

    with open(os.path.join(out_dir, "mesh_info.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return vtu_out


def main():
    args = parse_args()
    in_path = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.exists(in_path):
        return _die("modello non trovato: {} (esegui prima sv_apply).".format(in_path))

    # <proj>/Models/<name>.vtp  ->  project_dir, mesh_name
    models_dir = os.path.dirname(in_path)
    project_dir = os.path.dirname(models_dir)
    mesh_name = args.mesh_name or os.path.splitext(os.path.basename(in_path))[0]

    try:
        wall_id, _named = load_wall_id(out_dir)
    except Exception as exc:
        return _die(str(exc))

    bl = _str2bool(args.bl)

    # --- probe: nessun meshing, solo ispezione (validazione su pz000) ---------
    if args.probe:
        try:
            m = sv.meshing.TetGen()
            m.load_model(in_path)
            print("[probe] modello: {}".format(in_path), flush=True)
            print("[probe] ModelFaceID: {}".format(sorted(int(x) for x in m.get_model_face_ids())), flush=True)
            try:
                print("[probe] face_info: {}".format(m.get_model_face_info()), flush=True)
            except Exception as e:
                print("[probe] get_model_face_info non disponibile: {}".format(e), flush=True)
            print("[probe] wall_id (da faces_named.json): {}".format(wall_id), flush=True)
            print("[probe] TetGenOptions attrs: {}".format(
                [a for a in dir(sv.meshing.TetGenOptions) if not a.startswith("_")]), flush=True)
            return 0
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return _die("probe fallito: {}".format(exc))

    # --- sequenza di tentativi -------------------------------------------------
    if bl:
        portions = portion_sequence(args.portion_edge_size, args.step, args.floor)
        print("[sv_meshing] BL ON. gmes={} portion da provare={}".format(args.gmes, portions), flush=True)
    else:
        portions = [None]  # nessun BL: un solo tentativo, portion irrilevante
        print("[sv_meshing] BL OFF. gmes={} (nessun retry su portion)".format(args.gmes), flush=True)

    attempts = []
    for portion in portions:
        label = "no-BL" if portion is None else "portion={}".format(portion)
        print("[sv_meshing] >>> tentativo: gmes={} {}".format(args.gmes, label), flush=True)
        try:
            mesher, opt, _ = build_mesher(
                in_path, wall_id, args.gmes, bl, portion,
                args.bl_num_layers, args.bl_decreasing_ratio,
            )
            mesher.generate_mesh(opt)  # TetGen stampa qui i suoi errori (PLC ...)

            meta = {
                "units": "cm",
                "gmes": args.gmes,
                "boundary_layer": bl,
                "portion_edge_size_used": portion,
                "bl_num_layers": args.bl_num_layers if bl else None,
                "bl_decreasing_ratio": args.bl_decreasing_ratio if bl else None,
                "wall_id": wall_id,
                "mesh": mesh_name,
                "attempts": attempts + [{"portion": portion, "result": "ok"}],
            }
            vtu_out = write_outputs(mesher, project_dir, mesh_name, out_dir, meta)
            print("[OK] mesh generata ({}): {}  [celle={}]".format(
                label, vtu_out, meta["n_cells"]), flush=True)
            return 0

        except Exception as exc:
            msg = "{}".format(exc)
            print("[sv_meshing][FAIL] {} -> {}".format(label, msg), flush=True)
            attempts.append({"portion": portion, "result": "fail", "error": msg})

    # tutti i tentativi falliti -> esci con log (l'adapter li cattura e rilancia)
    print("[sv_meshing][FAIL] meshing fallito a tutti i portion.", flush=True)
    for a in attempts:
        print("   - portion={} : {}".format(a.get("portion"), a.get("error")), flush=True)
    return _die("meshing fallito (gmes={}, bl={}, portions={}).".format(
        args.gmes, bl, [a.get("portion") for a in attempts]))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        sys.exit(_die("eccezione non gestita: {}".format(exc)))