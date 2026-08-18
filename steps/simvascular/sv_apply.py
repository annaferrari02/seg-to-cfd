"""sv_apply.py 

crea sim vascular project headless, rinomina le facce secondo face_roles.json e scrive il solid model.

CONVENZIONE ID FINALI (stabile, leggibile):
    1 = wall, 2 = inlet, 3.. = outlet A, B, C, ...

Uso (di norma via adapter):
    simvascular --python -- sv_apply.py --input <out-dir>/model.vtp --out-dir <out-dir>
CONTRATTO
    input : --input   model.vtp (con array ModelFaceID, da extract_faces).
            --out-dir cartella del paziente; deve contenere face_roles.json.
    output: <out-dir>/cfd<pid>/  progetto SV headless (layout predefinito), con
            <out-dir>/cfd<pid>/Models/model.vtp  solid model, ID ricombinati.
            <out-dir>/faces_named.json           mappa ModelFaceID->ruolo (SSOT).
    exit  : != 0 a OGNI incoerenza (fail loud): array assente, ID nel modello
            non coperti da face_roles, face_roles malformato.
"""

import argparse
import json
import os
import sys
import time
import traceback

import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np

FACE_ID_ARRAY = "ModelFaceID"


def _die(msg):
    sys.stderr.write("ERRORE: {}\n".format(msg))
    sys.stderr.flush()
    return 1


def _letters(i):
    """0->A, 1->B, ... 25->Z, 26->AA (base-26 bijettiva)."""
    s, i = "", i + 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def parse_args():
    p = argparse.ArgumentParser(description="SimVascular: applica i ruoli (combine wall + naming).")
    p.add_argument("--input", required=True, help="model.vtp con ModelFaceID.")
    p.add_argument("--out-dir", required=True, help="cartella del paziente (contiene face_roles.json).")
    p.add_argument("--roles", default="face_roles.json", help="nome del file ruoli in out-dir.")
    p.add_argument("--model-name", default="model.vtp",
                   help="nome del solid model dentro <project>/Models/ (default model.vtp).")
    p.add_argument("--sv-version", default="23.03.27",
                   help="stringa versione scritta nel descrittore immagine (informativa).")
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    print("[DEBUG] sv_apply argv: {}".format(sys.argv), flush=True)
    return p.parse_args(argv)


def load_roles(path):
    r = json.loads(open(path).read())
    inlet = r.get("inlet", {}).get("face_id")
    wall = r.get("wall_face_ids")
    outlets = r.get("outlets")
    if inlet is None:
        raise ValueError("face_roles.json: manca inlet.face_id.")
    if not isinstance(wall, list) or not wall:
        raise ValueError("face_roles.json: wall_face_ids assente o vuoto.")
    if not isinstance(outlets, list):
        raise ValueError("face_roles.json: outlets assente.")
    outlet_ids = [o["face_id"] for o in outlets]  # ordine gia' deterministico da sv_match
    return int(inlet), [int(x) for x in wall], [int(x) for x in outlet_ids]


def build_mapping(inlet, wall, outlets):
    """Ritorna (old_id->new_id, faces[]) con la convenzione 1=wall,2=inlet,3.."""
    mapping = {}
    faces = []

    for w in wall:
        mapping[w] = 1
    faces.append({"model_face_id": 1, "name": "wall", "type": "wall",
                  "combined_from": sorted(set(wall))})

    if inlet in mapping:
        raise ValueError("inlet {} compare anche tra i wall: face_roles incoerente.".format(inlet))
    mapping[inlet] = 2
    faces.append({"model_face_id": 2, "name": "inlet", "type": "cap",
                  "combined_from": [inlet]})

    next_id = 3
    for i, o in enumerate(outlets):
        if o in mapping:
            raise ValueError("outlet {} duplicato/gia' assegnato: face_roles incoerente.".format(o))
        name = _letters(i)
        mapping[o] = next_id
        faces.append({"model_face_id": next_id, "name": name, "type": "cap",
                      "combined_from": [o]})
        next_id += 1

    return mapping, faces


def main():
    args = parse_args()
    in_path = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out_dir)
    pid = os.path.basename(out_dir.rstrip("/"))
    project = "cfd{}".format(pid)

    if not os.path.exists(in_path):
        return _die("input non trovato: {}".format(in_path))
    roles_path = os.path.join(out_dir, args.roles)
    if not os.path.exists(roles_path):
        return _die("face_roles non trovato: {} (esegui prima sv_match).".format(roles_path))

    # 1) leggi ruoli + modello
    try:
        inlet, wall, outlets = load_roles(roles_path)
    except Exception as exc:
        return _die(str(exc))

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(in_path)
    reader.Update()
    pd = reader.GetOutput()
    arr = pd.GetCellData().GetArray(FACE_ID_ARRAY)
    if arr is None:
        return _die("array '{}' assente in {}: non e' un modello SV valido.".format(FACE_ID_ARRAY, in_path))
    old = vtk_to_numpy(arr).astype(np.int64)

    # 2) costruisci la mappa old->new e verifica la COPERTURA (fail loud)
    try:
        mapping, faces = build_mapping(inlet, wall, outlets)
    except Exception as exc:
        return _die(str(exc))

    present = set(int(x) for x in np.unique(old))
    declared = set(mapping)
    missing_in_roles = present - declared      # celle del modello senza ruolo -> stop
    missing_in_model = declared - present      # ruoli che non toccano nessuna cella -> stop
    if missing_in_roles:
        return _die("ModelFaceID {} presenti nel modello ma non in face_roles.json: "
                    "classificazione incoerente.".format(sorted(missing_in_roles)))
    if missing_in_model:
        return _die("ModelFaceID {} in face_roles.json ma assenti dal modello: "
                    "face_roles non corrisponde a questo model.vtp.".format(sorted(missing_in_model)))

    # 3) riscrivi l'array ModelFaceID (COMBINE + rinomina in un colpo)
    new = np.fromiter((mapping[int(x)] for x in old), dtype=np.int32, count=len(old))
    new_arr = numpy_to_vtk(new, deep=1, array_type=vtk.VTK_INT)
    new_arr.SetName(FACE_ID_ARRAY)
    pd.GetCellData().RemoveArray(FACE_ID_ARRAY)
    pd.GetCellData().AddArray(new_arr)

    n_final = len(faces)
    print("[DEBUG] combine: {} facce originali -> {} facce finali (1 wall, 1 inlet, {} outlet)".format(
        len(present), n_final, n_final - 2), flush=True)

    # 4) cartella-progetto (layout predefinito SV, headless) + solid model in Models/
    proj_dir = os.path.join(out_dir, project)
    for sub in ("Images", "Paths", "Segmentations", "Models", "Meshes", "Simulations"):
        os.makedirs(os.path.join(proj_dir, sub), exist_ok=True)
    with open(os.path.join(proj_dir, "simvascular.proj"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<simvascular_project version="1.0"/>\n')

    # SV pretende il descrittore immagine anche senza immagine sorgente: senza
    # questo file l'apertura fallisce con "No project image location file found".
    # Campi immagine vuoti = progetto image-less (partiamo dal modello).
    now = int(time.time())
    with open(os.path.join(proj_dir, "Images", "image_information.xml"), "w") as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<ImageObjectInformation creation_time="{t}" version="1.0" modification_time="{t}">\n'
            '    <timestep id="0">\n'
            '        <created_with_simvascular_version>{ver}</created_with_simvascular_version>\n'
            '        <path></path>\n'
            '        <image_file_name></image_file_name>\n'
            '        <image_header_file_name></image_header_file_name>\n'
            '        <image_name></image_name>\n'
            '        <data_is_local_copy>false</data_is_local_copy>\n'
            '        <scale_factor>1</scale_factor>\n'
            '    </timestep>\n'
            '</ImageObjectInformation>\n'.format(t=now, ver=args.sv_version)
        )

    models_dir = os.path.join(proj_dir, "Models")
    model_out = os.path.join(models_dir, args.model_name)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(model_out)
    w.SetInputData(pd)
    w.Write()

    # 5) OPZIONALE: validazione con l'API SV (solo lettura verificata). Non fatale.
    try:
        import sv
        m = sv.modeling.Modeler(sv.modeling.Kernel.POLYDATA).read(model_out)
        got = sorted(int(i) for i in m.get_face_ids())
        exp = sorted(fc["model_face_id"] for fc in faces)
        if got != exp:
            print("[WARN] SV rilegge face_ids {} != attesi {}".format(got, exp), flush=True)
        else:
            print("[DEBUG] SV conferma le facce: {}".format(got), flush=True)
    except Exception as exc:
        print("[WARN] validazione sv.modeling saltata: {}".format(exc), flush=True)

    # 6) faces_named.json = SINGLE SOURCE OF TRUTH (post-combine)
    faces_named = {
        "units": "cm",
        "project": project,
        "model": os.path.relpath(model_out, out_dir),
        "id_convention": "1=wall, 2=inlet, 3..=outlet A,B,C,...",
        "wall_id": 1,
        "inlet_id": 2,
        "outlet_ids": [fc["model_face_id"] for fc in faces if fc["type"] == "cap" and fc["name"] != "inlet"],
        "faces": faces,
    }
    faces_json = os.path.join(out_dir, "faces_named.json")
    with open(faces_json, "w") as f:
        json.dump(faces_named, f, indent=2)

    print("[OK] modello nominato: {}".format(model_out), flush=True)
    print("[OK] scritto {}".format(faces_json), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        sys.exit(_die("eccezione non gestita: {}".format(exc)))