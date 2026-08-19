"""sv_apply.py 

Crea un progetto SimVascular headless, rinomina le facce secondo face_roles.json,
applica i nomi/tipi nativi tramite sv.modeling e registra il modello nel progetto.

CONVENZIONE ID FINALI:
    1 = wall, 2 = inlet, 3.. = outlet A, B, C, ...

Uso:
    simvascular --python -- sv_apply.py --input <out-dir>/model.vtp --out-dir <out-dir>
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

# API Nativa Modeler di SimVascular
import sv

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
    p = argparse.ArgumentParser(description="SimVascular: applica ruoli e registra il modello nel progetto.")
    p.add_argument("--input", required=True, help="model.vtp con ModelFaceID.")
    p.add_argument("--out-dir", required=True, help="cartella del paziente (contiene face_roles.json).")
    p.add_argument("--roles", default="face_roles.json", help="nome del file ruoli in out-dir.")
    p.add_argument("--model-name", default=None,
                   help="nome del solid model dentro <project>/Models/ (default <pid>).")
    p.add_argument("--sv-version", default="23.03.27", help="versione di SimVascular.")
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
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
    outlet_ids = [o["face_id"] for o in outlets]
    return int(inlet), [int(x) for x in wall], [int(x) for x in outlet_ids]


def build_mapping(inlet, wall, outlets):
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
    project_name = "cfd{}".format(pid)
    
    model_base_name = args.model_name if args.model_name else pid
    if model_base_name.endswith(".vtp"):
        model_base_name = os.path.splitext(model_base_name)[0]

    if not os.path.exists(in_path):
        return _die("input non trovato: {}".format(in_path))
    roles_path = os.path.join(out_dir, args.roles)
    if not os.path.exists(roles_path):
        return _die("face_roles non trovato: {} (esegui prima sv_match).".format(roles_path))

    # 1) Leggi ruoli + modello
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

    # 2) Costruisci mappa ed esegui la verifica di copertura
    try:
        mapping, faces = build_mapping(inlet, wall, outlets)
    except Exception as exc:
        return _die(str(exc))

    present = set(int(x) for x in np.unique(old))
    declared = set(mapping)
    missing_in_roles = present - declared
    missing_in_model = declared - present
    if missing_in_roles:
        return _die("ModelFaceID {} presenti nel modello ma non in face_roles.json".format(sorted(missing_in_roles)))
    if missing_in_model:
        return _die("ModelFaceID {} in face_roles.json ma assenti dal modello".format(sorted(missing_in_model)))

    # 3) Rimappa l'array ModelFaceID (Combine & Rename)
    new = np.fromiter((mapping[int(x)] for x in old), dtype=np.int32, count=len(old))
    new_arr = numpy_to_vtk(new, deep=1, array_type=vtk.VTK_INT)
    new_arr.SetName(FACE_ID_ARRAY)
    pd.GetCellData().RemoveArray(FACE_ID_ARRAY)
    pd.GetCellData().AddArray(new_arr)

    # 4) Generazione Albero Progetto
    proj_dir = os.path.join(out_dir, project_name)
    for sub in ("Images", "Paths", "Segmentations", "Models", "Meshes", "Simulations"):
        os.makedirs(os.path.join(proj_dir, sub), exist_ok=True)

    with open(os.path.join(proj_dir, "simvascular.proj"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<simvascular_project version="1.0"/>\n')

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

    # 5) Salvataggio File Solid Model (.vtp) e File Descrittori SimVascular (.mdl e .xml)
    models_dir = os.path.join(proj_dir, "Models")
    vtp_out = os.path.join(models_dir, "{}.vtp".format(model_base_name))
    mdl_out = os.path.join(models_dir, "{}.mdl".format(model_base_name))
    xml_out = os.path.join(models_dir, "{}.xml".format(model_base_name))

    # Scrittura VTP
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtp_out)
    writer.SetInputData(pd)
    writer.Write()

    # Scrittura .mdl con unità di misura esplicite
    mdl_lines = [
        '<?xml version="1.0" encoding="UTF-8" ?>',
        '<format version="1.0" />',
        '<model type="PolyData" units="cm">',  # <-- Aggiunto units="cm"
        '    <faces>'
    ]
    for fc in faces:
        mdl_lines.append('        <face id="{}" name="{}" type="{}" />'.format(
            fc["model_face_id"], fc["name"], fc["type"]
        ))
    mdl_lines.extend(['    </faces>', '</model>'])
    
    with open(mdl_out, "w") as f:
        f.write("\n".join(mdl_lines))

    # Scrittura .xml (File Registro Modello per la GUI)
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model_file version="1.0">',
        '    <model name="{}" type="PolyData">'.format(model_base_name),
        '        <vtp_file_name>{}</vtp_file_name>'.format(os.path.basename(vtp_out)),
        '        <mdl_file_name>{}</mdl_file_name>'.format(os.path.basename(mdl_out)),
        '    </model>',
        '</model_file>'
    ]
    with open(xml_out, "w") as f:
        f.write("\n".join(xml_lines))

    # 6) Validazione/Assegnazione facoltativa via sv.modeling (senza usare sv.project)
    try:
        modeler = sv.modeling.Modeler(sv.modeling.Kernel.POLYDATA)
        sv_model = modeler.read(vtp_out)
        got_ids = sorted(int(i) for i in sv_model.get_face_ids())
        print("[DEBUG] Modello letto correttamente da sv.modeling. Face IDs trovati: {}".format(got_ids), flush=True)
    except Exception as exc:
        print("[WARN] Validazione sv.modeling saltata: {}".format(exc), flush=True)

    # 7) faces_named.json (Single Source of Truth per la pipeline)
    faces_named = {
        "units": "cm",
        "project": project_name,
        "model": os.path.relpath(vtp_out, out_dir),
        "id_convention": "1=wall, 2=inlet, 3..=outlet A,B,C,...",
        "wall_id": 1,
        "inlet_id": 2,
        "outlet_ids": [fc["model_face_id"] for fc in faces if fc["type"] == "cap" and fc["name"] != "inlet"],
        "faces": faces,
    }
    faces_json = os.path.join(out_dir, "faces_named.json")
    with open(faces_json, "w") as f:
        json.dump(faces_named, f, indent=2)

    print("[OK] Progetto SimVascular salvato e registrato: {}".format(vtp_out), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        sys.exit(_die("eccezione non gestita: {}".format(exc)))