"""
probe_modeling_api.py  --  World 2 (SimVascular), SOLO ISPEZIONE, non modifica nulla.

Scopo: dumpare la superficie API REALE di `sv.modeling` (e le parti rilevanti di
`sv.meshing`) sul build SV in uso, PRIMA di scrivere `sv_apply`. Stesso principio
del `--probe` di mesh_model.py: nessuna assunzione sull'API, si guarda cosa c'e'.

Le domande a cui questo probe deve rispondere per decidere COME implementare
combine-wall / naming / cap-wall type in sv_apply:
  Q1. Come si legge il modello con le faces gia' estratte da extract_faces? (Modeler/kernel)
  Q2. Esiste get_face_ids() e come sono numerate le faces?
  Q3. Esiste un COMBINE di faces nativo? (altrimenti fallback: riscrivere ModelFaceID via VTK)
  Q4. Esiste assegnazione NOME faccia nativa? (set_face_name / rename)
  Q5. Esiste assegnazione TYPE cap/wall nativa? (storicamente NO: issue #867)
  Q6. Il polydata del modello porta l'array cell "ModelFaceID"? (serve al fallback VTK)
  Q7. sv.meshing.TetGen espone set_walls() e come propaga i nomi alle mesh-surface?

Uso (adatta il path del binario al tuo):
  /usr/local/sv/simvascular/2025-12-21/simvascular --python -- probe_modeling_api.py \
      --model /path/datalake/pz000/<modello_da_extract_faces>.vtp
"""

import argparse
import sys
import sv


def _hr(title):
    print("\n" + "=" * 78, flush=True)
    print(f"== {title}", flush=True)
    print("=" * 78, flush=True)


def _members(obj, keywords=None):
    """Elenca i membri pubblici di obj; se keywords, solo quelli che matchano."""
    out = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        low = name.lower()
        if keywords is None or any(k in low for k in keywords):
            kind = "callable" if callable(getattr(obj, name, None)) else "attr"
            out.append(f"  {name}  [{kind}]")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Probe API sv.modeling / sv.meshing")
    p.add_argument("--model", required=False, default=None,
                   help="Path al .vtp del modello con faces estratte (output di extract_faces)")
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    return p.parse_args(argv)


def main():
    args = parse_args()

    _hr("import sv")
    try:
        import sv
        print("import sv: OK", flush=True)
        print("sv.version:", getattr(sv, "version", getattr(sv, "__version__", "n/a")), flush=True)
    except Exception as e:
        print(f"import sv FALLITO: {e}", flush=True)
        sys.exit(1)

    import sv.modeling as modeling

    _hr("dir(sv.modeling)")
    for line in _members(modeling):
        print(line, flush=True)

    # Q1 -- Kernel disponibili
    _hr("Q1: Kernel disponibili")
    Kernel = getattr(modeling, "Kernel", None)
    if Kernel is not None:
        for line in _members(Kernel):
            print(line, flush=True)
    else:
        print("Nessun sv.modeling.Kernel (API vecchia?): guardare i costruttori sopra.", flush=True)

    # Costruzione Modeler + lettura modello
    model = None
    _hr("Q1: costruzione Modeler e lettura modello")
    if args.model is None:
        print("[!] Nessun --model passato: salto lettura. Passa il .vtp di extract_faces "
              "per ispezionare le faces reali.", flush=True)
    else:
        # Provo diversi modi di costruire il Modeler POLYDATA senza assumere lo spelling esatto
        modeler = None
        for attempt in ("Kernel.POLYDATA", "'PolyData'", "'POLYDATA'"):
            try:
                if attempt == "Kernel.POLYDATA" and Kernel is not None:
                    modeler = modeling.Modeler(Kernel.POLYDATA)
                else:
                    modeler = modeling.Modeler(attempt.strip("'"))
                print(f"Modeler creato con {attempt}", flush=True)
                break
            except Exception as e:
                print(f"Modeler({attempt}) -> {type(e).__name__}: {e}", flush=True)
        if modeler is not None:
            try:
                model = modeler.read(args.model)
                print(f"modeler.read('{args.model}') OK -> {type(model)}", flush=True)
            except Exception as e:
                print(f"modeler.read fallito: {type(e).__name__}: {e}", flush=True)
                # fallback: alcuni build hanno PolyData().set_surface(vtk_pd)
                try:
                    import vtk
                    r = vtk.vtkXMLPolyDataReader()
                    r.SetFileName(args.model)
                    r.Update()
                    model = modeling.PolyData()
                    model.set_surface(r.GetOutput())
                    print("fallback PolyData().set_surface(vtkPolyData) OK", flush=True)
                except Exception as e2:
                    print(f"fallback set_surface fallito: {type(e2).__name__}: {e2}", flush=True)

    # Introspezione dell'oggetto model
    if model is not None:
        _hr("dir(model) -- membri rilevanti (face/cap/wall/name/combine/id/remesh/boundary/delete)")
        kw = ["face", "cap", "wall", "name", "combine", "id", "remesh",
              "boundary", "delete", "identify", "surface", "polydata", "write"]
        hits = _members(model, kw)
        if hits:
            for line in hits:
                print(line, flush=True)
        else:
            print("Nessun membro corrispondente: stampo TUTTO dir(model):", flush=True)
            for line in _members(model):
                print(line, flush=True)

        # Q2 -- face ids
        _hr("Q2: get_face_ids()")
        get_ids = getattr(model, "get_face_ids", None)
        if callable(get_ids):
            try:
                ids = get_ids()
                print(f"face ids: {list(ids)}", flush=True)
            except Exception as e:
                print(f"get_face_ids() errore: {type(e).__name__}: {e}", flush=True)
                ids = None
        else:
            print("Nessun get_face_ids(): cerca l'equivalente nell'elenco sopra.", flush=True)
            ids = None

        # Q3/Q4/Q5 -- combine / name / type nativi
        _hr("Q3/Q4/Q5: combine / set_name / cap-wall type NATIVI?")
        for probe_name in ("combine_faces", "set_face_name", "get_face_name",
                            "set_face_names", "rename_face", "identify_caps",
                            "compute_boundary_faces", "get_face_polydata"):
            fn = getattr(model, probe_name, None)
            print(f"  {probe_name:22s}: {'PRESENTE' if callable(fn) else 'assente'}", flush=True)
        print("\nNota storica: nei build vecchi combine + cap/wall type NON esistevano nel "
              "Python API (SimVascular issue #867). Se sopra risultano 'assente', si usa il "
              "fallback VTK (riscrittura ModelFaceID) + set_walls al meshing.", flush=True)

        # Q6 -- array ModelFaceID sul polydata (indispensabile per il fallback VTK)
        _hr("Q6: array cell 'ModelFaceID' sul polydata del modello")
        get_pd = getattr(model, "get_polydata", None)
        if callable(get_pd):
            try:
                pd = get_pd()
                cd = pd.GetCellData()
                names = [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]
                print(f"cell arrays: {names}", flush=True)
                print(f"ModelFaceID presente: {'ModelFaceID' in names}", flush=True)
                print(f"n_celle: {pd.GetNumberOfCells()}, n_punti: {pd.GetNumberOfPoints()}", flush=True)
            except Exception as e:
                print(f"get_polydata()/ispezione errore: {type(e).__name__}: {e}", flush=True)
        else:
            print("Nessun get_polydata(): verifica come estrarre il vtkPolyData del modello.", flush=True)

        # Q2-bis -- per ogni faccia: centroide + planarita' (utile a sv_apply per cap vs wall)
        _hr("Q2-bis: centroide e n_celle per faccia (aiuta a distinguere cap planari)")
        get_face_pd = getattr(model, "get_face_polydata", None)
        if callable(get_face_pd) and ids:
            import vtk
            for fid in ids:
                try:
                    fpd = get_face_pd(fid)
                    com = vtk.vtkCenterOfMass()
                    com.SetInputData(fpd)
                    com.SetUseScalarsAsWeights(False)
                    com.Update()
                    c = com.GetCenter()
                    print(f"  face {fid}: celle={fpd.GetNumberOfCells():6d}  "
                          f"centroide=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})", flush=True)
                except Exception as e:
                    print(f"  face {fid}: errore {type(e).__name__}: {e}", flush=True)

    # Q7 -- meshing: set_walls e naming mesh-surface
    _hr("Q7: sv.meshing.TetGen -- set_walls / load_model / naming")
    try:
        import sv.meshing as meshing
        tg = meshing.TetGen()
        for line in _members(tg, ["wall", "face", "name", "load", "model", "surface", "boundary"]):
            print(line, flush=True)
    except Exception as e:
        print(f"sv.meshing.TetGen introspezione errore: {type(e).__name__}: {e}", flush=True)

    _hr("FINE PROBE")
    print("Riporta l'output: da li' decidiamo API nativa vs fallback VTK per sv_apply.", flush=True)


if __name__ == "__main__":
    main()
