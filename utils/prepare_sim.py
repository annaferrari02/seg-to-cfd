"""Prepara una simulazione svMultiPhysics completa SENZA la GUI di simulazione di SimVascular.

Da SimVascular servono solo il modello (Models/<m>.mdl, per i nomi delle facce) e la mesh
(Meshes/<mesh>.vtu volumetrica + Meshes/<mesh>.vtp di superficie con ModelFaceID).
Da setup_bcs_new.py servono rcrparams_<config>.txt e inflow.reference.scaled.

Genera in --out:
    mesh-complete/mesh-complete.mesh.vtu        mesh volumetrica (copia)
    mesh-complete/mesh-complete.exterior.vtp    superficie completa (copia)
    mesh-complete/mesh-surfaces/<nome>.vtp      una faccia per ModelFaceID, nome dal .mdl
    mesh-complete/walls_combined.vtp            tutte le facce di tipo wall
    inlet.flow                                  copia di inflow.reference.scaled
    init_pressure.vtu                           pressione iniziale uniforme del fluido (default MAP)
    solver.xml                                  dal template: facce, BC, RCR con Initial_pressure, n. passi

Uso:
    python3.10 prepare_sim.py --mdl P/Models/pz097_cm.mdl --mesh-vtu P/Meshes/pz097.vtu \
        --mesh-vtp P/Meshes/pz097.vtp --rcr P/rcrparams_les.txt --inflow P/inflow.reference.scaled \
        --template solver_template.xml --out P/Simulations/pz097_les --cycles 16

Validazione (consigliata la prima volta): con --compare-with <mesh-complete di SimVascular> confronta
le facce generate con quelle scritte da SimVascular per la STESSA mesh.
"""
import argparse
import copy
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.setup_bcs import map_from_mdl, read_flow, _trapz      # noqa: E402
from utils.init_pressure import write_init_pressure              # noqa: E402


def fail(msg):
    raise RuntimeError(msg)


def read_vtu(path):
    r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(path); r.Update()
    o = r.GetOutput()
    if o.GetNumberOfPoints() == 0:
        fail(f"mesh volumetrica vuota o non leggibile: {path}")
    return o


def read_vtp(path):
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(path); r.Update()
    o = r.GetOutput()
    if o.GetNumberOfCells() == 0:
        fail(f"superficie vuota o non leggibile: {path}")
    return o


def write_vtp(poly, path):
    w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(path); w.SetInputData(poly)
    if w.Write() != 1:
        fail(f"scrittura fallita: {path}")


def arr(data, name, where):
    a = (data.GetPointData() if where == "point" else data.GetCellData()).GetArray(name)
    return None if a is None else vtk_to_numpy(a)


def extract_faces(surf, face_ids):
    """Polydata con le sole celle il cui ModelFaceID e' in face_ids (array e punti usati conservati)."""
    mid = arr(surf, "ModelFaceID", "cell")
    sel = vtk.vtkIdTypeArray()
    for c in np.where(np.isin(mid, list(face_ids)))[0]:
        sel.InsertNextValue(int(c))
    node = vtk.vtkSelectionNode()
    node.SetFieldType(vtk.vtkSelectionNode.CELL)
    node.SetContentType(vtk.vtkSelectionNode.INDICES)
    node.SetSelectionList(sel)
    s = vtk.vtkSelection(); s.AddNode(node)
    ex = vtk.vtkExtractSelection(); ex.SetInputData(0, surf); ex.SetInputData(1, s); ex.Update()
    g = vtk.vtkGeometryFilter(); g.SetInputData(ex.GetOutput()); g.Update()
    out = vtk.vtkPolyData(); out.DeepCopy(g.GetOutput())
    for name in ("vtkOriginalCellIds", "vtkOriginalPointIds"):   # array di servizio aggiunti da VTK
        out.GetCellData().RemoveArray(name); out.GetPointData().RemoveArray(name)
    return out


def tri_areas(poly):
    """Area di ogni triangolo di un polydata."""
    pts = vtk_to_numpy(poly.GetPoints().GetData())
    conn = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    a, b, c = pts[conn[:, 0]], pts[conn[:, 1]], pts[conn[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def check_faces(vol, surf, faces, id2name, cap_ids, model_area, ext_max, max_cap_dev=None):
    """Controlli di coerenza delle facce, indipendenti da SimVascular. Ritorna il numero di errori."""
    errors = 0
    n = vol.GetNumberOfPoints()

    # indice volumetrico di ogni GlobalNodeID / GlobalElementID
    g_v = arr(vol, "GlobalNodeID", "point")
    node_idx = np.empty(n + 1, dtype=np.int64)
    node_idx[g_v if g_v is not None else np.arange(1, n + 1)] = np.arange(n)
    e_v = arr(vol, "GlobalElementID", "cell")
    m = vol.GetNumberOfCells()
    elem_idx = np.full((e_v.max() if e_v is not None else m) + 1, -1, dtype=np.int64)
    elem_idx[e_v if e_v is not None else np.arange(1, m + 1)] = np.arange(m)
    conn = vtk_to_numpy(vol.GetCells().GetConnectivityArray())
    offs = vtk_to_numpy(vol.GetCells().GetOffsetsArray())

    # 1. ogni triangolo appartiene al tetraedro indicato da GlobalElementID
    print("\nControlli di coerenza delle facce:")
    for f in sorted(faces, key=lambda f: id2name[f]):
        face = faces[f]
        g = arr(face, "GlobalNodeID", "point")
        e = arr(face, "GlobalElementID", "cell")
        tri = vtk_to_numpy(face.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        bad = 0
        for k in range(len(tri)):
            ci = elem_idx[e[k]] if 0 < e[k] < len(elem_idx) else -1
            if ci < 0 or not set(node_idx[g[tri[k]]]).issubset(conn[offs[ci]:offs[ci + 1]]):
                bad += 1
        if bad:
            print(f"  {id2name[f]}: {bad}/{len(tri)} triangoli NON appartengono al GlobalElementID indicato")
            errors += 1
    if not errors:
        print("  1. GlobalElementID: ogni triangolo appartiene al suo elemento volumetrico - ok")

    # 2. le facce coprono esattamente il bordo della mesh volumetrica
    sf = vtk.vtkDataSetSurfaceFilter(); sf.SetInputData(vol); sf.PassThroughPointIdsOn(); sf.Update()
    tf = vtk.vtkTriangleFilter(); tf.SetInputConnection(sf.GetOutputPort()); tf.Update()
    bnd = tf.GetOutput()
    orig = vtk_to_numpy(bnd.GetPointData().GetArray("vtkOriginalPointIds")).astype(np.int64)
    g_b = g_v[orig] if g_v is not None else orig + 1              # GlobalNodeID dei nodi di bordo
    tri_b = vtk_to_numpy(bnd.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    key_b = set(map(tuple, np.sort(g_b[tri_b], axis=1)))
    keys = []
    for face in faces.values():
        t = vtk_to_numpy(face.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        keys += map(tuple, np.sort(arr(face, "GlobalNodeID", "point")[t], axis=1))
    dup = len(keys) - len(set(keys))
    if dup or set(keys) != key_b:
        print(f"  2. copertura del bordo: ERRORE ({len(set(keys) - key_b)} triangoli fuori dal bordo, "
              f"{len(key_b - set(keys))} triangoli di bordo scoperti, {dup} duplicati)")
        errors += 1
    else:
        print(f"  2. copertura del bordo: {len(keys)} triangoli, ogni triangolo di bordo in una sola faccia - ok")

    # 3-4. area e planarita' dei cap
    print(f"  {'cap':>8} {'area mesh':>10} {'area modello':>12} {'rapporto':>9} {'non planarita':>14}")
    for f in sorted(cap_ids, key=lambda f: id2name[f]):
        face = faces[f]
        a_mesh = tri_areas(face).sum()
        ratio = a_mesh / model_area[f]
        pts = vtk_to_numpy(face.GetPoints().GetData())
        ctr = pts - pts.mean(axis=0)
        dev = np.abs(ctr @ np.linalg.svd(ctr, full_matrices=False)[2][-1]).max()
        r_eq = np.sqrt(a_mesh / np.pi)
        flag = ""
        if abs(ratio - 1) > 0.05:          # tolleranza: cap piccoli con pochi triangoli
            flag += " AREA"; errors += 1
        # pipeline: the VMTK remesh moves the cap rim by ~half an element on every cap, so the
        # tolerance is absolute (max_cap_dev = gmes); original rule (2% of r) without it
        if (dev > max_cap_dev) if max_cap_dev is not None else (dev / r_eq > 0.02):
            flag += " NON PIANO"; errors += 1
        print(f"  {id2name[f]:>8} {a_mesh:>10.5f} {model_area[f]:>12.5f} {ratio:>9.4f} "
              f"{dev / r_eq:>9.2%} ({dev:.3f} cm){flag}")
    return errors


def set_child(parent, tag, text):
    el = parent.find(tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    el.text = str(text)
    return el


def prepare(mdl, mesh_vtu, mesh_vtp, rcr, inflow, template, out, cycles, save_cycles=1,
            map_val=119989.8, distal_pressure=0.0, init_pressure_value=None, compare_with=None,
            force=False, max_cap_dev=None):
    """Same as the command line, callable from the pipeline (step6)."""
    from types import SimpleNamespace
    _prepare(SimpleNamespace(mdl=mdl, mesh_vtu=mesh_vtu, mesh_vtp=mesh_vtp, rcr=rcr, inflow=inflow,
                             template=template, out=out, cycles=cycles, save_cycles=save_cycles,
                             map=map_val, distal_pressure=distal_pressure,
                             init_pressure_value=init_pressure_value, compare_with=compare_with,
                             force=force, max_cap_dev=max_cap_dev))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mdl", required=True, help=".mdl del modello (nomi e tipi delle facce)")
    p.add_argument("--mesh-vtu", required=True, help="mesh volumetrica da Meshes/")
    p.add_argument("--mesh-vtp", required=True, help="superficie della mesh da Meshes/ (con ModelFaceID)")
    p.add_argument("--rcr", required=True, help="rcrparams_<config>.txt da setup_bcs_new.py")
    p.add_argument("--inflow", required=True, help="inflow.reference.scaled da setup_bcs_new.py")
    p.add_argument("--template", required=True, help="solver.xml validato da cui prendere i parametri fissi")
    p.add_argument("--out", required=True, help="cartella della simulazione da creare")
    p.add_argument("--cycles", type=int, required=True, help="numero di cicli cardiaci da simulare")
    p.add_argument("--save-cycles", type=int, default=1, help="cicli finali di cui salvare i VTU. Default: 1")
    p.add_argument("--map", type=float, default=119989.8, help="MAP [dyn/cm^2]. Default: 119989.8")
    p.add_argument("--distal-pressure", type=float, default=0.0, help="Distal_pressure. Default: 0")
    p.add_argument("--init-pressure-value", type=float, default=None, help="pressione iniziale fluido. Default: MAP")
    p.add_argument("--compare-with", default=None, help="mesh-complete di SimVascular per validare le facce")
    p.add_argument("--force", action="store_true", help="sovrascrive una cartella di simulazione esistente")
    p.add_argument("--max-cap-dev", type=float, default=None,
                   help="max distance of a cap from its plane [cm]. Default: 2%% of the cap radius")
    _prepare(p.parse_args())


def _prepare(a):
    out = os.path.abspath(a.out)
    if os.path.exists(os.path.join(out, "solver.xml")) and not a.force:
        fail(f"{out} contiene gia' un solver.xml (usare --force per sovrascrivere)")

    # ---------------- 1. nomi e tipi delle facce ----------------
    id2name, id2type = map_from_mdl(a.mdl)
    inlet = [f for f, n in id2name.items() if n == "inlet"]
    if len(inlet) != 1:
        fail(f"{len(inlet)} facce chiamate 'inlet' nel .mdl")
    inlet_id = inlet[0]
    wall_ids = {f for f, t in id2type.items() if t == "wall"}
    if not wall_ids:
        fail("nessuna faccia di tipo 'wall' nel .mdl")
    if len(set(id2name.values())) != len(id2name):
        fail("nomi di faccia duplicati nel .mdl")

    # ---------------- 2. mesh ----------------
    vol = read_vtu(a.mesh_vtu)
    surf = read_vtp(a.mesh_vtp)
    b = np.array(vol.GetBounds()); ext = b[1::2] - b[::2]
    print(f"Mesh volumetrica: {vol.GetNumberOfPoints()} nodi, {vol.GetNumberOfCells()} celle, "
          f"estensione {np.round(ext, 2)}")
    if ext.max() > 100:
        fail(f"estensione massima {ext.max():.1f}: la mesh sembra in mm, il solver lavora in cm")
    for name, where in (("ModelFaceID", "cell"), ("GlobalNodeID", "point"), ("GlobalElementID", "cell")):
        if arr(surf, name, where) is None:
            fail(f"array {where} '{name}' assente in {a.mesh_vtp}: non posso costruire le facce")
    gnid_vol = arr(vol, "GlobalNodeID", "point")
    gnid_surf = arr(surf, "GlobalNodeID", "point")
    n = vol.GetNumberOfPoints()
    if gnid_vol is not None and not np.array_equal(np.sort(gnid_vol), np.arange(1, n + 1)):
        fail("GlobalNodeID della mesh volumetrica non e' 1..N")
    if gnid_surf.min() < 1 or gnid_surf.max() > n:
        fail("GlobalNodeID della superficie fuori dall'intervallo dei nodi volumetrici")
    if gnid_vol is not None:   # la superficie deve stare sugli stessi nodi del volume
        pts_v = vtk_to_numpy(vol.GetPoints().GetData())
        pts_s = vtk_to_numpy(surf.GetPoints().GetData())
        order = np.empty(n, dtype=np.int64); order[gnid_vol - 1] = np.arange(n)
        err = np.abs(pts_v[order[gnid_surf - 1]] - pts_s).max()
        if err > 1e-6 * ext.max():
            fail(f"superficie e volume non coincidono (scarto {err:.3g}): file di mesh diverse?")
    mesh_ids = set(np.unique(arr(surf, "ModelFaceID", "cell")).astype(int))
    if mesh_ids != set(id2name):
        fail(f"ModelFaceID della mesh {sorted(mesh_ids)} diversi da quelli del .mdl {sorted(id2name)}")

    # ---------------- 3. RCR ----------------
    rcr, model_area = {}, {}
    for line in open(a.rcr, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            f, area, r1, c, r2 = line.split()[:5]
            rcr[int(f)] = (float(r1), float(c), float(r2))
            model_area[int(f)] = float(area)
    if set(rcr) != set(id2name):
        fail(f"face_id di {a.rcr} {sorted(rcr)} diversi da quelli del .mdl {sorted(id2name)}")
    outlets = sorted((f for f in rcr if f != inlet_id and f not in wall_ids), key=lambda f: id2name[f])
    for f in outlets:
        r1, c, r2 = rcr[f]
        if min(r1, c, r2) <= 0:
            fail(f"RCR nulla o negativa per l'outlet {id2name[f]} (id {f})")
        if not (r2 > r1 and c < 1e-2):
            fail(f"RCR sospetta per {id2name[f]}: R1={r1:g} C={c:g} R2={r2:g} (colonne scambiate?)")
    for f in [inlet_id, *wall_ids]:
        if any(rcr[f]):
            fail(f"la faccia {id2name[f]} (inlet/wall) ha valori RCR non nulli in {a.rcr}")

    # ---------------- 4. waveform e accoppiamento con le RCR ----------------
    t, q = read_flow(a.inflow)
    T = t[-1] - t[0]
    q_mean = _trapz(q, t) / T
    q_rcr = sum((a.map - a.distal_pressure) / (rcr[f][0] + rcr[f][2]) for f in outlets)
    print(f"Waveform: {len(t)} punti, T = {T:.4g} s, media {q_mean:.6g} mL/s | "
          f"somma flussi RCR {q_rcr:.6g} mL/s")
    if q_mean >= 0:
        fail("la media della waveform deve essere negativa (flusso entrante all'inlet)")
    if abs(-q_mean - q_rcr) / q_rcr > 1e-3:
        fail("la media della waveform non coincide con la somma dei flussi delle RCR: "
             "inflow e rcrparams vengono da esecuzioni diverse di setup_bcs_new.py")

    # ---------------- 5. scrittura mesh-complete ----------------
    ms = os.path.join(out, "mesh-complete", "mesh-surfaces")
    os.makedirs(ms, exist_ok=True)
    shutil.copyfile(a.mesh_vtu, os.path.join(out, "mesh-complete", "mesh-complete.mesh.vtu"))
    shutil.copyfile(a.mesh_vtp, os.path.join(out, "mesh-complete", "mesh-complete.exterior.vtp"))
    ncell, faces = {}, {}
    for f, name in sorted(id2name.items()):
        face = extract_faces(surf, {f})
        faces[f], ncell[f] = face, face.GetNumberOfCells()
        write_vtp(face, os.path.join(ms, f"{name}.vtp"))
    write_vtp(extract_faces(surf, wall_ids), os.path.join(out, "mesh-complete", "walls_combined.vtp"))
    print(f"Facce scritte in {ms}: " + ", ".join(f"{id2name[f]}({ncell[f]})" for f in sorted(ncell)))
    n_err = check_faces(vol, surf, faces, id2name, [inlet_id, *outlets], model_area, ext.max(),
                        getattr(a, "max_cap_dev", None))
    if n_err:
        fail(f"{n_err} controlli di coerenza delle facce falliti (vedi sopra): simulazione non preparata")

    # ---------------- 6. inlet.flow e init_pressure.vtu ----------------
    shutil.copyfile(a.inflow, os.path.join(out, "inlet.flow"))
    p0 = a.map if a.init_pressure_value is None else a.init_pressure_value
    write_init_pressure(os.path.join(out, "mesh-complete", "mesh-complete.mesh.vtu"),
                        os.path.join(out, "init_pressure.vtu"), p0)

    # ---------------- 7. solver.xml ----------------
    tree = ET.parse(a.template); root = tree.getroot()
    gen = root.find("GeneralSimulationParameters")
    seen = {}
    for el in list(gen):                       # tag duplicati: tengo il primo
        if el.tag in seen:
            if (el.text or "").strip() != (seen[el.tag].text or "").strip():
                fail(f"tag {el.tag} duplicato con valori diversi nel template")
            gen.remove(el); print(f"Template: rimosso duplicato {el.tag}")
        else:
            seen[el.tag] = el
    dt = float(gen.find("Time_step_size").text)
    steps_cycle = T / dt
    if abs(steps_cycle - round(steps_cycle)) > 1e-6:
        fail(f"il periodo {T} s non e' multiplo del passo {dt} s")
    steps_cycle = int(round(steps_cycle))
    if not (1 <= a.save_cycles <= a.cycles):
        fail("--save-cycles deve essere tra 1 e --cycles")
    n_steps = a.cycles * steps_cycle
    start_save = (a.cycles - a.save_cycles) * steps_cycle + 1
    set_child(gen, "Continue_previous_simulation", "false")
    set_child(gen, "Number_of_time_steps", n_steps)
    set_child(gen, "Start_saving_after_time_step", start_save)

    mesh = root.find("Add_mesh")
    for el in mesh.findall("Add_face") + mesh.findall("Initial_pressures_file_path"):
        mesh.remove(el)
    set_child(mesh, "Mesh_file_path", "mesh-complete/mesh-complete.mesh.vtu")
    ip = ET.Element("Initial_pressures_file_path"); ip.text = "init_pressure.vtu"
    mesh.insert(list(mesh).index(mesh.find("Mesh_file_path")) + 1, ip)
    for f in sorted(id2name, key=lambda f: id2name[f]):
        af = ET.SubElement(mesh, "Add_face", name=id2name[f])
        ET.SubElement(af, "Face_file_path").text = f"mesh-complete/mesh-surfaces/{id2name[f]}.vtp"

    eq = root.find("Add_equation[@type='fluid']")
    bcs = eq.findall("Add_BC")
    tpl_inlet = [b_ for b_ in bcs if b_.find("Temporal_values_file_path") is not None]
    tpl_rcr = [b_ for b_ in bcs if (b_.findtext("Time_dependence") or "").strip() == "RCR"]
    tpl_wall = [b_ for b_ in bcs if b_ not in tpl_inlet and b_ not in tpl_rcr]
    if len(tpl_inlet) != 1 or not tpl_rcr or len(tpl_wall) != 1:
        fail("il template deve avere 1 BC di inlet con file di flusso, almeno 1 RCR e 1 BC di parete")
    for b_ in bcs:
        eq.remove(b_)
    bc = copy.deepcopy(tpl_inlet[0]); bc.set("name", id2name[inlet_id])
    set_child(bc, "Temporal_values_file_path", "inlet.flow"); eq.append(bc)
    for f in sorted(wall_ids, key=lambda f: id2name[f]):
        bc = copy.deepcopy(tpl_wall[0]); bc.set("name", id2name[f]); eq.append(bc)
    print(f"\n{'outlet':>8} {'id':>4} {'R1':>11} {'C':>11} {'R2':>11} {'Initial_p':>11} {'q[mL/s]':>8}")
    for f in outlets:
        r1, c, r2 = rcr[f]
        p_init = (a.map - a.distal_pressure) * r2 / (r1 + r2) + a.distal_pressure
        bc = copy.deepcopy(tpl_rcr[0]); bc.set("name", id2name[f])
        rv = bc.find("RCR_values")
        for k in list(rv):
            rv.remove(k)
        for tag, val in (("Proximal_resistance", r1), ("Capacitance", c), ("Distal_resistance", r2),
                         ("Distal_pressure", a.distal_pressure), ("Initial_pressure", p_init)):
            ET.SubElement(rv, tag).text = f"{val:.12g}"
        eq.append(bc)
        print(f"{id2name[f]:>8} {f:>4} {r1:>11.5g} {c:>11.4g} {r2:>11.5g} {p_init:>11.6g} "
              f"{(a.map - a.distal_pressure) / (r1 + r2):>8.3f}")

    ET.indent(tree, space="    ")
    xml_out = os.path.join(out, "solver.xml")
    tree.write(xml_out, encoding="unicode")
    ET.parse(xml_out)                          # rilettura: XML valido
    print(f"\nsolver.xml: {n_steps} passi ({a.cycles} cicli x {steps_cycle}), dt = {dt}, "
          f"VTU dal passo {start_save} (ultimi {a.save_cycles} cicli)")

    # ---------------- 8. validazione facoltativa contro SimVascular ----------------
    if a.compare_with:
        ref = os.path.join(a.compare_with, "mesh-surfaces")
        print(f"\nConfronto con {ref}:")
        bad = 0
        for f, name in sorted(id2name.items()):
            rp = os.path.join(ref, f"{name}.vtp")
            if not os.path.isfile(rp):
                print(f"  {name}: assente nel riferimento"); bad += 1; continue
            mine, theirs = read_vtp(os.path.join(ms, f"{name}.vtp")), read_vtp(rp)
            ok = (mine.GetNumberOfCells() == theirs.GetNumberOfCells()
                  and set(arr(mine, "GlobalNodeID", "point")) == set(arr(theirs, "GlobalNodeID", "point"))
                  and set(arr(mine, "GlobalElementID", "cell")) == set(arr(theirs, "GlobalElementID", "cell")))
            bad += not ok
            print(f"  {name}: {'uguale' if ok else 'DIVERSA'} ({mine.GetNumberOfCells()} vs "
                  f"{theirs.GetNumberOfCells()} celle)")
        print("Facce identiche a quelle di SimVascular." if not bad else f"ATTENZIONE: {bad} facce diverse.")

    print(f"\nSimulazione pronta in {out}")


if __name__ == "__main__":
    main()