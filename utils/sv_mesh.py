"""TetGen meshing in SimVascular (SV world: sv + vtk).
Files as SimVascular writes them:
  GUI mesh   : <project>/Meshes/<name>.msh (+ .vtp surface, .vtu volume, same basename)
  mesh-complete (for the solver): mesh-complete.mesh.vtu, mesh-complete.exterior.vtp,
               walls_combined.vtp, mesh-surfaces/<face name>.vtp"""
import os
import sys

import vtk

from utils import config

FACE_ARRAY = "ModelFaceID"


class NativeOutput(object):
    """Redirect the C-level stdout/stderr (TetGen, VMTK) to a log file."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._out, self._err = os.dup(1), os.dup(2)
        self._fh = open(self.path, "a")
        os.dup2(self._fh.fileno(), 1)
        os.dup2(self._fh.fileno(), 2)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._out, 1)
        os.dup2(self._err, 2)
        os.close(self._out)
        os.close(self._err)
        self._fh.close()
        return False


def build_mesher(model_vtp, wall_ids, p, portion):
    """TetGen mesher with SimVascular GUI defaults, except use_mmg=False:
    MMG drops ModelFaceID, the VMTK path keeps it."""
    import sv
    mesher = sv.meshing.TetGen()
    mesher.load_model(model_vtp)
    ids = sorted(int(i) for i in mesher.get_model_face_ids())
    missing = [w for w in wall_ids if w not in ids]
    if missing:
        raise RuntimeError(f"Wall ids {missing} not in the model faces {ids}")
    mesher.set_walls(list(wall_ids))

    opt = sv.meshing.TetGenOptions(global_edge_size=p["gmes"], surface_mesh_flag=True, volume_mesh_flag=True)
    if not hasattr(opt, "use_mmg"):
        raise RuntimeError("TetGenOptions has no use_mmg in this SimVascular build (run probe_sv_mesh.py)")
    opt.use_mmg = False
    if getattr(opt, "use_mmg") not in (False, 0):  # MMG drops ModelFaceID and can rescale to a unit box
        raise RuntimeError(f"use_mmg is {opt.use_mmg!r} after setting it to False (run probe_sv_mesh.py)")
    # the options exactly as SimVascular reads them in generate_mesh
    print(f"  TetGenOptions.get_values(): {opt.get_values()}")
    bl = p["boundary_layer"]
    if bl["on"]:
        mesher.set_boundary_layer_options(number_of_layers=bl["layers"], edge_size_fraction=portion,
                                          layer_decreasing_ratio=bl["decreasing_ratio"],
                                          constant_thickness=bl["constant_thickness"])
    return mesher, opt, ids


def commands(p, portion, opt):
    """Command history of the .msh, as the SV GUI writes it (sv4guiMeshEdit::CreateCmdsT)."""
    bl = p["boundary_layer"]
    cmds = ["option surface 1", "option volume 1", "option UseMMG 0",
            f"option GlobalEdgeSize {p['gmes']}", "setWalls"]
    if bl["on"]:
        cmds += [f"boundaryLayer {bl['layers']} {portion} {bl['decreasing_ratio']} {int(bl['constant_thickness'])}",
                 "option BoundaryLayerDirection 1"]
    for attr, cmd in (("optimization", "option Optimization"), ("quality_ratio", "option QualityRatio")):
        if hasattr(opt, attr):
            cmds.append(f"{cmd} {getattr(opt, attr)}")
    if getattr(opt, "no_bisect", True):
        cmds.append("option NoBisect")
    return cmds + ["AllowMultipleRegions 0", "generateMesh", "writeMesh"]


def _write(writer_cls, data, path):
    config.assert_writable(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = writer_cls()
    w.SetFileName(path)
    w.SetInputData(data)
    if not w.Write():
        raise RuntimeError(f"Cannot write {path}")


def check_ids(data, what):
    """The solver needs GlobalNodeID (points) and, for surfaces, GlobalElementID (cells)."""
    if data is None or data.GetNumberOfCells() == 0:
        raise RuntimeError(f"{what}: empty")
    if data.GetPointData().GetArray("GlobalNodeID") is None:
        raise RuntimeError(f"{what}: no GlobalNodeID")
    if data.IsA("vtkPolyData") and data.GetCellData().GetArray("GlobalElementID") is None:
        raise RuntimeError(f"{what}: no GlobalElementID")


def _threshold(pd, fid):
    """Cells of one ModelFaceID as polydata (point and cell arrays carried over)."""
    th = vtk.vtkThreshold()
    th.SetInputData(pd)
    th.SetInputArrayToProcess(0, 0, 0, 1, FACE_ARRAY)
    th.SetLowerThreshold(fid)
    th.SetUpperThreshold(fid)
    g = vtk.vtkGeometryFilter()
    g.SetInputConnection(th.GetOutputPort())
    g.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(g.GetOutput())
    return out


def _area(pd):
    q = vtk.vtkMassProperties()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd)
    q.SetInputConnection(tri.GetOutputPort())
    q.Update()
    return q.GetSurfaceArea()


def _read_vtp(path):
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def _cell_centers(pd):
    from vtk.util.numpy_support import vtk_to_numpy
    cc = vtk.vtkCellCenters()
    cc.SetInputData(pd)
    cc.Update()
    return vtk_to_numpy(cc.GetOutput().GetPoints().GetData()).astype(float)


def fix_cap_rings(surface, faces, model_vtp, plane_tol_frac=0.02, radial_frac=1.05, min_cos=0.95):
    """SimVascular (VMTK remesh + boundary layer) fails to give the boundary-layer ring of each
    cap its cap's ModelFaceID ('Failure in resetting model face id on boundary layer caps'): the
    ring ends up with a provisional id (9999) or in the wall. A ring cell lies in the plane of
    its cap, inside its radius, with the normal parallel to the cap normal (a wall cell next to
    the rim is perpendicular). Every non-cap cell (wall or unknown id) that passes the three
    tests gets the cap id; an unknown id left over is an error.
    Cap planes from the model: centre, normal (SVD of cap cell centres), radius from area.
    Returns (fixed surface, number of cells reassigned)."""
    import numpy as np
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    model = _read_vtp(model_vtp)
    caps = {}
    for fid, name in faces.items():
        if name == "wall":
            continue
        face = _threshold(model, fid)
        c = _cell_centers(face)
        sz = vtk.vtkCellSizeFilter()
        sz.SetInputData(face)
        sz.ComputeAreaOn()
        sz.Update()
        w = vtk_to_numpy(sz.GetOutput().GetCellData().GetArray("Area")).astype(float)
        centre = (c * w[:, None]).sum(0) / w.sum()  # area-weighted: the triangulation may be uneven
        normal = np.linalg.svd((c - centre) * np.sqrt(w)[:, None])[2][2]
        caps[fid] = (centre, normal, (_area(face) / np.pi) ** 0.5)

    out = vtk.vtkPolyData()
    out.DeepCopy(surface)
    ids = vtk_to_numpy(out.GetCellData().GetArray(FACE_ARRAY)).astype(np.int32).copy()
    nrm = vtk.vtkPolyDataNormals()
    nrm.SetInputData(out)
    nrm.ComputeCellNormalsOn()
    nrm.ComputePointNormalsOff()
    nrm.SplittingOff()
    nrm.ConsistencyOff()
    nrm.AutoOrientNormalsOff()
    nrm.Update()
    cell_n = vtk_to_numpy(nrm.GetOutput().GetCellData().GetNormals()).astype(float)
    centers = _cell_centers(out)

    cand = np.nonzero(~np.isin(ids, list(caps)))[0]  # wall or unknown id
    x, n = centers[cand], cell_n[cand]
    best = np.full(len(cand), -1)
    for fid, (centre, normal, radius) in caps.items():
        v = x - centre
        d_plane = np.abs(v @ normal)
        d_radial = np.linalg.norm(v - np.outer(v @ normal, normal), axis=1)
        ok = ((d_plane <= plane_tol_frac * radius) & (d_radial <= radial_frac * radius)
              & (np.abs(n @ normal) >= min_cos))
        best[ok] = fid
    moved = best >= 0
    from_ids = sorted(set(ids[cand][moved].tolist()))
    ids[cand[moved]] = best[moved]
    left = sorted(set(ids[~np.isin(ids, list(faces))].tolist()))
    if left:
        raise RuntimeError(f"Surface cells with unknown ModelFaceID {left} fit no cap plane")
    arr = numpy_to_vtk(ids, deep=True, array_type=vtk.VTK_INT)
    arr.SetName(FACE_ARRAY)
    out.GetCellData().RemoveArray(FACE_ARRAY)
    out.GetCellData().AddArray(arr)
    per_cap = {faces[f]: int((best == f).sum()) for f in caps}
    print(f"  cap rings: {int(moved.sum())} cells (from ids {from_ids}) reassigned to their cap {per_cap}")
    return out, int(moved.sum())


def write_gui_mesh(mesher, surface, meshes_dir, name, model_name, cmds):
    """<name>.msh + <name>.vtp (surface with fixed face ids) + <name>.vtu in Meshes/."""
    _write(vtk.vtkXMLUnstructuredGridWriter, mesher.get_mesh(), os.path.join(meshes_dir, f"{name}.vtu"))
    _write(vtk.vtkXMLPolyDataWriter, surface, os.path.join(meshes_dir, f"{name}.vtp"))
    lines = ['<?xml version="1.0" encoding="UTF-8" ?>',
             f'<mitk_mesh type="TetGen" model_name="{model_name}" version="1.0">',
             '    <timestep id="0">',
             '        <mesh type="TetGen">',
             '            <command_history>']
    lines += [f'                <command content="{c}" />' for c in cmds]
    lines += ['            </command_history>', '        </mesh>', '    </timestep>', '</mitk_mesh>']
    msh = os.path.join(meshes_dir, f"{name}.msh")
    with open(msh, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {msh} (+ .vtp, .vtu)")


def check_faces(surface, model_vtp, faces, min_area_ratio, max_dev):
    """Face ids of the mesh surface, as needed by the boundary conditions:
    1. exactly the expected ModelFaceID values (no provisional 9999 left);
    2. every cap is one connected, flat disc: max distance of its points from the cap plane
       <= max_dev (pass gmes: the remesh moves the rim nodes by ~half an element, the same
       absolute amount on every cap; more than one element means wall cells in the cap);
    3. area mesh/model >= min_area_ratio (only a gross error: at coarse gmes the cap rim of a
       small vessel sits slightly inside, so its area is 10-25% smaller - reported, not fatal)."""
    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy
    ids = sorted(set(vtk_to_numpy(surface.GetCellData().GetArray(FACE_ARRAY)).astype(int).tolist()))
    if ids != sorted(faces):
        raise RuntimeError(f"Mesh surface ModelFaceID {ids} != expected {sorted(faces)}")
    model = _read_vtp(model_vtp)
    bad = []
    for fid, name in sorted(faces.items()):
        face = _threshold(surface, fid)
        ref = _area(_threshold(model, fid))
        ratio = _area(face) / ref if ref > 0 else 0.0
        line = f"  face {fid} {name}: area mesh/model {ratio:.3f}"
        if name != "wall":
            conn = vtk.vtkPolyDataConnectivityFilter()
            conn.SetInputData(face)
            conn.SetExtractionModeToAllRegions()
            conn.Update()
            n_reg = conn.GetNumberOfExtractedRegions()
            p = vtk_to_numpy(face.GetPoints().GetData()).astype(float)
            used = np.unique(vtk_to_numpy(face.GetPolys().GetConnectivityArray()))
            p = p[used]
            c = p.mean(axis=0)
            normal = np.linalg.svd(p - c)[2][2]
            dev = float(np.abs((p - c) @ normal).max())
            line += f", regions {n_reg}, max distance from plane {dev:.3f} cm"
            if n_reg != 1 or dev > max_dev:
                bad.append((name, "regions" if n_reg != 1 else "not flat"))
        if ratio < min_area_ratio:
            bad.append((name, f"area {ratio:.3f}"))
        print(line)
    if bad:
        raise RuntimeError(f"Mesh faces not valid for the boundary conditions: {bad}")


def write_mesh_complete(mesher, surface, out_dir, faces):
    """mesh-complete export with one surface per face (from the surface with fixed face ids),
    named after the face. faces: {face_id: name}; wall faces also go into walls_combined.vtp."""
    vol = mesher.get_mesh()
    surf = surface
    check_ids(vol, "volume mesh")
    check_ids(surf, "exterior surface")
    _write(vtk.vtkXMLUnstructuredGridWriter, vol, os.path.join(out_dir, "mesh-complete.mesh.vtu"))
    _write(vtk.vtkXMLPolyDataWriter, surf, os.path.join(out_dir, "mesh-complete.exterior.vtp"))

    walls = vtk.vtkAppendPolyData()
    for fid, name in sorted(faces.items()):
        pd = _threshold(surface, fid)
        check_ids(pd, f"face {fid} {name}")
        _write(vtk.vtkXMLPolyDataWriter, pd, os.path.join(out_dir, "mesh-surfaces", f"{name}.vtp"))
        if name == "wall":
            walls.AddInputData(pd)
    walls.Update()
    clean = vtk.vtkCleanPolyData()  # merge the points shared by several wall faces
    clean.SetInputConnection(walls.GetOutputPort())
    clean.Update()
    _write(vtk.vtkXMLPolyDataWriter, clean.GetOutput(), os.path.join(out_dir, "walls_combined.vtp"))
    print(f"  saved {out_dir}/ (mesh-complete.mesh.vtu, exterior, walls_combined, "
          f"mesh-surfaces/{{{', '.join(n for _, n in sorted(faces.items()))}}}.vtp)")
    return {"n_points": int(vol.GetNumberOfPoints()), "n_cells": int(vol.GetNumberOfCells())}