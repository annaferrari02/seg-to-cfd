"""SimVascular project utilities (SV world: vtk + numpy, `sv` only for the final check).
Formats as written by SimVascular itself (sv4gui_ProjectManager / sv4gui_ModelIO):
  <project>/simvascular.proj, Images/image_information.xml, one folder per plugin,
  Models/<name>.mdl (faces: id, name, type) + Models/<name>.vtp (same basename, ModelFaceID)."""
import json
import os
import time

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from utils import config

FACE_ARRAY = "ModelFaceID"
PLUGIN_DIRS = ("Images", "Meshes", "Models", "Paths", "Segmentations", "Simulations",
               "ROMSimulations", "MultiPhysics")
_COLORS = [(0.8, 0.8, 0.8), (0.9, 0.2, 0.2), (0.2, 0.5, 0.9), (0.2, 0.8, 0.3), (0.95, 0.75, 0.2),
           (0.7, 0.4, 0.85), (0.2, 0.8, 0.8), (0.95, 0.5, 0.2), (0.6, 0.3, 0.2), (0.9, 0.5, 0.7)]


def read_legacy(path):
    """Legacy .vtk surface -> triangles only, ModelFaceID as int, outward consistent normals
    (what the SV GUI does at import; inconsistent winding stalls SV face operations)."""
    r = vtk.vtkPolyDataReader()
    r.SetFileName(path)
    r.Update()
    src = r.GetOutput()
    if src.GetNumberOfPolys() == 0:
        raise RuntimeError(f"{path}: no polygons (not legacy VTK 4.2?)")
    if src.GetCellData().GetArray(FACE_ARRAY) is None:
        raise RuntimeError(f"{path}: no {FACE_ARRAY} cell array")

    pd = vtk.vtkPolyData()  # polygons only: a stray line/vertex cell breaks SV
    pd.SetPoints(src.GetPoints())
    pd.SetPolys(src.GetPolys())
    n_cells_before = src.GetNumberOfCells()
    first_poly = src.GetNumberOfVerts() + src.GetNumberOfLines()
    ids = vtk_to_numpy(src.GetCellData().GetArray(FACE_ARRAY)).astype(np.int32)
    ids = ids[first_poly:first_poly + src.GetNumberOfPolys()]
    arr = numpy_to_vtk(ids, deep=True, array_type=vtk.VTK_INT)
    arr.SetName(FACE_ARRAY)
    pd.GetCellData().AddArray(arr)
    if pd.GetNumberOfCells() != n_cells_before:
        print(f"  dropped {n_cells_before - pd.GetNumberOfCells()} non-polygon cells")

    # drop the points no polygon uses (left by the cells dropped above): invisible, but they
    # sit outside the vessel and end up in the SV model and in the mesh
    used = vtk.vtkCleanPolyData()
    used.SetInputData(pd)
    used.PointMergingOff()
    used.Update()
    n_orphan = pd.GetNumberOfPoints() - used.GetOutput().GetNumberOfPoints()
    if n_orphan:
        print(f"  dropped {n_orphan} points not used by any polygon")
    pd = used.GetOutput()

    n = vtk.vtkPolyDataNormals()
    n.SetInputData(pd)
    n.SplittingOff()
    n.ConsistencyOn()
    n.AutoOrientNormalsOn()
    n.ComputeCellNormalsOn()
    n.ComputePointNormalsOn()
    n.NonManifoldTraversalOff()
    n.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(n.GetOutput())
    if out.GetNumberOfCells() != pd.GetNumberOfCells():
        raise RuntimeError("normal orientation changed the number of cells")
    return out


def face_ids(pd):
    return sorted({int(v) for v in vtk_to_numpy(pd.GetCellData().GetArray(FACE_ARRAY))})


def face_centroids(pd):
    """{face_id: area-weighted centroid}."""
    cc = vtk.vtkCellCenters()
    cc.SetInputData(pd)
    cc.Update()
    centers = vtk_to_numpy(cc.GetOutput().GetPoints().GetData())
    q = vtk.vtkCellSizeFilter()
    q.SetInputData(pd)
    q.ComputeAreaOn()
    q.Update()
    area = vtk_to_numpy(q.GetOutput().GetCellData().GetArray("Area"))
    ids = vtk_to_numpy(pd.GetCellData().GetArray(FACE_ARRAY)).astype(int)
    return {int(f): (centers[ids == f] * area[ids == f, None]).sum(0) / area[ids == f].sum()
            for f in np.unique(ids)}


def load_faces(path):
    """final_tree_faces.json -> {face_id: name}."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("array") != FACE_ARRAY:
        raise RuntimeError(f"{path}: faces refer to array {d.get('array')}, not {FACE_ARRAY}")
    return {int(k): v for k, v in d["faces"].items()}


def write_project(proj_dir, model_name, pd, faces, sv_version):
    """SV project folder with one PolyData model. faces: {face_id: name}; 'wall' -> wall, else cap."""
    config.assert_writable(proj_dir)
    for sub in PLUGIN_DIRS:
        os.makedirs(os.path.join(proj_dir, sub), exist_ok=True)
    with open(os.path.join(proj_dir, "simvascular.proj"), "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<simvascular_project version="1.0"/>\n')
    now = int(time.time())
    with open(os.path.join(proj_dir, "Images", "image_information.xml"), "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<ImageObjectInformation version="1.0" creation_time="{now}" modification_time="{now}">\n'
                '    <timestep id="0">\n'
                f'        <created_with_simvascular_version>{sv_version}</created_with_simvascular_version>\n'
                '        <path></path>\n        <image_file_name></image_file_name>\n'
                '        <image_header_file_name></image_header_file_name>\n        <image_name></image_name>\n'
                '        <data_is_local_copy>false</data_is_local_copy>\n        <scale_factor>1</scale_factor>\n'
                '    </timestep>\n</ImageObjectInformation>\n')

    vtp = os.path.join(proj_dir, "Models", f"{model_name}.vtp")
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(vtp)
    w.SetInputData(pd)
    if not w.Write():
        raise RuntimeError(f"Cannot write {vtp}")

    lines = ['<?xml version="1.0" encoding="UTF-8" ?>',
             '<model type="PolyData" version="1.0">',
             '    <timestep id="0">',
             '        <model_element type="PolyData" num_sampling="0" use_uniform="0">',
             '            <segmentations />',
             '            <faces>']
    for i, (fid, name) in enumerate(sorted(faces.items())):
        c = _COLORS[i % len(_COLORS)]
        lines.append(f'                <face id="{fid}" name="{name}" type="{"wall" if name == "wall" else "cap"}" '
                     f'visible="true" opacity="1" color1="{c[0]}" color2="{c[1]}" color3="{c[2]}" />')
    lines += ['            </faces>', '            <blend_radii />', '        </model_element>',
              '    </timestep>', '</model>']
    mdl = os.path.join(proj_dir, "Models", f"{model_name}.mdl")
    with open(mdl, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {vtp}\n  saved {mdl}")
    return vtp, mdl


def check_with_sv(vtp, expected_ids):
    """Re-read the model with SimVascular's own modeler: face ids must be the expected ones."""
    import sv
    model = sv.modeling.Modeler(sv.modeling.Kernel.POLYDATA).read(vtp)
    got = sorted(int(i) for i in model.get_face_ids())
    if got != sorted(expected_ids):
        raise RuntimeError(f"SimVascular reads face ids {got}, expected {sorted(expected_ids)}")
    print(f"  SimVascular reads faces {got}")