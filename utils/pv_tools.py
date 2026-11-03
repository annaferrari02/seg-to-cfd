"""Surface utilities for the ParaView world (pvpython): paraview.simple + vtkmodules only, no slicer."""
import os

from paraview import servermanager
from paraview.simple import Clean, OpenDataFile, Transform, Triangulate
from vtkmodules.vtkFiltersCore import vtkFeatureEdges
from vtkmodules.vtkIOLegacy import vtkPolyDataWriter

from utils import config


def transform_triangulate_clean(path, scale):
    """Scale (mm -> cm with 0.1) + triangulate + clean. NO recentering: it would misalign the
    surface from the endpoints / centerline frame. Cell data (ModelFaceID) is carried through."""
    src = OpenDataFile(path)
    if src is None:
        raise RuntimeError(f"ParaView cannot read {path}")
    src.UpdatePipeline()
    print(f"  bounds in:  {[round(b, 2) for b in src.GetDataInformation().GetBounds()]}")

    scaled = Transform(Input=src)
    scaled.Transform.Scale = [scale, scale, scale]
    cleaned = Clean(Input=Triangulate(Input=scaled))
    cleaned.UpdatePipeline()

    pd = servermanager.Fetch(cleaned)
    if pd is None or not pd.IsA("vtkPolyData") or pd.GetNumberOfCells() == 0:
        raise RuntimeError("Transform/Triangulate/Clean produced no polydata")
    print(f"  bounds out: {[round(b, 2) for b in pd.GetBounds()]}")
    print(f"  points {pd.GetNumberOfPoints()}  cells {pd.GetNumberOfCells()}")
    return pd


def extent(pd):
    b = pd.GetBounds()
    return max(b[1] - b[0], b[3] - b[2], b[5] - b[4])


def face_ids(pd, array_name):
    arr = pd.GetCellData().GetArray(array_name)
    if arr is None:
        raise RuntimeError(f"No {array_name} cell array")
    return sorted({int(arr.GetValue(i)) for i in range(arr.GetNumberOfTuples())})


def open_edges(pd):
    """Number of boundary + non-manifold edges (0 = closed manifold surface)."""
    fe = vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    return fe.GetOutput().GetNumberOfCells()


def write_legacy42(pd, path, ascii=True):
    """Legacy VTK 4.2 (classic POLYGONS block): the format SimVascular reads.
    Do not use SaveData: it writes VTK 5.x."""
    config.assert_writable(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = vtkPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(pd)
    w.SetFileVersion(42)
    if ascii:
        w.SetFileTypeToASCII()
    else:
        w.SetFileTypeToBinary()
    if not w.Write() or not os.path.getsize(path):
        raise RuntimeError(f"Cannot write {path}")
    print(f"  saved {path}")