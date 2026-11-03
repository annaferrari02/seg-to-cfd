"""Binary mask operations (numpy + vtk, usable in Slicer and outside)."""
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def fill_cavities(mask):
    """Fill the background regions fully enclosed by the mask (6-connectivity of the background),
    e.g. calcifications removed inside the lumen. The outside = the largest background region.
    mask: 3D array [k, j, i]. Returns (filled uint8 array, number of filled voxels)."""
    m = (np.asarray(mask) > 0)
    bg = (~m).astype(np.uint8)
    img = vtk.vtkImageData()
    img.SetDimensions(m.shape[2], m.shape[1], m.shape[0])
    img.GetPointData().SetScalars(numpy_to_vtk(bg.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
    conn = vtk.vtkImageConnectivityFilter()
    conn.SetInputData(img)
    conn.SetScalarRange(1, 1)
    conn.SetExtractionModeToAllRegions()
    conn.SetLabelModeToSizeRank()  # label 1 = largest background region = outside
    conn.Update()
    labels = vtk_to_numpy(conn.GetOutput().GetPointData().GetScalars()).reshape(m.shape)
    holes = (bg > 0) & (labels != 1)
    out = m.astype(np.uint8)
    out[holes] = 1
    return out, int(np.count_nonzero(holes))