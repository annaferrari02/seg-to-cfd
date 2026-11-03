"""Crea init_pressure.vtu: copia della mesh volumetrica con un campo puntuale
'Pressure' uniforme, da usare come <Initial_pressures_file_path> nel solver.xml.

Uso da riga di comando:
    python3 make_init_pressure.py mesh-complete/mesh-complete.mesh.vtu init_pressure.vtu 119989.8

Uso da altro script:
    from make_init_pressure import write_init_pressure
    write_init_pressure(mesh_in, out, p0)
"""
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def write_init_pressure(mesh_in: str, out: str, p0: float, array_name: str = "Pressure") -> int:
    """Scrive `out` = mesh `mesh_in` + array puntuale `array_name` uniforme = p0.
    Ritorna il numero di nodi (stessi nodi, stesso ordine della mesh di partenza)."""
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(mesh_in)
    r.Update()
    grid = r.GetOutput()
    n = grid.GetNumberOfPoints()
    if n == 0:
        raise RuntimeError(f"Mesh vuota o non leggibile: {mesh_in}")

    arr = numpy_to_vtk(np.full(n, float(p0), dtype=np.float64), deep=1)
    arr.SetName(array_name)
    grid.GetPointData().AddArray(arr)

    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(out)
    w.SetInputData(grid)
    if w.Write() != 1:
        raise RuntimeError(f"Scrittura fallita: {out}")
    print(f"{out}: {n} nodi, {array_name} = {p0}")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    write_init_pressure(sys.argv[1], sys.argv[2], float(sys.argv[3]))
    
