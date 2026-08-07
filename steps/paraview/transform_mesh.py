# steps/paraview/transform_mesh.py
# Uso:  pvpython steps/paraview/transform_mesh.py <input.vtk> <output.vtk> --scale 0.1
#
# Contratto con l'adapter (MONDO 1):
#   - riceve input/output/scale da riga di comando (niente hardcoded);
#   - esce con codice != 0 a ogni problema;
#   - a fine corsa l'output DEVE esistere e non essere vuoto.
#
# MODIFICA (passo 1): l'output NON e' piu' scritto con SaveData (che su
# ParaView 5.11+ produce "DataFile Version 5.1", con i poligoni come
# OFFSETS+CONNECTIVITY che il VTK vecchio di SimVascular 2023-03-27 non sa
# leggere -> polys=0 -> freeze in set_surface). Ora scriviamo in legacy VTK
# 4.2, che usa il blocco POLYGONS classico leggibile da SV.

import argparse
import os
import sys

from paraview.simple import *
from paraview import servermanager

# vtkPolyDataWriter: import robusto tra versioni di ParaView.
try:
    from vtkmodules.vtkIOLegacy import vtkPolyDataWriter
except ImportError:  # ParaView piu' vecchi espongono il modulo monolitico "vtk"
    from vtk import vtkPolyDataWriter


def main() -> int:
    p = argparse.ArgumentParser(description="Scale + triangulate + clean di una mesh.")
    p.add_argument("input_file", help="mesh di partenza (.vtk)")
    p.add_argument("output_file", help="mesh di output (.vtk)")
    p.add_argument("--scale", type=float, default=0.1, help="fattore mm->cm")
    p.add_argument("--no-recenter", action="store_true", help="non ricentrare sull'origine")
    args = p.parse_args()

    # input pronto?
    if not os.path.exists(args.input_file):
        print(f"ERRORE: input non trovato: {args.input_file}", file=sys.stderr)
        return 1

    # carico. OpenDataFile ritorna None se non sa leggere: va controllato a mano.
    mesh = OpenDataFile(args.input_file)
    if mesh is None:
        print(f"ERRORE: ParaView non ha saputo leggere {args.input_file}", file=sys.stderr)
        return 1
    UpdatePipeline()
    print("Bounds iniziali:", mesh.GetDataInformation().GetBounds())

    # scale
    scaled = Transform(Input=mesh)
    scaled.Transform.Scale = [args.scale, args.scale, args.scale]
    UpdatePipeline()

    # recenter (opzionale, ON di default come nel tuo script)
    if not args.no_recenter:
        b = scaled.GetDataInformation().GetBounds()
        center = [(b[0]+b[1])/2.0, (b[2]+b[3])/2.0, (b[4]+b[5])/2.0]
        current = Transform(Input=scaled)
        current.Transform.Translate = [-center[0], -center[1], -center[2]]
        UpdatePipeline()
    else:
        current = scaled

    # triangulate + clean
    tri = Triangulate(Input=current)
    cleaned = Clean(Input=tri)
    UpdatePipeline()

    info = cleaned.GetDataInformation()
    n_points, n_cells = info.GetNumberOfPoints(), info.GetNumberOfCells()
    print("Bounds finali:", info.GetBounds())
    print(f"Punti: {n_points}  Celle: {n_cells}")

    # mesh vuota? fail loud
    if n_points == 0 or n_cells == 0:
        print("ERRORE: la mesh risultante è vuota.", file=sys.stderr)
        return 1

    # --- scrittura in legacy VTK 4.2 (leggibile dal VTK di SimVascular) ------
    # Portiamo i dati dal server al client come vtkPolyData. In pvpython/pvbatch
    # single-process Fetch restituisce l'intero oggetto sul client.
    poly = servermanager.Fetch(cleaned)
    if poly is None or not poly.IsA("vtkPolyData"):
        got = type(poly).__name__ if poly is not None else "None"
        print(f"ERRORE: l'output non è vtkPolyData (ottenuto: {got}).", file=sys.stderr)
        return 1

    writer = vtkPolyDataWriter()
    writer.SetFileName(args.output_file)
    writer.SetInputData(poly)
    writer.SetFileVersion(42)        # 42 == DataFile Version 4.2 (blocco POLYGONS classico)
    writer.SetFileTypeToASCII()      # come prima; usa SetFileTypeToBinary() per file più piccoli
    ok = writer.Write()              # ritorna 1 se ha scritto, 0 se ha fallito
    if not ok:
        print(f"ERRORE: vtkPolyDataWriter non ha scritto {args.output_file}", file=sys.stderr)
        return 1
    print("Scritto in legacy VTK 4.2 (ASCII)", flush=True)

    # verifica finale: il file è davvero uscito e non è vuoto?
    if not os.path.exists(args.output_file):
        print(f"ERRORE: l'output non è stato prodotto: {args.output_file}", file=sys.stderr)
        return 1
    if os.path.getsize(args.output_file) == 0:
        print(f"ERRORE: l'output è vuoto: {args.output_file}", file=sys.stderr)
        return 1

    print("Salvato:", args.output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())