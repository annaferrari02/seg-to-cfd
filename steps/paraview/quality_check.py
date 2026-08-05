"""Quality check script for ParaView.

Usage: pvpython steps/paraview/quality_check.py <input.vtp> <output.vtp> --threshold 400

Contract with adapter:
- argv[1] = input file
- argv[2] = output file
- optional: --threshold <float> (default 400)
- exit code 0 = pass, 1 = fail

The script performs a simple quality heuristic (number of points) and
returns non-zero if the mesh is below threshold. It writes the input
dataset to `output` so the pipeline can continue using the same path.
"""
#da controllare se funzia + aggiungere estrazione di model face id dell'INLET e WALL (a meno che non sia gestito direttamente da simvascular)
from __future__ import annotations

import argparse
import os
import sys

from paraview.simple import OpenDataFile, UpdatePipeline, SaveData, MeshQuality
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
from paraview.vtk.numpy_support import vtk_to_numpy


def main() -> int:
    p = argparse.ArgumentParser(description="Simple VTP quality check.")
    p.add_argument("input_file", help="input .vtp file")
    p.add_argument("output_file", help="output .vtp file (can be same as input)")
    p.add_argument("--threshold", type=float, default=400, help="min points threshold")
    args = p.parse_args()

    if not os.path.exists(args.input_file):
        print(f"ERRORE: input non trovato: {args.input_file}", file=sys.stderr)
        return 1

    data = OpenDataFile(args.input_file)
    if data is None:
        print(f"ERRORE: ParaView non ha saputo leggere {args.input_file}", file=sys.stderr)
        return 1
    UpdatePipeline()

#NON è IL NUMBER OF CELLS/POINTS MA PROPRIO MESH QUALITY IL TEMA. 
    # Apply MeshQuality filter
    mq = MeshQuality(Input=data)
    UpdatePipeline()

    # Try to fetch the output and find a quality array (cell or point)
    try:
        vtk_data = dsa.WrapDataObject(mq)
    except Exception:
        print("ERRORE: impossibile ottenere i dati da MeshQuality", file=sys.stderr)
        return 1

    # Search for an array whose name contains 'quality' (case-insensitive)
    quality_array = None
    # check cell data
    cd = vtk_data.GetCellData()
    for name in cd.keys():
        if "quality" in name.lower():
            quality_array = cd[name]
            break
    # check point data
    if quality_array is None:
        pd = vtk_data.GetPointData()
        for name in pd.keys():
            if "quality" in name.lower():
                quality_array = pd[name]
                break

    if quality_array is None:
        print("ERRORE: nessun array di 'quality' trovato dopo MeshQuality", file=sys.stderr)
        return 1

    # quality_array is a numpy-like sequence
    try:
        qvals = algs.as_array(quality_array)
    except Exception:
        # fallback: try converting via vtk_to_numpy
        try:
            # quality_array may be a vtkDataArray
            qvals = vtk_to_numpy(quality_array)
        except Exception as e:
            print(f"ERRORE: impossibile leggere l'array quality: {e}", file=sys.stderr)
            return 1

    # choose metric: use max quality
    qmax = float(qvals.max()) if qvals.size > 0 else 0.0
    print(f"Quality max: {qmax}")

    if qmax < args.threshold:
        print(f"QUALITY FAIL: max {qmax} < threshold {args.threshold}", file=sys.stderr)
        try:
            SaveData(args.output_file, proxy=data)
        except Exception as e:
            print(f"ERRORE: SaveData failed: {e}", file=sys.stderr)
        return 1

    try:
        SaveData(args.output_file, proxy=data)
    except Exception as e:
        print(f"ERRORE: SaveData failed: {e}", file=sys.stderr)
        return 1

    print(f"QUALITY PASS: max {qmax} >= threshold {args.threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
