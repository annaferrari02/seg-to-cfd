"""Quality check script for ParaView.

Usage: pvpython steps/paraview/quality_check.py <input.vtp> <output.vtp> --threshold 400

Contract with adapter:
- argv[1] = input file
- argv[2] = output file
- optional: --threshold <float> (default 400)
- exit code 0 = pass, 1 = fail

The script applies ParaView's MeshQuality filter and validates that all quality
values are within the inclusive range [1, threshold]. If any value falls outside
that range, it raises a ValueError with the message "mesh quality over <threshold>".
"""
from __future__ import annotations

import argparse
import os
import sys

from paraview import servermanager
from paraview.simple import MeshQuality, OpenDataFile, SaveData, UpdatePipeline
from vtk.util.numpy_support import vtk_to_numpy


def _extract_quality_values(dataset):
    """Return a flat list of quality values from the ParaView MeshQuality output."""
    fetched = servermanager.Fetch(dataset)
    if fetched is None:
        raise ValueError("mesh quality output is empty")

    quality_arrays = []
    for store in (fetched.GetCellData(), fetched.GetPointData()):
        if store is None:
            continue
        for i in range(store.GetNumberOfArrays()):
            arr = store.GetArray(i)
            if arr is not None and "quality" in arr.GetName().lower():
                quality_arrays.append(arr)

    if not quality_arrays:
        raise ValueError("mesh quality array not found after MeshQuality filter")

    values = []
    for arr in quality_arrays:
        qvals = vtk_to_numpy(arr)
        if qvals is None:
            continue
        values.extend(float(v) for v in qvals.ravel())

    if not values:
        raise ValueError("mesh quality array is empty")

    return values


def main() -> int:
    p = argparse.ArgumentParser(description="Validate ParaView MeshQuality values against a threshold.")
    p.add_argument("input_file", help="input .vtp file")
    p.add_argument("output_file", help="output .vtp file (can be same as input)")
    p.add_argument("--threshold", type=float, default=400.0, help="quality upper bound")
    args = p.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"input non trovato: {args.input_file}")

    data = OpenDataFile(args.input_file)
    if data is None:
        raise ValueError(f"ParaView non ha saputo leggere {args.input_file}")
    UpdatePipeline()

    mq = MeshQuality(Input=data)
    UpdatePipeline()

    try:
        values = _extract_quality_values(mq)
    except ValueError:
        raise

    invalid = [value for value in values if not (1.0 <= float(value) <= float(args.threshold))]
    if invalid:
        raise ValueError(f"mesh quality over {args.threshold}")

    try:
        SaveData(args.output_file, proxy=mq)
    except Exception as exc:  # pragma: no cover - depends on ParaView runtime
        raise RuntimeError(f"SaveData failed: {exc}") from exc

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
