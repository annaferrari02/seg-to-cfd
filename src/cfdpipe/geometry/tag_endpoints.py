#!/usr/bin/env python3
"""tag_endpoints.py  --  World 1 (orchestrator venv), pure transform.

Contract
--------
input :  <centerline.vtk|.vtp>   POLYDATA with LINES + a 'Radius' point-data
                                  array (Slicer "Extract Centerline" output),
                                  in the SAME frame as the markups.
         <endpoints.mrk.json>     Slicer markups fiducials. The inlet control
                                  point is labelled with the `inlet_prefix`
                                  (default: "inlet", e.g. "inlet_pz001").
output:  <endpoints_tagged.json>
             {
               "frame": "<LPS|RAS|...>",     # read from the markups, not assumed
               "units": "mm",
               "endpoints": [
                   {"id": <label>, "position": [x,y,z], "role": "inlet|outlet",
                    "radius_mm": <float>},
                   ...
               ]
             }

Design notes
------------
* Positions are passed through UNCHANGED (source frame + source units). The
  single Slicer->SV transform (scale 0.1 mm->cm, same frame, no translation)
  is applied downstream by the SV face matcher, in one place only.
* `role` comes from the label, not from a radius heuristic: the human designates
  the inlet in Slicer, the label persists in the .mrk.json, we read it here.
* Two fail-loud guards, in the spirit of "fail early and loudly":
    - exactly one inlet must exist;
    - every endpoint must sit within `max_snap_dist` mm of the centerline.
      A large snap distance means either a stray endpoint OR a frame mismatch
      (markups LPS vs centerline RAS) -- both must stop the pipeline, not
      silently mis-tag caps later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_centerline_points_radii(path, radius_array: str = "Radius"):
    """Return (points Nx3, radii N) from a legacy .vtk or XML .vtp polydata.

    vtk is imported lazily so the orchestrator can import this module's pure
    helpers without paying the vtk import cost until a file is actually read.
    """
    import vtk  # noqa: PLC0415  (lazy on purpose)
    from vtk.util.numpy_support import vtk_to_numpy

    p = str(path)
    reader = vtk.vtkXMLPolyDataReader() if p.lower().endswith(".vtp") else vtk.vtkPolyDataReader()
    reader.SetFileName(p)
    reader.Update()
    pd = reader.GetOutput()
    if pd is None or pd.GetNumberOfPoints() == 0:
        raise ValueError(f"No points read from centerline: {path}")

    pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    arr = pd.GetPointData().GetArray(radius_array)
    if arr is None:
        available = [pd.GetPointData().GetArrayName(i)
                     for i in range(pd.GetPointData().GetNumberOfArrays())]
        raise ValueError(
            f"Radius array '{radius_array}' not found in {path}. "
            f"Available point-data arrays: {available}"
        )
    return pts, vtk_to_numpy(arr).astype(float)


def _read_markups(path):
    """Return (frame, units, [(label, position Nx3), ...]) from a .mrk.json."""
    data = json.loads(Path(path).read_text())
    markups = data.get("markups") or []
    if not markups:
        raise ValueError(f"No 'markups' block in {path}")
    m = markups[0]
    frame = m.get("coordinateSystem", "UNKNOWN")
    units = m.get("coordinateUnits", "mm")
    eps = []
    for cp in m.get("controlPoints", []):
        label = cp.get("label") or cp.get("id") or "?"
        eps.append((label, np.asarray(cp["position"], dtype=float)))
    if not eps:
        raise ValueError(f"No control points in {path}")
    return frame, units, eps


def _role_from_label(label: str, inlet_prefix: str = "inlet") -> str:
    return "inlet" if label.lower().startswith(inlet_prefix.lower()) else "outlet"


def tag_endpoints(centerline_path, markups_path, radius_array: str = "Radius",
                  inlet_prefix: str = "inlet", max_snap_dist: float = 5.0) -> dict:
    """Core transform. Returns the endpoints_tagged dict (does not write it)."""
    pts, rad = _read_centerline_points_radii(centerline_path, radius_array)
    frame, units, eps = _read_markups(markups_path)

    endpoints = []
    n_inlet = 0
    worst_dist = 0.0
    worst_label = None

    for label, pos in eps:
        d2 = np.sum((pts - pos) ** 2, axis=1)
        j = int(np.argmin(d2))
        dist = float(np.sqrt(d2[j]))
        if dist > worst_dist:
            worst_dist, worst_label = dist, label

        role = _role_from_label(label, inlet_prefix)
        n_inlet += role == "inlet"
        endpoints.append({
            "id": label,
            "position": [float(x) for x in pos],
            "role": role,
            "radius_mm": float(rad[j]),
        })

    if n_inlet != 1:
        raise ValueError(
            f"Expected exactly 1 inlet (label starting with '{inlet_prefix}'), "
            f"found {n_inlet}. Labels: {[e['id'] for e in endpoints]}"
        )
    if worst_dist > max_snap_dist:
        raise ValueError(
            f"Endpoint '{worst_label}' is {worst_dist:.2f} mm from the nearest "
            f"centerline point (> {max_snap_dist} mm). Likely a coordinate-frame "
            f"mismatch (markups vs centerline, e.g. LPS vs RAS) or an endpoint "
            f"not placed on the centerline."
        )

    return {"frame": frame, "units": units, "endpoints": endpoints}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Tag Slicer endpoints with role + local radius from a centerline."
    )
    ap.add_argument("centerline", help="centerline polydata (.vtk legacy or .vtp) with a Radius array")
    ap.add_argument("markups", help="Slicer markups fiducials (.mrk.json)")
    ap.add_argument("output", help="output endpoints_tagged.json")
    ap.add_argument("--radius-array", default="Radius")
    ap.add_argument("--inlet-prefix", default="inlet")
    ap.add_argument("--max-snap-dist", type=float, default=5.0,
                    help="max mm allowed between an endpoint and the nearest centerline point")
    a = ap.parse_args(argv)

    result = tag_endpoints(a.centerline, a.markups, a.radius_array,
                           a.inlet_prefix, a.max_snap_dist)
    Path(a.output).write_text(json.dumps(result, indent=2))

    n = len(result["endpoints"])
    ni = sum(e["role"] == "inlet" for e in result["endpoints"])
    print(f"OK: {n} endpoints ({ni} inlet, {n - ni} outlet) | "
          f"frame={result['frame']} units={result['units']} -> {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())