#!/usr/bin/env python3
"""
Compute last-cardiac-cycle wall metrics from a folder of SimVascular VTU files
(allowing missing intermediate timesteps).

INPUTS
------
- A folder containing files like: result_0002.vtu, result_0004.vtu, ...
  Missing timesteps are OK.
- dt_step: physical time per integer solver timestep (seconds)
- T_cycle: cardiac cycle duration (seconds)

OUTPUT
------
A SURFACE VTP file (recommended for wall metrics) containing:
  - TAWSS
  - OSI
  - RRT (often called "Relative Residence Time"; you wrote RTT)
  - ECAP

Notes / Assumptions
-------------------
1) Each .vtu is a volume UnstructuredGrid for the same geometry; we extract the outer surface.
2) A wall shear stress VECTOR array exists on the extracted surface, either as PointData or CellData.
   If auto-detection fails, pass --wss-array with the exact array name.
3) Surface topology and point coordinates are assumed consistent across timesteps (rigid wall typical).
4) Integration uses trapezoidal rule with variable Δt based on timestep numbers:
      Δt = dt_step * (step_{i+1} - step_i)

Usage example
-------------
python compute_last_cycle_metrics_from_vtu.py \
  --folder /path/to/results \
  --dt 0.001 \
  --T 0.8 \
  --pattern "result_*.vtu" \
  --out last_cycle_metrics.vtp

Optional:
  --wss-array "wall_shear_stress"   (exact name in VTK arrays)
  --surface-only                    (if your .vtu already contains only the wall surface)
"""

import argparse
import glob
import os
import re
import sys
from typing import List, Tuple, Optional

import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("ERROR: pyvista is required. Install with: pip install pyvista", file=sys.stderr)
    sys.exit(1)


def extract_step_number(filename: str) -> Optional[int]:
    """Extract timestep number from a filename ending with digits before .vtu/.vtp."""
    base = os.path.basename(filename)
    m = re.search(r"(\d+)(?=\.(vtu|vtp)$)", base)
    return int(m.group(1)) if m else None


def list_files(folder: str, pattern: str) -> List[Tuple[int, str]]:
    paths = glob.glob(os.path.join(folder, pattern))
    items = []
    for p in paths:
        s = extract_step_number(p)
        if s is not None:
            items.append((s, p))
    items.sort(key=lambda x: x[0])
    return items


def is_vec3(arr) -> bool:
    a = np.asarray(arr)
    return a.ndim == 2 and a.shape[1] == 3


def find_wss_vector_array(mesh: pv.DataSet, user_name: Optional[str] = None) -> Tuple[str, str]:
    """
    Return (assoc, name) where assoc is 'point' or 'cell'.
    Heuristics:
      - If user_name provided and exists, use it.
      - Else search arrays whose name includes 'wss' and that are vec3.
      - Else fall back to any vec3 array (last resort).
    """
    if user_name:
        if user_name in mesh.point_data and is_vec3(mesh.point_data[user_name]):
            return ("point", user_name)
        if user_name in mesh.cell_data and is_vec3(mesh.cell_data[user_name]):
            return ("cell", user_name)
        raise RuntimeError(
            f"Requested WSS array '{user_name}' not found as vec3 in point/cell data."
        )

    candidates = []
    for name in mesh.point_data.keys():
        if "wss" in name.lower() and is_vec3(mesh.point_data[name]):
            candidates.append(("point", name))
    for name in mesh.cell_data.keys():
        if "wss" in name.lower() and is_vec3(mesh.cell_data[name]):
            candidates.append(("cell", name))

    if not candidates:
        for name in mesh.point_data.keys():
            if is_vec3(mesh.point_data[name]):
                candidates.append(("point", name))
        for name in mesh.cell_data.keys():
            if is_vec3(mesh.cell_data[name]):
                candidates.append(("cell", name))

    if not candidates:
        raise RuntimeError(
            "Could not find any vec3 array to use as WSS.\n"
            "Pass --wss-array with the exact array name. "
            "Check in ParaView (PointData/CellData arrays)."
        )

    # Prefer point data and names containing wss
    candidates.sort(
        key=lambda x: (0 if x[0] == "point" else 1,
                       0 if "wss" in x[1].lower() else 1,
                       len(x[1]))
    )
    return candidates[0]


def get_vec(mesh: pv.DataSet, assoc: str, name: str) -> np.ndarray:
    arr = mesh.point_data[name] if assoc == "point" else mesh.cell_data[name]
    return np.asarray(arr, dtype=np.float64)


def ensure_same_surface(a: pv.PolyData, b: pv.PolyData):
    """Ensure same surface mesh across timesteps."""
    if a.n_points != b.n_points or a.n_cells != b.n_cells:
        raise RuntimeError(
            "Surface topology changed across timesteps (different #points/#cells). "
            "This script assumes identical wall surface meshes."
        )
    if not np.allclose(a.points, b.points):
        maxdiff = float(np.max(np.linalg.norm(a.points - b.points, axis=1)))
        if maxdiff > 1e-6:
            raise RuntimeError(
                f"Surface point coordinates differ across timesteps (max |Δx|={maxdiff}). "
                "If you have moving walls, you must map fields to a common surface first."
            )


def read_wall_surface(path: str, surface_only: bool) -> pv.PolyData:
    """
    Read a timestep file and return wall surface as PolyData.
    - If surface_only=True, assumes the file is already a surface mesh (VTU or VTP).
    - Otherwise reads UnstructuredGrid and extracts outer surface.
    """
    ds = pv.read(path)

    if surface_only:
        # If ds is already PolyData, keep it; if it's an unstructured surface, extract_surface() is safe.
        if isinstance(ds, pv.PolyData):
            return ds
        return ds.extract_surface(algorithm="dataset_surface")

    # Typical case: volume .vtu -> surface PolyData
    return ds.extract_surface(algorithm="dataset_surface")


def trapezoid_integrate(
    cycle_steps: List[int],
    cycle_files: List[str],
    dt_step: float,
    T_cycle: float,
    wss_assoc: str,
    wss_name: str,
    surface_only: bool
) -> Tuple[pv.PolyData, np.ndarray, np.ndarray, float]:
    """
    Integrate over last cycle using trapezoid rule with variable Δt from missing timesteps.

    Returns:
      base_surface: PolyData to write output on (from last timestep)
      int_mag: ∫|τ|dt per point/cell
      int_vec: ∫τ dt per point/cell (vec3)
      window_duration: duration spanned by available files (seconds)
    """
    surfaces = [read_wall_surface(fp, surface_only) for fp in cycle_files]
    base = surfaces[-1]
    for s in surfaces[:-1]:
        ensure_same_surface(s, base)

    wss_list = [get_vec(s, wss_assoc, wss_name) for s in surfaces]
    mag_list = [np.linalg.norm(w, axis=1) for w in wss_list]

    int_mag = np.zeros_like(mag_list[0], dtype=np.float64)
    int_vec = np.zeros_like(wss_list[0], dtype=np.float64)

    times = [step * dt_step for step in cycle_steps]
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if dt <= 0:
            continue
        int_mag += 0.5 * (mag_list[i] + mag_list[i + 1]) * dt
        int_vec += 0.5 * (wss_list[i] + wss_list[i + 1]) * dt

    window_duration = times[-1] - times[0]

    # Normalize by requested cycle length; warn if your sampled window is too short.
    if window_duration < 0.9 * T_cycle:
        print(
            f"WARNING: Available last-cycle window is {window_duration:.6g}s but T_cycle={T_cycle:.6g}s. "
            "Results will still be normalized by T_cycle.",
            file=sys.stderr
        )

    return base, int_mag, int_vec, window_duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder containing .vtu timestep files")
    ap.add_argument("--dt", required=True, type=float, help="Physical time per integer timestep (seconds)")
    ap.add_argument("--T", required=True, type=float, help="Cardiac cycle duration (seconds)")
    ap.add_argument("--pattern", default="result_*.vtu", help="Glob pattern, default: result_*.vtu")
    ap.add_argument("--out", default="last_cycle_metrics.vtp", help="Output surface file (.vtp recommended)")
    ap.add_argument("--wss-array", default=None, help="Exact name of WSS vector array (optional)")
    ap.add_argument(
        "--surface-only",
        action="store_true",
        help="Set if your input files already contain only the wall surface (not a volume mesh)."
    )
    args = ap.parse_args()

    items = list_files(args.folder, args.pattern)
    if len(items) < 2:
        raise RuntimeError("Need at least 2 timestep files to integrate.")

    steps = np.array([s for s, _ in items], dtype=np.int64)
    files = [p for _, p in items]
    times = steps.astype(np.float64) * args.dt

    # Select last cardiac cycle
    t_end = times[-1]
    t_start = t_end - args.T
    idx = np.where(times >= t_start)[0]

    if len(idx) < 2:
        idx = np.array([len(times) - 2, len(times) - 1], dtype=int)
        print(
            "WARNING: Not enough files within the last cycle window; using last two timesteps only.",
            file=sys.stderr
        )

    cycle_steps = [int(steps[i]) for i in idx]
    cycle_files = [files[i] for i in idx]

    # Read first surface to detect WSS array name/association
    first_surface = read_wall_surface(cycle_files[0], args.surface_only)
    assoc, wss_name = find_wss_vector_array(first_surface, args.wss_array)
    print(f"Using WSS array '{wss_name}' from {assoc} data.")
    print(f"Using {len(cycle_files)} files from t={cycle_steps[0]*args.dt:.6g}s to t={cycle_steps[-1]*args.dt:.6g}s "
          f"(requested cycle length {args.T:.6g}s).")

    base, int_mag, int_vec, window_duration = trapezoid_integrate(
        cycle_steps=cycle_steps,
        cycle_files=cycle_files,
        dt_step=args.dt,
        T_cycle=args.T,
        wss_assoc=assoc,
        wss_name=wss_name,
        surface_only=args.surface_only
    )

    # Metrics
    T = float(args.T)
    eps = 1e-30

    TAWSS = int_mag / max(T, eps)

    # OSI = 0.5 * (1 - ||∫τ dt|| / ∫||τ|| dt)
    norm_int_vec = np.linalg.norm(int_vec, axis=1)
    denom = np.maximum(int_mag, eps)
    OSI = 0.5 * (1.0 - (norm_int_vec / denom))
    OSI = np.clip(OSI, 0.0, 0.5)

    # RRT = 1 / ((1 - 2*OSI) * TAWSS)
    one_minus_2osi = np.maximum(1.0 - 2.0 * OSI, eps)
    RRT = 1.0 / (one_minus_2osi * np.maximum(TAWSS, eps))

    # ECAP = OSI / TAWSS
    ECAP = OSI / np.maximum(TAWSS, eps)

    # Attach arrays to output mesh, preserving association of WSS
    if assoc == "point":
        base.point_data["TAWSS"] = TAWSS
        base.point_data["OSI"] = OSI
        base.point_data["RRT"] = RRT
        base.point_data["ECAP"] = ECAP
    else:
        base.cell_data["TAWSS"] = TAWSS
        base.cell_data["OSI"] = OSI
        base.cell_data["RRT"] = RRT
        base.cell_data["ECAP"] = ECAP

    out_path = os.path.join(args.folder, args.out) if not os.path.isabs(args.out) else args.out
    base.save(out_path)

    print(f"Wrote: {out_path}")
    print(f"Integrated over {len(cycle_files)} files spanning {window_duration:.6g}s; normalized by T_cycle={T:.6g}s.")


if __name__ == "__main__":
    main()