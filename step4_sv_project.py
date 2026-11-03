r"""Step 4 (SimVascular) - SV project with the named solid model

<simvascular> --python -- step4_sv_project.py <case_id>

Input:  Model/final_tree_cm.vtk        (step3: cm, ModelFaceID 1 wall, 2 inlet, 3+ outlets)
        Model/final_tree_faces.json    (step2: {face_id: name})
        Model/final_tree.vtk           (step2, mm LPS: frame check only)
        Centerline/endpoints_<case>.json (step1: LPS mm, frame check only)
Output: <case>/                        SV project, open it in SimVascular as a project folder
          simvascular.proj, Images/image_information.xml, Models/<case>.vtp + .mdl, empty plugin dirs
        faces_named.json               ids / names / types for the next SV steps

Face names come from ModelFaceID + final_tree_faces.json: no face recomputation by angle,
no geometric matching. Frame check: face centroids = step2 surface x 0.1 (step3 changed only
the units) and each cap within (flow extension + margin) of the endpoint with its name.
"""
import json
import os
import sys

PIPELINE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0]))
sys.path.insert(0, PIPELINE_DIR)

import numpy as np  # noqa: E402

from utils import config, sv_tools  # noqa: E402


def _args():
    argv = sys.argv[1:]
    if "--" in argv:  # SimVascular may put its own arguments before "--"
        argv = argv[argv.index("--") + 1:]
    if len(argv) != 1:
        sys.exit("Usage: simvascular --python -- step4_sv_project.py <case_id>")
    return argv[0]


def check_frame(centroids_cm, faces, final_tree_mm, endpoints_json, max_mm, tol_mm=1.0):
    """Names vs geometry (caps in cm, step2 surface and endpoints in mm LPS):
    1. every face centroid = the same face of the step2 surface (final_tree.vtk) x 0.1, within
       tol_mm: step3 changed only the units, ids and frame are intact. The names themselves
       were assigned in step2 along the centerline axis of each branch.
    2. each cap within max_mm of the endpoint carrying its name (gross errors).
    The closest-endpoint assignment is only reported: with 25 mm flow extensions on curved,
    close branches (e.g. iedx / iidx) the nearest endpoint is not always the own one."""
    from itertools import permutations
    ref = sv_tools.face_centroids(sv_tools.read_legacy(final_tree_mm))
    if sorted(ref) != sorted(centroids_cm):
        raise RuntimeError(f"Face ids {sorted(centroids_cm)} != step2 surface {sorted(ref)}")
    for fid in sorted(ref):
        d = float(np.linalg.norm(np.asarray(centroids_cm[fid]) * 10.0 - np.asarray(ref[fid])))
        if d > tol_mm:
            raise RuntimeError(f"Face {fid} ({faces[fid]}) moved {d:.2f} mm from step2 to step3 "
                               f"(max {tol_mm}): ids, units or frame changed")
    print(f"  face centroids = step2 surface (within {tol_mm} mm)")

    with open(endpoints_json, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("coordinate_system") != "LPS":
        raise RuntimeError(f"{endpoints_json}: endpoints not in LPS")
    pts = d["points"]
    caps = [(fid, name) for fid, name in sorted(faces.items()) if name != "wall"]
    missing = [name for _, name in caps if name not in pts]
    if missing:
        raise RuntimeError(f"No endpoint named {missing}")
    C = np.array([[np.linalg.norm(np.asarray(centroids_cm[fid]) * 10.0 - np.asarray(pts[n2]))
                   for _, n2 in caps] for fid, _ in caps])  # C[cap i, endpoint j] mm
    for i, (fid, name) in enumerate(caps):
        print(f"  cap {fid} {name}: {C[i, i]:.1f} mm from its endpoint")
        if C[i, i] > max_mm:
            raise RuntimeError(f"Cap {name} is {C[i, i]:.1f} mm from its endpoint (max {max_mm}): "
                               f"wrong names, units or frame")
    n = len(caps)
    rows = np.arange(n)
    best = min(permutations(range(n)), key=lambda perm: C[rows, perm].sum())
    other = [(caps[i][1], caps[j][1]) for i, j in enumerate(best) if i != j]
    if other:
        print(f"  note: nearest-endpoint assignment differs for (face, nearest endpoint) {other}; "
              f"names from step2 (centerline axis) kept - check them visually once")


def main(case_id):
    paths = config.case_paths(case_id)
    p = config.load_params()
    p4 = p["step4"]
    proj_dir = os.path.join(paths["db"], case_id)
    print(f">>> step4 {case_id}\n  input: {paths['final_tree_cm']}\n  project: {proj_dir}")
    for k in ("final_tree_cm", "final_tree", "final_faces", "endpoints_json"):
        if not os.path.exists(paths[k]):
            raise RuntimeError(f"Missing input: {paths[k]}")

    pd = sv_tools.read_legacy(paths["final_tree_cm"])
    faces = sv_tools.load_faces(paths["final_faces"])
    ids = sv_tools.face_ids(pd)
    if ids != sorted(faces):
        raise RuntimeError(f"ModelFaceID {ids} != faces in final_tree_faces.json {sorted(faces)}")
    if faces.get(1) != "wall" or faces.get(2) != "inlet":
        raise RuntimeError(f"Expected 1 = wall, 2 = inlet: {faces}")

    fe = p["step2"]["flow_ext"]
    if fe["mode"] == "mm":  # cap = clip point + extension: endpoint + extension + margin
        check_frame(sv_tools.face_centroids(pd), faces, paths["final_tree"], paths["endpoints_json"],
                    fe["value"] + p4["frame_margin_mm"])
    else:
        print("  frame check skipped (flow extension length relative to the radius)")

    vtp, _ = sv_tools.write_project(proj_dir, case_id, pd, faces, p4["sv_version"])
    sv_tools.check_with_sv(vtp, ids)

    out = os.path.join(paths["db"], "faces_named.json")
    config.assert_writable(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "units": "cm",
            "project": os.path.relpath(proj_dir, paths["db"]),
            "model": os.path.relpath(vtp, paths["db"]),
            "wall_id": 1,
            "inlet_id": 2,
            "outlets": {name: fid for fid, name in sorted(faces.items()) if fid > 2},
            "faces": [{"id": fid, "name": name, "type": "wall" if name == "wall" else "cap"}
                      for fid, name in sorted(faces.items())],
        }, f, indent=2)
    print(f"  saved {out}\n>>> step4 done")


if __name__ == "__main__":
    main(_args())