r"""Step 3 (ParaView) - mm -> cm, triangulate, clean

pvpython step3_transform.py <case_id>

Input:  Model/final_tree.vtk      (step2, mm, LPS, ModelFaceID)
Output: Model/final_tree_cm.vtk   (cm, same frame, ModelFaceID kept, legacy VTK 4.2)

Checks: extent in cm within params step3.extent_cm (catches a missing or double scaling),
same ModelFaceID values as the input, still a closed surface.
Unit conversion happens only here: everything before is mm, everything after is cm.
"""
import os
import sys

PIPELINE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.insert(0, PIPELINE_DIR)

from paraview.simple import OpenDataFile  # noqa: E402
from paraview import servermanager  # noqa: E402

from utils import config, pv_tools  # noqa: E402

FACE_ARRAY = "ModelFaceID"


def main(case_id):
    paths = config.case_paths(case_id)
    p = config.load_params()["step3"]
    print(f">>> step3 {case_id}\n  input: {paths['final_tree']}")
    if not os.path.exists(paths["final_tree"]):
        raise RuntimeError(f"Missing input (run step2): {paths['final_tree']}")

    src_pd = servermanager.Fetch(OpenDataFile(paths["final_tree"]))
    ids_in = pv_tools.face_ids(src_pd, FACE_ARRAY)

    pd = pv_tools.transform_triangulate_clean(paths["final_tree"], p["scale"])

    lo, hi = p["extent_cm"]
    ext = pv_tools.extent(pd)
    if not lo <= ext <= hi:
        raise RuntimeError(f"Extent {ext:.1f} cm outside [{lo}, {hi}]: input not in mm, or scaled twice?")
    ids_out = pv_tools.face_ids(pd, FACE_ARRAY)
    if ids_out != ids_in:
        raise RuntimeError(f"{FACE_ARRAY} changed: {ids_in} -> {ids_out}")
    n_open = pv_tools.open_edges(pd)
    if n_open:
        raise RuntimeError(f"Surface not closed after clean: {n_open} open/non-manifold edges")
    print(f"  extent {ext:.1f} cm, faces {ids_out}, closed")

    pv_tools.write_legacy42(pd, paths["final_tree_cm"], ascii=p["ascii"])
    print(">>> step3 done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: pvpython step3_transform.py <case_id>")
    main(sys.argv[1])