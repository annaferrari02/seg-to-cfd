r"""Step 1 - tree model, labelled endpoints, centerline

Slicer --python-script step1_extract_centerline.py <case_id>

Input:  Seg/<case>_seg_clean.seg.nrrd (or segmentation "seg" already in the scene) + CT
Output: Model/tree_model_<case>.vtk
        Centerline/Endpoints_<case>.mrk.json   endpoints labelled by branch, "inlet" first
        Centerline/endpoints_<case>.json       same points, plain JSON {label: [x,y,z]} in LPS
        Centerline/centerline_<case>.vtk       centerline model with Radius

1. segment "tree" -> model
2. endpoints: auto-detect (Extract Centerline), or reuse the node Endpoints_<case> if already
   in the scene (= manually corrected after a failed run)
3. check: exactly 1 inlet + N outlets (params.json step1)
4. label = nearest branch segment; the endpoint on the inlet segment (lumen) -> "inlet", first
5. centerline from the inlet to all outlets
On failure the scene stays open: fix the points of Endpoints_<case>, then main("<case>").
"""
import os
import sys

PIPELINE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0]))
sys.path.insert(0, PIPELINE_DIR)

import slicer  # noqa: E402

from utils import centerline as cl  # noqa: E402
from utils import config, state  # noqa: E402
from utils.seg_tools import (get_or_load_segmentation, get_or_load_volume, label_points,  # noqa: E402
                             save_node, segment_names, segment_to_model)


def _fail(case_id, msg):
    print(f"\n>>> step1 FAILED: {msg}")
    print(f">>> Fix the points of 'Endpoints_{case_id}' (move / delete / add), then: main('{case_id}')")
    print(f">>> To re-run the auto-detection: delete 'Endpoints_{case_id}', then main('{case_id}')")
    raise RuntimeError(msg)


def _new_node(cls, name):
    old = slicer.mrmlScene.GetFirstNodeByName(name)
    if old is not None:
        slicer.mrmlScene.RemoveNode(old)
    return slicer.mrmlScene.AddNewNodeByClass(cls, name)


def main(case_id):
    paths = config.case_paths(case_id)
    p = config.load_params()["step1"]
    inlet, outlets = p["inlet_segment"], p["outlet_segments"]
    print(f">>> step1 {case_id}")

    ct = get_or_load_volume("ct", paths["ct_in"])
    seg = get_or_load_segmentation("seg", paths["seg_clean"])

    # 1. tree model
    model = segment_to_model(seg, p["source_segment"], f"tree_model_{case_id}", p["surface_smoothing"])

    # 2. endpoints
    ep_name = f"Endpoints_{case_id}"
    ep = slicer.mrmlScene.GetFirstNodeByName(ep_name)
    if ep is not None and ep.GetNumberOfControlPoints() > 0:
        print(f"  reusing {ep.GetNumberOfControlPoints()} endpoints of {ep_name} (manual correction)")
    else:
        ep = ep or slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", ep_name)
        cl.detect_endpoints(model, ep, p["centerline"])
    points = cl.get_points(ep)

    # 3. count check
    expected = 1 + len(outlets)
    if len(points) != expected:
        _fail(case_id, f"{len(points)} endpoints, expected {expected} (1 inlet + {len(outlets)} outlets)")

    # 4. labels: nearest branch segment (all segments except the merged one)
    candidates = [n for n in segment_names(seg) if n != p["source_segment"]]
    labels = label_points(points, seg, candidates, ct)
    for (name, d), pt in zip(labels, points):
        print(f"  endpoint {[round(c, 1) for c in pt]} -> {name} ({d:.1f} mm)")
    # threshold per segment: the inlet endpoint can sit well off the large flat aortic end
    max_d = p["max_label_dist_mm"]
    max_d = max_d if isinstance(max_d, dict) else {"default": max_d}
    far = [(n, round(d, 1)) for n, d in labels if d > max_d.get(n, max_d["default"])]
    if far:
        _fail(case_id, f"endpoints too far from any segment: {far}")
    names = sorted(n for n, _ in labels)
    if names != sorted([inlet] + outlets):
        _fail(case_id, f"endpoint labels {names} != expected {sorted([inlet] + outlets)}")

    by_name = {n: pt for (n, _), pt in zip(labels, points)}
    order = [inlet] + outlets
    cl.set_points(ep, [by_name[n] for n in order], ["inlet"] + outlets)  # inlet = first = source

    # 5. centerline
    cl_model = _new_node("vtkMRMLModelNode", f"centerline_{case_id}")
    cl_curve = _new_node("vtkMRMLMarkupsCurveNode", f"centerline_curve_{case_id}")
    cl.extract_centerline(model, ep, cl_model, cl_curve, p["centerline"])

    # visual check: model + endpoints + centerline
    seg.GetDisplayNode().SetVisibility(False)
    model.GetDisplayNode().SetOpacity(0.4)
    state.reset_3d_view()

    def save():
        save_node(model, paths["tree_model"])
        save_node(ep, paths["endpoints_mrk"])
        cl.save_endpoints_json(ep, paths["endpoints_json"])
        save_node(cl_model, paths["centerline"])

    state.finish_step("step1", save)


state.run(main, sys.argv, "step1_extract_centerline.py")