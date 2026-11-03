r"""Step 2 - inset clip, flow extensions, caps, labelled faces

Slicer --python-script step2_cap_flow_ext.py <case_id>

Input:  Model/tree_model_<case>.vtk, Centerline/centerline_<case>.vtk,
        Centerline/Endpoints_<case>.mrk.json   (or the same nodes still in the scene)
Output: Model/final_tree.vtk            closed surface with ModelFaceID (1 wall, 2 inlet, 3+ outlets)
        Model/final_tree_faces.json     {face_id: name}
        Centerline/ClipPoints_<case>.mrk.json

1. clip points: each endpoint moved inward along the centerline by
   max(factor * radius, min_mm), capped at max_branch_fraction * branch length  (params step2.inset)
   (reuses the node ClipPoints_<case> if already in the scene = manually corrected)
2. clip (utils.clip.local_clip): at each clip point remove ONLY the vessel end connected to
   its endpoint; whatever else the plane crosses is re-stitched. Refused if the plane
   passes through a bifurcation. Then vmtk flow extensions, vmtk caps, ModelFaceID
   (cap -> label by the centerline axis at the clip point). No ClipVessel needed.
   Every stage is checked (open boundaries, non-manifold edges): the error says which.
3. checks: closed surface, contiguous ModelFaceID, 1 face per endpoint, no slivers
On failure: move the points of ClipPoints_<case>, then main("<case>").
"""
import os
import sys

PIPELINE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0]))
sys.path.insert(0, PIPELINE_DIR)

import slicer  # noqa: E402

from utils import centerline as cl  # noqa: E402
from utils import clip, config, state  # noqa: E402
from utils.seg_tools import save_node  # noqa: E402


def _get_or_load(name, path, loader):
    node = slicer.mrmlScene.GetFirstNodeByName(name)
    if node is None:
        node = loader(path)
        if node is None:
            raise RuntimeError(f"Cannot load {path}")
        node.SetName(name)
    return node


def _fail(case_id, msg):
    print(f"\n>>> step2 FAILED: {msg}")
    print(f">>> Move the points of 'ClipPoints_{case_id}' (e.g. further inside the vessel), then: main('{case_id}')")
    print(f">>> To recompute them from params.json: delete 'ClipPoints_{case_id}', then main('{case_id}')")
    raise RuntimeError(msg)


def _check_stage(pd, n_loops, clip_pts):
    """Exactly n_loops open boundaries and no non-manifold edge; locations in the error."""
    loops = clip.open_loops(pd)
    nm = clip.bad_edges(pd)["non_manifold"]
    pinched = clip.pinched_points(pd)
    warped = clip.nonplanar_loops(pd)
    if len(loops) != n_loops or nm or pinched or warped:
        where = sorted({clip.nearest_label(pos, clip_pts) for pos, _ in nm})
        where_p = sorted({clip.nearest_label(pos, clip_pts) for pos in pinched})
        raise RuntimeError(f"{len(loops)} open boundaries (expected {n_loops}) "
                           f"[(centre RAS, diameter mm): {[([round(c, 1) for c in x], round(d, 1)) for x, d in loops]}], "
                           f"{len(nm)} non-manifold edges near clip points {where}, "
                           f"{len(pinched)} pinched boundary points near clip points {where_p}, "
                           f"non-flat boundaries (centre, diameter, deviation mm) {warped}")


def _show_stage(case_id, pd, clip_pts, tree):
    """Surface of the failed stage + markers on open boundaries and non-manifold edges."""
    m = slicer.modules.models.logic().AddModel(pd)
    m.SetName(f"stage_surface_{case_id}")
    m.GetDisplayNode().SetScalarVisibility(False)
    m.GetDisplayNode().SetColor(0.3, 0.6, 0.9)
    tree.GetDisplayNode().SetVisibility(False)
    marks = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"stage_marks_{case_id}")
    for c, d in clip.open_loops(pd):
        marks.SetNthControlPointLabel(marks.AddControlPointWorld(c), f"open {d:.1f}mm")
    for pos in sorted({tuple(pos) for pos, _ in clip.bad_edges(pd)["non_manifold"]}):
        marks.SetNthControlPointLabel(marks.AddControlPointWorld(list(pos)), "non-manifold")
    for pos in clip.pinched_points(pd):
        marks.SetNthControlPointLabel(marks.AddControlPointWorld(pos), "pinched")


def main(case_id):
    paths = config.case_paths(case_id)
    p = config.load_params()["step2"]
    p1 = config.load_params()["step1"]
    print(f">>> step2 {case_id}")

    tree = _get_or_load(f"tree_model_{case_id}", paths["tree_model"], slicer.util.loadModel)
    cl_model = _get_or_load(f"centerline_{case_id}", paths["centerline"], slicer.util.loadModel)
    ep = _get_or_load(f"Endpoints_{case_id}", paths["endpoints_mrk"], slicer.util.loadMarkups)
    labels = [ep.GetNthControlPointLabel(i) for i in range(ep.GetNumberOfControlPoints())]
    endpoints = dict(zip(labels, cl.get_points(ep)))
    if labels[0] != "inlet" or len(labels) != 1 + len(p1["outlet_segments"]):
        raise RuntimeError(f"Endpoints must be 'inlet' first + {len(p1['outlet_segments'])} outlets: {labels}")

    # 1. clip points (inlet first: its cap gets face id 2)
    cp_name = f"ClipPoints_{case_id}"
    cp = slicer.mrmlScene.GetFirstNodeByName(cp_name)
    if cp is not None and cp.GetNumberOfControlPoints() == len(labels):
        print(f"  reusing clip points of {cp_name} (manual correction)")
    else:
        ins = p["inset"]
        pts = clip.inset_clip_points(cl_model.GetPolyData(), endpoints, "inlet", ins["factor"],
                                     ins["min_mm"], ins["max_branch_fraction"], ins["share_tol_mm"])
        for name in labels:
            _, d, L = pts[name]
            print(f"  clip {name}: inset {d:.1f} mm (segment {L:.1f} mm)")
        cp = cp or slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", cp_name)
        cl.set_points(cp, [pts[n][0] for n in labels], labels)

    # 2. clip -> flow extensions -> caps, each stage checked; on failure the last surface
    #    is shown with one marker per open boundary / bad edge
    for name in (f"stage_surface_{case_id}", f"stage_marks_{case_id}"):
        old = slicer.mrmlScene.GetFirstNodeByName(name)
        if old is not None:
            slicer.mrmlScene.RemoveNode(old)
    clip_pts = dict(zip([cp.GetNthControlPointLabel(i) for i in range(cp.GetNumberOfControlPoints())],
                        cl.get_points(cp)))
    cl_pd = cl_model.GetPolyData()
    stage, surface = "input tree model", tree.GetPolyData()
    try:
        _check_stage(surface, 0, clip_pts)                       # input must be closed
        stage, surface = "clip", clip.clean_surface(clip.clip_all(surface, cl_pd, clip_pts, endpoints, labels))
        _check_stage(surface, len(labels), clip_pts)
        # clean: the vmtk filter leaves coincident unmerged points on the extension (a zero-width
        # crack the capper fans over -> non-manifold edges on the caps)
        stage, surface = "flow extensions", clip.clean_surface(clip.flow_extensions(surface, cl_pd, p["flow_ext"]))
        _check_stage(surface, len(labels), clip_pts)
        stage = "caps"
        final_pd, faces = clip.cap_and_label(surface, cl_pd, clip_pts, endpoints, labels, p["cap_match_max_mm"])
    except RuntimeError as e:
        _show_stage(case_id, surface, clip_pts, tree)
        _fail(case_id, f"[{stage}] {e}")
    for fid, name in sorted(faces.items()):
        print(f"  face {fid}: {name}")

    # final surface in the scene (also on a failed check, to inspect it)
    for name in (f"final_tree_{case_id}", f"bad_edges_{case_id}"):
        old = slicer.mrmlScene.GetFirstNodeByName(name)
        if old is not None:
            slicer.mrmlScene.RemoveNode(old)
    final = slicer.modules.models.logic().AddModel(final_pd)
    final.SetName(f"final_tree_{case_id}")
    tree.GetDisplayNode().SetVisibility(False)
    # plain colour: scalar colouring by ModelFaceID renders nothing in this Slicer version
    final.GetDisplayNode().SetScalarVisibility(False)
    final.GetDisplayNode().SetColor(0.9, 0.3, 0.3)
    state.reset_3d_view()

    # 3. checks (bad edges marked with points named after their face)
    try:
        clip.check_closed_and_faces(final_pd, faces, labels)
    except RuntimeError as e:
        bad = clip.bad_edges(final_pd, faces)
        marks = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"bad_edges_{case_id}")
        for kind, items in bad.items():
            for pos, face in items:
                marks.SetNthControlPointLabel(marks.AddControlPointWorld(pos), f"{kind}:{face}")
        _fail(case_id, str(e))

    def save():
        save_node(final, paths["final_tree"])
        clip.save_faces_json(faces, paths["final_faces"])
        save_node(cp, paths["clip_points"])

    state.finish_step("step2", save)


state.run(main, sys.argv, "step2_cap_flow_ext.py")