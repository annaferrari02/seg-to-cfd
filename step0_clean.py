r"""Step 0 - clean the segmentation

Slicer --python-script step0_clean.py <case_id>

Input  (read-only): label map (*-label.nii.gz) + ORIGINAL CT (not normalized), from config
Output:             <database_root>/<case_id>/Seg/<case_id>_seg_clean.seg.nrrd
                    (branch segments + merged "tree")

1. label value -> segment name          (params.json "segments")
2. filters in order                     (params.json "step0.filters")
   "op" = SegmentEditorManager method, other keys = its arguments;
   "merge" runs once on the whole list, the other ops once per segment
      remove_above (in the default filters, before the merge): drops voxels with HU > threshold
   (calcifications). threshold = hu_max, or with "adaptive" max(hu_max, median + k*MAD of the
   HU in adaptive.ref, computed once on the unedited segment); "max_frac": a segment losing
   more than that fraction is left untouched (WARNING, fix by hand in the GUI).
   Before editing: CT scale check (standard HU or +1024 shift -> the threshold is shifted)
   and alignment check (mean HU in step0.hu_check.segment)
3. GUI: CT + 3D surface of "tree" for a visual check
"""
import os
import sys

PIPELINE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0]))
sys.path.insert(0, PIPELINE_DIR)

from utils import config, state  # noqa: E402
from utils.seg_tools import (SegmentEditorManager, check_segment_hu, ct_offset,  # noqa: E402
                             empty_segments, labelmap_to_segmentation, load_volume, save_node,
                             segment_names,
                             show_for_check)


def main(case_id):
    paths = config.case_paths(case_id)
    print(f"  CT ({paths['ct_source']}): {paths['ct_in']}")
    filters = config.load_params()["step0"]["filters"]
    print(f">>> step0 {case_id}\n  seg: {paths['seg_in']}\n  ct:  {paths['ct_in']}")

    ct = load_volume(paths["ct_in"], "ct")
    seg, _ = labelmap_to_segmentation(paths["seg_in"], ct, config.segment_map())
    has = lambda name: bool(seg.GetSegmentation().GetSegmentIdBySegmentName(name))

    # the CT is the original one (not normalized): detect its scale (standard HU or +1024
    # shift, the threshold follows it) and check it is aligned with the segmentation
    offset = ct_offset(ct)
    chk = config.load_params()["step0"]["hu_check"]
    check_segment_hu(seg, chk["segment"], ct, offset, *chk["mean_hu_range"])

    with SegmentEditorManager(seg, ct, hu_offset=offset) as sem:
        for f in filters:
            op = f["op"]
            args = {k: v for k, v in f.items() if k not in ("op", "segments")}
            targets = segment_names(seg) if f["segments"] == "all" else f["segments"]
            targets = [n for n in targets if has(n)]  # missing branch -> skip
            if op == "merge":
                sem.merge(args["name"], targets)
            else:
                for name in targets:
                    getattr(sem, op)(name, **args)

    show_for_check(seg, ct, visible=["tree"])
    def save():
        # an edit of "tree" in the GUI with 'overwrite all' empties the branch segments: refuse
        empty = empty_segments(seg, ct)
        if empty:
            raise RuntimeError(f"Empty segments, not saved: {empty}. Undo (Ctrl+Z) in the Segment "
                               f"Editor or re-run step0; edit with 'Modify other segments' = Allow overlap")
        save_node(seg, paths["seg_clean"])

    state.finish_step("step0", save)


state.run(main, sys.argv, "step0_clean.py")