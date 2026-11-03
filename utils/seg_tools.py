"""Segmentation utilities (Slicer world: Slicer API + numpy only).
SegmentEditorManager derived from the colleague's version, reduced to the methods the pipeline uses."""
import logging
import os

import numpy as np
import slicer

from utils import config, hu, mask_ops

# One colour per segment (cycled if more segments than colours)
_COLORS = [
    (0.90, 0.30, 0.30), (0.30, 0.60, 0.90), (0.30, 0.80, 0.40), (0.95, 0.75, 0.20),
    (0.70, 0.40, 0.85), (0.20, 0.80, 0.80), (0.95, 0.50, 0.20), (0.60, 0.60, 0.60),
    (0.85, 0.20, 0.20), (0.50, 0.30, 0.20), (0.90, 0.50, 0.70),
]


# ---------------------------------------------------------------- loading / saving

def load_volume(path, name=None):
    node = slicer.util.loadVolume(path, {"labelmap": False})
    if node is None:
        raise RuntimeError(f"Cannot load volume {path}")
    if name:
        node.SetName(name)
    return node


def ct_offset(volume):
    """0 for standard HU, 1024 for a CT stored with a +1024 shift (see utils.hu.hu_offset)."""
    off, report = hu.hu_offset(slicer.util.arrayFromVolume(volume))
    print(f"  {volume.GetName()}: {report} -> offset {off}")
    return off


def check_segment_hu(seg, segment_name, volume, offset, lo, hi):
    """Mean HU inside a segment in [lo, hi] (CT aligned with the segmentation, same scan)."""
    sid = seg.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
    if not sid:
        raise RuntimeError(f"Segment not found: {segment_name}")
    mask = slicer.util.arrayFromSegmentBinaryLabelmap(seg, sid, volume)
    mean = hu.check_mean(slicer.util.arrayFromVolume(volume), mask, offset, lo, hi, segment_name)
    print(f"  mean HU in {segment_name}: {mean:.0f}")


def get_or_load_volume(name, path):
    """Node already in the scene (e.g. previous step of the same session) or loaded from file."""
    node = slicer.mrmlScene.GetFirstNodeByName(name)
    return node if node is not None else load_volume(path, name)


def get_or_load_segmentation(name, path):
    """Same as above for a .seg.nrrd: in the same session manual (even unsaved) edits are kept."""
    node = slicer.mrmlScene.GetFirstNodeByName(name)
    if node is not None and node.IsA("vtkMRMLSegmentationNode"):
        return node
    node = slicer.util.loadSegmentation(path)
    if node is None:
        raise RuntimeError(f"Cannot load segmentation {path}")
    node.SetName(name)
    return node


def segment_to_model(seg, segment_name, model_name, smoothing=None):
    """Closed surface of one segment -> model node (replaces an existing model with that name)."""
    old = slicer.mrmlScene.GetFirstNodeByName(model_name)
    if old is not None:
        slicer.mrmlScene.RemoveNode(old)
    s = seg.GetSegmentation()
    sid = s.GetSegmentIdBySegmentName(segment_name)
    if not sid:
        raise RuntimeError(f"Segment not found: {segment_name}")
    if smoothing is not None:  # surface smoothing of the labelmap -> surface conversion
        s.SetConversionParameter("Smoothing factor", str(smoothing))
        seg.RemoveClosedSurfaceRepresentation()
    seg.CreateClosedSurfaceRepresentation()
    model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", model_name)
    model.CreateDefaultDisplayNodes()
    if not slicer.modules.segmentations.logic().ExportSegmentToRepresentationNode(s.GetSegment(sid), model):
        raise RuntimeError(f"Export of {segment_name} to model failed")
    if model.GetPolyData() is None or model.GetPolyData().GetNumberOfPoints() == 0:
        raise RuntimeError(f"Empty model from segment {segment_name}")
    print(f"  model {model_name}: {model.GetPolyData().GetNumberOfPoints()} points")
    return model


def segment_points_ras(seg, segment_name, volume):
    """RAS coordinates of the voxels of a segment (on the volume grid)."""
    import vtk
    sid = seg.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
    arr = slicer.util.arrayFromSegmentBinaryLabelmap(seg, sid, volume)  # [k, j, i]
    kji = np.argwhere(arr > 0)
    m = vtk.vtkMatrix4x4()
    volume.GetIJKToRASMatrix(m)
    ijk1 = np.c_[kji[:, ::-1], np.ones(len(kji))]
    return (ijk1 @ slicer.util.arrayFromVTKMatrix(m).T)[:, :3]


def label_points(points_ras, seg, candidates, volume):
    """For each point: (nearest segment among candidates, distance mm).
    Same purpose as the colleague's getSegmentationAtRASNeighborhood, but by nearest voxel
    instead of a fixed neighbourhood: works also when the point is just outside the segment."""
    clouds = {n: segment_points_ras(seg, n, volume) for n in candidates}
    empty = [n for n, c in clouds.items() if not len(c)]
    clouds = {n: c for n, c in clouds.items() if len(c)}
    if empty:
        print(f"  WARNING segments empty on the CT grid: {empty}")
    if not clouds:
        raise RuntimeError(
            f"No branch segment to label the endpoints: candidates {candidates}, empty on the CT "
            f"grid {empty}, segments in the segmentation {segment_names(seg)}. "
            f"Either the saved segmentation contains only the merged segment, or it does not "
            f"overlap the CT ({volume.GetName()})")
    out = []
    for p in np.asarray(points_ras, float):
        dist = {n: float(np.min(np.linalg.norm(c - p, axis=1))) for n, c in clouds.items()}
        best = min(dist, key=dist.get)
        out.append((best, dist[best]))
    return out


def labelmap_to_segmentation(label_path, ref_volume, label_names, node_name="seg"):
    """Import a label map as a segmentation; label value v -> segment named label_names[v].
    Fails on labels without a name; returns (node, names missing from the label map)."""
    lab = slicer.util.loadLabelVolume(label_path)
    if lab is None:
        raise RuntimeError(f"Cannot load label map {label_path}")

    present = set(np.unique(slicer.util.arrayFromVolume(lab)).astype(int)) - {0}
    unknown = present - set(label_names)
    if unknown:
        raise RuntimeError(f"Labels without a name in params.json: {sorted(unknown)}")
    missing = {label_names[v] for v in set(label_names) - present}
    if missing:
        logging.warning(f"Segments missing in {os.path.basename(label_path)}: {sorted(missing)}")

    # Segment names come from the colour table attached to the label map
    table = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", "seg_names")
    table.SetTypeToUser()
    table.SetNumberOfColors(max(label_names) + 1)
    table.SetColor(0, "Background", 0, 0, 0, 0)
    for v, name in label_names.items():
        table.SetColor(v, name, *_COLORS[(v - 1) % len(_COLORS)], 1.0)
    lab.GetDisplayNode().SetAndObserveColorNodeID(table.GetID())

    seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", node_name)
    seg.SetReferenceImageGeometryParameterFromVolumeNode(ref_volume)
    if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lab, seg):
        raise RuntimeError("Label map import failed")
    slicer.mrmlScene.RemoveNode(lab)
    slicer.mrmlScene.RemoveNode(table)

    names = set(segment_names(seg))
    expected = {label_names[v] for v in present}
    if names != expected:
        raise RuntimeError(f"Segment names after import {sorted(names)} != expected {sorted(expected)}")
    return seg, missing


def segment_names(seg):
    s = seg.GetSegmentation()
    return [s.GetNthSegment(i).GetName() for i in range(s.GetNumberOfSegments())]


def show_for_check(seg, volume, visible=None):
    """Visual check in the GUI: volume in the slice views, 3D surface of the visible segments."""
    if slicer.util.mainWindow() is None:  # --no-main-window
        return
    seg.CreateClosedSurfaceRepresentation()
    display = seg.GetDisplayNode()
    if visible is not None:
        display.SetAllSegmentsVisibility(False)
        for name in visible:
            display.SetSegmentVisibility(seg.GetSegmentation().GetSegmentIdBySegmentName(name), True)
    slicer.util.setSliceViewerLayers(background=volume, fit=True)
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()
    open_editor_no_overwrite(seg, volume)


def open_editor_no_overwrite(seg, volume):
    """Open the Segment Editor on seg with 'Modify other segments' = Allow overlap.
    The GUI default (overwrite all) empties every branch segment as soon as 'tree' is edited
    (e.g. smoothing): the branches are needed by step1 to name the endpoints."""
    try:
        slicer.util.selectModule("SegmentEditor")
        editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
        editor.setSegmentationNode(seg)
        editor.setSourceVolumeNode(volume)
        editor.mrmlSegmentEditorNode().SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
        print("  Segment Editor open, 'Modify other segments' = Allow overlap (keep it)")
    except Exception as e:  # GUI convenience only
        print(f"  WARNING could not set the Segment Editor overwrite mode: {e}. In the editor set "
              f"Masking > 'Modify other segments' = Allow overlap before editing")


def empty_segments(seg, volume, names=None):
    """Names of the segments with no voxel on the volume grid."""
    names = segment_names(seg) if names is None else names
    s = seg.GetSegmentation()
    return [n for n in names if not np.any(slicer.util.arrayFromSegmentBinaryLabelmap(
        seg, s.GetSegmentIdBySegmentName(n), volume))]


def save_node(node, path):
    """Save any node; never inside input_root."""
    config.assert_writable(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not slicer.util.saveNode(node, path):
        raise RuntimeError(f"Cannot save {path}")
    print(f"  saved {path}")


# ---------------------------------------------------------------- segment editor

class SegmentEditorManager:
    """Hidden Segment Editor on one segmentation. Segments accepted by ID or by name.
    Default overwrite = none: editing one segment never modifies the others."""

    def __init__(self, segmentation_node, volume_node=None, overwrite=None, mask_mode=None, hu_offset=0):
        self.segmentation_node = segmentation_node
        self.volume_node = volume_node  # original CT (not normalized): used by remove_above
        self.hu_offset = hu_offset      # 0 = standard HU, 1024 = CT stored with +1024 shift
        self._calc_thr = {}             # adaptive calcium thresholds, computed once per manager
        self.widget = slicer.qMRMLSegmentEditorWidget()
        self.widget.setMRMLScene(slicer.mrmlScene)
        self.editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.widget.setMRMLSegmentEditorNode(self.editor_node)
        self.widget.setSegmentationNode(segmentation_node)
        if volume_node is not None:
            self.widget.setSourceVolumeNode(volume_node)
        self.editor_node.SetOverwriteMode(
            slicer.vtkMRMLSegmentEditorNode.OverwriteNone if overwrite is None else overwrite)
        if mask_mode is not None:
            self.editor_node.SetMaskMode(mask_mode)
        segmentation_node.RemoveClosedSurfaceRepresentation()  # faster edits; rebuilt on demand

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()

    def get_segment_id(self, segment):
        """ID from ID or name; fails if the segment does not exist."""
        s = self.segmentation_node.GetSegmentation()
        if s.GetSegment(segment) is not None:
            return segment
        sid = s.GetSegmentIdBySegmentName(segment)
        if not sid:
            raise RuntimeError(f"Segment not found: {segment}")
        return sid

    def set_segment(self, segment):
        self.editor_node.SetSelectedSegmentID(self.get_segment_id(segment))

    def apply_effect(self, effect_name, params):
        self.widget.setActiveEffectByName(effect_name)
        effect = self.widget.activeEffect()
        if effect is None:
            raise RuntimeError(f"Segment Editor effect not found: {effect_name}")
        for key, val in params.items():
            effect.setParameter(key, str(val))
        effect.self().onApply()
        self.widget.setActiveEffect(None)

    def smoothing(self, segment, method="GAUSSIAN", size_mm=1.0):
        """method: MEDIAN | MORPHOLOGICAL_CLOSING | MORPHOLOGICAL_OPENING | GAUSSIAN.
        size_mm = kernel size, or standard deviation for GAUSSIAN."""
        size_key = "GaussianStandardDeviationMm" if method == "GAUSSIAN" else "KernelSizeMm"
        self.set_segment(segment)
        self.apply_effect("Smoothing", {
            "SmoothingMethod": method,
            size_key: size_mm,
            "ApplyToAllVisibleSegments": 0,
        })
        print(f"  smoothing {method} {size_mm} mm -> {segment}")

    def hollow(self, segment, thickness_mm=1.0, mode="OUTSIDE_SURFACE"):
        """mode: INSIDE_SURFACE | MEDIAL_SURFACE | OUTSIDE_SURFACE"""
        self.set_segment(segment)
        self.apply_effect("Hollow", {
            "ShellThicknessMm": thickness_mm,
            "ShellMode": mode,
            "ApplyToAllVisibleSegments": 0,
        })
        print(f"  hollow {mode} {thickness_mm} mm -> {segment}")

    def margin(self, segment, size_mm=1.0):
        """Grow (>0) or shrink (<0) the segment."""
        self.set_segment(segment)
        self.apply_effect("Margin", {"MarginSizeMm": size_mm, "ApplyToAllVisibleSegments": 0})
        print(f"  margin {size_mm} mm -> {segment}")

    def calc_threshold(self, ref_segment="lumen", hu_floor=400.0, k=3.0):
        """Per-patient calcium threshold = max(hu_floor, median + k*MAD) of the HU in ref_segment.
        With strong contrast (lumen ~450-600 HU) a fixed threshold removes the opacified blood.
        Median/MAD are robust to the calcifications inside the segment (if < ~50% of it).
        Computed once per manager (cached), i.e. on ref_segment BEFORE any removal."""
        key = (ref_segment, float(hu_floor), float(k))
        if key in self._calc_thr:
            return self._calc_thr[key]
        if self.volume_node is None:
            raise RuntimeError("calc_threshold needs the CT (volume_node)")
        sid = self.get_segment_id(ref_segment)
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(self.segmentation_node, sid, self.volume_node) > 0
        if not mask.any():
            raise RuntimeError(f"calc_threshold: segment {ref_segment} empty on the CT grid")
        vals = slicer.util.arrayFromVolume(self.volume_node)[mask].astype(float) - self.hu_offset
        med = float(np.median(vals))
        mad = 1.4826 * float(np.median(np.abs(vals - med)))
        thr = max(float(hu_floor), med + k * mad)
        print(f"  {ref_segment}: HU median {med:.0f}, MAD {mad:.0f} -> calcium threshold "
              f"max({hu_floor:.0f}, {med:.0f} + {k:g}*{mad:.0f}) = {thr:.0f} HU")
        self._calc_thr[key] = thr
        return thr

    def remove_above(self, segment, hu_max=350.0, adaptive=None, max_frac=None):
        """Remove from the segment every voxel with HU > threshold (calcifications); the threshold
        on the stored values is threshold + hu_offset. Same result as the colleague's threshold
        (masked to the segment) + subtract, in one numpy step, no mask left on the editor.
        The segment is read on the CT grid.
        adaptive: None -> threshold = hu_max (fixed, old behaviour);
                  {"ref": "lumen", "k": 3} -> threshold = calc_threshold(ref, hu_max, k).
        max_frac: if the removed fraction of the segment exceeds it, nothing is removed
                  (the threshold is hitting the contrast) and a WARNING is printed."""
        if self.volume_node is None:
            raise RuntimeError("remove_above needs the CT (volume_node)")
        thr = hu_max
        if adaptive:
            thr = self.calc_threshold(adaptive.get("ref", "lumen"), hu_max, adaptive.get("k", 3.0))
        sid = self.get_segment_id(segment)
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(self.segmentation_node, sid, self.volume_node)
        hu = slicer.util.arrayFromVolume(self.volume_node)
        if mask.shape != hu.shape:
            raise RuntimeError(f"Segment {segment} grid {mask.shape} != CT grid {hu.shape}")
        calc = (mask > 0) & (hu > thr + self.hu_offset)
        n_in, n_rm = int(np.count_nonzero(mask)), int(np.count_nonzero(calc))
        frac = n_rm / max(n_in, 1)
        if max_frac is not None and frac > max_frac:
            print(f"  WARNING remove HU > {thr:.0f}: {n_rm} of {n_in} voxels ({100.0 * frac:.1f}%) "
                  f"> max {100.0 * max_frac:.0f}% -> SKIPPED on {segment} (threshold hits the "
                  f"contrast: remove calcifications by hand in the GUI)")
            return
        mask[calc] = 0
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask, self.segmentation_node, sid, self.volume_node)
        print(f"  remove HU > {thr:.0f}: {n_rm} of {n_in} voxels ({100.0 * frac:.1f}%) -> {segment}")

    def fill_cavities(self, segment):
        """Fill the empty regions fully enclosed by the segment (e.g. calcifications removed
        inside the lumen): otherwise they become closed surfaces inside the vessel and
        separate regions in the volume mesh."""
        sid = self.get_segment_id(segment)
        ref = self.volume_node
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(self.segmentation_node, sid, ref)
        filled, n = mask_ops.fill_cavities(mask)
        if n:
            slicer.util.updateSegmentBinaryLabelmapFromArray(filled, self.segmentation_node, sid, ref)
        print(f"  fill cavities: {n} voxels -> {segment}")

    def islands(self, segment, operation="KEEP_LARGEST_ISLAND", min_size=1000):
        self.set_segment(segment)
        self.apply_effect("Islands", {"Operation": operation, "MinimumSize": min_size})
        print(f"  islands {operation} -> {segment}")

    def merge(self, new_name, segments):
        """New segment = union of segments (the originals are kept)."""
        s = self.segmentation_node.GetSegmentation()
        if s.GetSegmentIdBySegmentName(new_name):
            raise RuntimeError(f"Segment already exists: {new_name}")
        new_id = s.AddEmptySegment(new_name, new_name)
        for name in segments:
            self.editor_node.SetSelectedSegmentID(new_id)
            self.apply_effect("Logical operators", {
                "Operation": "UNION",
                "ModifierSegmentID": self.get_segment_id(name),
                "BypassMasking": 1,
            })
        print(f"  merge {segments} -> {new_name}")

    def cleanup(self):
        # Detach the widget from the segmentation BEFORE removing its editor node,
        # otherwise later edits of the segmentation still reach the widget (warnings / crash)
        self.widget.setActiveEffect(None)
        self.widget.setSegmentationNode(None)
        self.widget.setSourceVolumeNode(None)
        self.widget.setMRMLSegmentEditorNode(None)
        self.widget.setMRMLScene(None)
        slicer.mrmlScene.RemoveNode(self.editor_node)
        self.widget.deleteLater()