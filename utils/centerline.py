"""Centerline utilities (Slicer world, SlicerVMTK extension).
Same method as the colleague's SegmentEditorManager.extract_centerline: Extract Centerline module
driven through its widget (preprocess + decimation + target points), input = model node."""
import json
import os

import slicer

from utils import config


def _ui():
    widget = slicer.modules.extractcenterline.widgetRepresentation()
    if widget is None:
        raise RuntimeError("Extract Centerline module not available (SlicerVMTK installed?)")
    return widget.self().ui


def _set_input(ui, surface, endpoints, opts):
    ui.inputSurfaceSelector.setCurrentNode(surface)
    ui.endPointsMarkupsSelector.setCurrentNode(endpoints)
    ui.preprocessInputSurfaceModelCheckBox.checked = bool(opts["preprocess"])
    ui.decimationAggressivenessWidget.setValue(opts["decimation"])
    ui.targetKPointCountWidget.value = opts["target_kpoints"]
    ui.curveSamplingDistanceSpinBox.setValue(opts["sampling_distance"])


def detect_endpoints(surface, endpoints, opts):
    """'Auto-detect' of the module: fills the endpoints markup with the network end points."""
    ui = _ui()
    endpoints.RemoveAllControlPoints()
    _set_input(ui, surface, endpoints, opts)
    ui.autoDetectEndPointsPushButton.click()
    n = endpoints.GetNumberOfControlPoints()
    print(f"  auto-detected endpoints: {n}")
    return n


def get_points(markups):
    return [list(markups.GetNthControlPointPositionWorld(i)) for i in range(markups.GetNumberOfControlPoints())]


def set_points(markups, points, labels):
    """Replace the control points (order matters: the first one is the centerline source)."""
    markups.RemoveAllControlPoints()
    for p, label in zip(points, labels):
        i = markups.AddControlPointWorld(p)
        markups.SetNthControlPointLabel(i, label)


def extract_centerline(surface, endpoints, model, curve, opts):
    """Centerline from endpoints[0] (source) to the other endpoints. Output model has 'Radius'."""
    ui = _ui()
    _set_input(ui, surface, endpoints, opts)
    ui.outputCenterlineModelSelector.setCurrentNode(model)
    ui.outputCenterlineCurveSelector.setCurrentNode(curve)
    ui.applyButton.click()
    pd = model.GetPolyData()
    if pd is None or pd.GetNumberOfPoints() == 0:
        raise RuntimeError("Centerline extraction produced an empty model")
    if pd.GetPointData().GetArray("Radius") is None:
        raise RuntimeError("Centerline model has no 'Radius' array (needed by ClipVessel)")
    print(f"  centerline: {pd.GetNumberOfPoints()} points, {pd.GetNumberOfCells()} cells")


def save_endpoints_json(markups, path):
    """{label: [x, y, z]} in LPS, same frame as the .vtk files saved by Slicer.
    Plain JSON readable from ParaView / SimVascular without Slicer."""
    config.assert_writable(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pts = {}
    for i in range(markups.GetNumberOfControlPoints()):
        r, a, s = markups.GetNthControlPointPositionWorld(i)
        pts[markups.GetNthControlPointLabel(i)] = [-r, -a, s]  # RAS -> LPS
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"coordinate_system": "LPS", "points": pts}, f, indent=2)
    print(f"  saved {path}")