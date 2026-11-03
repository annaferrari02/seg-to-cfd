"""Clip (inset) + cap + flow extensions (Slicer world, SlicerVMTK ClipVessel)."""
import json
import os

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from utils import config

FACE_ARRAY = "ModelFaceID"


# ---------------------------------------------------------------- inset clip points

def _cells(pd):
    cells = []
    for c in range(pd.GetNumberOfCells()):
        ids = pd.GetCell(c).GetPointIds()
        cells.append(np.array([ids.GetId(i) for i in range(ids.GetNumberOfIds())]))
    return [c for c in cells if len(c) >= 2]


def _arclen(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def _min_dist(P, Q):
    """Distance of each point of P from the nearest point of Q."""
    return np.array([np.min(np.linalg.norm(Q - p, axis=1)) for p in P])


def inset_clip_points(cl_pd, endpoints, inlet_label, factor, min_mm, max_fraction, share_tol):
    """One clip point per endpoint, moved inward along the centerline by
         inset = max(factor * median radius, min_mm), capped at max_fraction * segment length
    segment = distal branch (from its bifurcation) for an outlet, trunk (to the first
    bifurcation) for the inlet. Centerline cells are full inlet->leaf paths.
    endpoints: {label: RAS}. Returns {label: (position, inset_mm, segment_length_mm)}."""
    pts = vtk_to_numpy(cl_pd.GetPoints().GetData()).astype(float)
    R = vtk_to_numpy(cl_pd.GetPointData().GetArray("Radius")).astype(float)
    cells = _cells(cl_pd)
    if len(cells) < 2:
        raise RuntimeError(f"Centerline has {len(cells)} branches")

    # orient every cell inlet -> leaf
    inlet = np.asarray(endpoints[inlet_label], float)
    cells = [c if np.linalg.norm(pts[c[0]] - inlet) <= np.linalg.norm(pts[c[-1]] - inlet) else c[::-1]
             for c in cells]

    def inset(c_from_terminus, radii, length):
        t = min(max(factor * float(np.median(radii)), min_mm), max_fraction * length)
        s = _arclen(pts[c_from_terminus])
        i = min(int(np.searchsorted(s, t)), len(c_from_terminus) - 1)
        return pts[c_from_terminus[i]].tolist(), float(s[i])

    out = {}

    # inlet: trunk = start of cell 0 shared with every other cell
    c0 = cells[0]
    trunk_end = len(c0) - 1
    for c in cells[1:]:
        far = np.nonzero(_min_dist(pts[c0], pts[c]) > share_tol)[0]
        if len(far):
            trunk_end = min(trunk_end, int(far[0]))
    L = _arclen(pts[c0[:trunk_end + 1]])[-1]
    pos, d = inset(c0, R[c0[:trunk_end + 1]], L)
    out[inlet_label] = (pos, d, L)

    # outlets: endpoint -> cell with the nearest leaf (must be one-to-one)
    leaves = np.array([pts[c[-1]] for c in cells])
    used = {}
    for label, p in endpoints.items():
        if label == inlet_label:
            continue
        k = int(np.argmin(np.linalg.norm(leaves - np.asarray(p, float), axis=1)))
        if k in used:
            raise RuntimeError(f"Endpoints {used[k]} and {label} map to the same centerline branch")
        used[k] = label
        c = cells[k]
        others = np.vstack([pts[cc] for j, cc in enumerate(cells) if j != k])
        shared = np.nonzero(_min_dist(pts[c], others) < share_tol)[0]
        b = int(shared[-1]) if len(shared) else 0  # bifurcation index
        L = _arclen(pts[c[b:]])[-1]
        pos, d = inset(c[::-1], R[c[b:]], L)
        out[label] = (pos, d, L)
    return out


# ---------------------------------------------------------------- clip + cap + extensions

def local_clip(pd, origin, normal, endpoint, other_endpoints, eps_mm=0.01):
    """Remove ONLY the vessel end beyond the plane (origin, normal -> toward the end):
    the piece on the far side of the plane that is connected to `endpoint`.
    Anything else the infinite plane crosses (a neighbouring vessel) is cut and immediately
    re-stitched: both sides share the same new points, so they merge back exactly.
    Fails if the cut does not open exactly one new boundary around `origin`, or if the
    removed piece also contains another vessel end (plane through a bifurcation)."""
    import vtk
    P = vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    o, n = np.asarray(origin, float), np.asarray(normal, float) / np.linalg.norm(normal)
    for _ in range(50):  # keep the plane off existing vertices: no sliver triangles
        if np.min(np.abs((P - o) @ n)) > eps_mm:
            break
        o = o + 2 * eps_mm * n

    plane = vtk.vtkPlane()
    plane.SetOrigin(o.tolist())
    plane.SetNormal(n.tolist())
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(pd)
    clipper.SetClipFunction(plane)
    clipper.SetInsideOut(1)           # output: interior side; clipped output: end side
    clipper.GenerateClippedOutputOn()
    clipper.Update()
    prox, far = clipper.GetOutput(), clipper.GetClippedOutput()
    if far.GetNumberOfCells() == 0:
        raise RuntimeError("nothing beyond the plane (clip point at or past the vessel end)")

    # drop the unused points far shares with prox, else RegionId and points do not line up
    far_clean = vtk.vtkCleanPolyData()
    far_clean.SetInputData(far)
    far_clean.PointMergingOff()
    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputConnection(far_clean.GetOutputPort())
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    d = vtk.vtkPolyData()
    d.DeepCopy(conn.GetOutput())
    D = vtk_to_numpy(d.GetPoints().GetData()).astype(float)
    rid_pt = vtk_to_numpy(d.GetPointData().GetArray("RegionId")).astype(int)
    # region of a cell = region of its first point (cell RegionId is not written by every VTK)
    rid_cell = np.array([rid_pt[d.GetCell(i).GetPointId(0)] for i in range(d.GetNumberOfCells())])
    end_region = rid_pt[np.argmin(np.linalg.norm(D - np.asarray(endpoint, float), axis=1))]

    # the removed piece must not contain another vessel end
    in_end = rid_pt == end_region
    for lab, q in other_endpoints.items():
        q = np.asarray(q, float)
        d_all = np.min(np.linalg.norm(P - q, axis=1))
        d_end = np.min(np.linalg.norm(D[in_end] - q, axis=1))
        if d_end <= d_all + 0.5:
            raise RuntimeError(f"the cut would also remove the end of {lab} (plane through a bifurcation)")

    d.BuildLinks()
    for c in np.nonzero(rid_cell == end_region)[0]:
        d.DeleteCell(int(c))
    d.RemoveDeletedCells()

    app = vtk.vtkAppendPolyData()
    app.AddInputData(prox)
    app.AddInputData(d)
    out = clean_surface(app_output(app))

    before, after = open_loops(pd), open_loops(out)
    if len(after) != len(before) + 1:
        raise RuntimeError(f"cut opened {len(after) - len(before)} boundaries instead of 1")
    c, diam = min(after, key=lambda l: np.linalg.norm(np.asarray(l[0]) - o))
    if np.linalg.norm(np.asarray(c) - o) > diam / 2:
        raise RuntimeError("new boundary is not around the clip point")
    return out


def app_output(alg):
    alg.Update()
    return alg.GetOutput()


def clip_all(pd, cl_pd, clip_points, endpoints, labels):
    """local_clip at every clip point, inlet first. clip_points, endpoints: {label: RAS}."""
    cl_pts = vtk_to_numpy(cl_pd.GetPoints().GetData()).astype(float)
    surface = pd
    for lab in labels:
        o = np.asarray(clip_points[lab], float)
        n = _axis_dir(cl_pts, o, np.asarray(endpoints[lab], float))
        others = {k: v for k, v in endpoints.items() if k != lab}
        try:
            surface = local_clip(surface, o, n, endpoints[lab], others)
        except RuntimeError as e:
            raise RuntimeError(f"clip {lab}: {e}")
        print(f"  clipped {lab}")
    return surface


def flow_extensions(pd, cl_pd, fe):
    """vmtk flow extensions on every open boundary (same defaults as ClipVessel).
    fe: {"mode": "mm" | "ratio", "value": length mm | multiple of the boundary radius,
         "direction": "CENTERLINE_DIRECTION" | "BOUNDARY_NORMAL"}"""
    import vtk
    import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
    if fe["mode"] not in ("mm", "ratio"):
        raise RuntimeError(f"flow_ext.mode must be 'mm' or 'ratio', not {fe['mode']}")
    f = vtkvmtkComputationalGeometry.vtkvmtkPolyDataFlowExtensionsFilter()
    f.SetInputData(pd)
    f.SetCenterlines(cl_pd)
    if hasattr(f, "SetSigma"):
        f.SetSigma(1.0)
    f.SetAdaptiveExtensionLength(1 if fe["mode"] == "ratio" else 0)
    f.SetExtensionLength(float(fe["value"]) if fe["mode"] == "mm" else 0.0)
    f.SetExtensionRatio(float(fe["value"]) if fe["mode"] == "ratio" else 2.0)
    f.SetAdaptiveExtensionRadius(1)
    f.SetExtensionRadius(1.0)
    f.SetAdaptiveNumberOfBoundaryPoints(0)
    f.SetNumberOfBoundaryPoints(50)
    f.SetTransitionRatio(0.25)
    f.SetCenterlineNormalEstimationDistanceRatio(1.0)
    if fe["direction"] == "CENTERLINE_DIRECTION":
        f.SetExtensionModeToUseCenterlineDirection()
    else:
        f.SetExtensionModeToUseNormalToBoundary()
    f.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(f.GetOutput())
    if out.GetNumberOfPoints() <= pd.GetNumberOfPoints():
        raise RuntimeError("flow extensions not added")
    return out


def _axis_dir(cl_pts, origin, toward, radius_mm=3.0):
    """Centerline direction at origin (main axis of the centerline points within radius_mm),
    oriented toward the endpoint = direction along which the flow extension grows."""
    near = cl_pts[np.linalg.norm(cl_pts - origin, axis=1) < radius_mm]
    if len(near) < 3:
        d = toward - origin
    else:
        d = np.linalg.svd(near - near.mean(axis=0))[2][0]
        if np.dot(d, toward - origin) < 0:
            d = -d
    return d / np.linalg.norm(d)


def boundary_topology(pd):
    """Boundary of a surface from its point ids (no point merging, unlike vtkFeatureEdges).
    Returns (loops, pinched): loops = [point ids of each connected boundary],
    pinched = ids of boundary points with != 2 boundary edges (loop touching itself or another
    loop: a fan cap on it makes non-manifold edges)."""
    import vtk
    tri = vtk.vtkTriangleFilter()  # keeps point ids
    tri.SetInputData(pd)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    t = vtk_to_numpy(tri.GetOutput().GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    b = uniq[cnt == 1]
    if len(b) == 0:
        return [], []
    valence = np.bincount(b.ravel(), minlength=pd.GetNumberOfPoints())
    pinched = np.where(valence > 2)[0].tolist()
    # connected components of the boundary edges (union-find)
    parent = {int(i): int(i) for i in np.unique(b)}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in b:
        parent[root(int(i))] = root(int(j))
    comps = {}
    for i in parent:
        comps.setdefault(root(i), []).append(i)
    return list(comps.values()), pinched


def open_loops(pd):
    """Boundary loops of a surface: [(centre RAS, diameter mm)], diameter = max point distance.
    Loops from point ids: coincident but unmerged points (a crack) are NOT hidden."""
    loops, _ = boundary_topology(pd)
    pts_all = vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    out = []
    for ids in loops:
        p = pts_all[ids]
        diam = float(np.max(np.linalg.norm(p[:, None] - p[None], axis=2))) if len(p) > 1 else 0.0
        out.append((p.mean(axis=0).tolist(), diam))
    return out


def nonplanar_loops(pd, tol_frac=0.1):
    """Loops farther than tol_frac * diameter from their best-fit plane: after a planar clip
    and a flow extension every loop is flat; a non-flat one hides a slit / fold that the
    capper would fan over. Returns [(centre RAS, diameter mm, max deviation mm)]."""
    loops, _ = boundary_topology(pd)
    pts_all = vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    bad = []
    for ids in loops:
        p = pts_all[ids]
        c = p.mean(axis=0)
        normal = np.linalg.svd(p - c)[2][-1]
        dev = float(np.max(np.abs((p - c) @ normal)))
        diam = float(np.max(np.linalg.norm(p[:, None] - p[None], axis=2)))
        if dev > tol_frac * diam:
            bad.append(([round(float(x), 1) for x in c], round(diam, 1), round(dev, 1)))
    return bad


def pinched_points(pd):
    """RAS of boundary points where a boundary touches itself (see boundary_topology)."""
    _, pinched = boundary_topology(pd)
    return [[round(float(c), 1) for c in pd.GetPoint(i)] for i in pinched]


def clean_surface(pd, fill_holes_max_mm=0.0, merge_tol_mm=1e-4):
    """Merge coincident points (vessel / flow extension seam, within merge_tol_mm), drop
    degenerate cells and unused points and, if fill_holes_max_mm > 0, close holes smaller than
    that diameter (far below any outlet)."""
    import vtk
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(pd)
    clean.PointMergingOn()
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(merge_tol_mm)
    clean.ConvertLinesToPointsOff()
    clean.ConvertPolysToLinesOff()
    clean.ConvertStripsToPolysOff()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(clean.GetOutputPort())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    out = tri.GetOutput()
    if fill_holes_max_mm > 0:
        small = [d for _, d in open_loops(out) if d <= fill_holes_max_mm]
        if small:
            fill = vtk.vtkFillHolesFilter()
            fill.SetInputData(out)
            fill.SetHoleSize(fill_holes_max_mm / 2.0)  # radius of the hole's bounding sphere
            tri2 = vtk.vtkTriangleFilter()
            tri2.SetInputConnection(fill.GetOutputPort())
            tri2.Update()
            out = tri2.GetOutput()
            print(f"  filled {len(small)} small holes (diameter {[round(d, 2) for d in small]} mm)")
    res = vtk.vtkPolyData()
    res.DeepCopy(out)
    return res


def fan_caps(pd):
    """Close every boundary loop with a fan of triangles from its centre (replaces the vmtk
    capper, which made non-manifold edges on some flow-extension ends).
    Built from the point ids: each boundary edge a->b of a wall triangle gets the cap triangle
    (c, b, a) -> consistent orientation, every boundary edge shared by exactly 2 triangles.
    Requires simple loops (no pinched points: checked before). Cell array CapIds: 1 = wall,
    2.. = one id per loop."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    t = vtk_to_numpy(tri.GetOutput().GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(float)

    loops, pinched = boundary_topology(pd)
    if pinched:
        raise RuntimeError(f"{len(pinched)} pinched boundary points: cannot cap")
    directed = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    und = np.sort(directed, axis=1)
    uniq, inv, cnt = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    bdir = directed[cnt[inv.ravel()] == 1]  # boundary edges, wall orientation a->b

    loop_of = {}
    for k, ids in enumerate(loops):
        for i in ids:
            loop_of[i] = k
    new_pts, caps, cap_ids = [], [], []
    for k, ids in enumerate(loops):
        e = bdir[[loop_of[int(a)] == k for a in bdir[:, 0]]]
        mid = (pts[e[:, 0]] + pts[e[:, 1]]) / 2
        w = np.linalg.norm(pts[e[:, 1]] - pts[e[:, 0]], axis=1)
        c_id = len(pts) + k
        new_pts.append((mid * w[:, None]).sum(axis=0) / w.sum())  # centre of the loop polyline
        caps.append(np.column_stack([np.full(len(e), c_id), e[:, 1], e[:, 0]]))
        cap_ids.append(np.full(len(e), 2 + k))

    all_pts = np.vstack([pts, np.array(new_pts)]) if new_pts else pts
    all_tri = np.vstack([t] + caps)
    ids_arr = np.concatenate([np.ones(len(t), int)] + cap_ids)
    out = vtk.vtkPolyData()
    vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(all_pts, deep=True))
    out.SetPoints(vp)
    conn = np.column_stack([np.full(len(all_tri), 3), all_tri]).ravel().astype(np.int64)
    cells = vtk.vtkCellArray()
    cells.SetCells(len(all_tri), numpy_to_vtkIdTypeArray(conn, deep=True))
    out.SetPolys(cells)
    arr = numpy_to_vtk(ids_arr, deep=True, array_type=vtk.VTK_INT)
    arr.SetName("CapIds")
    out.GetCellData().AddArray(arr)
    return out


def cap_and_label(pd, cl_pd, clip_points, endpoints, labels, max_dist_mm):
    """Cap every open boundary (vmtk capper) and write ModelFaceID:
    1 = wall, 2 + i = labels[i]  (labels[0] = inlet -> 2, outlets 3+: SimVascular convention).
    A cap belongs to the label whose axis (clip point + centerline direction) passes closest
    to its centre: the flow extension grows along that axis, whatever its length.
    pd must be cleaned (clean_surface) and have exactly len(labels) open loops.
    clip_points, endpoints: {label: RAS}. Returns (polydata, {face_id: name})."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    loops = open_loops(pd)
    if len(loops) != len(labels):
        raise RuntimeError(f"{len(loops)} open boundaries before capping, expected {len(labels)}; "
                           f"(centre RAS, diameter mm): {[([round(c, 1) for c in x], round(d, 1)) for x, d in loops]}")

    out = fan_caps(pd)
    ids = vtk_to_numpy(out.GetCellData().GetArray("CapIds")).astype(int)
    uniq, counts = np.unique(ids, return_counts=True)
    wall = uniq[np.argmax(counts)]  # the wall has far more cells than any cap
    caps = [int(u) for u in uniq if u != wall]
    if len(caps) != len(labels):
        raise RuntimeError(f"{len(caps)} caps, expected {len(labels)}")

    cc = vtk.vtkCellCenters()
    cc.SetInputData(out)
    cc.Update()
    centers = vtk_to_numpy(cc.GetOutput().GetPoints().GetData())
    cap_center = {u: centers[ids == u].mean(axis=0) for u in caps}

    cl_pts = vtk_to_numpy(cl_pd.GetPoints().GetData()).astype(float)
    axes = {}
    for lab in labels:
        o = np.asarray(clip_points[lab], float)
        axes[lab] = (o, _axis_dir(cl_pts, o, np.asarray(endpoints[lab], float)))

    def ray_dist(c, label):
        o, d = axes[label]
        t = float(np.dot(c - o, d))
        return np.inf if t < 0 else float(np.linalg.norm(c - o - t * d))

    pairs = sorted((ray_dist(cap_center[u], lab), u, lab) for u in caps for lab in labels)
    cap_of, used = {}, set()
    for dist, u, lab in pairs:  # greedy one-to-one, closest first
        if lab not in cap_of and u not in used:
            print(f"  cap {lab}: {dist:.1f} mm from its axis")
            if dist > max_dist_mm:
                raise RuntimeError(f"Cap for {lab} not found ({dist:.1f} mm from its axis, max {max_dist_mm})")
            cap_of[lab] = u
            used.add(u)

    face = np.ones_like(ids)
    for i, lab in enumerate(labels):
        face[ids == cap_of[lab]] = 2 + i
    arr = numpy_to_vtk(face, deep=True, array_type=vtk.VTK_INT)
    arr.SetName(FACE_ARRAY)
    out.GetCellData().RemoveArray("CapIds")
    out.GetCellData().AddArray(arr)
    out.GetCellData().SetActiveScalars(FACE_ARRAY)

    faces = {1: "wall"}
    faces.update({2 + i: lab for i, lab in enumerate(labels)})
    return out, faces


# ---------------------------------------------------------------- checks / output

def nearest_label(pos, points):
    """Label of the point (dict label -> RAS) nearest to pos."""
    return min(points, key=lambda k: np.linalg.norm(np.asarray(points[k], float) - np.asarray(pos, float)))


def bad_edges(pd, faces=None):
    """Boundary and non-manifold edges: {"boundary"/"non_manifold": [(midpoint RAS, face name)]}."""
    import vtk
    out = {}
    cell_loc = vtk.vtkCellLocator()
    cell_loc.SetDataSet(pd)
    cell_loc.BuildLocator()
    face_arr = pd.GetCellData().GetArray(FACE_ARRAY)
    for kind in ("boundary", "non_manifold"):
        fe = vtk.vtkFeatureEdges()
        fe.SetInputData(pd)
        fe.FeatureEdgesOff()
        fe.ManifoldEdgesOff()
        fe.SetBoundaryEdges(kind == "boundary")
        fe.SetNonManifoldEdges(kind == "non_manifold")
        fe.Update()
        e = fe.GetOutput()
        found = []
        for i in range(e.GetNumberOfCells()):
            ids = e.GetCell(i).GetPointIds()
            mid = (np.array(e.GetPoint(ids.GetId(0))) + np.array(e.GetPoint(ids.GetId(1)))) / 2
            closest, cid, sub, d2 = [0.0, 0.0, 0.0], vtk.mutable(0), vtk.mutable(0), vtk.mutable(0.0)
            cell_loc.FindClosestPoint(mid, closest, cid, sub, d2)
            fid = int(face_arr.GetValue(int(cid))) if face_arr is not None else None
            found.append(([round(float(c), 1) for c in mid], (faces or {}).get(fid, fid)))
        out[kind] = found
    return out


def check_closed_and_faces(pd, faces, expected_names):
    """Closed surface, ModelFaceID present and contiguous 1..N, one face per expected name."""
    import vtk
    bad = bad_edges(pd, faces)
    if bad["boundary"] or bad["non_manifold"]:
        raise RuntimeError(f"Final surface not closed/manifold (RAS midpoint, face): "
                           f"boundary {bad['boundary']}, non-manifold {bad['non_manifold']}")

    arr = pd.GetCellData().GetArray(FACE_ARRAY)
    if arr is None:
        raise RuntimeError(f"No {FACE_ARRAY} cell array on the final surface")
    ids = sorted(set(vtk_to_numpy(arr).astype(int).tolist()))
    if ids != list(range(1, len(ids) + 1)) or ids != sorted(faces):
        raise RuntimeError(f"{FACE_ARRAY} values {ids} != faces {sorted(faces)}")
    if sorted(faces.values()) != sorted(["wall"] + list(expected_names)):
        raise RuntimeError(f"Face names {sorted(faces.values())} != expected")

    # tiny triangles at the clip boundaries break MMG downstream: report them
    q = vtk.vtkCellSizeFilter()
    q.SetInputData(pd)
    q.ComputeAreaOn()
    q.Update()
    areas = vtk_to_numpy(q.GetOutput().GetCellData().GetArray("Area"))
    n_sliver = int(np.sum(areas <= 1e-9))
    if n_sliver:
        raise RuntimeError(f"{n_sliver} sliver triangles (area <= 1e-9 mm2) on the final surface")


def save_faces_json(faces, path):
    config.assert_writable(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"array": FACE_ARRAY, "faces": {str(k): v for k, v in sorted(faces.items())}}, f, indent=2)
    print(f"  saved {path}")