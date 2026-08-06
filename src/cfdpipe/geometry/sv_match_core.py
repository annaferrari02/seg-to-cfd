#!/usr/bin/env python3
"""sv_match_core.py  --  World 1 (orchestrator venv), pure transform.

Match SimVascular cap faces to tagged endpoints.

Contract
--------
input :  tagged      = endpoints_tagged.json dict, produced by tag_endpoints.py
                       {"frame","units","endpoints":[{id,position,role,radius_mm}]}
                       positions in SOURCE frame + mm.
         cap_faces   = [{"face_id": <int|str>,
                         "centroid": [x,y,z],   # cm, in the SV frame
                         "radius":   <float>}]  # cm, effective cap radius
                       (the SV glue isolates cap faces from wall faces and
                        computes these before calling this core.)
output:  list of assignments, one per cap face:
             {"face_id", "endpoint_id", "role" ("inlet"|"outlet"),
              "name" ("inlet" or the endpoint id), "match_dist_cm",
              "margin_cm", "ambiguous"}
         plus a summary dict with the worst distance / whether any pair is
         ambiguous, so the adapter's validate() can gate on it.

Why this shape
--------------
* The single Slicer->SV transform lives here, in one place: scale (mm->cm) and,
  only if the source and SV frames differ, the RAS<->LPS flip. Nothing upstream
  converts coordinates.
* Assignment is OPTIMAL and one-to-one (Hungarian), not greedy nearest: two close
  outlets of similar radius (the renals) can swap under greedy; a bijective
  optimum keeps them apart.
* The flow-extension length is NOT an input. It offsets every true pair by a
  near-constant amount and so does not change which assignment is optimal. If you
  want a geometric sanity check against the extension length from params.yaml,
  do it at the adapter boundary using `match_dist_cm` from the summary.
* Ambiguity is surfaced, never hidden: if a cap is nearly equidistant between two
  endpoints, it is flagged for a human gate instead of being tagged silently.
"""
from __future__ import annotations

import numpy as np

# RAS<->LPS is an involution: negate x and y, keep z. Same flip both directions.
_RAS_LPS_FLIP = np.array([-1.0, -1.0, 1.0])


def _to_sv_frame(pos_src, src_frame: str, sv_frame: str, scale: float) -> np.ndarray:
    """Transform a source-frame position (mm) into the SV frame (cm)."""
    p = np.asarray(pos_src, dtype=float)
    if src_frame != sv_frame:
        if {src_frame, sv_frame} == {"RAS", "LPS"}:
            p = p * _RAS_LPS_FLIP
        else:
            raise ValueError(
                f"Unsupported frame conversion '{src_frame}' -> '{sv_frame}'. "
                f"Only RAS<->LPS is handled."
            )
    return p * scale


def _hungarian(cost: np.ndarray) -> np.ndarray:
    """Optimal assignment for a square cost matrix. Returns col index per row.

    Compact O(n^3) Kuhn-Munkres via the shortest-augmenting-path formulation.
    """
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)   # p[j] = row assigned to column j (1-indexed)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assign = np.zeros(n, dtype=int)   # row -> col (0-indexed)
    for j in range(1, n + 1):
        assign[p[j] - 1] = j - 1
    return assign


def match_caps(tagged: dict, cap_faces: list, *, scale: float = 0.1,
               sv_frame: str = "LPS", radius_weight: float = 1.0,
               ambiguity_margin_cm: float = 0.2):
    """Assign each cap face to an endpoint. See module docstring for the contract."""
    endpoints = tagged["endpoints"]
    src_frame = tagged.get("frame", "UNKNOWN")

    n_caps = len(cap_faces)
    n_eps = len(endpoints)
    if n_caps != n_eps:
        raise ValueError(
            f"Cap/endpoint count mismatch: {n_caps} cap faces vs {n_eps} endpoints. "
            f"SV face extraction (angle) or the endpoint set is wrong."
        )

    # endpoints -> SV frame (cm)
    ep_pos = np.array([_to_sv_frame(e["position"], src_frame, sv_frame, scale)
                       for e in endpoints])
    ep_rad = np.array([e["radius_mm"] * scale for e in endpoints])  # mm -> cm

    cap_pos = np.array([f["centroid"] for f in cap_faces], dtype=float)
    cap_rad = np.array([f["radius"] for f in cap_faces], dtype=float)

    # cost[i,j] = ||cap_i - ep_j|| + w * |cap_r_i - ep_r_j|   (all in cm)
    dpos = np.linalg.norm(cap_pos[:, None, :] - ep_pos[None, :, :], axis=2)
    drad = np.abs(cap_rad[:, None] - ep_rad[None, :])
    cost = dpos + radius_weight * drad

    assign = _hungarian(cost)

    results = []
    worst_dist = 0.0
    any_ambiguous = False
    for i, j in enumerate(assign):
        e = endpoints[j]
        # geometric match distance (position only) for reporting / gating
        mdist = float(dpos[i, j])
        worst_dist = max(worst_dist, mdist)
        # ambiguity: how much better is the assigned endpoint than the next best?
        other = np.delete(cost[i], j)
        margin = float(other.min() - cost[i, j]) if other.size else float("inf")
        ambiguous = margin < ambiguity_margin_cm
        any_ambiguous = any_ambiguous or ambiguous
        results.append({
            "face_id": cap_faces[i]["face_id"],
            "endpoint_id": e["id"],
            "role": e["role"],
            "name": "inlet" if e["role"] == "inlet" else e["id"],
            "match_dist_cm": round(mdist, 4),
            "margin_cm": round(margin, 4),
            "ambiguous": ambiguous,
        })

    summary = {
        "n_caps": n_caps,
        "worst_match_dist_cm": round(worst_dist, 4),
        "any_ambiguous": any_ambiguous,
        "sv_frame": sv_frame,
        "src_frame": src_frame,
    }
    return results, summary
