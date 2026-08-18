#!/usr/bin/env python3
"""sv_match.py  --  MONDO 1 (venv orchestratore), trasformazione pura.

Scopo-->
Garantire la corrispondenza INLET: quale `face_id` di `cap_faces.json` e' il
cap di ingresso (aorta), coerente con l'endpoint di inlet in `Endpoints_pz*.json`.
Gli altri cap sono, di conseguenza, outlet. 


input :  cap_faces.json      prodotto da extract_faces.py:
                             {units:"cm", wall_face_ids:[...],
                              cap_faces:[{face_id, centroid[cm,SV], radius, area}]}
         Endpoints_pz*.json  markups Slicer (LPS, mm). L'inlet e' l'unico
                             control point con "selected": false (convenzione
                             VMTK: deselezionato = sorgente). Se e' presente
                             anche un label che inizia per "inlet", DEVE
                             concordare col flag, altrimenti si esce.
output:  face_roles.json     {units, sv_frame, detected_frame, inlet{...},
                              outlets[...], wall_face_ids[...], crosscheck{...}}
exit  :  != 0 a OGNI ambiguita' (fail loud): frame markups inatteso, #cap!=#ep,
         zero/ >1 inlet, label vs flag discordi, cap piu' grande non netto,
         frame non separabile, inlet-piu-grande != inlet-piu-vicino.

note
- per ovviare ai problemi di traslazione/scala tra SV e Slicer, si allineano i due modelli 
per centroide prima di misurare le distanze (invariante alla traslazione).
- inlet= cap di area max (robsusto a traslazioni scale etc)

* Assegnamento endpoint->cap con Hungarian (bijettivo, ottimo): due outlet
  vicini di raggio simile (i renali) sotto un nearest greedy potrebbero
  scambiarsi; l'ottimo biunivoco li tiene separati. Serve solo a scegliere il
  frame e a confermare l'inlet; sull'inlet non ci si affida comunque al
  greedy, ma al cap di area massima.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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

# RAS<->LPS: involuzione, nega x,y e tieni z.
_XY_FLIP = np.array([-1.0, -1.0, 1.0])


class MatchError(Exception):
    """Ambiguita' o incoerenza: fa uscire la pipeline con codice != 0."""


def load_caps(path):
    """Ritorna (face_ids[N], centroids[N,3] cm, areas[N], radii[N], wall_ids)."""
    d = json.loads(Path(path).read_text())
    caps = d.get("cap_faces") or []
    if len(caps) < 2:
        raise MatchError(f"{path}: attesi >=2 cap, trovati {len(caps)}.")
    fid = [c["face_id"] for c in caps]
    cen = np.array([c["centroid"] for c in caps], dtype=float)
    area = np.array([c["area"] for c in caps], dtype=float)
    rad = np.array([c["radius"] for c in caps], dtype=float)
    wall = list(d.get("wall_face_ids") or [])
    return fid, cen, area, rad, wall


def load_endpoints(path, inlet_prefix: str = "inlet"):
    """Ritorna (positions[M,3] mm, ids[M], inlet_idx) dalla markups list.

    L'inlet e' l'unico control point con selected == False. Se un label inizia
    per `inlet_prefix`, deve indicare lo stesso punto del flag (fail loud).
    """
    d = json.loads(Path(path).read_text())
    mk = (d.get("markups") or [None])[0]
    if mk is None:
        raise MatchError(f"{path}: nessun blocco 'markups'.")
    if mk.get("coordinateSystem") != "LPS":
        raise MatchError(f"{path}: frame {mk.get('coordinateSystem')!r}, atteso LPS.")
    cps = mk.get("controlPoints") or []
    if len(cps) < 2:
        raise MatchError(f"{path}: attesi >=2 control point, trovati {len(cps)}.")

    ids = [c.get("label") or c.get("id") or f"cp{i}" for i, c in enumerate(cps)]
    pos = np.array([c["position"] for c in cps], dtype=float)

    src = [i for i, c in enumerate(cps) if c.get("selected") is False]
    if len(src) != 1:
        raise MatchError(
            f"{path}: atteso 1 inlet (selected=false), trovati {len(src)} "
            f"(indici {src}). Convenzione VMTK: deselezionato = sorgente."
        )
    inlet_idx = src[0]

    # controprova sul label, se c'e'
    labelled = [i for i, c in enumerate(cps)
                if (c.get("label") or "").lower().startswith(inlet_prefix.lower())]
    if labelled and labelled != [inlet_idx]:
        raise MatchError(
            f"{path}: label '{inlet_prefix}*' su {labelled} ma flag selected=false "
            f"su {inlet_idx}: label e flag discordano."
        )
    return pos, ids, inlet_idx


def _to_sv(pos_mm, flip: bool, scale: float) -> np.ndarray:
    p = np.asarray(pos_mm, dtype=float)
    if flip:
        p = p * _XY_FLIP
    return p * scale


def _assign_cost(ep_cm, cap_cm):
    """Allinea per centroide (invariante alla traslazione) e ritorna
    (assegnamento ep->cap, matrice distanze, costo totale)."""
    ep = ep_cm - ep_cm.mean(axis=0)
    cap = cap_cm - cap_cm.mean(axis=0)
    dist = np.linalg.norm(ep[:, None, :] - cap[None, :, :], axis=2)
    assign = _hungarian(dist)               # riga(ep) -> colonna(cap)
    total = float(dist[np.arange(len(assign)), assign].sum())
    return assign, dist, total


def match(cap_path, ep_path, *, scale: float = 0.1,
          area_margin_ratio: float = 1.3, frame_margin_ratio: float = 1.3,
          inlet_prefix: str = "inlet") -> dict:
    """Cuore puro. Ritorna il dict face_roles (non lo scrive)."""
    fid, cap_cm, area, rad, wall = load_caps(cap_path)
    ep_mm, ep_ids, inlet_idx = load_endpoints(ep_path, inlet_prefix)

    n_cap, n_ep = len(fid), len(ep_ids)
    if n_cap != n_ep:
        raise MatchError(
            f"#cap ({n_cap}) != #endpoint ({n_ep}): separation_angle o set "
            f"endpoint incoerenti."
        )

    # 1) DECISIONE PRIMARIA: inlet cap = area massima, con margine netto.
    order = np.argsort(area)[::-1]
    big, second = int(order[0]), int(order[1])
    if area[big] < area_margin_ratio * area[second]:
        raise MatchError(
            f"cap piu' grande non netto: area {area[big]:.4f} vs {area[second]:.4f} "
            f"cm2 (ratio {area[big]/area[second]:.2f} < {area_margin_ratio}). "
            f"L'inlet aortico dovrebbe dominare: verificare la classificazione."
        )
    inlet_cap = big

    # 2) FRAME: prova {LPS, LPS+flipXY}, tieni il residuo minore e pretendi netto.
    trials = {}
    for name, flip in (("LPS", False), ("LPS_XYFLIP", True)):
        ep_cm = _to_sv(ep_mm, flip, scale)
        assign, dist, total = _assign_cost(ep_cm, cap_cm)
        trials[name] = (total, assign, dist)
    win = min(trials, key=lambda k: trials[k][0])
    lose = "LPS_XYFLIP" if win == "LPS" else "LPS"
    tw, assign, dist = trials[win]
    tl = trials[lose][0]
    if tw > 0 and tl < frame_margin_ratio * tw:
        raise MatchError(
            f"frame non separabile: residuo {win}={tw:.3f} vs {lose}={tl:.3f} cm "
            f"(serve ratio >= {frame_margin_ratio}). Geometria/frame incerti."
        )

    # 3) CROSS-CHECK: sotto il frame vincente, il cap assegnato all'inlet
    #    endpoint DEVE essere il cap di area massima.
    inlet_cap_by_match = int(assign[inlet_idx])
    inlet_dist = float(dist[inlet_idx, inlet_cap_by_match])
    if inlet_cap_by_match != inlet_cap:
        raise MatchError(
            f"incoerenza inlet: area-massima -> face {fid[inlet_cap]}, ma "
            f"endpoint inlet '{ep_ids[inlet_idx]}' cade su face "
            f"{fid[inlet_cap_by_match]} (frame {win}). Non tag silenzioso: gate umano."
        )

    # 4) risultato. Outlet = tutti gli altri cap; corrispondenza endpoint fornita
    #    solo come riferimento (la naming alfabetica la fa sv_apply).
    inlet = {
        "face_id": fid[inlet_cap],
        "endpoint_id": ep_ids[inlet_idx],
        "area_cm2": round(float(area[inlet_cap]), 6),
        "radius_cm": round(float(rad[inlet_cap]), 6),
        "match_dist_cm": round(inlet_dist, 4),
    }
    outlets = []
    for ep_i, cap_j in enumerate(assign):
        if ep_i == inlet_idx:
            continue
        outlets.append({
            "face_id": fid[int(cap_j)],
            "endpoint_ref": ep_ids[ep_i],
            "radius_cm": round(float(rad[int(cap_j)]), 6),
            "match_dist_cm": round(float(dist[ep_i, int(cap_j)]), 4),
        })
    outlets.sort(key=lambda o: (o["face_id"] if isinstance(o["face_id"], (int, float))
                                else str(o["face_id"])))

    return {
        "units": "cm",
        "sv_frame": "LPS",
        "scale_mm_to_cm": scale,
        "detected_frame": win,
        "inlet": inlet,
        "outlets": outlets,
        "wall_face_ids": wall,
        "crosscheck": {
            "n_caps": n_cap,
            "n_endpoints": n_ep,
            "largest_over_second_area_ratio": round(float(area[big] / area[second]), 3),
            "inlet_is_largest_and_nearest": True,
            "frame_residual_cm": round(tw, 4),
            "frame_margin_ratio": round(float(tl / tw), 3) if tw > 0 else None,
            "inlet_match_dist_cm": round(inlet_dist, 4),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Match inlet cap <-> inlet endpoint (World 1, puro)."
    )
    ap.add_argument("cap_faces", help="cap_faces.json (da extract_faces.py)")
    ap.add_argument("endpoints", help="Endpoints_pz*.json (markups Slicer, LPS mm)")
    ap.add_argument("output", help="face_roles.json in uscita")
    ap.add_argument("--scale", type=float, default=0.1, help="mm->cm (default 0.1)")
    ap.add_argument("--area-margin", type=float, default=1.3,
                    help="l'inlet cap deve avere area >= questo x il secondo")
    ap.add_argument("--frame-margin", type=float, default=1.3,
                    help="residuo del frame perdente >= questo x il vincente")
    ap.add_argument("--inlet-prefix", default="inlet")
    a = ap.parse_args(argv)

    try:
        roles = match(a.cap_faces, a.endpoints, scale=a.scale,
                      area_margin_ratio=a.area_margin,
                      frame_margin_ratio=a.frame_margin,
                      inlet_prefix=a.inlet_prefix)
    except (MatchError, KeyError, ValueError, OSError) as e:
        print(f"ERRORE: {e}", file=sys.stderr)
        return 1

    Path(a.output).write_text(json.dumps(roles, indent=2))
    i = roles["inlet"]
    print(f"OK: inlet = face {i['face_id']} (ep '{i['endpoint_id']}', "
          f"frame {roles['detected_frame']}, d={i['match_dist_cm']} cm) + "
          f"{len(roles['outlets'])} outlet -> {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())