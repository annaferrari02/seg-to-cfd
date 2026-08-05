#!/usr/bin/env python3
# steps/endpoints/tag_endpoints.py
#
# Deriva il RUOLO (inlet/outlet) di ogni endpoint incrociando la markups list
# di Slicer con l'albero di centerline, ed emette endpoints_tagged.json.
#
# PERCHE' SERVE
#   Gli endpoint di Slicer non portano ruolo: sono Endpoints-1..N, tutti uguali.
#   Il ruolo e' pero' deducibile dalla TOPOLOGIA delle centerline: ogni linea va
#   dalla sorgente (= inlet) a una foglia (= outlet). Il nodo condiviso da tutte
#   le linee e' l'inlet; le foglie sono gli outlet. Il raggio e' solo controprova.
#
# CONTRATTO (come gli step ParaView)
#   input : <centerlines.vtk>   POLYDATA con LINES + array 'Radius', frame LPS mm
#           <endpoints.mrk.json> markups fiducial di Slicer, LPS mm
#   output: <endpoints_tagged.json>
#             {frame, units, endpoints:[{id, position, role, radius_mm}]}
#   exit  : != 0 a OGNI ambiguita' (fail loud): frame non LPS, #endpoint != #nodi,
#           zero o piu' sorgenti, match non biunivoco, vincitore non netto.
#
# Uso:
#   python3 tag_endpoints.py <centerlines.vtk> <endpoints.mrk.json> <out.json>

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import vtk


NODE_TOL_MM = 1.0      # due estremi piu' vicini di cosi' = stesso nodo dell'albero
CLEAR_WIN = 0.5        # il match e' netto se dist_prima < CLEAR_WIN * dist_seconda


def _die(msg: str) -> "int":
    print(f"ERRORE: {msg}", file=sys.stderr)
    return 1


def _load_centerline_nodes(path: str):
    """Ritorna (posizioni_nodi[N,3], gradi[N], raggi[N], n_linee).

    Un 'nodo' e' un gruppo di estremi di linea coincidenti entro NODE_TOL_MM;
    il suo grado e' quante linee lo toccano.
    """
    r = vtk.vtkPolyDataReader()
    r.SetFileName(path)
    r.Update()
    pd = r.GetOutput()
    if pd is None or pd.GetNumberOfLines() == 0:
        raise ValueError(f"{path}: nessuna linea di centerline leggibile.")

    rad_arr = pd.GetPointData().GetArray("Radius")
    if rad_arr is None:
        raise ValueError(f"{path}: manca l'array 'Radius'.")
    pts = np.array([pd.GetPoint(i) for i in range(pd.GetNumberOfPoints())])
    rad = np.array(rad_arr)

    lines = pd.GetLines()
    lines.InitTraversal()
    idl = vtk.vtkIdList()
    end_ids = []
    n_lines = 0
    while lines.GetNextCell(idl):
        n_lines += 1
        end_ids += [idl.GetId(0), idl.GetId(idl.GetNumberOfIds() - 1)]
    end_pts = pts[end_ids]
    end_rad = rad[end_ids]

    # clustering greedy degli estremi coincidenti -> nodi + grado
    used = np.zeros(len(end_pts), bool)
    node_pos, node_deg, node_rad = [], [], []
    for i in range(len(end_pts)):
        if used[i]:
            continue
        m = np.linalg.norm(end_pts - end_pts[i], axis=1) < NODE_TOL_MM
        used |= m
        node_pos.append(end_pts[m].mean(axis=0))
        node_deg.append(int(m.sum()))
        node_rad.append(float(end_rad[m].mean()))
    return np.array(node_pos), np.array(node_deg), np.array(node_rad), n_lines


def _load_endpoints(path: str):
    """Ritorna (ids, posizioni[M,3]) dalla markups list, verificando il frame."""
    doc = json.load(open(path))
    mk = doc["markups"][0]
    if mk.get("coordinateSystem") != "LPS":
        raise ValueError(
            f"{path}: frame {mk.get('coordinateSystem')!r}, atteso LPS "
            "(un flip RAS/LPS sballa il match in silenzio)."
        )
    cps = mk["controlPoints"]
    ids = [c["id"] for c in cps]
    pos = np.array([c["position"] for c in cps])
    return ids, pos


def main() -> int:
    p = argparse.ArgumentParser(description="Tag inlet/outlet degli endpoint da centerline.")
    p.add_argument("centerlines", help="centerlines .vtk (LINES + Radius, LPS mm)")
    p.add_argument("endpoints", help="endpoints .mrk.json di Slicer (LPS mm)")
    p.add_argument("output", help="endpoints_tagged.json in uscita")
    args = p.parse_args()

    try:
        node_pos, node_deg, node_rad, n_lines = _load_centerline_nodes(args.centerlines)
        ep_ids, ep_pos = _load_endpoints(args.endpoints)
    except (KeyError, ValueError, OSError) as e:
        return _die(str(e))

    # --- l'albero e' sano? esattamente una sorgente, foglie = n_linee ---
    n_nodes = len(node_pos)
    if n_nodes != len(ep_ids):
        return _die(f"#nodi terminali ({n_nodes}) != #endpoint ({len(ep_ids)}).")

    src = np.where(node_deg == n_lines)[0]
    if len(src) != 1:
        return _die(f"attesa 1 sorgente (grado {n_lines}), trovate {len(src)}.")
    src = int(src[0])

    leaves = np.where(node_deg == 1)[0]
    if len(leaves) != n_lines:
        return _die(f"attese {n_lines} foglie (grado 1), trovate {len(leaves)}.")

    # controprova: la sorgente e' anche il raggio massimo?
    if node_rad[src] < node_rad.max() - 1e-6:
        print(f"ATTENZIONE: la sorgente (r={node_rad[src]:.2f}) non e' il raggio "
              f"massimo (r={node_rad.max():.2f}). Verificare.", file=sys.stderr)

    # --- match markup -> nodo terminale (biunivoco e netto) ---
    claimed = set()
    tagged = []
    for eid, e in zip(ep_ids, ep_pos):
        d = np.linalg.norm(node_pos - e, axis=1)
        order = np.argsort(d)
        j = int(order[0])
        if j in claimed:
            return _die(f"endpoint {eid}: nodo {j} gia' assegnato (match non biunivoco).")
        if len(order) > 1 and d[j] > CLEAR_WIN * d[order[1]]:
            return _die(f"endpoint {eid}: match non netto "
                        f"(prima {d[j]:.2f} vs seconda {d[order[1]]:.2f} mm).")
        claimed.add(j)
        role = "inlet" if j == src else "outlet"
        tagged.append({
            "id": eid,
            "position": [round(float(x), 4) for x in e],
            "role": role,
            "radius_mm": round(float(node_rad[j]), 3),
        })

    out = {"frame": "LPS", "units": "mm", "endpoints": tagged}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    n_out = sum(t["role"] == "outlet" for t in tagged)
    print(f"OK: 1 inlet + {n_out} outlet -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())