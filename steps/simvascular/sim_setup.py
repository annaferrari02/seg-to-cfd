#!/usr/bin/env python3
"""
sim_setup.py 
scrive il job .sjb con BC e solver params.

    <sv> --python -- sim_setup.py --input <cfd{pid}/Models/{pid}.vtp> \
         --out-dir <pz_root> [--map .. --tot-comp .. --timesteps .. ...]

MAP / tot_comp: default di coorte da params.yaml (via extra_args). Se esiste
<pz_root>/bcs.json con {"map":..,"tot_comp":..}, ha la precedenza (per-paziente).
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import sorelle
import setup_bcs      # noqa: E402  (motore BC, vtk+numpy)
import build_sjb      # noqa: E402  (assembla .sjb, stdlib)


def parse_args():
    p = argparse.ArgumentParser(description="Predispone il job CFD (.sjb) di un pz.")
    p.add_argument("--input", required=True, help="modello .vtp con ModelFaceID")
    p.add_argument("--out-dir", required=True, help="cartella del paziente")
    # BC di coorte (override per-paziente via <out-dir>/bcs.json)
    p.add_argument("--map", type=float, required=True)
    p.add_argument("--tot-comp", type=float, required=True)
    # inflow di riferimento (asset di coorte, gia' su 0.8 s e media -Qref)
    p.add_argument("--reference-inflow", required=True,
                   help="path a inflow.reference (da make_inflow_reference)")
    # solver / scaling
    p.add_argument("--timesteps", type=int, default=960)
    p.add_argument("--vtk-increment", type=int, default=2)
    p.add_argument("--period", type=float, default=0.8)
    p.add_argument("--time-step-size", type=float, default=None)
    p.add_argument("--area-to-cm2", type=float, default=1.0)
    p.add_argument("--res1-split", type=float, default=0.15)
    argv = sys.argv[1:]
    if "--" in argv:                       # come sv_apply: scarta i flag di lancio SV
        argv = argv[argv.index("--") + 1:]
    return p.parse_args(argv)


def _assert_convention(mdl_path, inlet_id=2, wall_id=1):
    """Verifica che il .mdl rispetti la convenzione di sv_apply. Fail loud."""
    faces = build_sjb.parse_mdl(mdl_path)
    if inlet_id not in faces or faces[inlet_id]["type"] != "cap":
        raise RuntimeError(
            f"{mdl_path}: id {inlet_id} non e' un cap (atteso inlet). "
            f"Facce: { {i: f['type'] for i, f in faces.items()} }")
    if wall_id not in faces or faces[wall_id]["type"] != "wall":
        raise RuntimeError(
            f"{mdl_path}: id {wall_id} non e' 'wall'. Convenzione sv_apply violata.")
    if faces[inlet_id]["name"] != "inlet":
        print(f"[WARN] id {inlet_id} si chiama '{faces[inlet_id]['name']}', "
              f"non 'inlet' (procedo, ma verifica sv_apply).", file=sys.stderr)
    return faces


def main():
    a = parse_args()
    out_dir = os.path.abspath(a.out_dir)
    pid = os.path.basename(out_dir.rstrip("/"))
    project = os.path.join(out_dir, f"cfd{pid}")
    mdl = os.path.join(project, "Models", f"{pid}.mdl")
    vtp = os.path.abspath(a.input)

    for label, path in (("modello .vtp", vtp), (".mdl", mdl),
                        ("inflow.reference", a.reference_inflow)):
        if not os.path.isfile(path):
            raise RuntimeError(f"{pid}: manca {label}: {path}")

    job_dir = os.path.join(project, "Simulations", pid)
    os.makedirs(job_dir, exist_ok=True)
    rcr_path = os.path.join(job_dir, "rcrparams.txt")
    scaled_path = os.path.join(job_dir, "inflow.reference.scaled")
    sjb_path = os.path.join(project, "Simulations", f"{pid}.sjb")

    # convenzione ID (fail loud) -> inlet/wall fissi, niente input a mano
    _assert_convention(mdl, inlet_id=2, wall_id=1)

    # override BC per-paziente se presente
    map_val, tot_comp = a.map, a.tot_comp
    bcs_json = os.path.join(out_dir, "bcs.json")
    if os.path.isfile(bcs_json):
        b = json.loads(open(bcs_json).read())
        map_val = float(b.get("map", map_val))
        tot_comp = float(b.get("tot_comp", tot_comp))
        print(f"[{pid}] BC per-paziente da bcs.json: map={map_val} tot_comp={tot_comp}")

    # (1) BC + inflow scalato -> rcrparams.txt + inflow.reference.scaled nel job dir
    setup_bcs.main(
        vtp_path=vtp,
        out_path=rcr_path,
        face_id_array_name="ModelFaceID",
        inlet_id=2,
        wall_ids={1},
        map_val=map_val,
        tot_comp=tot_comp,
        area_to_cm2=a.area_to_cm2,
        res1_split=a.res1_split,
        inflow_file=os.path.abspath(a.reference_inflow),
    )
    if not os.path.isfile(scaled_path):
        raise RuntimeError(f"{pid}: setup_bcs non ha prodotto {scaled_path}")

    # (2) assembla il .sjb
    build_sjb.main([
        "--mdl", mdl,
        "--rcr", rcr_path,
        "--inflow", scaled_path,
        "--out", sjb_path,
        "--model-name", pid,
        "--mesh-name", pid,
        "--inlet-id", "2",
        "--wall-ids", "1",
        "--timesteps", str(a.timesteps),
        "--vtk-increment", str(a.vtk_increment),
        "--period", str(a.period),
        *(["--time-step-size", str(a.time_step_size)]
          if a.time_step_size is not None else []),
    ])

    if not os.path.isfile(sjb_path):
        raise RuntimeError(f"{pid}: .sjb non prodotto: {sjb_path}")
    print(f"[{pid}] OK -> {sjb_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())