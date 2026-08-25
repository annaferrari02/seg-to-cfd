#!/usr/bin/env python3
"""
build_sjb.py — assemble a SimVascular "Simulations" job file (.sjb).


Inputs (contracts):
  --mdl     <model>.mdl                 : ModelFaceID <-> face name/type bridge
  --rcr     rcrparams.txt               : setup_bcs output  (face_id area R1 C R2)
  --inflow  inflow.reference.scaled     : inlet waveform     (time  flow), 2 cols

Output:
  <out>.sjb  loadable by the SV Simulations GUI, with:
    inlet  (--inlet-id) -> BC Type "Prescribed Velocities" + embedded waveform
    outlets            -> BC Type "RCR", Values "R1 C R2" matched *by ModelFaceID*
    solver             -> "Number of Timesteps", "Increment in saving VTK files"
"""
import sys
import argparse
import xml.etree.ElementTree as ET


# ---- solver defaults, lifted verbatim from SV solvertemplate.xml -------------
SOLVER_TIME = {
    "Number of Timesteps": None,                       # <- overridden
    "Time Step Size": None,                             # <- overridden
    "Spectral radius of infinite time step": "0.5",
}
SOLVER_OUTPUT = {
    "Save results in folder": "N-procs",
    "Increment in saving restart files": "10",
    "Start saving after time step": "1",
    "Save results to VTK format": "true",
    "Name prefix of saved VTK files": "result",
    "Increment in saving VTK files": None,              # <- overridden
}
NONLINEAR = {
    "Min iterations": "3", "Max iterations": "12",
    "Tolerance": "1e-5", "Backflow stabilization coefficient": "0.2",
}
LINEAR = {
    "Solver": "NS", "Max iterations": "15",
    "NS GM max iterations": "10", "NS GM tolerance": "1e-3",
    "NS CG max iterations": "300", "NS CG tolerance": "1e-3",
    "Krylov space dimension": "200", "Tolerance": "1e-4",
    "Absolute tolerance": "1e-10",
}


def parse_mdl(path):
    """Return {face_id:int -> {'name':str, 'type':str}} from a .mdl model file."""
    root = ET.parse(path).getroot()
    faces = {}
    for fe in root.iter("face"):
        fid = fe.get("id")
        name = fe.get("name")
        ftype = fe.get("type")
        if fid is None or name is None:
            continue
        faces[int(fid)] = {"name": name, "type": (ftype or "").strip()}
    if not faces:
        raise RuntimeError(f"No <face> entries found in {path}")
    return faces


def parse_rcr(path):
    """Return {face_id:int -> (R1,C,R2) as strings} from setup_bcs output."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            fid = int(parts[0])
            r1, c, r2 = parts[2], parts[3], parts[4]
            out[fid] = (r1, c, r2)
    if not out:
        raise RuntimeError(f"No data rows parsed from {path}")
    return out


def read_inflow(path):
    """Validate 2-column numeric waveform; return (raw_text, period)."""
    with open(path) as f:
        raw = f.read()
    tmin = tmax = None
    for ln in raw.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        c = s.split()
        try:
            t = float(c[0]); float(c[1])
        except (ValueError, IndexError):
            raise RuntimeError(
                f"Inflow file '{path}' has a non 2-column-numeric line: {ln!r}. "
                f"The GUI parses this verbatim; strip any header (e.g. '400 100')."
            )
        tmin = t if tmin is None else min(tmin, t)
        tmax = t if tmax is None else max(tmax, t)
    if tmin is None or tmax is None or tmax - tmin <= 0:
        raise RuntimeError(f"Cannot derive a positive period from '{path}'.")
    return raw, tmax - tmin


def _props(parent, tag, kv):
    sec = ET.SubElement(parent, tag)
    for k, v in kv.items():
        ET.SubElement(sec, "prop", {"key": k, "value": str(v)})
    return sec


def build(args):
    faces = parse_mdl(args.mdl)
    rcr = parse_rcr(args.rcr)
    flow_text, period = read_inflow(args.inflow)
    if args.period is not None:
        period = args.period
    dt = args.time_step_size if args.time_step_size is not None \
        else period / args.timesteps  # default: 960 steps == one period

    wall_ids = set(args.wall_ids)

    if args.inlet_id not in faces:
        raise RuntimeError(f"inlet-id {args.inlet_id} not in .mdl faces {sorted(faces)}")

    # ---- classify caps ------------------------------------------------------
    cap_entries = {}   # name -> {key:val}
    outlets = []
    for fid, meta in faces.items():
        if meta["type"] != "cap":
            continue                      # walls live in wall_props, not cap_props
        if fid in wall_ids:
            continue
        name = meta["name"]
        if fid == args.inlet_id:
            cap_entries[name] = {
                "BC Type": "Prescribed Velocities",
                "Analytic Shape": args.shape,
                "Point Number": str(args.point_number),
                "Fourier Modes": str(args.fourier_modes),
                "Period": f"{period:.10g}",
                "Flip Normal": "True" if args.flip_normal else "False",
                "Original File": args.inflow,
                "Flow Rate": flow_text,
            }
        else:
            if fid not in rcr:
                raise RuntimeError(
                    f"outlet face_id {fid} ('{name}') has no RCR row in {args.rcr}")
            r1, c, r2 = rcr[fid]
            if float(r1) <= 0:
                raise RuntimeError(f"non-positive R1 for face {fid} ('{name}'): {r1}")
            cap_entries[name] = {
                "BC Type": "RCR",
                "Values": f"{r1} {c} {r2}",
                "R Values": f"{r1} {r2}",
                "C Values": f"{c}",
            }
            outlets.append((fid, name))

    if not outlets:
        raise RuntimeError("No outlet caps found (every cap was inlet or wall?).")

    # ---- assemble XML (child order matches SV writer) -----------------------
    mj = ET.Element("mitk_job", {
        "model_name": args.model_name,
        "mesh_name": args.mesh_name,
        "status": "No Data Files",
        "version": "1.0",
    })
    job = ET.SubElement(mj, "job")

    _props(job, "basic_props", {
        "Fluid Density": str(args.density),
        "Fluid Viscosity": str(args.viscosity),
        "Initial Pressure": "0",
        "Initial Velocities": "0.0001 0.0001 0.0001",
    })
    _props(job, "zerod_interface_props", {})

    cp = ET.SubElement(job, "cap_props")
    for name, kv in cap_entries.items():
        ce = ET.SubElement(cp, "cap", {"name": name})
        for k, v in kv.items():
            ET.SubElement(ce, "prop", {"key": k, "value": str(v)})

    _props(job, "wall_props", {"Type": "rigid"})
    _props(job, "cmm_props", {})

    out_props = dict(SOLVER_OUTPUT)
    out_props["Increment in saving VTK files"] = str(args.vtk_increment)
    _props(job, "solver_output_props", out_props)

    time_props = dict(SOLVER_TIME)
    time_props["Number of Timesteps"] = str(args.timesteps)
    time_props["Time Step Size"] = f"{dt:.10g}"
    _props(job, "solver_time_props", time_props)

    _props(job, "nonlinear_solver_props", NONLINEAR)
    _props(job, "linear_solver_props", LINEAR)
    _props(job, "run_props", {"Number of Processes": "1"})

    ET.indent(mj, space="    ")
    ET.ElementTree(mj).write(args.out, encoding="UTF-8", xml_declaration=True)

    # ---- fail-loud report ---------------------------------------------------
    print(f"Wrote {args.out}")
    print(f"  model={args.model_name}  mesh={args.mesh_name}")
    print(f"  inlet  id={args.inlet_id}  name='{faces[args.inlet_id]['name']}'  "
          f"file='{args.inflow}'  period={period:.6g}s")
    print(f"  outlets ({len(outlets)}): "
          + ", ".join(f"{fid}:{nm}" for fid, nm in sorted(outlets)))
    print(f"  timesteps={args.timesteps}  dt={dt:.6g}s  "
          f"(total {args.timesteps*dt:.4g}s = {args.timesteps*dt/period:.3g} cycles)  "
          f"vtk_increment={args.vtk_increment}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Build a SimVascular .sjb job file.")
    p.add_argument("--mdl", required=True)
    p.add_argument("--rcr", required=True)
    p.add_argument("--inflow", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--mesh-name", required=True)
    p.add_argument("--inlet-id", type=int, required=True)
    p.add_argument("--wall-ids", type=int, nargs="*", default=[])
    p.add_argument("--timesteps", type=int, default=960)
    p.add_argument("--vtk-increment", type=int, default=2)
    p.add_argument("--time-step-size", type=float, default=None,
                   help="dt [s]; default = period / timesteps (one cycle).")
    p.add_argument("--period", type=float, default=None,
                   help="override; default read from the inflow file.")
    p.add_argument("--density", type=float, default=1.06)
    p.add_argument("--viscosity", type=float, default=0.04)
    p.add_argument("--shape", default="parabolic",
                   choices=["parabolic", "plug", "womersley"])
    p.add_argument("--point-number", type=int, default=201)
    p.add_argument("--fourier-modes", type=int, default=10)
    p.add_argument("--flip-normal", action="store_true")
    build(p.parse_args(argv))


if __name__ == "__main__":
    main()