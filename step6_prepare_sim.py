r"""Step 6 - svMultiPhysics simulation, ready to launch (no SimVascular GUI)

<simvascular> --python -- step6_prepare_sim.py <case_id>      (any python with vtk + numpy works)

Input:  <case>/Models/<case>.mdl                    face names and types (step4)
        <case>/Meshes/<case>.vtu, <case>.vtp        volume mesh, surface with fixed ModelFaceID (step5)
        sim_templates/solver_template.xml, inflow_3d.flow, run_pbs.template
Output (in the SV project <case>/):
        Simulations/                           all simulation files:
            run.pbs                            PBS job script (job name = case id)
            <case>.sjb                         SimVascular simulation job (same BCs, for the GUI)
            solver.xml                         faces, inlet flow, RCR with Initial_pressure, n. of steps
            mesh-complete/                     mesh + mesh-surfaces/<face>.vtp + walls_combined.vtp
            inlet.flow                         waveform scaled to the inlet area and to the period
            init_pressure.vtu                  fluid pressure at t=0 = MAP
        <db>/<case>/bc_<config>/               rcrparams_<config>.txt, _split.csv, _solverxml.txt,
                                               inflow.reference.scaled (outside the SV project)

1. RCR (utils/setup_bcs.py = setup_bcs_new.py): flow from the inlet area (Cheng), split <config>
   (les: 33/22.3/44.6 % upper branches, 70/30 external/internal iliac), R1 = res1_split * R.
   Areas from the MESH surface: the same faces the solver integrates on.
2. Pressure at steady state from t=0: fluid initialised at MAP (init_pressure.vtu), every RCR
   capacitor at its mean steady-state pressure P_c = (MAP - P_d) R2/(R1+R2) + P_d.
3. utils/prepare_sim.py: faces rebuilt from ModelFaceID and checked (GlobalElementID, boundary
   coverage, cap area, cap flatness within one gmes), waveform vs RCR consistency, solver.xml.
4. utils/sjb.py (old build_sjb.py): .sjb with the same RCR, waveform, steps and dt.
Parameters: params.json step6.
"""
import os
import stat
import sys
import xml.etree.ElementTree as ET

PIPELINE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0]))
sys.path.insert(0, PIPELINE_DIR)

from utils import config, prepare_sim, setup_bcs, sjb  # noqa: E402


def _args():
    argv = sys.argv[1:]
    if "--" in argv:  # SimVascular may put its own arguments before "--"
        argv = argv[argv.index("--") + 1:]
    if len(argv) != 1:
        sys.exit("Usage: simvascular --python -- step6_prepare_sim.py <case_id>")
    return argv[0]


def _project_file(rel):
    return os.path.join(config.PROJECT_DIR, rel)


def write_run_pbs(template, out, case_id):
    """run_pbs from the template, job name = case id (plain replace: the script has ${...})."""
    with open(template, encoding="utf-8") as f:
        text = f.read().replace("{case}", case_id)
    config.assert_writable(out)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.chmod(out, os.stat(out).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  saved {out}")


def main(case_id):
    paths = config.case_paths(case_id)
    params = config.load_params()
    p = params["step6"]
    cfg_name = p["config"]
    proj = os.path.join(paths["db"], case_id)
    mdl = os.path.join(proj, "Models", f"{case_id}.mdl")
    mesh_vtu = os.path.join(proj, "Meshes", f"{case_id}.vtu")
    mesh_vtp = os.path.join(proj, "Meshes", f"{case_id}.vtp")
    sims = os.path.join(proj, "Simulations")
    # Keep generated simulation files directly in Simulations/. Only
    # mesh-complete remains a directory.
    sim_dir = sims
    bc_dir = os.path.join(paths["db"], f"bc_{cfg_name}")
    template, inflow = _project_file(p["template"]), _project_file(p["inflow"])
    pbs_template = _project_file(p["pbs_template"])
    print(f">>> step6 {case_id} ({cfg_name})\n  simulation: {sim_dir}")
    for f in (mdl, mesh_vtu, mesh_vtp, template, inflow, pbs_template):
        if not os.path.exists(f):
            raise RuntimeError(f"Missing input: {f}")
    for d in (sim_dir, bc_dir):
        config.assert_writable(d)
    os.makedirs(bc_dir, exist_ok=True)

    # 1. RCR + scaled waveform, from the mesh surface (areas the solver integrates on)
    setup_bcs.main(vtp_path=mesh_vtp, out_path=bc_dir, mdl_path=mdl,
                   map_val=p["map"], tot_comp=p["tot_comp"], res1_split=p["res1_split"],
                   ir_sc=p["ir_sc"], renal_rp=p["renal_rp"], dp_max=p["dp_max"],
                   configs=[cfg_name], inflow_file=inflow, target_duration=p["period"],
                   distal_pressure=p["distal_pressure"], init_pressure_hint=False)
    rcr = os.path.join(bc_dir, f"rcrparams_{cfg_name}.txt")
    scaled = os.path.join(bc_dir, "inflow.reference.scaled")

    # 2-3. Simulations/: mesh-complete, init_pressure.vtu, inlet.flow, solver.xml
    prepare_sim.prepare(mdl=mdl, mesh_vtu=mesh_vtu, mesh_vtp=mesh_vtp, rcr=rcr, inflow=scaled,
                        template=template, out=sim_dir, cycles=p["cycles"],
                        save_cycles=p["save_cycles"], map_val=p["map"],
                        distal_pressure=p["distal_pressure"], init_pressure_value=None,
                        force=True, max_cap_dev=params["step5"]["gmes"])

    # 4. Simulations/<case>.sjb with the same settings as solver.xml
    root = ET.parse(os.path.join(sim_dir, "solver.xml")).getroot()
    gen = root.find("GeneralSimulationParameters")
    fluid = root.find("Add_equation[@type='fluid']")
    id2name, id2type = setup_bcs.map_from_mdl(mdl)
    sjb.write_sjb(mdl=mdl, rcr=rcr, inflow=scaled, out=os.path.join(sims, f"{case_id}.sjb"),
                  model_name=case_id, mesh_name=case_id,
                  inlet_id=[f for f, n in id2name.items() if n == "inlet"][0],
                  wall_ids=[f for f, t in id2type.items() if t == "wall"],
                  timesteps=int(gen.findtext("Number_of_time_steps")),
                  dt=float(gen.findtext("Time_step_size")),
                  vtk_increment=int(gen.findtext("Increment_in_saving_VTK_files")),
                  density=float(fluid.findtext("Density")),
                  viscosity=float(fluid.find("Viscosity").findtext("Value")),
                  initial_pressure=p["map"])

    # 5. run.pbs next to solver.xml
    write_run_pbs(pbs_template, os.path.join(sims, "run.pbs"), case_id)
    print(f">>> step6 done")


if __name__ == "__main__":
    main(_args())