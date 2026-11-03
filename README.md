# seg-to-cfd (from Segmentation to CFD Simulation)

Takes a vascular segmentation (label map + CT) and turns it into an svMultiPhysics CFD simulation ready to launch. 

## How it works

The pipeline is a chain of 8 steps. Each one runs in its own tool (Slicer, ParaView or SimVascular).

| # | Script | Runs in | What it does |
|---|--------|---------|--------------|
| 0 | `step0_clean.py` | Slicer | Cleans the segmentation (HU filters, smoothing, merge) |
| 1 | `step1_extract_centerline.py` | Slicer | Builds the tree model, labels the endpoints, extracts the centerline |
| 2 | `step2_cap_flow_ext.py` | Slicer | Clips, adds flow extensions and caps, tags faces (`ModelFaceID`) |
| 3 | `step3_transform.py` | pvpython | Converts mm → cm |
| 4 | `step4_sv_project.py` | SimVascular | Creates the SV project, faces named after each artery |
| 5 | `step5_sv_mesh.py` | SimVascular | Volume mesh (TetGen + boundary layer) |
| 6 | `step6_prepare_sim.py` | SimVascular | Writes `Simulations/` (`solver.xml`, `.sjb`, `run.pbs`) |
| 7 | `step7_upload.py` | python | Copies the simulations to the external disk |

## Setup

You need 3D Slicer (with SlicerVMTK), ParaView, SimVascular, and Python 3 with `vtk` and `numpy`.

Then fill in two files:

- `utils/config.json`: where things are. Input folders, `database_root`, paths to the Slicer / pvpython / SimVascular executables, external disk, cluster.
- `utils/params.json`: how each step behaves. Segment names, filters, centerline settings, flow extension length, mesh size, boundary conditions. Modify this file to adapt this pipeline to your parameters for ct processing, mesh generation and CFD configuration. 

The simulation templates (`solver.xml`, inflow waveform, PBS script) live in `sim_templates/`.

## Running it

```bash
python3 run_pipeline.py pz006                   # everything, steps 0-7
python3 run_pipeline.py pz006 pz088             # several patients, one after the other
python3 run_pipeline.py pz088 --from 3          # restart from step 3 (e.g. after a manual fix)
python3 run_pipeline.py pz088 --from 1 --to 2   # just steps 1-2
python3 run_pipeline.py pz088 --gui             # Slicer steps with the GUI open
python3 run_pipeline.py pz088 --no-review       # skip the manual check after step 0
```

### Manual check after step 0

By default the pipeline pauses after step 0 and leaves Slicer open, so you can look at the segmentation and fix it if needed. When you're happy, hit **SAVE step0 AND CONTINUE** in the toolbar. Slicer closes and the pipeline carries on by itself.

### Checking progress

```bash
python3 run_pipeline.py --status                # patient × step table, plus why things failed
python3 run_pipeline.py --status pz088          # full record for one patient
python3 run_pipeline.py pz006 pz088 --resume    # each patient picks up from its first unfinished step
python3 run_pipeline.py --retry-failed          # rerun only the patients that failed
```

If one patient fails, it stops there and the next one keeps going.

## Where stuff ends up

- `<database_root>/<pz>/`: everything for that patient. The SimVascular project is in `<pz>/<pz>/`.
- `<database_root>/<pz>/logs/stepN_<name>.log`: one log per step. Look here first when something breaks.
- `<database_root>/trackpaz.json`: status of every patient and step.
