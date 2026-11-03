# seg-to-cfd

Pipeline che porta una segmentazione vascolare (label map + TAC) a una simulazione CFD svMultiPhysics pronta da lanciare.

## Step

| # | Script | Interprete | Output |
|---|--------|-----------|--------|
| 0 | `step0_clean.py` | Slicer | `Seg/<pz>_seg_clean.seg.nrrd` (filtri HU, smoothing, merge) |
| 1 | `step1_extract_centerline.py` | Slicer | tree model, endpoint etichettati, centerline |
| 2 | `step2_cap_flow_ext.py` | Slicer | `Model/final_tree.vtk` (clip, flow extension, cap, `ModelFaceID`) |
| 3 | `step3_transform.py` | pvpython | `Model/final_tree_cm.vtk` (mm → cm) |
| 4 | `step4_sv_project.py` | SimVascular | progetto SV con facce nominate per arteria |
| 5 | `step5_sv_mesh.py` | SimVascular | mesh volumetrica TetGen + boundary layer |
| 6 | `step6_prepare_sim.py` | SimVascular | `Simulations/` (`solver.xml`, `.sjb`, `run.pbs`) |
| 7 | `step7_upload.py` | python | copia delle simulazioni su disco esterno |

Ogni step gira nel proprio interprete; gli step comunicano solo tramite file.

## Configurazione

- `utils/config.json` — percorsi: input (segmentazioni, TAC), `database_root`, eseguibili (Slicer, pvpython, SimVascular), disco esterno, cluster.
- `utils/params.json` — parametri per step (segmenti, filtri, centerline, flow extension, mesh, BC).
- `sim_templates/` — template di `solver.xml`, onda di inflow, script PBS.

## Uso

```bash
python3 run_pipeline.py pz006                   # step 0-7
python3 run_pipeline.py pz006 pz088             # più pazienti in sequenza
python3 run_pipeline.py pz088 --from 3          # riprende dallo step 3
python3 run_pipeline.py pz088 --from 1 --to 2   # solo step 1-2
python3 run_pipeline.py pz088 --gui             # step Slicer con GUI
python3 run_pipeline.py pz088 --no-review       # salta la revisione manuale dopo step0
python3 run_pipeline.py --status [pz088]        # stato (tabella o record del paziente)
python3 run_pipeline.py pz006 pz088 --resume    # riparte dal primo step non completato
python3 run_pipeline.py --retry-failed          # solo i pazienti falliti
```

**Revisione manuale:** dopo lo step 0 Slicer resta aperto; correggere la segmentazione e premere *SAVE step0 AND CONTINUE*. La pipeline prosegue da sola.

## Output

- `<database_root>/<pz>/` — risultati per paziente; progetto SimVascular in `<pz>/<pz>/`
- `<database_root>/<pz>/logs/stepN_<nome>.log` — log per step
- `<database_root>/trackpaz.json` — stato di ogni paziente e step

## Requisiti

3D Slicer + SlicerVMTK, ParaView (pvpython), SimVascular, Python 3 con `vtk` e `numpy`.
