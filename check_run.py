from pathlib import Path
from cfdpipe.config import load_paths, load_params
from cfdpipe.patient import Patient
from cfdpipe.adapters import build_adapters

root = Path(".")   # repo root
adapters = build_adapters(
    load_paths(root / "config" / "paths.yaml"),
    load_params(root / "config" / "params.yaml"),
)
pv = adapters["paraview"]
pt = Patient(root / "database" / "pz000")

pv.preconditions(pt)        # non solleva -> mondo pronto
pv.run(pt)                  # lancia pvpython, aspetta
print(pv.validate(pt))      # {'clip_cm': '.../work/lumen_tree_cfd_clip_cm.vtk'}