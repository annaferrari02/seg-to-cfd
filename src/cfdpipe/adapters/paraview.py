"""Adapter per il passo ParaView (MONDO 1).

Non conosce vtk: lancia pvpython su steps/2_paraview.py e comunica via file.
Implementa il contratto base.Adapter.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .base import Adapter
from ..patient import Patient

OUTPUT_FILENAME = "lumen_tree_cfd_clip_cm.vtk"


class ParaViewAdapter(Adapter):
    stage = "paraview"

    def __init__(self, pvpython: str, script: Path, scale: float) -> None:
        self.pvpython = pvpython
        self.script = Path(script)
        self.scale = scale

    def _output_path(self, patient: Patient) -> Path:
        return patient.root / OUTPUT_FILENAME

    def preconditions(self, patient: Patient) -> None:
        # solo LETTURA: il mondo è pronto adesso? (nessun mkdir qui)
        if not patient.input_vtk.exists():
            raise FileNotFoundError(f"{patient.id}: manca l'input {patient.input_vtk}")
        if not Path(self.pvpython).exists():
            raise FileNotFoundError(f"pvpython non trovato: {self.pvpython} (vedi paths.yaml)")
        if not self.script.exists():
            raise FileNotFoundError(f"script MONDO 2 non trovato: {self.script}")

    def run(self, patient: Patient) -> None:
        
        output = self._output_path(patient)

        cmd = [
            self.pvpython,
            str(self.script),
            str(patient.input_vtk),
            str(output),
            "--scale", str(self.scale),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"{patient.id}: pvpython uscito con codice {proc.returncode}.\n"
                f"--- stderr ---\n{proc.stderr}\n--- stdout ---\n{proc.stdout}"
            )

    def validate(self, patient: Patient) -> dict[str, str]:
        # legge SOLO dal disco: niente vtk nel MONDO 1
        output = self._output_path(patient)
        if not output.exists():
            raise FileNotFoundError(f"{patient.id}: output non prodotto: {output}")
        if output.stat().st_size == 0:
            raise ValueError(f"{patient.id}: output vuoto: {output}")
        return {"clip_cm": str(output)}