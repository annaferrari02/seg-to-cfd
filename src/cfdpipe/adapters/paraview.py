"""Adapter per il passo ParaView.

Non conosce vtk: lancia pvpython su steps/2_paraview.py e comunica via file.
Implementa il contratto base.Adapter.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .base import Adapter
from ..patient import Patient


class ParaViewAdapter(Adapter):
    """Adapter generico per uno script ParaView che legge e scrive un file."""

    def __init__(
        self,
        stage: str,
        pvpython: str,
        script: Path,
        input_filename: str,
        output_filename: str,
        output_artifact_name: str,
        extra_args: list[str] | None = None,
    ) -> None:
        self.stage = stage
        self.pvpython = pvpython
        self.script = Path(script)
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.output_artifact_name = output_artifact_name
        self.extra_args = extra_args or []

    def _input_path(self, patient: Patient) -> Path:
        name = self.input_filename.format(patient=patient.id)
        return patient.root / name

    def _output_path(self, patient: Patient) -> Path:
        name = self.output_filename.format(patient=patient.id)
        return patient.root / name

    def preconditions(self, patient: Patient) -> None:
        if not self._input_path(patient).exists():
            raise FileNotFoundError(
                f"{patient.id}: manca l'input {self._input_path(patient)}"
            )
        if not Path(self.pvpython).exists():
            raise FileNotFoundError(
                f"pvpython non trovato: {self.pvpython} (vedi paths.yaml)"
            )
        if not self.script.exists():
            raise FileNotFoundError(f"script non trovato: {self.script}")

    def run(self, patient: Patient) -> None:
        output = self._output_path(patient)
        cmd = [
            self.pvpython,
            str(self.script),
            str(self._input_path(patient)),
            str(output),
            *self.extra_args,
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{patient.id}: pvpython uscito con codice {proc.returncode}.\n"
                f"--- stderr ---\n{proc.stderr}\n--- stdout ---\n{proc.stdout}"
            )

    def validate(self, patient: Patient) -> dict[str, str]:
        output = self._output_path(patient)
        if not output.exists():
            raise FileNotFoundError(f"{patient.id}: output non prodotto: {output}")
        if output.stat().st_size == 0:
            raise ValueError(f"{patient.id}: output vuoto: {output}")
        return {self.output_artifact_name: str(output)}