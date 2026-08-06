"""Astrazione su una cartella ``pz***`` del database.

Un ``Patient`` non e' altro che un wrapper su una cartella: conosce i percorsi
standard degli artefatti e delega al ledger la lettura/scrittura dello stato.
E' l'unico punto in cui il codice sa "com'e' fatta dentro" la cartella di un pz.
"""
from __future__ import annotations

from pathlib import Path

from .ledger import (
    HistoryEntry,
    PatientStatus,
    StageStatus,
    now,
    read_status,
    write_status,
)

from .stages import Pipeline 
# La sequenza completa vivra' in stages.py / pipeline.yaml (prossimo passo).

INPUT_FILENAME = "lumen_tree_cfd.vtk"


class Patient:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.id = self.root.name

    @property
    def path(self) -> Path:
        return self.root

    # percorsi dentro la cartella di un pz
    @property
    def input_vtk(self) -> Path:
        return self.root / INPUT_FILENAME

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def preview_dir(self) -> Path:
        return self.root / "preview"

    @property
    def sim_dir(self) -> Path:
        return self.root / "sim"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    # check status
    def has_status(self) -> bool:
        return self.status_path.exists()

    def load_status(self) -> PatientStatus:
        return read_status(self.status_path)

    def initialize(self, pipeline: Pipeline) -> PatientStatus:
        """Crea il ledger per un pz nuovo. """
        if self.has_status():
            return self.load_status()
        if not self.input_vtk.exists():
            raise FileNotFoundError(
                f"{self.id}: manca l'input {INPUT_FILENAME} in {self.root}"
            )
        ts = now()
        st = PatientStatus(
            patient_id=self.id,
            stage= pipeline.first_stage(),
            status=StageStatus.PENDING.value,
            history=[HistoryEntry(stage="ingest", status="done", at=ts,
                                  message="pz inizializzato")],
        )
        write_status(self.status_path, st)
        return st

    def set_stage(
        self,
        stage: str,
        status: StageStatus,
        message: str | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> PatientStatus:
        """Aggiorna stadio+stato e appende una riga di storia. Unico modo di
        mutare il ledger: cosi' ogni transizione lascia sempre una traccia."""
        st = self.load_status()
        st.stage = stage
        st.status = status.value
        if artifacts:
            st.artifacts.update(artifacts)
        st.history.append(
            HistoryEntry(stage=stage, status=status.value, at=now(), message=message)
        )
        write_status(self.status_path, st)
        return st

    # --- scoperta ---
    @classmethod
    def discover(cls, database_root: Path) -> list["Patient"]:
        """Tutti i pz nel datalake, ordinati per id."""
        root = Path(database_root)
        if not root.exists():
            return []
        return sorted(
            (cls(p) for p in root.iterdir() if p.is_dir() and p.name.startswith("pz")),
            key=lambda pt: pt.id,
        )