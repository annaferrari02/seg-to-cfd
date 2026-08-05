"""Persistenza dello stato per-paziente (il "ledger").

Ogni paziente ha un file ``status.json`` nella propria cartella del database.
Questo modulo sa solo (1) definire gli stati possibili di uno stadio e
(2) leggere/scrivere quel file in modo atomico. NON conosce la sequenza degli
stadi: quello e' compito di ``stages.py`` / dell'orchestratore. Tenere il
ledger "stupido" e' voluto: cosi' un cambio di pipeline non tocca la
persistenza.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

SCHEMA_VERSION = 1


class StageStatus(str, Enum):
    """Stato dello stadio *corrente* di un paziente."""
    PENDING = "pending"                # in coda
    RUNNING = "running"                # in esecuzione adesso
    AWAITING_HUMAN = "awaiting_human"  # fermo: aspetta la revisione umana
    DONE = "done"                      # completato con successo
    FAILED = "failed"                  # errore: da investigare


def now() -> str:
    """Timestamp ISO-8601 in UTC, al secondo."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class HistoryEntry:
    stage: str
    status: str
    at: str
    message: str | None = None


@dataclass
class PatientStatus:
    patient_id: str
    stage: str                                     # stadio corrente / prossimo
    status: str                                    # valore di StageStatus
    schema_version: int = SCHEMA_VERSION
    updated_at: str = field(default_factory=now)
    artifacts: dict[str, str] = field(default_factory=dict)  # nome logico -> path
    history: list[HistoryEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "patient_id": self.patient_id,
            "stage": self.stage,
            "status": self.status,
            "updated_at": self.updated_at,
            "artifacts": self.artifacts,
            "history": [asdict(h) for h in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PatientStatus":
        return cls(
            patient_id=d["patient_id"],
            stage=d["stage"],
            status=d["status"],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            updated_at=d.get("updated_at", now()),
            artifacts=d.get("artifacts", {}),
            history=[HistoryEntry(**h) for h in d.get("history", [])],
        )


def read_status(path: Path) -> PatientStatus:
    with open(path, "r", encoding="utf-8") as f:
        return PatientStatus.from_dict(json.load(f))


def write_status(path: Path, status: PatientStatus) -> None:
    """Scrittura ATOMICA del ledger.

    Scrive su un file temporaneo nella stessa cartella, fa fsync, poi
    ``os.replace`` (rename atomico su POSIX). Se il processo muore a meta',
    lo ``status.json`` originale resta intatto: mai un ledger corrotto.
    """
    path = Path(path)
    status.updated_at = now()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".status.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(status.to_dict(), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)