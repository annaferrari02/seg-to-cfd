# adapters/base.py  (MONDO 1)
from __future__ import annotations
from abc import ABC, abstractmethod
from ..patient import Patient


class Adapter(ABC):
    stage: str   # il nome dello stadio che gestisce, es. "paraview"
                 # DEVE combaciare col name in pipeline.yaml

    @abstractmethod
    def preconditions(self, patient: Patient) -> None:
        """Solleva se il mondo non è pronto (input mancante, binario assente)."""

    @abstractmethod
    def run(self, patient: Patient) -> None:
        """Lancia il tool come sottoprocesso. Solleva se exit code != 0."""

    @abstractmethod
    def validate(self, patient: Patient) -> dict[str, str]:
        """Controlla l'output. Solleva se non valido.
        Ritorna gli artifacts prodotti (nome_logico -> path)."""