"""
La spina dorsale della pipeline.

Legge config/pipeline.yaml (i DATI: sequenza + tipo di ogni stadio),
lo VALIDA all'avvio, ed espone all'orchestratore poche domande pulite:
  - qual e' il primo stadio?            -> first_stage()
  - dato uno stadio, qual e' il prossimo? -> next_stage(current)
  - di che tipo e' uno stadio?           -> stage_type(name)

L'orchestratore parla solo con questo modulo e non sa che dietro c'e' YAML.
Confine: cambi la SEQUENZA -> tocchi il .yaml; cambi il COMPORTAMENTO -> qui.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


SUPPORTED_SCHEMA_VERSION = 1


class StageType(str, Enum):
    """
    mappatura tipi di "stadi"-> come vanno "trattati"
    """
    AUTO = "auto"     # l'orchestratore lo esegue da solo (sottoprocesso)
    INTERACTIVE= "interactive"   # richiede una persona -> awaiting_human
    GATE = "gate"     # controllo pass/fail sull'output precedente
    ASYNC = "async"   # lavoro lungo -> va in coda separata


class PipelineConfigError(Exception):
    """
    problemi con file di configurazione
    """


@dataclass(frozen=True)
class Stage:
    """Un singolo stadio: nome logico + tipo. Immutabile."""
    name: str
    type: StageType


class Pipeline:
    """
    La sequenza ordinata di stadi, in memoria.

    Costruisce un indice nome->posizione per rispondere
    in O(1).
    """

    def __init__(self, stages: list[Stage]) -> None:
        self._stages = stages
        self._index = {s.name: i for i, s in enumerate(stages)}

    def first_stage(self) -> str:
        """Il primo stadio eseguibile: valore di partenza per un pz nuovo."""
        return self._stages[0].name

    def next_stage(self, current: str) -> str | None:
        """
        Lo stadio dopo `current`, oppure None se `current` e' l'ultimo
        (= pipeline finita).

        Alza KeyError se `current` non esiste
        """
        pos = self._index[current]          # KeyError se sconosciuto
        if pos + 1 >= len(self._stages):
            return None
        return self._stages[pos + 1].name

    def stage_type(self, name: str) -> StageType:
        """Il tipo di uno stadio"""
        return self._stages[self._index[name]].type

    def stage_names(self) -> list[str]:
        """Tutti i nomi, in ordine. Utile per 'status' e per i test."""
        return [s.name for s in self._stages]

    def __contains__(self, name: str) -> bool:
        return name in self._index


def load_pipeline(path: str | Path) -> Pipeline:
    """
    Legge e VALIDA config/pipeline.yaml, poi ritorna un Pipeline.

    Ogni problema -> PipelineConfigError con messaggio chiaro (dove + perche'),
    mai un traceback grezzo di yaml o un KeyError opaco.
    """
    path = Path(path)

    if not path.exists():
        raise PipelineConfigError(f"File di configurazione non trovato: {path}")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise PipelineConfigError(f"YAML non valido in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise PipelineConfigError(
            f"{path}: il contenuto non è una mappa du chiavi, eg name: x"
            f"trovato {type(raw).__name__}."
        )

    # stages deve esistere ed essere una lista non vuota di dict {name, type} (DEBUG) 
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise PipelineConfigError(f"{path}: manca una lista 'stages' non vuota.")

    stages: list[Stage] = []
    seen: set[str] = set() #elimina stage duplicati 

    for i, entry in enumerate(raw_stages):
        where = f"{path}: stages[{i}]"

        if not isinstance(entry, dict):
            raise PipelineConfigError(
                f"{where}: ogni stadio deve avere 'name' e 'type'."
            )

        name = entry.get("name")
        type_str = entry.get("type")

        if not isinstance(name, str) or not name.strip():
            raise PipelineConfigError(f"{where}: 'name' mancante o vuoto.")

        if name in seen:
            raise PipelineConfigError(
                f"{where}: nome duplicato {name!r}. I nomi devono essere unici"
            )

        try:
            stage_type = StageType(type_str)
        except ValueError:
            valid = ", ".join(t.value for t in StageType)
            raise PipelineConfigError(
                f"{where}: type {type_str!r} sconosciuto. Ammessi: {valid}."
            ) from None

        stages.append(Stage(name=name, type=stage_type))
        seen.add(name) #logica per evitare duplicati

    return Pipeline(stages)