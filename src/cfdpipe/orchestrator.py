"""Orchestratore di tutta sta pipeline.

Il "cervello" stateless della pipeline. Non conosce YAML, non conosce vtk, non
sa com'e' fatto un adapter dentro: riceve dal confine CLI una ``Pipeline`` gia'
validata e un dizionario ``{nome_stadio -> Adapter}`` gia' costruito, poi:

  1. scandisce il datalake (``Patient.discover``);
  2. per ogni paziente legge lo stato da disco e compie UNA transizione alla
     volta, finche' il pz non arriva a un punto di riposo;
  3. isola i guasti: un'eccezione su un pz lo marca ``failed`` e prosegue con
     gli altri.

STATELESSNESS: nessuno stato vive in memoria tra due ``run``. Ogni invocazione
ricostruisce tutto rileggendo gli ``status.json``. Si puo' interrompere e
riavviare in qualsiasi momento senza corrompere nulla.

CONFINE: la SEQUENZA e i TIPI degli stadi vivono in ``stages.py`` /
``pipeline.yaml``; il COME si esegue ogni tipo di stadio vive qui, in ``_step``.

La primitiva e' ``_step``: UNA lettura, UNA decisione, UNA scrittura. Tutto il
resto (chaining tra stadi, isolamento) e' fatto da un ciclo che chiama ``_step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ledger import StageStatus
from .patient import Patient
from .stages import Pipeline, StageType
from .adapters.base import Adapter

# Tetto anti-loop: un pz non dovrebbe mai richiedere piu' transizioni del numero
# di stadi (+ margine). Se le supera, c'e' un bug: ci fermiamo invece di girare
# all'infinito.
MAX_STEPS = 50


@dataclass
class Step:
    """Esito di UNA transizione di ``_step``."""
    cont: bool      # True = ho fatto progresso, il ciclo puo' ritentare
    action: str     # etichetta leggibile di cosa e' successo
    note: str       # dettaglio umano (per la stampa)


@dataclass
class PatientRun:
    """Riassunto di un ``run`` su un singolo paziente."""
    patient_id: str
    stage: str          # stadio in cui il pz si e' fermato
    status: str         # stato finale (valore di StageStatus)
    steps: list[Step]   # traccia delle transizioni compiute in questo run



# La primitiva: una lettura -> una decisione -> una scrittura.

def _step(patient: Patient, pipeline: Pipeline, adapters: dict[str, Adapter]) -> Step:
    """Compie AL PIU' una transizione sul ledger del paziente e la descrive.

    Non cattura eccezioni: l'isolamento dei guasti e' responsabilita' del
    chiamante (``_process_patient``), cosi' qui la logica di stato resta pulita.
    """
    st = patient.load_status()
    stage, status = st.stage, st.status

    # 1) Stadio finito -> avanza al successivo.
    #    E' anche il punto di RIPRESA: se un run precedente e' morto tra il
    #    'done' e l'avanzamento, qui recuperiamo (done e' un checkpoint durevole).
    if status == StageStatus.DONE.value:
        nxt = pipeline.next_stage(stage)          # None se 'stage' e' l'ultimo
        if nxt is None:
            return Step(False, "completed",
                        f"pipeline completata (ultimo stadio: {stage})")
        patient.set_stage(nxt, StageStatus.PENDING, message=f"avanzo da {stage}")
        return Step(True, "advanced", f"{stage} -> {nxt}")

    # 2) Stati che l'orchestratore NON tocca in questo giro.
    #    running        = in corso, oppure residuo di un crash (lo lascia a un
    #                     futuro --retry, non lo riesegue a caso).
    #    failed         = isolato: lasciato fermo, gli altri pz proseguono.
    #    awaiting_human = tocca a una persona (futuro comando 'review').
    if status != StageStatus.PENDING.value:
        return Step(False, "skipped",
                    f"{stage}: {status} (non gestito in questo run)")

    # 3) Stadio PENDING: la decisione dipende dal TIPO dello stadio.
    stype = pipeline.stage_type(stage)

    if stype is StageType.INTERACTIVE:
        # Non lo esegue l'orchestratore: segnala che tocca all'umano.
        patient.set_stage(stage, StageStatus.AWAITING_HUMAN,
                          message="in attesa di revisione umana")
        return Step(False, "await_human", f"{stage}: attende revisione umana")

    if stype is StageType.ASYNC:
        # Lavoro lungo (8-10 h): NON gira qui. Andra' alla coda simulazioni
        # (simqueue.py, da costruire). Per ora lo lasciamo pending e lo segnaliamo.
        return Step(False, "queued",
                    f"{stage}: async, destinato alla coda (non ancora attiva)")

    # AUTO o GATE: serve un adapter registrato per questo stadio.
    adapter = adapters.get(stage)
    if adapter is None:
        return Step(False, "skipped",
                    f"{stage}: nessun adapter registrato (da implementare)")

    return _execute(patient, stage, adapter)


def _execute(patient: Patient, stage: str, adapter: Adapter) -> Step:
    """Esegue uno stadio auto/gate: pending -> running -> done.

    ``preconditions`` PRIMA di scrivere 'running': se il mondo non e' pronto non
    sporchiamo il ledger con un 'running' mai partito. Se una qualsiasi delle
    tre fasi solleva, l'eccezione risale a ``_process_patient``, che marca il pz
    'failed'.
    """
    adapter.preconditions(patient)                      # puo' sollevare
    patient.set_stage(stage, StageStatus.RUNNING, message="avvio")
    adapter.run(patient)                                # bloccante (sottoprocesso)
    artifacts = adapter.validate(patient)               # puo' sollevare
    patient.set_stage(stage, StageStatus.DONE,
                      message="completato e validato", artifacts=artifacts)
    return Step(True, "executed", f"{stage}: eseguito")


# --------------------------------------------------------------------------- #
# Il ciclo per-paziente: chaining + isolamento dei guasti.
# --------------------------------------------------------------------------- #
def _current_stage(patient: Patient) -> str:
    """Stadio corrente dal disco, in modo difensivo (per marcare i 'failed')."""
    try:
        return patient.load_status().stage
    except Exception:
        return "?"


def _mark_failed(patient: Patient, stage: str, exc: Exception) -> None:
    """Marca 'failed' senza mai far crashare il ciclo (best-effort)."""
    try:
        patient.set_stage(stage, StageStatus.FAILED,
                          message=f"{type(exc).__name__}: {exc}")
    except Exception:
        pass


def _process_patient(patient: Patient, pipeline: Pipeline,
                     adapters: dict[str, Adapter]) -> PatientRun:
    """Porta UN paziente avanti finche' puo', poi si ferma.

    Continua a chiamare ``_step`` finche' una transizione dice 'basta'
    (awaiting_human / failed / async / running / nessun adapter / pipeline
    finita). Ogni chiamata e' avvolta in try/except: QUI vive l'isolamento dei
    guasti richiesto dall'architettura.
    """
    if not patient.has_status():
        return PatientRun(patient.id, "-", "(no ledger)",
                          [Step(False, "skipped",
                                "nessun ledger: esegui prima 'cfdpipe init'")])

    steps: list[Step] = []
    for _ in range(MAX_STEPS):
        try:
            step = _step(patient, pipeline, adapters)
        except Exception as exc:                        # ISOLAMENTO
            stage = _current_stage(patient)
            _mark_failed(patient, stage, exc)
            steps.append(Step(False, "failed",
                              f"{stage}: {type(exc).__name__}: {exc}"))
            break
        steps.append(step)
        if not step.cont:
            break
    else:
        steps.append(Step(False, "skipped",
                          f"raggiunto il limite di {MAX_STEPS} step (anti-loop)"))

    final = patient.load_status()
    return PatientRun(patient.id, final.stage, final.status, steps)


# --------------------------------------------------------------------------- #
# API pubblica.
# --------------------------------------------------------------------------- #
def run(database_root: Path, pipeline: Pipeline,
        adapters: dict[str, Adapter], only: str | None = None) -> list[PatientRun]:
    """Scandisce il datalake ed elabora ogni paziente in isolamento.

    ``only`` limita il run a un singolo pz (utile per i test). La configurazione
    (pipeline, adapter) arriva gia' pronta dal confine CLI: l'orchestratore non
    legge YAML.
    """
    patients = Patient.discover(database_root)
    if only is not None:
        patients = [p for p in patients if p.id == only]
    return [_process_patient(p, pipeline, adapters) for p in patients]