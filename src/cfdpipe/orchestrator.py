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
    print(f"[DEBUG] _step: patient={patient.id} stage={stage} status={status}")

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

    # 1b) Stallo da ultimo tentativo: se era ancora running, riproviamo.
    if status == StageStatus.RUNNING.value:
        print(f"[DEBUG] _step: patient={patient.id} stage={stage} status=running -> retry")
        patient.set_stage(stage, StageStatus.PENDING,
                          message="stale running: retry")
        status = StageStatus.PENDING.value

    # 1c) Se lo stadio era fallito, proviamo a recuperarlo automaticamente.
    if status == StageStatus.FAILED.value:
        stype = pipeline.stage_type(stage)
        adapter = adapters.get(stage)
        if adapter is not None:
            print(f"[DEBUG] _step: patient={patient.id} stage={stage} status=failed -> attempt recovery")
            try:
                artifacts = adapter.validate(patient)
            except Exception as exc:
                print(f"[DEBUG] _step: recovery failed for patient={patient.id} stage={stage}: {exc}")
                if stype is not StageType.ASYNC:
                    patient.set_stage(stage, StageStatus.PENDING,
                                      message="failed: retry")
                    status = StageStatus.PENDING.value
            else:
                patient.set_stage(stage, StageStatus.DONE,
                                  message="recovered failed stage",
                                  artifacts=artifacts)
                return Step(True, "recovered",
                            f"{stage}: recovered from failed stage")

    # 2) Stati che l'orchestratore NON tocca in questo giro.
    #    failed         = isolato: lasciato fermo, gli altri pz proseguono.
    #    awaiting_human = tocca a una persona (futuro comando 'review').
    if status != StageStatus.PENDING.value:
        print(f"[DEBUG] _step: skipping patient={patient.id} stage={stage} status={status}")
        return Step(False, "skipped",
                    f"{stage}: {status} (non gestito in questo run)")

    # 3) Stadio PENDING: la decisione dipende dal TIPO dello stadio.
    stype = pipeline.stage_type(stage)

    if stype is StageType.ASYNC:
        # Lavoro lungo (8-10 h): NON gira qui. Andra' alla coda simulazioni
        # (simqueue.py, da costruire). Per ora lo lasciamo pending e lo segnaliamo.
        print(f"[DEBUG] _step: patient={patient.id} stage={stage} is ASYNC")
        return Step(False, "queued",
                    f"{stage}: async, destinato alla coda (non ancora attiva)")

    print(f"[DEBUG] _step: patient={patient.id} stage={stage} type={stype.value}")
    # AUTO, INTERACTIVE o GATE: serve un adapter registrato per questo stadio.
    adapter = adapters.get(stage)
    if adapter is None:
        print(f"[DEBUG] _step: no adapter for stage={stage}")
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
    print(f"[DEBUG] _execute: patient={patient.id} stage={stage} adapter={type(adapter).__name__}")
    adapter.preconditions(patient)                      # puo' sollevare
    print(f"[DEBUG] _execute: preconditions ok for patient={patient.id} stage={stage}")
    patient.set_stage(stage, StageStatus.RUNNING, message="avvio")
    print(f"[DEBUG] _execute: set stage RUNNING for patient={patient.id} stage={stage}")
    adapter.run(patient)                                # bloccante (sottoprocesso)
    print(f"[DEBUG] _execute: run complete for patient={patient.id} stage={stage}")
    artifacts = adapter.validate(patient)               # puo' sollevare
    print(f"[DEBUG] _execute: validate ok for patient={patient.id} stage={stage}")
    patient.set_stage(stage, StageStatus.DONE,
                      message="completato e validato", artifacts=artifacts)
    return Step(True, "executed", f"{stage}: eseguito")


def _reconcile_step(patient: Patient, pipeline: Pipeline, adapters: dict[str, Adapter]) -> Step:
    """Allinea il ledger allo stato effettivo dei file gia' presenti su disco.

    Non esegue nessun adapter: usa solo ``validate`` per capire se lo stadio
    corrente e' gia' completato fuori pipeline (es. da GUI o da un tool esterno).
    Se e' completato, lo marca ``done`` e lascia che il ciclo passi allo stadio
    successivo.
    """
    st = patient.load_status()
    stage, status = st.stage, st.status
    print(f"[DEBUG] _reconcile: patient={patient.id} stage={stage} status={status}")

    if status == StageStatus.DONE.value:
        nxt = pipeline.next_stage(stage)
        if nxt is None:
            return Step(False, "completed", f"pipeline completata (ultimo stadio: {stage})")
        patient.set_stage(nxt, StageStatus.PENDING, message=f"avanzo da {stage} (sync)")
        return Step(True, "advanced", f"{stage} -> {nxt}")

    if status not in {StageStatus.PENDING.value, StageStatus.RUNNING.value, StageStatus.FAILED.value}:
        print(f"[DEBUG] _reconcile: skipping patient={patient.id} stage={stage} status={status}")
        return Step(False, "skipped", f"{stage}: {status} (non sincronizzabile)")

    adapter = adapters.get(stage)
    if adapter is None:
        return Step(False, "skipped", f"{stage}: nessun adapter registrato (sync rimandato)")

    try:
        artifacts = adapter.validate(patient)
    except Exception as exc:
        print(f"[DEBUG] _reconcile: stage={stage} non ancora valido: {exc}")
        return Step(False, "skipped", f"{stage}: artefatti non ancora completi")

    patient.set_stage(stage, StageStatus.DONE,
                      message="sincronizzato da filesystem", artifacts=artifacts)
    return Step(True, "recovered", f"{stage}: riconosciuto da filesystem")


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


def _sync_patient(patient: Patient, pipeline: Pipeline,
                  adapters: dict[str, Adapter]) -> PatientRun:
    """Sincronizza un paziente con i file gia' presenti, senza rieseguire stadi.

    Utile dopo correzioni manuali dalla GUI: se i file attesi dallo stadio
    corrente esistono gia', il ledger avanza da solo fino al primo stadio che
    manca davvero.
    """
    if not patient.has_status():
        return PatientRun(patient.id, "-", "(no ledger)",
                          [Step(False, "skipped",
                                "nessun ledger: esegui prima 'cfdpipe init'")])

    steps: list[Step] = []
    for _ in range(MAX_STEPS):
        try:
            step = _reconcile_step(patient, pipeline, adapters)
        except Exception as exc:
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


def sync(database_root: Path, pipeline: Pipeline,
         adapters: dict[str, Adapter], only: str | None = None) -> list[PatientRun]:
    """Allinea i ledger ai file gia' presenti sul disco e avanza quanto possibile."""
    patients = Patient.discover(database_root)
    if only is not None:
        patients = [p for p in patients if p.id == only]
    return [_sync_patient(p, pipeline, adapters) for p in patients]