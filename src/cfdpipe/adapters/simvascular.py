"""Adapter per gli step SimVascular lanciati da riga di comando

PERCHE' extract e apply sono due step distinti
    - sv_extract -> carica il modello, identifica le facce per angolo di
      separazione, distingue parete (wall) dai tappi (cap) ed ESTRAE la
      geometria dei cap (centroide + raggio, frame SV, cm). NON sa ancora
      quale cap sia l'inlet: lo decide sv_match incrociando con gli endpoint
      taggati.
    - sv_apply -> (step 6) rinomina/tipizza le facce secondo gli assegnamenti
      di sv_match e genera la mesh di volume.

"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .base import Adapter
from ..patient import Patient


class SimVascularAdapter(Adapter):
    """Lancia uno script SimVascular che legge un modello e scrive artefatti.

    Parametri
    ---------
    stage           : nome dello stadio (DEVE combaciare con pipeline.yaml).
    simvascular     : path all'eseguibile (sv.bat / simvascular), da paths.yaml.
    script          : script del MONDO 2 da eseguire dentro l'interprete SV.
    input_filename  : file di input, relativo alla cartella del pz. Supporta il
                      template "{patient}".
    outputs         : {nome_logico -> filename} degli artefatti attesi, scritti
                      nella cartella del paziente. Anche qui vale il template "{patient}".
    extra_args      : argomenti extra per lo script (es. --separation-angle 50).
    launch_flags    : flag che precedono lo script nella riga di comando SV.
                      Default ["--python", "--"]. DA CONFERMARE sulla tua build
                      2023-03-27: la forma esatta dipende da versione/OS, come e'
                      stato per il path di pvpython.

    CONTRATTO CON LO SCRIPT
        Ogni script SV riceve sempre, in aggiunta agli extra_args:
            --input   <path del file di input>
            --out-dir <cartella del paziente>
        e scrive i suoi output dentro --out-dir.
    """

    def __init__(
        self,
        stage: str,
        simvascular: str,
        script: Path,
        input_filename: str,
        outputs: Dict[str, str],
        extra_args: Optional[List[str]] = None,
        launch_flags: Optional[List[str]] = None,
    ) -> None:
        self.stage = stage
        self.simvascular = simvascular
        self.script = Path(script)
        self.input_filename = input_filename
        self.outputs = outputs
        self.extra_args = extra_args or []
        self.launch_flags = (
            launch_flags if launch_flags is not None else ["--python", "--"]
        )

    # --- percorsi ---
    def _input_path(self, patient: Patient) -> Path:
        return patient.root.resolve() / self.input_filename.format(patient=patient.id)

    def _output_path(self, patient: Patient, filename: str) -> Path:
        return patient.root.resolve() / filename.format(patient=patient.id)

    # --- contratto base.Adapter ---
    def preconditions(self, patient: Patient) -> None:
        """Solleva se il mondo non e' pronto: input, binario o script assenti."""
        inp = self._input_path(patient)
        if not inp.exists():
            raise FileNotFoundError(f"{patient.id}: manca l'input {inp}")
        if not Path(self.simvascular).exists():
            raise FileNotFoundError(
                f"SimVascular non trovato: {self.simvascular} (vedi config/paths.yaml)"
            )
        if not self.script.exists():
            raise FileNotFoundError(f"script SimVascular non trovato: {self.script}")

    def run(self, patient: Patient) -> None:
        """Lancia SV come sottoprocesso bloccante. Solleva se exit code != 0."""
        cmd = [
            self.simvascular,
            *self.launch_flags,
            str(self.script),
            "--input", str(self._input_path(patient)),
            "--out-dir", str(patient.root.resolve()),
            *self.extra_args,
        ]
        print(f"[DEBUG] SimVascularAdapter.run: cmd={' '.join(cmd)}")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = (proc.stdout or "") + (proc.stderr or "")

        # Log per-paziente, come per Slicer: post-mortem senza dover riprodurre.
        log_path = patient.root / f"{self.stage}.log"
        try:
            log_path.write_text(output, encoding="utf-8")
            print(f"[DEBUG] SimVascularAdapter.run: log in {log_path}")
        except Exception as e:
            print(f"[DEBUG] SimVascularAdapter.run: log non scritto: {e}")

        if proc.returncode != 0:
            raise RuntimeError(
                f"{patient.id}: SimVascular uscito con codice {proc.returncode}.\n"
                f"--- output ---\n{output}"
            )

    def validate(self, patient: Patient) -> dict[str, str]:
        """Verifica che ogni artefatto atteso esista e non sia vuoto.

        La correttezza del CONTENUTO (facce classificate, cap validi) e'
        garantita dallo script, che esce != 0 se fallisce; qui controlliamo solo
        che i file siano materializzati. Ritorna {nome_logico -> path}.
        """
        artifacts: dict[str, str] = {}
        for name, filename in self.outputs.items():
            path = self._output_path(patient, filename)
            if not path.exists():
                raise FileNotFoundError(
                    f"{patient.id}: artefatto '{name}' non prodotto: {path}"
                )
            if path.stat().st_size == 0:
                raise ValueError(f"{patient.id}: artefatto '{name}' vuoto: {path}")
            artifacts[name] = str(path)
        return artifacts