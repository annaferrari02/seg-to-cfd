"""Adapter per stadio sv_matchh (no interpreti esterni)
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import Adapter
from ..patient import Patient
from ..geometry.sv_match import match, MatchError  # noqa: F401 (MatchError = fail loud)


class SvMatchAdapter(Adapter):
    """Stadio `sv_match`: inlet cap <-> inlet endpoint, produce face_roles.json.

    Parametri
    ---------
    stage        : nome dello stadio (DEVE combaciare con pipeline.yaml).
    cap_faces    : filename di cap_faces.json, relativo alla cartella del pz
                   (supporta il template "{patient}").
    endpoints    : filename del markups Slicer (es. "Endpoints_{patient}.mrk.json").
    output       : filename del face_roles.json prodotto.
    scale        : mm->cm (0.1). area_margin/frame_margin: soglie fail-loud.
    """

    def __init__(
        self,
        stage: str,
        cap_faces: str,
        endpoints: str,
        output: str,
        scale: float = 0.1,
        area_margin: float = 1.3,
        frame_margin: float = 1.3,
    ) -> None:
        self.stage = stage
        self.cap_faces = cap_faces
        self.endpoints = endpoints
        self.output = output
        self.scale = scale
        self.area_margin = area_margin
        self.frame_margin = frame_margin

    def _p(self, patient: Patient, name: str) -> Path:
        return patient.root.resolve() / name.format(patient=patient.id)

    # --- contratto base.Adapter ---
    def preconditions(self, patient: Patient) -> None:
        """Servono gli output di sv_extract (cap_faces) e di Slicer (endpoints)."""
        for logical, name in (("cap_faces", self.cap_faces),
                              ("endpoints", self.endpoints)):
            p = self._p(patient, name)
            if not p.exists():
                raise FileNotFoundError(f"{patient.id}: manca '{logical}': {p}")

    def run(self, patient: Patient) -> None:
        """Esegue il match IN-PROCESS e scrive face_roles.json.

        `match` solleva MatchError a ogni ambiguita' (frame non separabile, cap
        piu' grande non netto, inlet-area != inlet-vicino, ...): l'eccezione
        risale all'orchestratore, che marca il pz 'failed' -> ispezione umana.
        """
        roles = match(
            self._p(patient, self.cap_faces),
            self._p(patient, self.endpoints),
            scale=self.scale,
            area_margin_ratio=self.area_margin,
            frame_margin_ratio=self.frame_margin,
        )
        self._p(patient, self.output).write_text(json.dumps(roles, indent=2))

    def validate(self, patient: Patient) -> dict[str, str]:
        """Verifica che face_roles.json esista, non sia vuoto e abbia un inlet."""
        out = self._p(patient, self.output)
        if not out.exists():
            raise FileNotFoundError(f"{patient.id}: face_roles non prodotto: {out}")
        if out.stat().st_size == 0:
            raise ValueError(f"{patient.id}: face_roles vuoto: {out}")
        roles = json.loads(out.read_text())
        if not roles.get("inlet", {}).get("face_id") is not None:
            raise ValueError(f"{patient.id}: face_roles.json senza inlet valido.")
        return {"face_roles": str(out)}