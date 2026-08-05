"""CLI dell'orchestratore (scheletro).

Comandi disponibili in questo primo passo:
  cfdpipe init <pz>      inizializza il ledger di un paziente
  cfdpipe status [pz]    mostra lo stato di uno o di tutti i pz

Il database e' preso da $CFDPIPE_DATAbase (default ./database). Al prossimo
passo passera' da config/paths.yaml.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .patient import Patient
from .stages import load_pipeline, PipelineConfigError

PIPELINE_CONFIG= Path(__file__).resolve().parents[2]/ "config" / "pipeline.yaml"
def _database_root() -> Path:
    return Path(os.environ.get("CFDPIPE_DATABASE", "./database"))


def cmd_init(args) -> None:
    root = _database_root() / args.patient
    if not root.exists():
        raise SystemExit(f"cartella non trovata: {root}")
    try: 
        pipeline= load_pipeline(PIPELINE_CONFIG)
    except PipelineConfigError as e:
        raise SystemExit(f"[init] errore nella pipeline: {e}")
    try:
        st = Patient(root).initialize(pipeline)
    except FileNotFoundError as e:
        raise SystemExit(f"[init] errore: {e}")
    print(f"[init] {args.patient}: stage={st.stage} status={st.status}")


def cmd_status(args) -> None:
    if args.patient:
        patients = [Patient(_database_root() / args.patient)]
    else:
        patients = Patient.discover(_database_root())
    if not patients:
        print("nessun paziente trovato")
        return
    print(f"{'PZ':<10} {'STAGE':<12} {'STATUS':<16} UPDATED")
    print("-" * 62)
    for p in patients:
        if not p.has_status():
            print(f"{p.id:<10} {'-':<12} {'(no ledger)':<16}")
            continue
        st = p.load_status()
        print(f"{p.id:<10} {st.stage:<12} {st.status:<16} {st.updated_at}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfdpipe")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="inizializza il ledger di un pz")
    p_init.add_argument("patient")
    p_init.set_defaults(func=cmd_init)

    p_status = sub.add_parser("status", help="mostra lo stato dei pz")
    p_status.add_argument("patient", nargs="?", default=None)
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()