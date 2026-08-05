"""Caricamento della configurazione NON-sequenziale.

Possiede due file, distinti da pipeline.yaml (di cui e' padrone stages.py):
  - paths.yaml   -> i path degli esecutori (pvpython, simvascular, ...)
  - params.yaml  -> i parametri della pipeline (scale, soglia QC, GMES, ...)

Valida che siano BEN FORMATI (esiste, YAML valido, e' una mappa) e ritorna
dizionari gia' pronti. NON verifica che i binari esistano su disco: quello e'
un check ambientale e vive in preconditions() dei singoli adapter.
"""

from __future__ import annotations

from pathlib import Path

import yaml


class ConfigError(Exception):
    """Problemi con un file di configurazione (paths/params)."""



#func di check iniziale --> Legge un file YAML e garantisce che sia una mappa chiave->valore (nel caso, da fare)


def load_paths(path: str | Path) -> dict[str, str]:
    """I path degli esecutori. sono inseriti da ogni user perche i path dipendono dall'ambiente.
      Ogni valore deve essere una stringa non vuota."""
    path= Path(path)

    raw = yaml.safe_load(path.read_text())
    for key, value in raw.items():
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"{path}: il path '{key}' deve essere una stringa non vuota."
            )
    return raw


def load_params(path: str | Path) -> dict:
    """I parametri numerici. Eterogenei per tipo -> nessuna validazione qui;
    ogni adapter controlla in preconditions le chiavi che gli servono."""
    path = Path(path)

    if not path.exists():
        raise ConfigError(f"File di configurazione non trovato: {path}")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML non valido in {path}: {e}") from e

    if raw is None:
        return {}

    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path}: il contenuto deve essere una mappa 'chiave: valore', "
            f"trovato {type(raw).__name__}."
        )

    return raw


def require(mapping: dict, key: str, source: str):
    """Estrae una chiave obbligatoria con messaggio chiaro se manca.
    Da usare in build_adapters / preconditions, non qui dentro."""
    if key not in mapping:
        raise ConfigError(f"{source}: manca la chiave obbligatoria '{key}'.")
    return mapping[key]