#ponte tra script di processing e script di orchestrazione.

from pathlib import Path

from .base import Adapter
from .paraview import ParaViewAdapter

# repo_root/steps  (stessa "risalita" che fa cli.py per config/)
STEPS_DIR = Path(__file__).resolve().parents[3] / "steps"

__all__ = ["Adapter", "build_adapters"]


def build_adapters(paths: dict, params: dict) -> dict[str, Adapter]:
    """stage-name -> adapter, con config iniettata dall'esterno."""
    return {
        "paraview": ParaViewAdapter(
            pvpython=paths["pvpython"],
            script=STEPS_DIR / "paraview" / "transform_mesh.py",
            scale=params.get("scale", 0.1),
        ),
        # "sv_mesh": ...,  # arriveranno gli altri
    }