#ponte tra script di processing e script di orchestrazione.

from pathlib import Path

from .base import Adapter
from .paraview import ParaViewAdapter
from .slicer import SlicerInteractiveAdapter


# repo_root/steps  (stessa "risalita" che fa cli.py per config/)
STEPS_DIR = Path(__file__).resolve().parents[3] / "steps"

__all__ = ["Adapter", "build_adapters"]


def build_adapters(paths: dict, params: dict) -> dict[str, Adapter]:
    """stage-name -> adapter, con config iniettata dall'esterno."""
    # Import adapters that may trigger heavy imports lazily to avoid
    # circular import problems at package import time.
    try:
        from .simvascular import SimVascularAdapter
    except Exception as e:
        SimVascularAdapter = None
        print(f"[DEBUG] build_adapters: could not import SimVascularAdapter: {e}")

    adapters = {
        "paraview": ParaViewAdapter(
            stage="paraview",
            pvpython=paths["pvpython"],
            script=STEPS_DIR / "paraview" / "transform_mesh.py",
            input_filename="lumen_tree_cfd_cap.vtk",
            output_filename="lumen_tree_cfd_clip_cm.vtk",
            output_artifact_name="clip_cm",
            extra_args=["--scale", str(params.get("scale", 0.1))],
        ),
        "mesh_qc": ParaViewAdapter(
            stage="mesh_qc",
            pvpython=paths["pvpython"],
            script=STEPS_DIR / "paraview" / "quality_check.py",
            input_filename="{patient}.vtp",
            output_filename="{patient}.vtp",
            output_artifact_name="mesh_qc",
            extra_args=["--threshold", str(params.get("quality_threshold", 400))],
        ),
        "slicer": SlicerInteractiveAdapter(
            stage="slicer",
            slicer_bin=paths.get("slicer", "Slicer"),
            script=STEPS_DIR / "slicer" / "extract_tree_and_extensions.py",
            extensions_length=params.get("extensions_length", 25.0)
        ),
    }

    if SimVascularAdapter is not None:
        adapters["sv_extract"] = SimVascularAdapter(
            stage="sv_extract",
            simvascular=paths["simvascular"],
            script=STEPS_DIR / "simvascular" / "extract_faces.py",
            input_filename="lumen_tree_cfd_clip_cm.vtk",
            outputs={
                "sv_model": "model.vtp",       # modello con ModelFaceID
                "cap_faces": "cap_faces.json",  # consumato da sv_match
            },
            extra_args=["--separation-angle", str(params.get("separation_angle", 50))],
        )

    return adapters