#ponte tra script di processing e script di orchestrazione.

from pathlib import Path

from .base import Adapter
from .paraview import ParaViewAdapter
from .slicer import SlicerInteractiveAdapter
from .geometry import SvMatchAdapter

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
            input_filename="cfd{patient}/Meshes/{patient}/{patient}.vtu",
            output_filename="cfd{patient}/Meshes/{patient}/{patient}_qc.vtu",
            output_artifact_name="mesh_qc",
            extra_args=["--threshold", str(params.get("quality_threshold", 400))],
        ),
        "slicer": SlicerInteractiveAdapter(
            stage="slicer",
            slicer_bin=paths.get("slicer", "Slicer"),
            script=STEPS_DIR / "slicer" / "extract_tree_and_extensions.py",
            extensions_length=params.get("extensions_length", 25.0)
        ),
        "sv_match": SvMatchAdapter(
            stage="sv_match",
            cap_faces="cap_faces.json",
            endpoints="Endpoints_{patient}.mrk.json",
            output="face_roles.json",
            scale=params.get("scale", 0.1),
            area_margin=params.get("area_margin", 1.3),
            frame_margin=params.get("frame_margin", 1.3),
        ),
        "sv_apply": SimVascularAdapter(
            stage="sv_apply",
            script= STEPS_DIR / "simvascular" / "sv_apply.py",
            input_filename="model.vtp",
            outputs= {"faces_named": "faces_named.json"},
            simvascular=paths["simvascular"],
        ),
        "sv_meshing": SimVascularAdapter(
            stage="sv_meshing",
            simvascular=paths["simvascular"],
            script=STEPS_DIR / "simvascular" / "sv_meshing.py",
            input_filename="cfd{patient}/Models/{patient}.vtp",
            outputs={
                "volume_mesh": "cfd{patient}/Meshes/{patient}/{patient}.vtu",
                "mesh_info":   "mesh_info.json",
            },
            extra_args=[
            "--gmes", str(params.get("gmes", params.get("global_max_edge_size", 0.25))),
            "--bl", str(params.get("bl", True)).lower(),
            "--portion-edge-size", str(params.get("portion_edge_size", 0.5)),
            "--bl-num-layers", str(params.get("bl_num_layers", 2)),
            "--bl-decreasing-ratio", str(params.get("bl_decreasing_ratio", 0.1)),
        ]
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
            extra_args=[
            "--separation-angle", str(params.get("separation_angle", 50)),
            "--remesh ",
            "--remesh-hmin",  str(params.get("remesh_hmin", 0.05)),
            "--remesh-hmax",  str(params.get("remesh_hmax", 0.05)),
            "--remesh-hausd", str(params.get("remesh_hausd", 0.01)),
        ],
        )

    return adapters