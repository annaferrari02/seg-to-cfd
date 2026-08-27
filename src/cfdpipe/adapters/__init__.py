#ponte tra script di processing e script di orchestrazione.

from pathlib import Path

from .base import Adapter
from .paraview import ParaViewAdapter
from .slicer import SlicerInteractiveAdapter, SlicerConversionAdapter
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
            input_filename="cfd{patient}/Meshes/{patient}.vtp",
            output_filename="cfd{patient}/Meshes/{patient}.vtp",
            output_artifact_name="mesh_qc",
            extra_args=["--threshold", str(params.get("quality_threshold", 400))],
        ),
        "slicer": SlicerInteractiveAdapter(
            stage="slicer",
            slicer_bin=paths.get("slicer", "Slicer"),
            script=STEPS_DIR / "slicer" / "extract_tree_and_extensions.py",
            flow_ext_length=params.get("flow_ext_length", 25.0),
            clip_inset=params.get("clip_inset", 2.0),
        ),
        "start": SlicerConversionAdapter(
            stage="start",
            slicer_bin=paths.get("slicer", "Slicer"),
            script=STEPS_DIR / "slicer" / "export_lumen.py",
            output_filename="lumen_tree_cfd.vtk",
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
                "volume_mesh": "cfd{patient}/Meshes/{patient}.vtu",
                "mesh_info":   "mesh_info.json",
            },
            extra_args=[
                "--gmes", str(params.get("gmes", params.get("global_max_edge_size", 0.25))),
                "--bl", str(params.get("bl", True)).lower(),
                "--portion-edge-size", str(params.get("portion_edge_size", 0.2)),
                "--bl-num-layers", str(params.get("bl_num_layers", 2)),
                "--bl-decreasing-ratio", str(params.get("bl_decreasing_ratio", 0.6)),
                "--remesh", str(params.get("remesh", False)).lower(),
                "--remesh-hmin", str(params.get("remesh_hmin", max(0.01, float(params.get("gmes", params.get("global_max_edge_size", 0.25))) / 5.0))),
                "--remesh-hmax", str(params.get("remesh_hmax", params.get("gmes", params.get("global_max_edge_size", 0.25)))),
                "--remesh-hausd", str(params.get("remesh_hausd", 0.01)),
                "--remesh-angle", str(params.get("remesh_angle", 50.0)),
            ]
        ),
        "sim_setup": SimVascularAdapter(
            stage="sim_setup",
            simvascular=paths["simvascular"],
            script=STEPS_DIR / "simvascular" / "sim_setup.py",
            input_filename="cfd{patient}/Models/{patient}.vtp",
            outputs={
                "sjb": "cfd{patient}/Simulations/{patient}.sjb",
                "rcr": "cfd{patient}/Simulations/{patient}/rcrparams.txt",
                "inflow_scaled": "cfd{patient}/Simulations/{patient}/inflow.reference.scaled",},
        extra_args=[
            "--map",              str(params["map"]),
            "--tot-comp",         str(params["tot_comp"]),
            "--reference-inflow", str(STEPS_DIR.parent / params.get("reference_inflow", "config/inflow.reference")),
            "--timesteps",        str(params.get("timesteps", 960)),
            "--vtk-increment",    str(params.get("vtk_increment", 2)),
            "--period",           str(params.get("period", 0.8)),
            "--area-to-cm2",      str(params.get("area_to_cm2", 1.0)),
            "--res1-split",       str(params.get("res1_split", 0.15)),
        ],
        )
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
        ],
        )

    return adapters