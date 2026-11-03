r"""Step 5 (SimVascular) - TetGen volume mesh

<simvascular> --python -- step5_sv_mesh.py <case_id>

Input:  <case>/Models/<case>.vtp + faces_named.json   (step4)
Output: <case>/Meshes/<case>.msh + .vtp + .vtu         mesh in the SV project (GUI)
        <case>/Meshes/mesh-complete/                   for the solver: mesh-complete.mesh.vtu,
            mesh-complete.exterior.vtp, walls_combined.vtp, mesh-surfaces/<face name>.vtp
        mesh_info.json, sv_meshing.log

Cap rings: SimVascular leaves the boundary-layer ring of each cap with a provisional
ModelFaceID (9999); each ring cell is given the id of the cap in whose plane it lies.
Checks on the face ids (what the boundary conditions use): exactly the expected ModelFaceID,
every cap one flat disc (within one gmes of its plane), area mesh/model >= step5.face_min_area_ratio (at coarse gmes the
cap of a small vessel is 10-25% smaller than in the model: reported, not an error).
Parameters: params.json step5 (gmes, boundary layer). SimVascular GUI defaults otherwise,
except use_mmg=False (MMG drops ModelFaceID). If TetGen fails with the boundary layer,
the next portion of params step5.boundary_layer.portions is tried; any other error
(missing parameter, failed check) stops the step at once.
"""
import json
import os
import sys
import time

PIPELINE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0]))
sys.path.insert(0, PIPELINE_DIR)

from utils import config, sv_mesh  # noqa: E402


def _args():
    argv = sys.argv[1:]
    if "--" in argv:  # SimVascular may put its own arguments before "--"
        argv = argv[argv.index("--") + 1:]
    if len(argv) != 1:
        sys.exit("Usage: simvascular --python -- step5_sv_mesh.py <case_id>")
    return argv[0]


def main(case_id):
    paths = config.case_paths(case_id)
    p = config.load_params()["step5"]
    db = paths["db"]
    proj = os.path.join(db, case_id)
    model_vtp = os.path.join(proj, "Models", f"{case_id}.vtp")
    meshes = os.path.join(proj, "Meshes")
    log = os.path.join(db, "sv_meshing.log")
    with open(os.path.join(db, "faces_named.json"), encoding="utf-8") as f:
        named = json.load(f)
    faces = {int(x["id"]): x["name"] for x in named["faces"]}
    walls = [fid for fid, name in faces.items() if name == "wall"]
    print(f">>> step5 {case_id}\n  model: {model_vtp}\n  gmes {p['gmes']} cm, boundary layer "
          f"{p['boundary_layer']['on']}\n  TetGen output -> {log}")
    if not os.path.exists(model_vtp):
        raise RuntimeError(f"Missing model (run step4): {model_vtp}")

    # all parameters read up front: a missing key stops here, not inside the retry loop
    bl = p["boundary_layer"]
    portions = bl["portions"] if bl["on"] else [None]
    min_area_ratio = p["face_min_area_ratio"]
    for key in ("layers", "decreasing_ratio", "constant_thickness"):
        if bl["on"] and key not in bl:
            raise KeyError(f"params.json step5.boundary_layer.{key} missing")
    info = {"units": "cm", "gmes": p["gmes"], "boundary_layer": bl, "attempts": []}

    # only a TetGen failure moves on to the next portion; any other error stops the step
    mesher = opt = used = None
    for portion in portions:
        label = "no BL" if portion is None else f"BL portion {portion}"
        t0 = time.time()
        mesher, opt, ids = sv_mesh.build_mesher(model_vtp, walls, p, portion)
        if ids != sorted(faces):
            raise RuntimeError(f"Model faces {ids} != faces_named.json {sorted(faces)}")
        try:
            with sv_mesh.NativeOutput(log):
                mesher.generate_mesh(opt)
        except Exception as e:
            print(f"  {label}: TetGen FAILED ({e}), see {log}")
            info["attempts"].append({"portion": portion, "error": str(e)})
            continue
        print(f"  {label}: meshed in {time.time() - t0:.0f} s")
        info["attempts"].append({"portion": portion, "result": "ok"})
        used = portion
        break
    else:
        raise RuntimeError(f"TetGen failed for every portion {portions}: see {log}")

    # GUI mesh first (inspectable in SimVascular even if the check fails), then the check,
    # then mesh-complete for the solver only if the faces are right
    surface, n_fixed = sv_mesh.fix_cap_rings(mesher.get_surface(), faces, model_vtp)
    sv_mesh.write_gui_mesh(mesher, surface, meshes, case_id, case_id, sv_mesh.commands(p, used, opt))
    sv_mesh.check_faces(surface, model_vtp, faces, min_area_ratio, max_dev=p["gmes"])
    stats = sv_mesh.write_mesh_complete(mesher, surface, os.path.join(meshes, "mesh-complete"), faces)
    info["cap_ring_cells_reassigned"] = n_fixed
    print(f"  {stats['n_cells']} cells, {stats['n_points']} points")
    info.update(stats, portion_used=used, mesh=os.path.relpath(meshes, db))
    out = os.path.join(db, "mesh_info.json")
    config.assert_writable(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"  saved {out}\n>>> step5 done")


if __name__ == "__main__":
    main(_args())