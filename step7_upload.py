r"""Step 7 - copy the simulation to the external disk (database-tango)

python3 step7_upload.py pz006 [pz088 ...]

<database_root>/trackpaz.json: if Simulations/ of the patient is complete -> "ready for upload",
scp -r content of Simulations/ -> <tango_root>/<pz>/ (same level, no Simulations folder) -> "on tango"
Config: utils/config.json "tango_root".
"""
import json
import os
import subprocess
import sys
import time

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)

from utils import config  # noqa: E402

READY_UPLOAD, ON_TANGO = "ready for upload", "on tango"


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(path, data):
    """Atomic write (a crash never leaves a half-written json)."""
    config.assert_writable(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _run(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------- database -> tango

def sim_missing(case):
    """Files of a complete simulation (step6, everything flat in Simulations/) that are missing."""
    sim = os.path.join(config.case_paths(case)["db"], case, "Simulations")
    need = [f"{case}.sjb", "solver.xml", "inlet.flow", "init_pressure.vtu", "run.pbs",
            os.path.join("mesh-complete", "mesh-complete.mesh.vtu")]
    return [os.path.join(sim, f) for f in need if not os.path.exists(os.path.join(sim, f))]


def copy_to_tango(case, cfg, track_path):
    track = _load(track_path)
    rec = track.setdefault(case, {})
    missing = sim_missing(case)
    if missing:
        rec.update(status="simulation incomplete", missing=missing, updated=_now())
        _save(track_path, track)
        print(f"  {case}: Simulations incomplete, not copied: {missing}")
        return False
    rec.pop("missing", None)
    if rec.get("status") != ON_TANGO:
        rec.update(status=READY_UPLOAD, updated=_now())
        _save(track_path, track)

    tango = cfg["tango_root"]
    if not os.path.isdir(tango):
        print(f"  {case}: {tango} not found (disk not mounted?) -> stays '{READY_UPLOAD}'")
        return False
    proj = os.path.join(config.case_paths(case)["db"], case)
    dst = os.path.join(tango, case)
    config.assert_writable(dst)
    os.makedirs(dst, exist_ok=True)
    # content of Simulations/, directly under tango/pzXXX/
    sim = os.path.join(proj, "Simulations")
    srcs = sorted(os.path.join(sim, n) for n in os.listdir(sim))
    for src in srcs:
        r = _run(["scp", "-r", src, dst + "/"])
        if r.returncode != 0:
            rec.update(error=r.stderr.strip(), updated=_now())
            _save(track_path, track)
            print(f"  {case}: copy FAILED: {r.stderr.strip()}")
            return False
    rec.pop("error", None)
    rec.update(status=ON_TANGO, tango=dst, updated=_now())
    _save(track_path, track)
    print(f"  {case}: copied to {dst} -> '{ON_TANGO}'")
    return True


def main(cases):
    cfg = config.load_config()
    if "tango_root" not in cfg:
        raise SystemExit("utils/config.json: missing 'tango_root'")
    if not cases:
        raise SystemExit("Usage: python3 step7_upload.py <case_id> [<case_id> ...]")
    track_path = os.path.join(cfg["database_root"], "trackpaz.json")
    print(">>> step7 copy to tango")
    ok = all([copy_to_tango(c, cfg, track_path) for c in cases])  # every case, also after a failure
    print(">>> step7 done" if ok else ">>> step7 FAILED (see above)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)