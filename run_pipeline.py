r"""Whole pipeline for one or more patients, with a single command.

python3 run_pipeline.py pz006                  steps 0-7 (7 = copy to the external disk)
python3 run_pipeline.py pz006 pz088            several patients, one after the other
python3 run_pipeline.py pz088 --from 3         resume from step 3 (after a manual fix)
python3 run_pipeline.py pz088 --from 1 --to 2  only steps 1-2
python3 run_pipeline.py pz088 --gui            Slicer steps with the GUI: check / fix, then close Slicer
python3 run_pipeline.py pz088 --no-review      no stop for the manual review after step0

Tracking (<database_root>/trackpaz.json):
python3 run_pipeline.py --status               table patient x step + cause of the last failure
python3 run_pipeline.py --status pz088         full record of one patient (error, log, timings)
python3 run_pipeline.py pz006 pz088 --resume   each patient restarts from its first step not done
python3 run_pipeline.py --retry-failed         only failed / interrupted patients, from the failed step

Manual review (params.json "pipeline": {"review_steps": [0]}, default [0]): after the step Slicer
stays open; fix the segmentation, then press "SAVE step0 AND CONTINUE" (toolbar) or run
save_checkpoint(): outputs saved, Slicer closes, the pipeline goes on by itself.

Each step runs in its own interpreter (Slicer / pvpython / SimVascular, paths in utils/config.json);
the steps communicate only through files. A step is OK when its outputs exist and are newer than
its start; otherwise the patient stops there (the next patient goes on).
Log of each step: <database_root>/<case>/logs/stepN_<name>.log
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time

from utils import config

PROJECT_DIR = config.PROJECT_DIR


def _outputs(case):
    """Expected outputs of each step (= its checkpoint)."""
    p = config.case_paths(case)
    proj = os.path.join(p["db"], case)
    return {
        0: [p["seg_clean"]],
        1: [p["tree_model"], p["endpoints_json"], p["centerline"]],
        2: [p["final_tree"], p["final_faces"]],
        3: [p["final_tree_cm"]],
        4: [os.path.join(proj, "Models", f"{case}.vtp"), os.path.join(proj, "Models", f"{case}.mdl"),
            os.path.join(p["db"], "faces_named.json")],
        5: [os.path.join(proj, "Meshes", f"{case}.vtu"),
            os.path.join(proj, "Meshes", "mesh-complete", "mesh-complete.mesh.vtu"),
            os.path.join(p["db"], "mesh_info.json")],
        6: [os.path.join(proj, "Simulations", f) for f in ("solver.xml", f"{case}.sjb", "run.pbs")],
        7: [],  # no own outputs (json shared by all patients): exit code of step7_upload.py
    }


# step -> (script, interpreter)
STEPS = {
    0: ("step0_clean.py", "slicer"),
    1: ("step1_extract_centerline.py", "slicer"),
    2: ("step2_cap_flow_ext.py", "slicer"),
    3: ("step3_transform.py", "pvpython"),
    4: ("step4_sv_project.py", "simvascular"),
    5: ("step5_sv_mesh.py", "simvascular"),
    6: ("step6_prepare_sim.py", "simvascular"),
    7: ("step7_upload.py", "python"),
}


# ---------------------------------------------------------------------------- tracking
# trackpaz.json = {case: {"status", "current_step", "failed_step", "last_update",
#                      "steps": {"0": {"status", "t_start", "t_end", "duration_s", "exit_code",
#                                      "missing_outputs", "error", "log", "params_sha", "attempts"}}}}
# step status: pending | running | done | failed | interrupted
# "running" left in the file = the pipeline itself died during that step (kill, crash, reboot)

def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def _track_path():
    root = config.load_config().get("database_root", os.path.join(PROJECT_DIR, "database"))
    return os.path.join(root, "trackpaz.json")


def _track_load():
    path = _track_path()
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _track_update(mutate):
    """Read-modify-write under a file lock, atomic replace (safe with parallel runs)."""
    path = _track_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = _track_load()
        mutate(data)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)


def _set_step(case, step, **fields):
    def m(data):
        rec = data.setdefault(case, {"steps": {}})
        s = rec["steps"].setdefault(str(step), {})
        s.update(fields)
        st = fields.get("status")
        if st == "running":
            s["attempts"] = s.get("attempts", 0) + 1
            rec.update(status="running", current_step=step, failed_step=None)
        elif st in ("failed", "interrupted"):
            rec.update(status=st, current_step=step, failed_step=step)
        rec["last_update"] = _now()
    _track_update(m)


def _set_case(case, **fields):
    def m(data):
        data.setdefault(case, {"steps": {}}).update(fields, last_update=_now())
    _track_update(m)


def _invalidate_after(case, step):
    """A step re-run makes every downstream result stale."""
    def m(data):
        steps = data.setdefault(case, {"steps": {}})["steps"]
        for s in STEPS:
            if s > step and str(s) in steps:
                steps[str(s)] = {"status": "pending", "invalidated_by": f"step{step} @ {_now()}",
                                 "attempts": steps[str(s)].get("attempts", 0)}
    _track_update(m)


def _params_sha():
    """Fingerprint of params.json: tells whether a failed step ran with the current params."""
    path = os.path.join(PROJECT_DIR, "utils", "params.json")
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:10]
    except OSError:
        return None


_ERR = re.compile(r"(Traceback|Error|Exception|FAILED|ERROR|Segmentation fault|Abort|Killed)")


def _extract_error(log, max_chars=3000):
    """Last traceback of the log if any, else the last error-like lines, else the tail."""
    try:
        with open(log, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    tb = [i for i, l in enumerate(lines) if l.startswith("Traceback")]
    if tb:
        chunk = lines[tb[-1]:tb[-1] + 40]
    else:
        hits = [l for l in lines if _ERR.search(l)]
        chunk = hits[-15:] if hits else lines[-20:]
    return "\n".join(chunk)[-max_chars:]


def _step_done(case, step, rec):
    """Done = marked done in trackpaz.json AND outputs still on disk."""
    if rec.get("steps", {}).get(str(step), {}).get("status") != "done":
        return False
    return all(os.path.exists(o) for o in _outputs(case)[step])


def _first_todo(case, first, last):
    rec = _track_load().get(case, {})
    for s in range(first, last + 1):
        if not _step_done(case, s, rec):
            return s
    return None


SYM = {"done": "✓", "failed": "✗", "interrupted": "!", "running": "?", "pending": "·"}


def _print_status(cases):
    data = _track_load()
    if not data:
        sys.exit(f"No track file: {_track_path()}")
    if len(cases) == 1:  # full record of one patient
        print(json.dumps(data.get(cases[0], {}), indent=2, ensure_ascii=False))
        return
    names = cases or sorted(data)
    print("case    " + " ".join(f"{s}" for s in STEPS) + "   status            last update")
    for c in names:
        rec = data.get(c, {})
        row = " ".join(SYM.get(rec.get("steps", {}).get(str(s), {}).get("status"), "-") for s in STEPS)
        print(f"{c:7} {row}   {rec.get('status', '-'):16}{rec.get('last_update', '')}")
        fs = rec.get("failed_step")
        if fs is not None and rec.get("status") in ("failed", "interrupted"):
            err = (rec["steps"].get(str(fs), {}).get("error") or "").strip().splitlines()
            print(f"        step{fs}: {err[-1][:110] if err else '(no error text, see log)'}")
    print("\n✓ done  ✗ failed  ! interrupted  ? running/died  · pending  - never run")


# ---------------------------------------------------------------------------- run
def _command(cfg, kind, script, case, gui):
    script = os.path.join(PROJECT_DIR, script)  # Slicer wants absolute paths
    if kind == "slicer":
        # always with the main window: the Segment Editor / Extract Centerline widgets crash Slicer
        # with --no-main-window. Batch: the window closes by itself at the end of the step
        return [cfg["slicer_bin"], "--no-splash", "--python-script", script, case]
    if kind == "python":
        return [sys.executable, script, case]
    if kind == "pvpython":
        return [cfg["pvpython_bin"], script, case]
    return [cfg["simvascular_bin"], "--python", "--", script, case]


def _check_bins(cfg, steps):
    keys = {f"{STEPS[s][1]}_bin" for s in steps if STEPS[s][1] != "python"}
    bad = [f"{k}: {cfg.get(k)}" for k in sorted(keys) if not cfg.get(k) or not os.access(cfg[k], os.X_OK)]
    if bad:
        sys.exit("Fix utils/config.json, executable not found:\n  " + "\n  ".join(bad))


def run_step(cfg, case, step, gui, review_steps=()):
    script, kind = STEPS[step]
    db = config.case_paths(case)["db"]
    log_dir = os.path.join(db, "logs")
    config.assert_writable(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f"step{step}_{script[6:-3]}.log")
    cmd = _command(cfg, kind, script, case, gui)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if kind == "slicer" and not gui:
        env["SEG2CFD_BATCH"] = "1"  # the step exits Slicer with 0 / 1 (utils/state.py)
        if step in review_steps:
            env["SEG2CFD_REVIEW"] = "1"  # Slicer stays open until "save and continue"
            print(f">>> step{step}: MANUAL REVIEW in Slicer, then 'SAVE ... AND CONTINUE'", flush=True)

    print(f"\n{'=' * 70}\n[{case}] step{step} {script}  ({kind})\n  log: {log}\n{'=' * 70}", flush=True)
    _invalidate_after(case, step)
    _set_step(case, step, status="running", t_start=_now(), t_end=None, duration_s=None,
              exit_code=None, missing_outputs=None, error=None, log=log, gui=gui,
              params_sha=_params_sha())
    t0 = time.time()
    proc = None
    try:
        with open(log, "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n\n")
            proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
            for line in proc.stdout:  # terminal + log
                sys.stdout.write(line)
                f.write(line)
            rc = proc.wait()
    except KeyboardInterrupt:
        if proc and proc.poll() is None:
            proc.terminate()
        _set_step(case, step, status="interrupted", t_end=_now(),
                  duration_s=round(time.time() - t0), error="KeyboardInterrupt (Ctrl+C)")
        raise

    # OK = all outputs written during this run (exit codes of SV / Slicer are not reliable)
    stale = [o for o in _outputs(case)[step] if not os.path.exists(o) or os.path.getmtime(o) < t0]
    dur = time.time() - t0
    if not _outputs(case)[step] and rc != 0:
        stale = [f"exit code {rc}"]
    if stale:
        print(f"\n>>> [{case}] step{step} FAILED (exit {rc}, {dur:.0f} s). Outputs missing / not updated:")
        for o in stale:
            print(f"      {o}")
        _set_step(case, step, status="failed", t_end=_now(), duration_s=round(dur), exit_code=rc,
                  missing_outputs=stale, error=_extract_error(log))
        return False
    if rc != 0:
        print(f">>> [{case}] step{step}: exit code {rc}, but all outputs updated -> considered OK")
    print(f">>> [{case}] step{step} OK ({dur:.0f} s)")
    _set_step(case, step, status="done", t_end=_now(), duration_s=round(dur), exit_code=rc)
    return True


def _hint(case, step):
    script, kind = STEPS[step]
    print(f"\n>>> [{case}] stopped at step{step}. Cause: python3 run_pipeline.py --status {case}")
    if kind == "slicer":
        print(f"      fix by hand:  python3 run_pipeline.py {case} --from {step} --to {step} --gui")
        print(f"                    (in Slicer: fix, main('{case}') or save_checkpoint(), close Slicer)")
        print(f"      then resume:  python3 run_pipeline.py {case} --resume")
    else:
        print(f"      fix params.json / previous step, then: python3 run_pipeline.py {case} --resume")


def main():
    ap = argparse.ArgumentParser(description="seg-to-cfd: segmentation -> svMultiPhysics case")
    ap.add_argument("cases", nargs="*", help="patient ids, e.g. pz006 pz088")
    ap.add_argument("--from", dest="first", type=int, default=0, choices=STEPS, help="first step (default 0)")
    ap.add_argument("--to", dest="last", type=int, default=7, choices=STEPS, help="last step (default 7)")
    ap.add_argument("--gui", action="store_true", help="Slicer steps with the GUI (close Slicer to go on)")
    ap.add_argument("--no-review", action="store_true", help="skip the manual review (params pipeline.review_steps)")
    ap.add_argument("--resume", action="store_true", help="skip steps already done (trackpaz.json + outputs on disk)")
    ap.add_argument("--retry-failed", action="store_true", help="all failed/interrupted patients, from the failed step")
    ap.add_argument("--status", action="store_true", help="print trackpaz.json and exit")
    a = ap.parse_args()

    if a.status:
        _print_status(a.cases)
        return
    if a.retry_failed:
        data = _track_load()
        a.cases = a.cases or sorted(c for c, r in data.items()
                                    if r.get("status") in ("failed", "interrupted", "running"))
        a.resume = True
        if not a.cases:
            sys.exit("No failed patients in trackpaz.json")
        print("Retry:", " ".join(a.cases))
    if not a.cases:
        ap.error("give at least one patient (or --status / --retry-failed)")
    if a.first > a.last:
        sys.exit("--from must be <= --to")
    cfg = config.load_config()
    # Slicer steps that wait for a manual check before saving (default: step0)
    review_steps = [] if a.no_review else config.load_params().get("pipeline", {}).get("review_steps", [0])
    _check_bins(cfg, list(range(a.first, a.last + 1)))

    summary = {}
    for case in a.cases:
        try:
            config.case_paths(case)  # input files present (read-only dir, only globbed)
        except RuntimeError as e:
            print(f"\n>>> [{case}] skipped: {e}")
            summary[case] = "inputs not found"
            _set_case(case, status="inputs not found", error=str(e))
            continue
        first = _first_todo(case, a.first, a.last) if a.resume else a.first
        if first is None:
            print(f"\n>>> [{case}] steps {a.first}-{a.last} already done, skipped")
            summary[case] = "OK, already done"
            continue
        if a.resume and first > a.first:
            print(f"\n>>> [{case}] resume from step{first}")
        t0 = time.time()
        for step in range(first, a.last + 1):
            if not run_step(cfg, case, step, a.gui, review_steps):
                _hint(case, step)
                summary[case] = f"FAILED at step{step}"
                break
        else:
            summary[case] = f"OK, steps {first}-{a.last} ({(time.time() - t0) / 60:.1f} min)"
            _set_case(case, status="done" if a.last == max(STEPS) else f"done up to step{a.last}",
                      current_step=a.last, failed_step=None)

    print(f"\n{'=' * 70}\nSUMMARY")
    for case, res in summary.items():
        print(f"  {case}: {res}")
    sys.exit(0 if all(r.startswith("OK") for r in summary.values()) else 1)


if __name__ == "__main__":
    main()