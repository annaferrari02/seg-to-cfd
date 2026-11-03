"""Config/params loading and per-case paths. Pure Python (usable from any interpreter)."""
import glob
import json
import os

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(UTILS_DIR)

DEFAULT_CT_FALLBACK_PATTERN = "*0002.nii.gz"


def _load(name):
    # Re-read at every call: params.json can be edited between steps of the same Slicer session
    with open(os.path.join(UTILS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def load_config():
    cfg = _load("config.json")
    for key in ("input_root", "database_root"):  # relative paths are relative to the project dir
        cfg[key] = os.path.realpath(os.path.join(PROJECT_DIR, cfg[key]))
    if cfg.get("ct_fallback_root"):  # optional; absolute, or relative to the project dir
        cfg["ct_fallback_root"] = os.path.realpath(os.path.join(PROJECT_DIR, cfg["ct_fallback_root"]))
    if _is_inside(cfg["database_root"], cfg["input_root"]):
        raise RuntimeError("database_root must not be inside input_root")
    if cfg.get("ct_fallback_root") and _is_inside(cfg["database_root"], cfg["ct_fallback_root"]):
        raise RuntimeError("database_root must not be inside ct_fallback_root")
    return cfg


def load_params():
    return _load("params.json")


def segment_map():
    """{label value (int): segment name}"""
    return {int(k): v for k, v in load_params()["segments"].items()}


def _is_inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def assert_writable(path):
    """Fail if path is inside a read-only input tree (input_root or ct_fallback_root)."""
    cfg = load_config()
    for key in ("input_root", "ct_fallback_root"):
        root = cfg.get(key)
        if root and _is_inside(path, root):
            raise RuntimeError(f"Refusing to write inside {key}: {path}")


def _one(folder, pattern, exclude=()):
    hits = [p for p in glob.glob(os.path.join(folder, pattern)) if p not in exclude]
    if len(hits) != 1:
        raise RuntimeError(f"Expected 1 file '{pattern}' in {folder}, found {len(hits)}: {hits}")
    return hits[0]


def _header_problem(path):
    """None if the file header looks like its extension says, else the reason.
    Cheap check without Slicer: catches empty files, git-lfs pointers, HTML/text placeholders,
    files with a wrong extension (Slicer: 'This is not a nrrd file')."""
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except OSError as e:
        return f"cannot open ({e})"
    if not head:
        return "empty file"
    low = path.lower()
    if low.endswith((".nrrd", ".nhdr")) and head != b"NRRD":
        return f"not a NRRD file (header {head!r})"
    if low.endswith(".gz") and head[:2] != b"\x1f\x8b":
        return f"not a gzip file (header {head!r})"
    return None


def _find_ct(cfg, case_id, inp_ct, seg_in):
    """CT path and its source ('primary' | 'fallback').
    Primary: input_root/inp_ct/<case_id>/<ct_pattern>.
    Fallback: ct_fallback_root/<case_id>/<ct_fallback_pattern>, used when the primary
    has no CT or its only CT has an unreadable header.
    Several CTs in the primary folder is an ambiguity, not an absence: no fallback."""
    hits = []
    if os.path.isdir(inp_ct):
        hits = [p for p in glob.glob(os.path.join(inp_ct, cfg["ct_pattern"])) if p != seg_in]
    if len(hits) > 1:
        raise RuntimeError(f"Expected 1 file '{cfg['ct_pattern']}' in {inp_ct}, found {len(hits)}: {hits}")
    if len(hits) == 1:
        problem = _header_problem(hits[0])
        if problem is None:
            return hits[0], "primary"
        why = f"{hits[0]} unreadable: {problem}"
    else:
        why = f"no '{cfg['ct_pattern']}' in {inp_ct}"

    root = cfg.get("ct_fallback_root")
    if not root:
        raise RuntimeError(f"CT not usable ({why}) and no ct_fallback_root in config")
    fb_dir = os.path.join(root, case_id)
    pattern = cfg.get("ct_fallback_pattern", DEFAULT_CT_FALLBACK_PATTERN)
    if not os.path.isdir(fb_dir):
        raise RuntimeError(f"CT not usable ({why}); fallback folder {fb_dir} missing")
    ct = _one(fb_dir, pattern, exclude=(seg_in,))
    problem = _header_problem(ct)
    if problem:
        raise RuntimeError(f"CT not usable ({why}); fallback {ct} unreadable: {problem}")
    return ct, f"fallback ({why})"


def case_paths(case_id):
    """All paths of a case. Inputs are read-only, outputs live in database_root/<case_id>."""
    cfg = load_config()
    inp = os.path.join(cfg["input_root"], case_id)
    inp_seg = os.path.join(cfg["input_root"], cfg["inp_seg"], case_id)
    inp_ct = os.path.join(cfg["input_root"], cfg["inp_ct"], case_id)
    db = os.path.join(cfg["database_root"], case_id)
    seg_in = _one(inp_seg, cfg["seg_pattern"])
    ct_in, ct_source = _find_ct(cfg, case_id, inp_ct, seg_in)
    seg_dir = os.path.join(db, "Seg")
    model_dir = os.path.join(db, "Model")
    cl_dir = os.path.join(db, "Centerline")
    return {
        "input_dir": inp,
        "seg_in": seg_in,
        "ct_in": ct_in,
        "ct_source": ct_source,  # "primary" | "fallback"
        "db": db,
        # step0
        "seg_dir": seg_dir,
        "seg_clean": os.path.join(seg_dir, f"{case_id}_seg_clean.seg.nrrd"),
        # step1
        "model_dir": model_dir,
        "tree_model": os.path.join(model_dir, f"tree_model_{case_id}.vtk"),
        "cl_dir": cl_dir,
        "endpoints_mrk": os.path.join(cl_dir, f"Endpoints_{case_id}.mrk.json"),
        "endpoints_json": os.path.join(cl_dir, f"endpoints_{case_id}.json"),
        "centerline": os.path.join(cl_dir, f"centerline_{case_id}.vtk"),
        # step2
        "clip_points": os.path.join(cl_dir, f"ClipPoints_{case_id}.mrk.json"),
        "final_tree": os.path.join(model_dir, "final_tree.vtk"),
        "final_faces": os.path.join(model_dir, "final_tree_faces.json"),
        # step3 (ParaView)
        "final_tree_cm": os.path.join(model_dir, "final_tree_cm.vtk"),
    }