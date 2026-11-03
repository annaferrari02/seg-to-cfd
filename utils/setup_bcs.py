"""RCR boundary conditions (Les / Moore&Ku split) - from setup_bcs_new.py, used by step6."""
import sys
import os
import glob                       # per elencare i .vtp dentro mesh-surfaces/
from collections import defaultdict

import vtk
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x rinomina trapz -> trapezoid

N_FOURIER = 100                   # MOD: numero di modi di Fourier nell'header del file di flusso


def read_flow(path):
    """MOD: legge un file (t, Q) saltando l'eventuale header svMultiPhysics "<n_punti> <n_modi>".
    Prima l'header veniva letto come prima riga di dati (t=400, Q=100) senza errori."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().split()
    skip = 1 if (len(first) == 2 and all(tok.isdigit() for tok in first)) else 0
    data = np.loadtxt(path, skiprows=skip)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Expected at least 2 columns in {path}")
    return data[:, 0].astype(float), data[:, 1].astype(float)


# Scaling waveform (durata target + media target) dal modulo esterno; fallback se assente.
try:
    from make_inflow_reference import scale_waveform_to_1s_and_mean
    _SCALER = "make_inflow_reference (modulo esterno)"
except ImportError:
    _SCALER = "fallback interno"

    def scale_waveform_to_1s_and_mean(in_path: str, out_path: str,
                                      target_duration: float = 0.8,
                                      target_mean_flow: float = -38.3333):
        t, q = read_flow(in_path)                             # MOD: salta l'header se presente
        if np.any(np.diff(t) <= 0):
            idx = np.argsort(t)
            t, q = t[idx], q[idx]
        t = t - t[0]
        duration_old = t[-1] - t[0]
        q_mean_old = _trapz(q, t) / duration_old
        q_scaled = q * (target_mean_flow / q_mean_old)
        t_scaled = t * (target_duration / duration_old)
        t_scaled[0], t_scaled[-1] = 0.0, float(target_duration)
        np.savetxt(out_path, np.column_stack([t_scaled, q_scaled]), fmt="%.10g")


# etichetta del cap (nome file SimVascular) -> ruolo anatomico (gruppi di Les)
NAME2ROLE = {
    'ca':   'celiac',                          # tripode celiaco
    'sma':  'sma',                             # arteria mesenterica superiore
    'ardx': 'renal', 'arsx': 'renal',          # renali destra/sinistra
    'iedx': 'iliac_ext', 'iesx': 'iliac_ext',  # iliache esterne
    'iidx': 'iliac_int', 'iisx': 'iliac_int',  # iliache interne
}
# nomi non in NAME2ROLE riconosciuti per prefisso (es. 'ardx2', 'ar_acc' = renale accessoria)
PREFIX2ROLE = [('ar', 'renal'), ('ie', 'iliac_ext'), ('ii', 'iliac_int')]

# configurazioni predefinite generate con --configs (un file rcrparams_<nome>.txt ciascuna)
CONFIG_PRESETS = {
    # split di letteratura puro: IR/SC, 33/22.3/44.6 tra i rami superiori, renali 50/50, iliache 70/30
    "les":     dict(upper_split="les", iliac_split="les", within_group="equal", dp_max=None),
    # come les, con tetto sulla caduta stimata (5 mmHg): limita solo i rami troppo stretti
    "lescap":  dict(upper_split="les", iliac_split="les", within_group="equal", dp_max=5.0),
    # come lescap, ma dentro ogni gruppo il flusso si divide per area della cap
    "lesarea": dict(upper_split="les", iliac_split="les", within_group="area", dp_max=5.0),
}

UPPER_ROLES = ('celiac', 'sma', 'renal')       # rami superiori (quota UBVF = 1 - IR/SC)
ILIAC_ROLES = ('iliac_ext', 'iliac_int')       # rami infrarenali (quota IR/SC)
RHO = 1.06                                     # densita' sangue [g/cm^3]
MMHG = 1333.22                                 # dyn/cm^2 per mmHg


def role_of(name):
    """Ruolo anatomico dal nome del cap: prima match esatto, poi per prefisso."""
    if name in NAME2ROLE:
        return NAME2ROLE[name]
    for pre, role in PREFIX2ROLE:
        if name is not None and name.startswith(pre):
            return role
    return None


def triangle_area(polydata: vtk.vtkPolyData, cell_id: int) -> float:
    """Compute area of a single triangle cell by its cell_id."""
    cell = polydata.GetCell(cell_id)
    if cell.GetNumberOfPoints() != 3:
        return 0.0
    pts = cell.GetPoints()
    return vtk.vtkTriangle.TriangleArea(pts.GetPoint(0), pts.GetPoint(1), pts.GetPoint(2))


def parse_id_list(s: str):
    """Parse '1,2,5' -> {1,2,5}. Empty/None -> empty set."""
    if s is None or str(s).strip() == "":
        return set()
    return {int(x.strip()) for x in str(s).split(",") if x.strip()}


def map_from_mdl(mdl_path):
    """MOD: {face_id -> nome} e {face_id -> tipo} dal file .mdl di SimVascular (cartella Models).
    I nomi sono quelli dati alle facce in SimVascular, gli stessi dei file di mesh-surfaces."""
    import re
    import xml.etree.ElementTree as ET
    with open(mdl_path, "r", encoding="utf-8") as fh:
        txt = fh.read()
    # i .mdl di SimVascular hanno due elementi radice (<format/> e <model>): li avvolgo in un'unica radice
    txt = re.sub(r"<\?xml[^>]*\?>", "", txt)
    root = ET.fromstring(f"<root>{txt}</root>")
    id2name, id2type = {}, {}
    for f in root.iter("face"):
        if f.get("id") is None or f.get("name") is None:
            continue
        fid = int(f.get("id"))
        id2name[fid] = f.get("name").strip().lower()
        if f.get("type") is not None:
            id2type[fid] = f.get("type").strip().lower()
    if not id2name:
        raise RuntimeError(f"Nessuna faccia con id e name trovata in {mdl_path}")
    return id2name, id2type


def map_from_mesh_surfaces(mesh_surfaces_dir, face_id_array_name="ModelFaceID"):
    """{face_id -> nome cap} leggendo i .vtp di mesh-surfaces (nome file = nome cap)."""
    id2name = {}
    files = glob.glob(os.path.join(mesh_surfaces_dir, "*.vtp"))
    if not files:
        raise RuntimeError(f"Nessun .vtp trovato in {mesh_surfaces_dir}")
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0].strip().lower()
        r = vtk.vtkXMLPolyDataReader(); r.SetFileName(path); r.Update()
        arr = r.GetOutput().GetCellData().GetArray(face_id_array_name)
        if arr is None:
            continue
        ids = {int(arr.GetTuple1(i)) for i in range(arr.GetNumberOfTuples())}
        if len(ids) == 1:
            id2name[ids.pop()] = name
    return id2name


# ============================== split compatibile con la geometria ==============================
def pulsatility_factor(waveform_path):
    """<Q^2> / <Q>^2 dell'onda d'ingresso (per stimare <v^2> = pf * v_media^2 a ogni outlet,
    assumendo che ogni outlet riceva un'onda di forma simile a quella d'ingresso)."""
    t, q = read_flow(waveform_path)                           # MOD: header gestito da read_flow
    T = t[-1] - t[0]
    return float((_trapz(q ** 2, t) / T) / (_trapz(q, t) / T) ** 2)


def dp_est_mmHg(q, A, pf):
    """Caduta di pressione convettiva stimata [mmHg] per un outlet: K * 1/2 rho <v^2>, con K = 1,
    v = q/A [cm/s], <v^2> = pf * v^2. Stima d'ordine di grandezza: ignora la perdita viscosa e
    un eventuale restringimento a monte della cap (es. stenosi ostiale)."""
    v = q / A
    return 0.5 * RHO * pf * v ** 2 / MMHG


def vmax_from_dp(dp_max_mmHg, pf):
    """Velocita' media massima per cui dp_est_mmHg <= dp_max_mmHg."""
    return float(np.sqrt(2.0 * dp_max_mmHg * MMHG / (RHO * pf)))


def within_weights(fids, A, mode):
    """Pesi per dividere la portata di un gruppo tra i suoi vasi.
    equal: parti uguali; area: proporzionale all'area (regola di Les per le renali accessorie);
    murray: proporzionale a r^3 = (A/pi)^1.5."""
    if mode == "equal":
        w = {f: 1.0 for f in fids}
    elif mode == "area":
        w = {f: A[f] for f in fids}
    elif mode == "murray":
        w = {f: (A[f] / np.pi) ** 1.5 for f in fids}
    else:
        raise ValueError(f"within-group mode sconosciuto: {mode}")
    s = sum(w.values())
    return {f: w[f] / s for f in fids}


def cap_velocity(q, A, fids, vmax):
    """Tetto alla velocita' media dentro un livello (rami superiori o iliache), conservandone la portata
    totale: gli outlet oltre vmax scendono a vmax*A, l'eccesso va agli altri outlet dello stesso livello
    in proporzione alla loro portata. Si ripete finche' nessuno supera il tetto (water-filling)."""
    q = dict(q); capped = set()
    for _ in range(len(fids)):
        over = [f for f in fids if f not in capped and q[f] > vmax * A[f] * (1 + 1e-9)]
        if not over:
            break
        excess = sum(q[f] - vmax * A[f] for f in over)
        for f in over:
            q[f] = vmax * A[f]; capped.add(f)
        free = [f for f in fids if f not in capped]
        if not free:
            raise RuntimeError("Tetto di velocita' non soddisfacibile: tutti gli outlet del livello sono "
                               "oltre vmax. Aumentare --dp-max o controllare le aree delle cap.")
        tot = sum(q[f] for f in free)
        for f in free:
            q[f] += excess * q[f] / tot
    return q, capped


def compute_split(Q_sc, A, outlet_role, ir_sc, upper_split="les", within="equal", dp_max=None, pf=2.0,
                  iliac_split="les"):
    """Portata media target per outlet [mL/s].
    - IR/SC fisso: iliache = ir_sc * Q_sc, rami superiori = (1 - ir_sc) * Q_sc.
    - Rami superiori: 'les' = 33 / 22.3 / 44.6 % di UBVF per gruppo, poi divisione interna secondo
      `within`; 'murray'/'area' = tutta la UBVF divisa tra i rami superiori per r^3 / per area.
    - Iliache: 'les' = 70 % esterne / 30 % interne del flusso IR, poi divisione interna secondo `within`;
      'murray'/'area' = tutto il flusso IR diviso tra le iliache per r^3 / per area.
    - dp_max [mmHg] (opzionale): tetto di velocita' per livello, vedi cap_velocity.
    Ritorna (q, capped, vmax)."""
    u, r = 1.0 - ir_sc, ir_sc
    groups = defaultdict(list)
    for f, role in outlet_role.items():
        groups[role].append(f)

    q = {}
    upper = [f for f, role in outlet_role.items() if role in UPPER_ROLES]
    iliac = [f for f, role in outlet_role.items() if role in ILIAC_ROLES]
    if upper_split == "les":
        GROUP_UPPER = {'celiac': 0.330, 'sma': 0.223, 'renal': 0.446}
        present = {g: v for g, v in GROUP_UPPER.items() if groups.get(g)}
        norm = sum(present.values())                # se manca un gruppo, rinormalizza sugli altri
        for g, frac in present.items():
            for f, w in within_weights(groups[g], A, within).items():
                q[f] = Q_sc * u * (frac / norm) * w
    elif upper_split in ("murray", "area"):
        for f, w in within_weights(upper, A, upper_split).items():
            q[f] = Q_sc * u * w
    else:
        raise ValueError(f"upper-split sconosciuto: {upper_split}")

    if iliac_split == "les":
        GROUP_ILIAC = {'iliac_ext': 0.700, 'iliac_int': 0.300}
        present = {g: v for g, v in GROUP_ILIAC.items() if groups.get(g)}
        norm = sum(present.values())
        for g, frac in present.items():
            for f, w in within_weights(groups[g], A, within).items():
                q[f] = Q_sc * r * (frac / norm) * w
    elif iliac_split in ("murray", "area"):
        for f, w in within_weights(iliac, A, iliac_split).items():
            q[f] = Q_sc * r * w
    else:
        raise ValueError(f"iliac-split sconosciuto: {iliac_split}")

    capped, vmax = set(), None
    if dp_max is not None:
        vmax = vmax_from_dp(dp_max, pf)
        for level in (upper, iliac):                # il tetto non sposta flusso tra livelli: IR/SC resta fisso
            if level:
                q, c = cap_velocity(q, A, level, vmax)
                capped |= c
    return q, capped, vmax
# =========================================================================================================


def main(
    vtp_path: str,
    out_path: str,
    face_id_array_name: str = "ModelFaceID",
    inlet_id: int = None,
    wall_ids=None,
    map_val: float = None,
    tot_comp: float = None,
    area_to_cm2: float = 1.0,
    res1_split: float = 0.15,
    inflow_file: str = "inflow_3d.flow",
    target_duration: float = 0.8,
    mesh_surfaces_dir: str = None,
    ir_sc: float = 0.343,
    renal_rp: float = None,
    upper_split: str = "les",
    within_group: str = "equal",
    dp_max: float = None,
    iliac_split: str = "les",
    configs: list = None,
    distal_pressure: float = 0.0,      # MOD: P_d delle RCR (dyn/cm^2), serve per Initial_pressure
    init_pressure_value: float = None, # MOD: pressione uniforme del fluido a t=0; None = MAP
    volume_mesh: str = None,           # MOD: mesh volumetrica; se None init_pressure.vtu non viene generato
    init_pressure_out: str = None,     # MOD: None = cartella che contiene mesh-complete/
    mdl_path: str = None,              # MOD: .mdl con i nomi delle facce; None = stesso nome del .vtp
    init_pressure_hint: bool = True,   # pipeline: False, init_pressure.vtu is written by prepare_sim
):
    # MOD: nomi delle facce dal .mdl (cartella Models), mesh-surfaces solo come alternativa
    if mdl_path is None:
        cand = os.path.splitext(vtp_path)[0] + ".mdl"
        mdl_path = cand if os.path.isfile(cand) else None
    id2type = {}
    if mdl_path is not None:
        id2name, id2type = map_from_mdl(mdl_path)
        print(f"Nomi delle facce da: {mdl_path}")
    elif mesh_surfaces_dir is not None:
        id2name = map_from_mesh_surfaces(mesh_surfaces_dir, face_id_array_name)
        print(f"Nomi delle facce da: {mesh_surfaces_dir}")
    else:
        raise RuntimeError("Servono i nomi delle facce: nessun .mdl accanto al .vtp. "
                           "Usare --mdl oppure --mesh-surfaces.")
    print("id -> nome:", dict(sorted(id2name.items())))

    # MOD: inlet dal nome 'inlet' se --inlet-id non e' dato
    if inlet_id is None:
        ids = [f for f, n in id2name.items() if n == "inlet"]
        if len(ids) != 1:
            raise RuntimeError(f"{len(ids)} facce chiamate 'inlet': usare --inlet-id.")
        inlet_id = ids[0]
        print(f"inlet_id dal nome 'inlet': {inlet_id}")
    # MOD: pareti dal tipo 'wall' del .mdl se --wall-ids non e' dato
    if not wall_ids and id2type:
        wall_ids = {f for f, t in id2type.items() if t == "wall"}
        print(f"wall_ids dal tipo 'wall' nel .mdl: {sorted(wall_ids)}")

    if map_val is None or tot_comp is None:                   # MOD: controllo spostato prima dell'uso
        raise RuntimeError("map_val and tot_comp must be provided.")
    if not (0.0 <= res1_split <= 1.0):                        # MOD: controllato una volta sola, all'inizio
        raise ValueError("res1_split must be in [0, 1].")
    if renal_rp is not None and not (0.0 <= renal_rp <= 1.0):
        raise ValueError("renal_rp must be in [0, 1].")
    wall_ids = set(wall_ids) if wall_ids is not None else set()

    # Reference scaling constants (Cheng): SC = 2.3 L/min = 38.3333 mL/s a 3.6 cm^2
    Aref_cm2 = 3.6
    Qref_ml_s = 38.3333

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfCells() == 0:
        raise RuntimeError(f"No cells found in: {vtp_path}")

    face_id_arr = poly.GetCellData().GetArray(face_id_array_name)
    if face_id_arr is None:
        raise RuntimeError(
            f"Cell data array '{face_id_array_name}' not found.\n"
            f"Available cell arrays: {[poly.GetCellData().GetArrayName(i) for i in range(poly.GetCellData().GetNumberOfArrays())]}"
        )

    areas = defaultdict(float)
    for cid in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(cid)
        if cell.GetCellType() == vtk.VTK_TRIANGLE and cell.GetNumberOfPoints() == 3:
            fid = int(face_id_arr.GetTuple1(cid))
            areas[fid] += triangle_area(poly, cid)

    if inlet_id not in areas:
        raise RuntimeError(f"inlet_id={inlet_id} not found among face IDs: {sorted(areas.keys())}")
    missing_walls = [wid for wid in wall_ids if wid not in areas]
    if missing_walls:
        print(f"Warning: wall_ids not found in mesh (ignored): {missing_walls}")

    AREA_cm2 = areas[inlet_id] * float(area_to_cm2)
    Q_to_distribute = Qref_ml_s * (AREA_cm2 / Aref_cm2)

    if configs is not None:                       # con --configs out_path e' la cartella di output
        out_dir = os.path.abspath(out_path)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    inflow_in_path = inflow_file
    if not os.path.isabs(inflow_in_path):
        candidate = os.path.join(out_dir, inflow_file)
        if os.path.isfile(candidate):
            inflow_in_path = candidate
    inflow_out_path = os.path.join(out_dir, "inflow.reference.scaled")

    print(f"\n--- Riscalamento waveform [{_SCALER}] da {inflow_in_path} (T={target_duration}s) ---")
    scale_waveform_to_1s_and_mean(
        in_path=inflow_in_path,
        out_path=inflow_out_path,
        target_duration=target_duration,
        target_mean_flow=-1.0 * Q_to_distribute,
    )
    # MOD: header "<n_punti> <n_modi>" calcolato dal file (prima era fisso a "400 100" e, se il modulo
    # esterno scriveva gia' un header diverso, ne veniva aggiunto un secondo)
    t_in, q_in = read_flow(inflow_out_path)
    with open(inflow_out_path, "w", encoding="utf-8") as f:
        f.write(f"{len(t_in)} {N_FOURIER}\n")
        np.savetxt(f, np.column_stack([t_in, q_in]), fmt="%.10g")

    # MOD: controllo che il file scritto abbia davvero media -Q_SC e periodo target
    q_mean_file = _trapz(q_in, t_in) / (t_in[-1] - t_in[0])
    err_mean = abs(q_mean_file + Q_to_distribute) / Q_to_distribute
    print(f"Check waveform: {len(t_in)} punti, t = [{t_in[0]:.4g}, {t_in[-1]:.4g}] s, "
          f"media = {q_mean_file:.6g} mL/s (target {-Q_to_distribute:.6g}, scarto {100*err_mean:.3f}%)")
    if err_mean > 1e-3:
        raise RuntimeError("La media della waveform scalata non coincide con -Q_SC: controllare lo scaler.")
    if abs(t_in[0]) > 1e-9 or abs(t_in[-1] - target_duration) > 1e-6:
        raise RuntimeError(f"Periodo della waveform scalata diverso da [0, {target_duration}] s.")
    print(f"--- Scalamento completato: {inflow_out_path} ---\n")

    tot_comp = tot_comp * (AREA_cm2 / Aref_cm2)

    print(f"Inlet face_id: {inlet_id}")
    print(f"Inlet AREA (cm^2): {AREA_cm2:.6g}  (area_to_cm2={area_to_cm2})")
    print(f"Q_SC (mL/s): {Q_to_distribute:.6g}  (Qref={Qref_ml_s}, Aref={Aref_cm2})")

    outlets_excluded = {inlet_id} | (wall_ids & set(areas.keys()))
    outlet_face_ids = [fid for fid in areas.keys() if fid not in outlets_excluded]
    if not outlet_face_ids:
        raise RuntimeError("No outlet faces found after excluding inlet and wall ids.")

    mancanti = [f for f in outlet_face_ids if f not in id2name]
    if mancanti:
        print("outlet senza nome:", mancanti)
    outlet_role = {}
    for fid in outlet_face_ids:
        name = id2name.get(fid)
        role = role_of(name)
        if role is None:
            raise RuntimeError(f"Outlet face_id={fid} nome='{name}' non mappato a un ruolo.")
        outlet_role[fid] = role

    A = {fid: areas[fid] * float(area_to_cm2) for fid in outlet_face_ids}   # aree delle cap [cm^2]
    pf = pulsatility_factor(inflow_out_path)

    def write_bc(out_file, upper_split, iliac_split, within_group, dp_max):
        """Split, tabella di controllo (<out_file>_split.csv), tabella RCR e blocchi XML per UNA configurazione."""
        flow_distribution, capped, vmax = compute_split(
            Q_to_distribute, A, outlet_role, ir_sc,
            upper_split=upper_split, within=within_group, dp_max=dp_max, pf=pf, iliac_split=iliac_split)

        q_sum = sum(flow_distribution.values())
        print(f"Check conservazione split: somma outlet {q_sum:.6g} vs Q_SC {Q_to_distribute:.6g} "
              f"(ratio {q_sum/Q_to_distribute:.4f})")
        q_ir = sum(q for f, q in flow_distribution.items() if outlet_role[f] in ILIAC_ROLES)
        print(f"IR/SC realizzato nello split: {q_ir/Q_to_distribute:.4f} (target {ir_sc})")
        print(f"Split: upper={upper_split}, iliac={iliac_split}, within={within_group}, dp_max={dp_max} mmHg, "
              f"pulsatilita' <Q^2>/<Q>^2 = {pf:.2f}" + (f", v_max = {vmax:.1f} cm/s" if vmax else ""))

        resistances = {fid: float(map_val) / q for fid, q in flow_distribution.items()}
        comp_distribution = {fid: float(tot_comp) * q / Q_to_distribute
                             for fid, q in flow_distribution.items()}
        rp_frac = {fid: (renal_rp if (renal_rp is not None and outlet_role[fid] == 'renal') else res1_split)
                   for fid in flow_distribution}

        # MOD: pressione media sul condensatore a regime = Initial_pressure della RCR.
        # A regime, in media: P_out = q (R1 + R2) + P_d = MAP  e  P_c = q R2 + P_d
        #   =>  P_c = (MAP - P_d) * R2 / (R1 + R2) + P_d = (MAP - P_d) * (1 - rp) + P_d
        # Con rp uniforme vale 0.85 MAP per tutte le outlet; con --renal-rp le renali hanno un valore diverso.
        p_init = {fid: (float(map_val) - distal_pressure) * (1.0 - rp_frac[fid]) + distal_pressure
                  for fid in flow_distribution}

        rows = []
        for fid in sorted(flow_distribution):
            q = flow_distribution[fid]
            d_mm = 20.0 * np.sqrt(A[fid] / np.pi)                 # diametro equivalente [mm]
            rows.append((fid, id2name.get(fid), outlet_role[fid], A[fid], d_mm, q, 100 * q / Q_to_distribute,
                         q / A[fid], dp_est_mmHg(q, A[fid], pf), p_init[fid], fid in capped))
        print("\nFlow distribution:")
        print(f"{'Face':>5} {'nome':>8} {'ruolo':>10} {'A[cm2]':>8} {'d[mm]':>6} {'Q[mL/s]':>9} {'%SC':>6} "
              f"{'v[cm/s]':>8} {'dP~[mmHg]':>10} {'R1':>11} {'C':>11} {'R2':>11} {'P_init':>10} tetto")
        for (fid, nm, role, a, d_mm, q, pct, v, dp, pi_, cp) in rows:
            R = resistances[fid]; rp = rp_frac[fid]
            # MOD: colonne nell'ordine del solver.xml (R1 C R2), non piu' R1 R2 C (origine dello swap di pz101)
            print(f"{fid:>5} {str(nm):>8} {role:>10} {a:>8.4f} {d_mm:>6.1f} {q:>9.3f} {pct:>6.1f} {v:>8.1f} "
                  f"{dp:>10.2f} {R*rp:>11.5g} {comp_distribution[fid]:>11.4g} {R*(1-rp):>11.5g} "
                  f"{pi_:>10.6g} {'si' if cp else ''}")
        split_csv = os.path.splitext(out_file)[0] + "_split.csv"
        with open(split_csv, "w", encoding="utf-8") as f:
            f.write("face_id,nome,ruolo,area_cm2,d_eq_mm,q_ml_s,pct_SC,v_media_cm_s,dP_stimata_mmHg,"
                    "P_init_dyn_cm2,tetto\n")
            for r_ in rows:
                f.write(",".join(str(x) for x in r_[:3]) + "," +
                        ",".join(f"{x:.6g}" for x in r_[3:10]) + f",{int(r_[10])}\n")
        print(f"Tabella di controllo: {split_csv}")

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("# face_id area R1 C R2\n")
            for fid in sorted(areas.keys()):
                if fid in flow_distribution:
                    R = resistances[fid]; rp = rp_frac[fid]
                    R1, R2, C = R * rp, R * (1 - rp), comp_distribution[fid]
                else:
                    R1 = R2 = C = 0.0
                f.write(f"{fid} {areas[fid]:.12g} {R1:.12g} {C:.12g} {R2:.12g}\n")
        print(f"\nWrote BC file: {out_file}")

        # MOD: blocchi <RCR_values> pronti da incollare nel solver.xml, gia' nell'ordine giusto e con
        # Initial_pressure: niente piu' copia a mano dei valori dal terminale.
        xml_file = os.path.splitext(out_file)[0] + "_solverxml.txt"
        with open(xml_file, "w", encoding="utf-8") as f:
            f.write(f"<!-- {os.path.basename(out_file)} | A_in = {AREA_cm2:.6g} cm^2 | "
                    f"Q_SC = {Q_to_distribute:.6g} mL/s | MAP = {map_val} dyn/cm^2 -->\n")
            for fid in sorted(flow_distribution):
                R = resistances[fid]; rp = rp_frac[fid]
                f.write(f"\n<!-- face_id {fid} : {id2name.get(fid)} ({outlet_role[fid]}) -->\n"
                        f"<RCR_values>\n"
                        f"  <Proximal_resistance> {R*rp:.10g} </Proximal_resistance>\n"
                        f"  <Capacitance> {comp_distribution[fid]:.10g} </Capacitance>\n"
                        f"  <Distal_resistance> {R*(1-rp):.10g} </Distal_resistance>\n"
                        f"  <Distal_pressure> {distal_pressure:.10g} </Distal_pressure>\n"
                        f"  <Initial_pressure> {p_init[fid]:.10g} </Initial_pressure>\n"
                        f"</RCR_values>\n")
        print(f"Blocchi RCR per solver.xml: {xml_file}")

    if configs is None:
        write_bc(out_path, upper_split, iliac_split, within_group, dp_max)
    else:
        for name in configs:
            if name not in CONFIG_PRESETS:
                raise ValueError(f"Configurazione sconosciuta: {name}. Disponibili: {', '.join(CONFIG_PRESETS)}")
            cfg = dict(CONFIG_PRESETS[name])
            if cfg["dp_max"] is not None and dp_max is not None:
                cfg["dp_max"] = dp_max                 # --dp-max sovrascrive il tetto dei preset che lo usano
            print(f"\n==================== configurazione: {name} ====================")
            write_bc(os.path.join(out_path, f"rcrparams_{name}.txt"), **cfg)

    # MOD: campo iniziale di pressione del fluido (stessa mesh volumetrica della simulazione)
    # MOD: solo se la mesh volumetrica viene indicata (per sim nuove la mesh non esiste ancora)
    p0 = float(map_val) if init_pressure_value is None else float(init_pressure_value)
    if volume_mesh is None and not init_pressure_hint:
        pass
    elif volume_mesh is None:
        print(f"\nPressione iniziale del fluido: nessuna --volume-mesh, init_pressure.vtu non generato. "
              f"Dopo la mesh: python3 make_init_pressure.py mesh-complete/mesh-complete.mesh.vtu "
              f"init_pressure.vtu {p0}")
    else:
        from utils.init_pressure import write_init_pressure
        vol = os.path.abspath(volume_mesh)
        if not os.path.isfile(vol):
            raise RuntimeError(f"Mesh volumetrica non trovata: {vol}")
        sim_dir = os.path.dirname(os.path.dirname(vol))              # cartella che contiene mesh-complete/
        out_vtu = init_pressure_out or os.path.join(sim_dir, "init_pressure.vtu")
        print(f"\n--- Pressione iniziale del fluido: {p0} dyn/cm^2 su {vol} ---")
        write_init_pressure(vol, out_vtu, p0)
        print(f"Nel solver.xml, dentro <Add_mesh>: <Initial_pressures_file_path> "
              f"{os.path.relpath(out_vtu, sim_dir)} </Initial_pressures_file_path>")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="RCR boundary conditions con flow split fisiologico (Les/Moore&Ku), "
                    "Initial_pressure delle RCR e campo iniziale di pressione (init_pressure.vtu)."
    )
    p.add_argument("vtp", help="mesh di superficie (.vtp) con ModelFaceID")
    p.add_argument("out", help="file di output della tabella RCR; con --configs, la cartella di output")
    p.add_argument("--map", type=float, required=True, dest="map_val",
                   help="pressione arteriosa media MAP (dyn/cm^2)")
    p.add_argument("--tot-comp", type=float, required=True,
                   help="compliance totale da distribuire")
    p.add_argument("--inlet-id", type=int, default=None,
                   help="ModelFaceID dell'inlet (sopraceliaco). Default: la faccia chiamata 'inlet' nel .mdl")
    p.add_argument("--mdl", default=None,
                   help="file .mdl con i nomi delle facce. Default: stesso nome del .vtp (cartella Models)")
    p.add_argument("--mesh-surfaces", default=None,
                   help="alternativa al .mdl: cartella mesh-surfaces (nome file = nome vaso)")
    p.add_argument("--wall-ids", default="",
                   help="id facce wall separati da virgola (es. 1 o 1,2). Default: facce di tipo 'wall' nel .mdl")
    p.add_argument("--face-id-array", default="ModelFaceID",
                   help="nome array cell-data con gli id faccia. Default: ModelFaceID")
    p.add_argument("--area-to-cm2", type=float, default=1.0,
                   help="fattore area -> cm^2. Mesh in mm: 0.01. Default: 1.0")
    p.add_argument("--res1-split", type=float, default=0.15,
                   help="frazione Rp/Rtot uniforme. Default: 0.15")
    p.add_argument("--ir-sc", type=float, default=0.343,
                   help="rapporto IR/SC (0.343 Les CVET; 0.373; 0.391). Default: 0.343")
    p.add_argument("--renal-rp", type=float, default=None,
                   help="Rp/Rtot per le renali (es. 0.28, Les). Default: come res1-split")
    p.add_argument("--upper-split", choices=["les", "murray", "area"], default="les",
                   help="divisione della UBVF tra i rami superiori. Default: les (33/22.3/44.6)")
    p.add_argument("--iliac-split", choices=["les", "murray", "area"], default="les",
                   help="divisione del flusso IR tra le iliache. Default: les (70/30 est/int)")
    p.add_argument("--within-group", choices=["equal", "area", "murray"], default="equal",
                   help="divisione dentro un gruppo (es. renali dx/sx/accessoria). Default: equal")
    p.add_argument("--dp-max", type=float, default=None,
                   help="tetto sulla caduta convettiva stimata per outlet [mmHg], es. 5 o 10. Default: nessuno")
    p.add_argument("--configs", default=None,
                   help="preset da generare, separati da virgola (es. les,lescap,lesarea). Con questa "
                        "opzione 'out' e' una cartella e i file sono rcrparams_<nome>.txt; i flag di split "
                        "(--upper-split, --iliac-split, --within-group) sono ignorati, --dp-max sovrascrive "
                        "il tetto dei preset che lo usano.")
    p.add_argument("--target-duration", type=float, default=0.8,
                   help="durata ciclo per lo scaling waveform (s). Default: 0.8")
    p.add_argument("--inflow-file", default="inflow_3d.flow",
                   help="waveform (t,Q) da riscalare. Default: inflow_3d.flow")
    # MOD: opzioni per la pressione iniziale
    p.add_argument("--distal-pressure", type=float, default=0.0,
                   help="Distal_pressure delle RCR (dyn/cm^2). Default: 0")
    p.add_argument("--init-pressure-value", type=float, default=None,
                   help="pressione uniforme del fluido a t=0 (dyn/cm^2). Default: MAP")
    p.add_argument("--volume-mesh", default=None,
                   help="mesh volumetrica (mesh-complete.mesh.vtu); se data genera anche init_pressure.vtu")
    p.add_argument("--init-pressure-out", default=None,
                   help="file di output. Default: <cartella che contiene mesh-complete>/init_pressure.vtu")
    a = p.parse_args()

    main(
        vtp_path=a.vtp,
        out_path=a.out,
        face_id_array_name=a.face_id_array,
        inlet_id=a.inlet_id,
        wall_ids=parse_id_list(a.wall_ids),
        map_val=a.map_val,
        tot_comp=a.tot_comp,
        area_to_cm2=a.area_to_cm2,
        res1_split=a.res1_split,
        inflow_file=a.inflow_file,
        target_duration=a.target_duration,
        mesh_surfaces_dir=a.mesh_surfaces,
        ir_sc=a.ir_sc,
        renal_rp=a.renal_rp,
        upper_split=a.upper_split,
        within_group=a.within_group,
        dp_max=a.dp_max,
        iliac_split=a.iliac_split,
        configs=[c.strip() for c in a.configs.split(",") if c.strip()] if a.configs else None,
        distal_pressure=a.distal_pressure,
        init_pressure_value=a.init_pressure_value,
        volume_mesh=a.volume_mesh,
        init_pressure_out=a.init_pressure_out,
        mdl_path=a.mdl,
    )