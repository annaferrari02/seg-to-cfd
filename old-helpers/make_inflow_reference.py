#!/usr/bin/env python3
import sys
import numpy as np


def scale_waveform_to_1s_and_mean(in_path: str, out_path: str,
                                 target_duration: float = 0.8,
                                 target_mean_flow: float = -38.3333):
    """
    Reads a 2-column file (time, flow). Produces a new waveform with:
      - duration scaled to `target_duration` (time re-mapped linearly)
      - cardiac-cycle mean flow scaled to `target_mean_flow`
    Writes the result to `out_path` as two columns: time, flow.

    Notes:
      - Duration is taken as t[-1] - t[0]
      - Mean flow is computed as (1/duration) * integral(Q dt) via trapezoidal rule
      - Output time starts at 0 and ends at target_duration
    """
    data = np.loadtxt(in_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Expected at least 2 columns (time, flow) in {in_path}")

    t = data[:, 0].astype(float)
    q = data[:, 1].astype(float)

    if len(t) < 2:
        raise RuntimeError(f"Need at least 2 samples in {in_path}")

    # Sort by time if needed
    if np.any(np.diff(t) <= 0):
        idx = np.argsort(t)
        t = t[idx]
        q = q[idx]
        if np.any(np.diff(t) <= 0):
            raise RuntimeError("Time column must be strictly increasing (or sortable to strictly increasing).")

    # Shift time to start at 0
    t0 = t[0]
    t = t - t0

    duration_old = t[-1] - t[0]
    if duration_old <= 0:
        raise RuntimeError("Waveform duration <= 0 (check time column).")

    # Mean flow over original duration
    q_mean_old = np.trapz(q, t) / duration_old
    if q_mean_old == 0:
        raise RuntimeError("Old waveform mean flow is zero; cannot scale.")

    # Flow scaling to match target mean
    flow_scale = target_mean_flow / q_mean_old
    q_scaled = q * flow_scale

    # Time scaling to match target duration
    time_scale = target_duration / duration_old
    t_scaled = t * time_scale

    # Ensure output ends exactly at target_duration (numerical roundoff)
    t_scaled[0] = 0.0
    t_scaled[-1] = float(target_duration)

    # (Optional sanity) recompute mean on scaled time; should be target_mean_flow
    duration_new = t_scaled[-1] - t_scaled[0]
    q_mean_new = np.trapz(q_scaled, t_scaled) / duration_new

    np.savetxt(out_path, np.column_stack([t_scaled, q_scaled]), fmt="%.10g")

    print(f"Read: {in_path}")
    print(f"Original duration: {duration_old:.10g} s")
    print(f"Original mean flow: {q_mean_old:.10g} mL/s")
    print(f"target_mean_flow: {target_mean_flow:.10g} mL/s")
    print(f"Time scale factor: {time_scale:.10g}  -> duration {target_duration} s")
    print(f"Flow scale factor: {flow_scale:.10g}  -> mean {target_mean_flow} mL/s")
    print(f"New mean flow (check): {q_mean_new:.10g} mL/s")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "inflow_3d.flow"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "inflow.reference"

    scale_waveform_to_1s_and_mean(
        in_path=in_path,
        out_path=out_path,
        target_duration=0.8,
        target_mean_flow=-38.3333
    )
