"""CT intensity checks (numpy only, usable from any interpreter)."""
import numpy as np

OFFSETS = (0, 1024)  # standard HU (water 0, air -1000) / unsigned storage with +1024 shift


def hu_offset(ct, air_tol=150.0, min_air_fraction=0.01):
    """Offset of the CT scale w.r.t. standard HU, from where the air is:
    standard -> air near -1000; shifted by +1024 -> air near +24 and nothing below -500
    (padding outside the scan field, e.g. -3024 / -2048, is ignored by the windows).
    Returns (offset, report). Fails if neither scale fits (e.g. a normalized CT)."""
    a = np.asarray(ct).ravel()
    if a.size > 4_000_000:  # subsample: the fractions do not need every voxel
        a = a[:: a.size // 4_000_000]
    n = float(a.size)
    air = {off: np.count_nonzero(np.abs(a - (off - 1000.0)) <= air_tol) / n for off in OFFSETS}
    # soft tissue / blood (-100..200 HU) must also be where the scale puts it
    tissue = {off: np.count_nonzero((a >= off - 100.0) & (a <= off + 200.0)) / n for off in OFFSETS}
    below = np.count_nonzero((a < -500.0) & (a > -1500.0)) / n  # tissue/air range below -500
    report = (f"range [{float(a.min()):.0f}, {float(a.max()):.0f}], air at -1000: {100 * air[0]:.1f}%, "
              f"air at +24: {100 * air[1024]:.1f}%, tissue at 0: {100 * tissue[0]:.1f}%, "
              f"tissue at +1024: {100 * tissue[1024]:.1f}%")
    if air[0] >= min_air_fraction and tissue[0] >= min_air_fraction:
        return 0, report
    if (air[1024] >= min_air_fraction and tissue[1024] >= min_air_fraction
            and below < min_air_fraction / 10):
        return 1024, report
    raise RuntimeError(f"CT is neither standard HU nor HU+1024 ({report}): normalized or wrong CT?")


def check_mean(ct, mask, offset, lo, hi, name):
    """Mean HU (offset removed) inside a segment must lie in [lo, hi]: catches a CT that is not
    aligned with the segmentation, or a different scan."""
    vals = np.asarray(ct)[np.asarray(mask) > 0]
    if vals.size == 0:
        raise RuntimeError(f"Segment {name} is empty on the CT grid (CT not aligned with the segmentation?)")
    mean = float(vals.mean()) - offset
    if not lo <= mean <= hi:
        raise RuntimeError(f"Mean HU in {name} = {mean:.0f}, expected [{lo}, {hi}]: "
                           f"CT not aligned with the segmentation, or not the same scan?")
    return mean