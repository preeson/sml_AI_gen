"""
ioi_sign.py -- visual field sign map, and the V1 patch.

THE METHOD
----------
Cortical visual areas alternate between mirror-image and non-mirror-image
representations of the visual field.  The sign of the mapping is the sine of
the angle between the azimuth and elevation gradients:

    VFS = sin( angle(grad azimuth) - angle(grad elevation) )

V1 appears as one large patch of a single sign, bounded by opposite-sign
higher areas (LM, AL, RL...).  This, not the phase maps themselves, is what
delineates V1.

WHY GATING IS NOT OPTIONAL
--------------------------
VFS is a sine, so it passes through zero wherever the two gradients are
near-parallel -- and there the sign flips on noise alone.  In the M2 data,
27-36% of pixels had gradients within 30 degrees of parallel or antiparallel,
which is exactly what turned the sign map into stripes rather than areas.

So we compute a per-pixel reliability and refuse to assign a sign where it is
low.  A border drawn only where the sign is actually determined is worth more
than a complete one that is partly arbitrary.  Reliability combines:

  * |sin(angle)| itself -- how far from degenerate the gradient pair is
  * gradient magnitude in both axes -- a flat map has no meaningful direction
  * the SNR of the underlying maps

REPORTING
---------
patches() labels contiguous same-sign regions and reports their area in mm^2.
Mouse V1 is roughly 2-3 mm^2; a "V1" that comes out at 0.2 or 15 mm^2 is
telling you something is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, label, binary_opening, binary_closing


def _smooth_masked(x, mask, sigma):
    if sigma is None or sigma <= 0:
        return np.array(x, dtype=float)
    den = gaussian_filter(mask.astype(float), sigma)
    num = gaussian_filter(np.where(mask, np.nan_to_num(x), 0.0), sigma)
    return num / np.where(den > 1e-6, den, np.nan)


@dataclass
class SignMap:
    vfs: np.ndarray          # (h, w) sine of the gradient angle, -1..1
    angle: np.ndarray        # (h, w) angle between gradients, radians
    reliability: np.ndarray  # (h, w) 0..1
    gated: np.ndarray        # (h, w) vfs where reliable, NaN elsewhere
    mask: np.ndarray         # (h, w) bool, pixels with any estimate
    reliable: np.ndarray     # (h, w) bool, pixels passing the gate
    grad_mag_el: np.ndarray
    grad_mag_az: np.ndarray
    um_per_pixel: float


def visual_field_sign(elevation, azimuth, mask, um_per_pixel: float,
                      sigma: float = 3.0, snr=None,
                      min_grad_pct: float = 25.0,
                      min_abs_sin: float = 0.35,
                      snr_min: float = 3.0) -> SignMap:
    """
    elevation, azimuth : position maps in degrees
    mask               : where those maps are valid
    um_per_pixel       : for reporting areas
    sigma              : smoothing before differentiating (pixels).  Gradients
                         amplify noise, so this is larger than the smoothing
                         used for the phase maps themselves.
    min_grad_pct       : gradient magnitude percentile below which a pixel is
                         called flat and rejected
    min_abs_sin        : reject pixels whose |VFS| is below this -- the
                         near-degenerate gradient pairs that cause stripes
    snr, snr_min       : optional extra gate on the underlying map SNR
    """
    e = _smooth_masked(elevation, mask, sigma)
    a = _smooth_masked(azimuth, mask, sigma)

    gey, gex = np.gradient(e)
    gay, gax = np.gradient(a)
    mag_e = np.hypot(gey, gex)
    mag_a = np.hypot(gay, gax)

    ang = np.angle(np.exp(1j * (np.arctan2(gay, gax) - np.arctan2(gey, gex))))
    vfs = np.sin(ang)

    ok = mask & np.isfinite(vfs) & np.isfinite(mag_e) & np.isfinite(mag_a)
    if ok.sum() == 0:
        raise ValueError("no valid pixels for the sign map")
    ge_thr = np.percentile(mag_e[ok], min_grad_pct)
    ga_thr = np.percentile(mag_a[ok], min_grad_pct)

    rel_sin = np.abs(vfs)
    rel_grad = np.minimum(mag_e / max(ge_thr, 1e-12),
                          mag_a / max(ga_thr, 1e-12))
    rel_grad = np.clip(rel_grad, 0, 1)
    reliability = np.where(ok, rel_sin * rel_grad, np.nan)

    reliable = ok & (np.abs(vfs) >= min_abs_sin) & \
        (mag_e >= ge_thr) & (mag_a >= ga_thr)
    if snr is not None:
        reliable &= (snr >= snr_min)

    return SignMap(vfs=np.where(ok, vfs, np.nan), angle=np.where(ok, ang, np.nan),
                   reliability=reliability,
                   gated=np.where(reliable, vfs, np.nan),
                   mask=ok, reliable=reliable,
                   grad_mag_el=mag_e, grad_mag_az=mag_a,
                   um_per_pixel=um_per_pixel)


@dataclass
class Patch:
    index: int
    sign: int
    n_pixels: int
    area_mm2: float
    centroid: tuple
    mask: np.ndarray


def patches(sm: SignMap, min_area_mm2: float = 0.15,
            open_iter: int = 1, close_iter: int = 2):
    """Label contiguous same-sign reliable regions, largest first."""
    px_mm2 = (sm.um_per_pixel / 1000.0) ** 2
    out = []
    for sign in (+1, -1):
        b = sm.reliable & (np.sign(sm.vfs) == sign)
        if open_iter:
            b = binary_opening(b, iterations=open_iter)
        if close_iter:
            b = binary_closing(b, iterations=close_iter)
        lab, n = label(b)
        for i in range(1, n + 1):
            m = lab == i
            area = m.sum() * px_mm2
            if area < min_area_mm2:
                continue
            ys, xs = np.nonzero(m)
            out.append(Patch(index=len(out), sign=sign, n_pixels=int(m.sum()),
                             area_mm2=float(area),
                             centroid=(float(ys.mean()), float(xs.mean())),
                             mask=m))
    out.sort(key=lambda p: -p.area_mm2)
    for i, p in enumerate(out):
        p.index = i
    return out


def report(sm: SignMap, ps) -> str:
    lines = [
        f"  pixels with a sign estimate : {int(sm.mask.sum())}",
        f"  passing the reliability gate: {int(sm.reliable.sum())} "
        f"({100*sm.reliable.sum()/max(sm.mask.sum(),1):.0f}%)",
        f"  median |VFS| where reliable : "
        f"{np.nanmedian(np.abs(sm.vfs[sm.reliable])):.2f}",
        f"  near-degenerate gradients   : "
        f"{100*np.nanmean(np.abs(sm.vfs[sm.mask]) < 0.35):.0f}% of estimated px",
        "",
        f"  {'patch':>6}{'sign':>6}{'area mm2':>11}{'centroid (row, col)':>22}",
    ]
    for p in ps[:8]:
        lines.append(f"  {p.index:>6}{p.sign:>+6d}{p.area_mm2:>11.2f}"
                     f"{f'({p.centroid[0]:.0f}, {p.centroid[1]:.0f})':>22}")
    if ps:
        lines += ["",
                  f"  largest patch: {ps[0].area_mm2:.2f} mm2, sign "
                  f"{ps[0].sign:+d}  (mouse V1 is typically 2-3 mm2)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Averaging repeat recordings
# --------------------------------------------------------------------------

def estimate_shift(ref_img, img, max_shift: int = 12):
    """
    Integer (row, col) shift to APPLY to `img` so that it matches `ref_img`,
    i.e. np.roll(img, shift) aligns with the reference.  Brute force over a
    small window.  Repeat recordings of the same animal are
    usually within a few pixels, but averaging misaligned maps would blur
    exactly the borders we are trying to find, so this is checked rather than
    assumed.
    """
    a = ref_img - ref_img.mean()
    best, best_r = (0, 0), -np.inf
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            b = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            b = b - b.mean()
            sl = (slice(max_shift, -max_shift), slice(max_shift, -max_shift))
            r = float((a[sl] * b[sl]).sum() /
                      np.sqrt((a[sl] ** 2).sum() * (b[sl] ** 2).sum()))
            if r > best_r:
                best_r, best = r, (dy, dx)
    return best, best_r


def align_phase(comps, mask):
    """
    Remove the constant phase offset between repeat recordings.

    Each recording has its own stimulus onset relative to its camera clock, and
    its own overall haemodynamic lag, so its Fourier components are rotated by
    some constant angle relative to another recording's.  That constant cancels
    inside a recording (it is common to forward and reverse, so the position
    estimate is immune -- which is why cross-trial POSITION agreement can be
    excellent while the raw components disagree).  But it does NOT cancel when
    averaging components across recordings: rotate one by 90 degrees and the
    average of two perfectly reproducible recordings is smaller than either.

    So estimate the rotation against the first recording and undo it before
    averaging.  Returns (rotated_comps, angles_deg, coherence) where coherence
    is |<exp(i.dphi)>| over the mask -- near 1 means the two recordings really
    do differ by a single constant, which is the assumption being made.
    """
    comps = [np.asarray(c) for c in comps]
    ref = comps[0]
    out, angles, coh = [ref], [0.0], [1.0]
    for c in comps[1:]:
        z = c * np.conj(ref)
        z = z[mask & np.isfinite(z) & (np.abs(z) > 0)]
        u = z / np.abs(z)
        mu = u.mean()
        ang = np.angle(mu)
        out.append(c * np.exp(-1j * ang))
        angles.append(float(np.degrees(ang)))
        coh.append(float(np.abs(mu)))
    return out, angles, coh


def average_components(comps, weights=None, mask=None, align=True):
    """
    Average complex Fourier components across repeat recordings.

    Averaging the COMPLEX values (not the amplitudes) is what buys SNR: noise
    adds incoherently while a reproducible signal adds coherently.  It is also
    a test -- once the constant rotation is removed (see align_phase), an
    average SMALLER than the individual recordings means they genuinely
    disagree.

    Set align=False only if the components are already on a common phase.
    """
    comps = [np.asarray(c) for c in comps]
    info = None
    if align and len(comps) > 1:
        if mask is None:
            mask = np.ones(comps[0].shape, dtype=bool)
        comps, angles, coh = align_phase(comps, mask)
        info = (angles, coh)
    if weights is None:
        weights = np.ones(len(comps))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    avg = sum(wi * c for wi, c in zip(w, comps))
    return (avg, info) if info is not None else avg
