"""
ioi_maps.py -- combine opposite-direction Fourier components into haemodynamic
delay and absolute retinotopic position.

WHY NOT THE TEXTBOOK HALF-DIFFERENCE
------------------------------------
The usual recipe reads position off (phi_F - phi_R)/2.  That only works when
the bar's travel is no wider than the visible screen.  Here it is not: the bar
CENTRE travels 86.9 deg in elevation while the screen spans only 66.8 deg (the
extra 20 deg is the bar walking off each edge), and 119.5 vs 99.4 in azimuth.

Writing u for a pixel's preferred position as a fraction of the bar's travel,

    phi_F / 2pi = (tau/T + u)     mod 1
    phi_R / 2pi = (tau/T + 1 - u) mod 1

the difference gives (2u - 1) mod 1.  Visible screen corresponds to u in
[0.115, 0.885], so 2u - 1 spans 1.54 -- wider than the unit interval, and
everything beyond +-21.7 deg elevation aliases back on itself.  The map still
looks plausible, which is what makes it dangerous.

WHAT WE DO INSTEAD
------------------
The SUM is well behaved: (phi_F + phi_R)/2pi = 2 tau/T (mod 1), so

    tau = T * wrap01((phi_F + phi_R)/2pi) / 2

which is unambiguous as long as tau < T/2 (8.1 s for elevation, 11.2 s for
azimuth).  Haemodynamic delays are 1-3 s, so there is ample headroom.

Then subtract the delay from each direction SEPARATELY and read position off
that direction alone, where one full cycle maps to the full travel with no
ambiguity:

    u = wrap01(phi/2pi - tau/T)
    P = p_first + u * (p_last - p_first)

SMOOTHING THE DELAY IS NOT COSMETIC
-----------------------------------
With a per-pixel tau, P_F and P_R are algebraically the same number and their
agreement proves nothing.  Delay varies slowly and smoothly across cortex
(it tracks vasculature), so smoothing tau before subtracting is physically
right AND makes the two position estimates genuinely independent.  Their
difference is then a free per-pixel validation of the entire pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

TWO_PI = 2.0 * np.pi


def wrap01(x):
    """Wrap to [0, 1)."""
    return np.mod(x, 1.0)


def wrap_pi(x):
    """Wrap radians to [-pi, pi)."""
    return np.angle(np.exp(1j * np.asarray(x)))


def smooth_masked(x, mask, sigma, eps=1e-6):
    """Gaussian smoothing that ignores masked-out pixels (works on complex)."""
    if sigma is None or sigma <= 0:
        return x.astype(x.dtype, copy=True)
    m = mask.astype(np.float64)
    den = gaussian_filter(m, sigma)
    if np.iscomplexobj(x):
        num = (gaussian_filter(np.where(mask, x.real, 0.0), sigma)
               + 1j * gaussian_filter(np.where(mask, x.imag, 0.0), sigma))
    else:
        num = gaussian_filter(np.where(mask, x, 0.0), sigma)
    return num / np.where(den > eps, den, np.nan)


@dataclass
class AxisMaps:
    axis: str                # 'elevation' or 'azimuth'
    label_fwd: str
    label_rev: str
    delay: np.ndarray        # (h, w) seconds, spatially smoothed
    delay_raw: np.ndarray    # (h, w) seconds, per pixel
    position: np.ndarray     # (h, w) degrees, combined estimate
    pos_fwd: np.ndarray      # (h, w) degrees, from the forward block alone
    pos_rev: np.ndarray      # (h, w) degrees, from the reverse block alone
    agreement: np.ndarray    # (h, w) degrees, pos_fwd - pos_rev (wrapped)
    consistency: np.ndarray  # (h, w) 0-1, |mean of the two unit phasors|
    amplitude: np.ndarray    # (h, w) mean dR/R amplitude of the two blocks
    snr: np.ndarray          # (h, w) combined SNR
    mask: np.ndarray         # (h, w) bool
    travel: float            # degrees swept by the bar centre
    visible: float           # degrees of that travel actually on screen
    period: float            # seconds


def combine_axis(sp_fwd, blk_fwd, sp_rev, blk_rev,
                 phase_sigma: float = 2.0, delay_sigma: float = 4.0,
                 snr_min: float = 2.0, visible_half: float | None = None):
    """
    sp_*  : BlockSpectrum from ioi_fourier.analyse
    blk_* : the matching StimBlock (supplies bar_deg_first/last and f_stim)
    phase_sigma : smoothing of the complex component before taking phase
    delay_sigma : smoothing of the delay map before it is subtracted
    visible_half: half-extent of the screen along this axis, in degrees; used
                  only for reporting how much of the travel was visible
    """
    mask = sp_fwd.mask & sp_rev.mask & (sp_fwd.snr > snr_min) & (sp_rev.snr > snr_min)

    cF = smooth_masked(sp_fwd.component, sp_fwd.mask, phase_sigma)
    cR = smooth_masked(sp_rev.component, sp_rev.mask, phase_sigma)

    # numpy's rfft gives X = (N/2) exp(-i theta) for cos(wt - theta),
    # so the response time is encoded in MINUS the angle.
    phiF = wrap01(-np.angle(cF) / TWO_PI)
    phiR = wrap01(-np.angle(cR) / TWO_PI)

    T = 0.5 * (1.0 / blk_fwd.f_stim + 1.0 / blk_rev.f_stim)

    delay_raw = T * wrap01(phiF + phiR) / 2.0
    delay = smooth_masked(delay_raw, mask, delay_sigma)

    p0F, p1F = blk_fwd.bar_deg_first, blk_fwd.bar_deg_last
    p0R, p1R = blk_rev.bar_deg_first, blk_rev.bar_deg_last

    uF = wrap01(phiF - delay / T)
    uR = wrap01(phiR - delay / T)
    posF = p0F + uF * (p1F - p0F)
    posR = p0R + uR * (p1R - p0R)

    # combine in the forward block's fractional-travel coordinate, circularly
    travel = p1F - p0F
    gF = (posF - p0F) / travel
    gR = (posR - p0F) / travel
    z = 0.5 * (np.exp(1j * TWO_PI * gF) + np.exp(1j * TWO_PI * gR))
    position = p0F + wrap01(np.angle(z) / TWO_PI) * travel
    consistency = np.abs(z)

    agreement = wrap_pi(TWO_PI * (gF - gR)) / TWO_PI * travel

    amp = 0.5 * (sp_fwd.amplitude + sp_rev.amplitude)
    snr = np.sqrt(sp_fwd.snr ** 2 + sp_rev.snr ** 2)

    return AxisMaps(
        axis=blk_fwd.axis, label_fwd=blk_fwd.label, label_rev=blk_rev.label,
        delay=delay, delay_raw=delay_raw,
        position=position, pos_fwd=posF, pos_rev=posR,
        agreement=agreement, consistency=consistency,
        amplitude=amp, snr=snr, mask=mask,
        travel=abs(travel),
        visible=2 * visible_half if visible_half else np.nan,
        period=T,
    )


def report(am: AxisMaps) -> str:
    m = am.mask
    if not m.any():
        return f"{am.axis}: mask empty"
    d = am.delay[m]
    a = am.agreement[m]
    lines = [
        f"{am.axis} ({am.label_fwd} / {am.label_rev})",
        f"  pixels in mask        : {int(m.sum())} ({100*m.mean():.1f}%)",
        f"  bar travel            : {am.travel:.1f} deg  "
        f"(screen spans {am.visible:.1f} deg)",
        f"  cycle period          : {am.period:.3f} s  "
        f"(delay unambiguous below {am.period/2:.2f} s)",
        f"  haemodynamic delay    : median {np.nanmedian(d):.2f} s, "
        f"IQR {np.nanpercentile(d,25):.2f}-{np.nanpercentile(d,75):.2f} s",
        f"  fwd/rev agreement     : median |diff| "
        f"{np.nanmedian(np.abs(a)):.2f} deg, "
        f"90th pct {np.nanpercentile(np.abs(a),90):.2f} deg",
        f"  position range        : "
        f"{np.nanpercentile(am.position[m],2):.1f} .. "
        f"{np.nanpercentile(am.position[m],98):.1f} deg",
    ]
    return "\n".join(lines)
