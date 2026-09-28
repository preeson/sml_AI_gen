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


# --------------------------------------------------------------------------
# Joint delay across both axes
# --------------------------------------------------------------------------

def _phase01(c):
    """Response phase as a fraction of a cycle, from the rfft convention."""
    return wrap01(-np.angle(c) / TWO_PI)


def combine_axes_joint(specs, blocks, pairs=(("elevation", "B2U", "U2B"),
                                             ("azimuth", "L2R", "R2L")),
                       phase_sigma: float = 2.0, delay_sigma: float = 4.0,
                       mask=None, snr_min: float = 0.0):
    """
    Estimate ONE haemodynamic delay map from all four blocks, then use it for
    both axes.

    WHY
    ---
    Haemodynamic delay is a property of the tissue: it cannot depend on which
    way the bar swept.  Estimating it separately per axis therefore throws
    information away, and worse, it is much noisier for the slower axis.  The
    delay from one axis pair is

        tau = T * wrap01(phi_F + phi_R) / 2

    so its error scales with the cycle period T:

        sigma_tau = T/(4*pi) * sqrt(1/snr_F^2 + 1/snr_R^2)

    Azimuth here has T = 22.4 s against elevation's 14.4 s, so at equal SNR the
    azimuth delay is ~1.6x noisier.  That is why azimuth delay ran ~1 s longer
    than elevation and drifted toward its own chance level: the estimate was
    being pulled by noise, and the error fed straight into azimuth position.

    Inverse-variance weighting fixes this.  The joint delay is dominated by the
    better-conditioned elevation estimate, and azimuth position inherits it.

    WHERE THE DELAY ACTUALLY MATTERS
    --------------------------------
    Worth being precise, because it is not where you would guess.  Writing
    g_F and g_R for the two directions' position estimates as a fraction of
    travel, the combined estimate is the circular mean of exp(2i.pi.g), and

        g_F + g_R = 1 + phi_F - phi_R      -- tau cancels exactly
        g_F - g_R = phi_F + phi_R - 2.tau/T - 1

    The first line means the combined position is INDEPENDENT of tau within a
    branch: only the half-difference of the phases survives.  The second means
    tau decides which branch the circular mean lands in, because the mean
    flips by half a cycle when cos(pi(g_F - g_R)) changes sign.

    So a better delay does not nudge positions slightly -- it moves pixels
    between branches, i.e. it fixes the ones that were wrapped by half the bar
    travel.  That is exactly the failure that put ~15% of azimuth pixels
    outside the screen extent, and why this matters for the sign map: a
    wrapped pixel has a meaningless gradient.

    Returns (maps, info) where maps is {axis: AxisMaps} and info carries the
    per-axis delay and its residual against the joint estimate -- if those
    residuals are large and structured, the model is wrong and you should know.
    """
    lab0 = pairs[0][1]
    shape = specs[lab0].component.shape
    if mask is None:
        mask = np.ones(shape, dtype=bool)
        for _, f, r in pairs:
            mask &= specs[f].mask & specs[r].mask
            if snr_min > 0:
                mask &= (specs[f].snr > snr_min) & (specs[r].snr > snr_min)

    per_axis = {}
    tau_num = np.zeros(shape)
    tau_den = np.zeros(shape)

    for axis, f, r in pairs:
        cF = smooth_masked(specs[f].component, specs[f].mask, phase_sigma)
        cR = smooth_masked(specs[r].component, specs[r].mask, phase_sigma)
        phiF, phiR = _phase01(cF), _phase01(cR)
        T = 0.5 * (1.0 / blocks[f].f_stim + 1.0 / blocks[r].f_stim)
        tau = T * wrap01(phiF + phiR) / 2.0

        snrF = np.maximum(specs[f].snr, 1e-3)
        snrR = np.maximum(specs[r].snr, 1e-3)
        sig = T / (2 * TWO_PI) * np.sqrt(1.0 / snrF ** 2 + 1.0 / snrR ** 2)
        w = np.where(np.isfinite(tau) & mask, 1.0 / np.maximum(sig, 1e-9) ** 2, 0.0)

        per_axis[axis] = dict(phiF=phiF, phiR=phiR, T=T, tau=tau, sigma=sig,
                              fwd=f, rev=r)
        tau_num += np.nan_to_num(tau) * w
        tau_den += w

    tau_joint = np.where(tau_den > 0, tau_num / np.maximum(tau_den, 1e-30), np.nan)
    tau_joint = smooth_masked(tau_joint, mask & np.isfinite(tau_joint), delay_sigma)

    maps, info = {}, {"delay_joint": tau_joint}
    for axis, f, r in pairs:
        a = per_axis[axis]
        T = a["T"]
        p0F, p1F = blocks[f].bar_deg_first, blocks[f].bar_deg_last
        p0R, p1R = blocks[r].bar_deg_first, blocks[r].bar_deg_last
        uF = wrap01(a["phiF"] - tau_joint / T)
        uR = wrap01(a["phiR"] - tau_joint / T)
        posF = p0F + uF * (p1F - p0F)
        posR = p0R + uR * (p1R - p0R)

        travel = p1F - p0F
        gF = (posF - p0F) / travel
        gR = (posR - p0F) / travel
        z = 0.5 * (np.exp(1j * TWO_PI * gF) + np.exp(1j * TWO_PI * gR))
        position = p0F + wrap01(np.angle(z) / TWO_PI) * travel

        maps[axis] = AxisMaps(
            axis=axis, label_fwd=f, label_rev=r,
            delay=tau_joint, delay_raw=a["tau"],
            position=position, pos_fwd=posF, pos_rev=posR,
            agreement=wrap_pi(TWO_PI * (gF - gR)) / TWO_PI * travel,
            consistency=np.abs(z),
            amplitude=0.5 * (specs[f].amplitude + specs[r].amplitude),
            snr=np.sqrt(specs[f].snr ** 2 + specs[r].snr ** 2),
            mask=mask, travel=abs(travel),
            visible=2 * blocks[f].visible_half, period=T,
        )
        info[f"delay_{axis}"] = a["tau"]
        info[f"delay_resid_{axis}"] = a["tau"] - tau_joint
        info[f"sigma_{axis}"] = a["sigma"]
    return maps, info
