"""
ioi_leak.py -- separate a spatially-common stimulus-locked contamination from
anything with genuine spatial structure.

THE MODEL
---------
Block 1/2 established that the dominant stimulus-locked signal here is not a
tissue signal: its dR/R amplitude scales as I^-0.71 and its SNR is 4x HIGHER
in the darkest sixth of the field than the brightest.  That is the signature
of light reaching the detector additively, in counts, rather than reflectance
modulation.

Fitting amplitude = A/I + c to the B2U data gives A ~ 0.84 counts and
c ~ 0.0037 %dR/R, so we model the complex Fourier coefficient of each pixel as

    c(x)  =  alpha / I(x)  +  beta  +  s(x)

with alpha, beta COMPLEX and global:
  alpha/I  -- stray light, additive in counts, so fractional size goes as 1/I
  beta     -- any globally uniform fractional modulation (screen light
              reflected in proportion to reflectance, or a global
              haemodynamic response -- both are non-retinotopic)
  s(x)     -- what we are looking for

WHY THIS IS THE RIGHT SUBTRACTION
---------------------------------
alpha and beta are two complex numbers for the whole field.  Retinotopy is by
definition spatially structured with a phase that varies across cortex, so it
cannot be absorbed into a spatially constant term -- it survives in s(x).
Only the globally common part is removed.

This is deliberately weaker than nulling the leak's phase (which would throw
away the in-phase half of any real signal) and weaker than fitting a smooth
spatial model (which could eat a retinotopic gradient).  `order` can add
polynomial terms in x and y if you want to test sensitivity, but raising it
risks absorbing exactly what you are hunting for -- the default of 0 is the
conservative choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualSpectrum:
    """Duck-types ioi_fourier.BlockSpectrum so combine_axis() accepts it."""
    label: str
    component: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    noise: np.ndarray
    snr: np.ndarray
    mask: np.ndarray


@dataclass
class LeakFit:
    alpha: complex           # counts, additive stray light
    beta: complex            # fractional, globally uniform modulation
    coefs: np.ndarray        # full coefficient vector
    predicted: np.ndarray    # (h, w) complex, the fitted common mode
    var_removed: float       # fraction of masked complex variance removed
    order: int


def _basis(I, mask, order):
    h, w = I.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xx = (xx - w / 2) / (w / 2)
    yy = (yy - h / 2) / (h / 2)
    cols = [1.0 / np.where(I > 0, I, np.nan), np.ones_like(I, dtype=float)]
    names = ["1/I", "1"]
    for o in range(1, order + 1):
        for p in range(o + 1):
            cols.append((xx ** (o - p)) * (yy ** p))
            names.append(f"x^{o-p}y^{p}")
    B = np.stack([c.ravel() for c in cols], axis=1)
    return B, names


def fit_common_mode(component, I, noise, mask, order: int = 0) -> LeakFit:
    """
    Weighted complex least squares of component ~ alpha/I + beta (+ poly).

    Weighting is 1/noise so that bright, low-noise pixels are not drowned out
    by the dim ones where the leak is loudest.
    """
    B, names = _basis(I, mask, order)
    y = component.ravel()
    m = mask.ravel() & np.isfinite(y) & np.isfinite(B).all(axis=1)
    wgt = 1.0 / np.where(noise.ravel() > 0, noise.ravel(), np.inf)

    A = B[m] * wgt[m, None]
    b = y[m] * wgt[m]
    coefs, *_ = np.linalg.lstsq(A, b, rcond=None)

    pred = (B @ coefs).reshape(component.shape)
    resid = component - pred
    v0 = np.nanvar(component[mask])
    v1 = np.nanvar(resid[mask])
    return LeakFit(alpha=complex(coefs[0]), beta=complex(coefs[1]),
                   coefs=coefs, predicted=pred,
                   var_removed=float(1 - v1 / v0) if v0 > 0 else np.nan,
                   order=order)


def remove_common_mode(sp, I, mask, order: int = 0):
    """Return (ResidualSpectrum, LeakFit) for one block."""
    fit = fit_common_mode(sp.component, I, sp.noise, mask, order=order)
    resid = sp.component - fit.predicted
    amp = np.abs(resid)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = amp / sp.noise
    return ResidualSpectrum(
        label=sp.label + "-resid", component=resid, amplitude=amp,
        phase=np.angle(resid), noise=sp.noise, snr=snr, mask=mask,
    ), fit


# --------------------------------------------------------------------------
# Structure tests
# --------------------------------------------------------------------------

def phase_concentration(component, mask):
    """|<exp(i phi)>|: 1 = every pixel shares one phase, 0 = uniform spread."""
    z = component[mask]
    z = z[np.isfinite(z) & (np.abs(z) > 0)]
    if z.size == 0:
        return np.nan
    u = z / np.abs(z)
    return float(np.abs(u.mean()))


def neighbour_phase_difference(phase, mask):
    """Mean |phase difference| between adjacent pixels, in degrees."""
    z = np.exp(1j * phase)
    gy = np.angle(z[1:, :] * np.conj(z[:-1, :]))
    gx = np.angle(z[:, 1:] * np.conj(z[:, :-1]))
    my = mask[1:, :] & mask[:-1, :]
    mx = mask[:, 1:] & mask[:, :-1]
    vals = np.r_[np.abs(gy[my]), np.abs(gx[mx])]
    return float(np.degrees(vals.mean())) if vals.size else np.nan


def structure_score(phase, mask, n_shuffle=20, seed=0):
    """
    Compare neighbour phase coherence to spatially shuffled controls.

    Returns (observed_deg, shuffled_mean_deg, z).  A z well below zero means
    the phase map is more spatially organised than chance.  This test is run
    on UNSMOOTHED phase -- smoothing manufactures organisation and would make
    the test meaningless.
    """
    rng = np.random.default_rng(seed)
    obs = neighbour_phase_difference(phase, mask)
    idx = np.flatnonzero(mask.ravel())
    sh = []
    for _ in range(n_shuffle):
        p = phase.copy().ravel()
        p[idx] = p[idx][rng.permutation(idx.size)]
        sh.append(neighbour_phase_difference(p.reshape(phase.shape), mask))
    sh = np.asarray(sh)
    z = (obs - sh.mean()) / sh.std() if sh.std() > 0 else np.nan
    return obs, float(sh.mean()), float(z)
