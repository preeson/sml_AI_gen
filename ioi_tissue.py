"""
ioi_tissue.py -- find the perfused cortex automatically, with no hand-drawn ROI.

THE PROBLEM
-----------
Everything downstream is evaluated over a mask, and until now that mask was a
brightness band.  Brightness cannot tell cortex from skull, dental cement, the
headplate, agarose or a specular highlight: those are often the BRIGHTEST things
in the field, so they pass the band and then contribute pure noise to every
median, every gradient and every patch.  Hand-drawing an ROI fixes one recording
and has to be redone for the next, differently for a thinned skull than for a
cranial window, which makes it both laborious and unauditable.

THE CRITERION
-------------
Ask each pixel whether it has a pulse.

Perfused cortex modulates its 625 nm reflectance at the heart rate, because
every systole changes the blood volume in the pixel.  Bone, cement, metal and
glass do not.  The cardiac modulation is one to two orders of magnitude larger
than an evoked intrinsic signal and needs no stimulus, so it is available in any
recording -- including one that failed.

Crucially this is a test of each pixel against ITS OWN spectral noise floor:

    ratio = |X(f_hr)| / sigma_pixel

so it is automatically invariant to the things that differ between preparations
and between sessions -- absolute light level, window clarity, exposure, how much
of the field is brain.  A thinned skull returns less light than a glass window,
so both numerator and denominator shrink and the ratio is unchanged.  That is
the property that lets one threshold serve every animal.

WHY AT A KNOWN FREQUENCY, NOT A BAND MAXIMUM
--------------------------------------------
The heart rate is a property of the animal, not of the pixel, so it is measured
once from the spatial mean and then every pixel is interrogated at that one
frequency.  This matters more than it looks.  Searching a 4.5-14 Hz band is a
search over ~285 bins, and the largest of 285 noise bins is large by
construction -- it needs ~4 sigma before it means anything, and every pixel in
the field gets one.  Asking a single known bin needs only ~3.2 sigma for the
same false-positive rate over the whole field, and a non-perfused pixel has
nothing there at all.  The separation between tissue and not-tissue goes from
marginal to obvious.

THRESHOLD
---------
For complex Gaussian noise |X|^2 is exponential, so P(|X| > x.sigma) = exp(-x^2)
and the expected number of false pixels in a field of n is n.exp(-x^2).  Setting
that to `expected_false` gives

    x = sqrt( ln(n / expected_false) )

about 3.2 for 21600 binned pixels and one expected false positive.  No tuning
knob, and it does not move when the prep changes.

FALLBACKS, IN ORDER
-------------------
1. cardiac amplitude at the measured heart rate   -- preferred
2. cardiac band maximum, Rayleigh threshold       -- if the global HR is not
                                                     detectable (reported)
3. variance in excess of photon shot noise        -- if there is no pulse
                                                     anywhere.  This is an
                                                     ANATOMICAL mask, not a
                                                     physiological one, and it
                                                     says so loudly: a field
                                                     with no pulse anywhere is
                                                     a failed preparation, and
                                                     the mask is then the least
                                                     of the problems.

The fallback actually used is recorded in the result and printed, because a
mask built by rule 3 does not support the same conclusions as one built by
rule 1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import (binary_closing, binary_erosion, binary_fill_holes,
                           gaussian_filter, label)

import ioi_core as ioi

# ---------------------------------------------------------------- config ----
PULSE_BAND = (4.5, 14.0)     # mouse heart rate, 270-840 /min
RESP_BAND = (0.7, 4.0)       # respiration, 42-240 /min
NOISE_LO = 15.0              # bottom of the band used to measure each pixel's
                             # own noise floor.  Above every physiological
                             # rhythm and its low harmonics.
OFFSET = 1540.0              # camera dark level, counts
MIN_COUNTS = 500.0           # below this a pixel is not illuminated
SAT_LEVEL = 55000.0
GAIN = 2.18                  # counts per photoelectron, PCO Edge 5.5
EXPECTED_FALSE = 1.0         # how many false-positive pixels to tolerate in
                             # the whole field
N_HARMONICS = 2              # cardiac harmonics summed (1 = fundamental only)
SMOOTH_SIGMA = 3.0           # binned px.  Coherent spatial averaging of the
                             # cardiac component before thresholding; see
                             # _coherent_statistic.  0 disables it, which
                             # returns the vasculature and little else.
MIN_AREA_MM2 = 0.5           # discard components smaller than this
CLOSE_ITER = 2               # bridge between vessels into a filled region
ERODE_ITER = 1               # pull the mask off the window rim
LARGEST_ONLY = False         # True keeps only the biggest component


@dataclass
class TissueMask:
    mask: np.ndarray             # (h, w) bool -- USE THIS
    raw: np.ndarray              # (h, w) bool, before morphology
    ratio: np.ndarray            # (h, w) pulse amplitude / own noise floor
    threshold: float
    f_pulse: float               # Hz, heart rate used
    f_resp: float                # Hz, respiration (reported only)
    pulse_amp: np.ndarray        # (h, w) cardiac amplitude, FRACTION of dR/R
    excess_var: np.ndarray       # (h, w) measured var / shot-noise var
    mean_img: np.ndarray
    method: str                  # 'cardiac', 'cardiac_band' or 'excess_var'
    n_frames: int
    fs: float
    bin_xy: int
    um_per_pixel: float
    smooth_sigma: float = 0.0    # binned px of coherent spatial averaging
    n_eff: float = 0.0           # independent pixels actually tested
    global_ratio: float = np.nan  # pulse / noise in the coherent spatial mean
    lit: np.ndarray | None = None  # illuminated, unsaturated field
    notes: list = field(default_factory=list)

    @property
    def area_mm2(self) -> float:
        return float(self.mask.sum()) * (self.um_per_pixel / 1000.0) ** 2

    @property
    def n_components(self) -> int:
        return int(label(self.mask)[1])

    def summary(self) -> str:
        pct = 100.0 * self.mask.mean()
        med = float(np.nanmedian(self.ratio[self.mask])) if self.mask.any() \
            else np.nan
        amp = float(np.nanmedian(self.pulse_amp[self.mask])) if self.mask.any() \
            else np.nan
        lines = [
            f"  method              : {self.method}",
            f"  heart rate          : {self.f_pulse:.3f} Hz "
            f"({self.f_pulse * 60:.0f}/min)",
            f"  respiration         : {self.f_resp:.3f} Hz "
            f"({self.f_resp * 60:.0f}/min)",
            f"  pulse in the mean   : {self.global_ratio:.1f} x "
            f"(coherent spatial mean over the lit field)",
            f"  spatial averaging   : sigma {self.smooth_sigma:g} binned px "
            f"({self.smooth_sigma * self.um_per_pixel:.0f} um) = "
            f"{1/max(k_from_sigma(self.smooth_sigma), 1e-9):.0f} px coherently"
            f", {1/np.sqrt(max(k_from_sigma(self.smooth_sigma), 1e-9)):.1f}x "
            f"less noise" if self.smooth_sigma else
            "  spatial averaging   : none",
            f"  threshold           : {self.threshold:.2f} x own noise floor "
            f"({EXPECTED_FALSE:.0f} expected false px in "
            f"{self.n_eff:.0f} independent px)",
            f"  pixels kept         : {int(self.mask.sum())} / {self.mask.size}"
            f" ({pct:.0f}%)  = {self.area_mm2:.2f} mm2",
            f"  connected pieces    : {self.n_components}",
            f"  median ratio inside : {med:.1f} x",
            f"  median pulse inside : {100 * amp:.4f} %dR/R"
            if np.isfinite(amp) else "  median pulse inside : --",
        ]
        for n in self.notes:
            lines.append(f"  NOTE: {n}")
        return "\n".join(lines)


# ------------------------------------------------------------- internals ----

def _binned_stack(mov, start, n, bin_xy, chunk=250):
    H, W = mov.shape[1], mov.shape[2]
    if H % bin_xy or W % bin_xy:
        raise ValueError(f"bin_xy={bin_xy} does not divide {H}x{W}")
    h, w = H // bin_xy, W // bin_xy
    out = np.empty((n, h, w), dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d = np.asarray(mov[start + a:start + b], dtype=np.float32)
        out[a:b] = d.reshape(b - a, h, bin_xy, w, bin_xy).mean(axis=(2, 4))
    return out


def _detrend(x, order=4):
    """Remove a low-order polynomial along axis 0.  x is (n, ...)."""
    n = x.shape[0]
    v = np.vander(np.linspace(-1.0, 1.0, n), order)
    flat = x.reshape(n, -1)
    coef, *_ = np.linalg.lstsq(v, flat, rcond=None)
    return (flat - v @ coef).reshape(x.shape)


def _amp_at(f, X, target, halfwidth=1):
    """
    Amplitude at a frequency, taking the largest bin within +-halfwidth.

    X may be (nf,) or (nf, npx).  A real rhythm is not exactly on a bin centre
    and drifts slightly over the window, so a single bin underestimates it;
    +-1 bin costs a factor sqrt(3) in the effective number of trials searched,
    which the threshold already absorbs.
    """
    if not (np.isfinite(target) and 0 < target < f[-1]):
        return np.zeros(X.shape[1:], dtype=float)
    i = int(np.searchsorted(f, target))
    a, b = max(0, i - halfwidth), min(X.shape[0], i + halfwidth + 1)
    return np.asarray(np.abs(X[a:b]).max(axis=0), dtype=float)


def _notch(f, X, f0, n_harm=10, halfwidth=2):
    """Blank a harmonic comb (used to keep breathing out of the heart band)."""
    Xn = np.array(X, copy=True)
    if not (np.isfinite(f0) and f0 > 0):
        return Xn
    floor = float(np.median(np.abs(X[np.abs(X) > 0]))) if np.any(X) else 0.0
    for k in range(1, n_harm + 1):
        fk = f0 * k
        if fk >= f[-1]:
            break
        i = int(np.searchsorted(f, fk))
        a, b = max(0, i - halfwidth), min(Xn.shape[0], i + halfwidth + 1)
        Xn[a:b] = floor
    return Xn


def _fundamental(f, X, lo, hi, weights=(1.0, 0.5, 0.25)):
    """Harmonic-sum fundamental in a band, then snapped to the local peak."""
    b = np.flatnonzero((f >= lo) & (f < hi))
    if b.size == 0:
        return np.nan
    A = np.abs(X)
    score = np.zeros(b.size)
    for k, wk in enumerate(weights, start=1):
        score += wk * np.array([float(_amp_at(f, A, fb * k)) for fb in f[b]])
    j = int(b[int(np.argmax(score))])
    a, c = max(0, j - 1), min(A.size, j + 2)
    return float(f[a + int(np.argmax(A[a:c]))])


def _rayleigh_threshold(n_bins, n_px, expected_false=EXPECTED_FALSE):
    """Amplitude multiple of the noise sigma for `expected_false` false px."""
    trials = max(n_bins, 1) * max(n_px, 1)
    return float(np.sqrt(np.log(max(trials / max(expected_false, 1e-9), 1.0001))))


def _gamma_threshold(m, n_eff, expected_false=EXPECTED_FALSE):
    """
    Threshold in sigma units for a sum of m independent |Z|^2/sigma^2 terms.

    Each term is exponential, so the sum is Gamma(m, 1) and the threshold
    solves n_eff * P(Gamma(m,1) > x) = expected_false.  Returned as sqrt(x) so
    it is directly comparable with an amplitude ratio.
    """
    from scipy.stats import gamma as _gamma
    q = min(max(expected_false, 1e-12) / max(n_eff, 1.0), 0.5)
    return float(np.sqrt(_gamma.isf(q, max(int(m), 1))))


def _gauss_kernel(sigma, truncate=4.0):
    """Explicit normalised 2-D Gaussian kernel, and its element-wise square."""
    if sigma is None or sigma <= 0:
        k = np.ones((1, 1))
        return k, k
    r = max(int(truncate * sigma + 0.5), 1)
    x = np.arange(-r, r + 1, dtype=float)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g /= g.sum()
    w = np.outer(g, g)
    return w, w ** 2


def k_from_sigma(sigma):
    """Sum of squared kernel weights -- 1/k is the number of pixels averaged."""
    return float(_gauss_kernel(sigma)[1].sum())


def _smooth_complex(z, mask, sigma):
    """Normalised-convolution smoothing of a complex map over `mask`."""
    from scipy.ndimage import convolve
    w, _ = _gauss_kernel(sigma)
    m = mask.astype(float)
    den = convolve(m, w, mode="nearest")
    num = (convolve(np.where(mask, z.real, 0.0), w, mode="nearest")
           + 1j * convolve(np.where(mask, z.imag, 0.0), w, mode="nearest"))
    good = den > 1e-6
    return np.where(good, num / np.where(good, den, 1.0), 0.0 + 0j)


def _coherent_statistic(X, sigma, bins, mask, shape, smooth_sigma):
    """
    Spatially averaged detection statistic at known DFT bins.

    WHY SMOOTH AT ALL
    -----------------
    At 4x binning and 30 s, a single pixel's cardiac modulation over
    parenchyma sits near its own noise floor, while over a surface vessel it
    is tens of times larger.  Thresholding per pixel therefore returns the
    vasculature and nothing else -- a thin crescent instead of a window, which
    is exactly what the first real run produced.

    But the pulse arrives across a few hundred microns essentially
    simultaneously: at 8 Hz, a 10 ms transit is 29 degrees of phase.  So
    neighbouring pixels can be averaged COMPLEX, where the signal adds
    coherently and the noise does not.  At sigma = 2 binned pixels (77 um)
    that is a 7x gain in sensitivity, and because smoothing also reduces the
    number of independent pixels being tested, the threshold falls as well.

    The cost is resolution: the mask boundary is blurred by about the
    smoothing scale, which is why the result is eroded afterwards.  Smoothing
    the AMPLITUDE instead would buy nothing -- noise amplitude is positive and
    averages to its own mean rather than to zero.

    Returns (S, sigma_smoothed, k) with S = sum over bins of |Zs|^2/sigma_s^2.
    """
    from scipy.ndimage import convolve
    w, w2 = _gauss_kernel(smooth_sigma)
    k = float(w2.sum())
    m = mask.astype(float)
    den = convolve(m, w, mode="nearest")
    good = den > 1e-6

    # Var(sum w_i z_i / sum w_i) = sum w_i^2 sigma_i^2 / (sum w_i)^2
    s2 = np.nan_to_num(sigma.reshape(shape) ** 2)
    var = convolve(np.where(mask, s2, 0.0), w2, mode="nearest")
    var = np.where(good, var / np.where(good, den ** 2, 1.0), np.nan)

    S = np.zeros(shape)
    for b in bins:
        zs = _smooth_complex(np.asarray(X[b]).reshape(shape), mask,
                             smooth_sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            S = S + np.abs(zs) ** 2 / var
    return S, np.sqrt(var), k


def _bleed_px(ratio, mask, thr, smooth_sigma, cap=3.0):
    """
    How far the coherent smoothing pushed the mask past the real boundary.

    Smoothing dilates: across a step edge with signal S inside, the smoothed
    amplitude at distance d outside is S * Phi(-d / sigma), so the threshold
    is still cleared out to

        d = -sigma * Phi^-1(thr / S)

    which grows only logarithmically with S -- 1.6 sigma for a signal 12x
    threshold, 3.7 sigma for 70x.  Measuring S from the data and eroding by
    exactly that distance is tighter than a fixed margin: a weak recording
    loses almost no rim, a strong one loses what it has to.  Capped, because
    beyond ~3 sigma the kernel is truncated anyway.
    """
    from scipy.stats import norm
    if smooth_sigma <= 0 or not np.any(mask):
        return 0
    S = float(np.nanpercentile(ratio[mask], 99.0))
    if not np.isfinite(S) or S <= thr:
        return 0
    d = -norm.ppf(min(thr / S, 0.5)) * smooth_sigma
    return int(round(min(d, cap * smooth_sigma)))


def _morphology(raw, um_per_pixel, min_area_mm2=MIN_AREA_MM2,
                close_iter=CLOSE_ITER, erode_iter=ERODE_ITER,
                largest_only=LARGEST_ONLY):
    """
    Turn a thresholded pixel set into a region.

    The cardiac signal is strongest over vessels, so the thresholded map is a
    vascular network with parenchyma between the branches.  Closing and
    hole-filling turn that network into the territory it drains, which is the
    thing we actually want.  Eroding afterwards removes the window rim, where
    apparent modulation comes from the brain pulsing against a high-contrast
    edge rather than from blood volume in the pixel.
    """
    m = np.array(raw, dtype=bool)
    if close_iter:
        m = binary_closing(m, iterations=close_iter)
    m = binary_fill_holes(m)
    if erode_iter:
        m = binary_erosion(m, iterations=erode_iter, border_value=0)
    px_mm2 = (um_per_pixel / 1000.0) ** 2
    lab, n = label(m)
    if n == 0:
        return m
    areas = np.array([(lab == i).sum() * px_mm2 for i in range(1, n + 1)])
    if largest_only:
        keep = {1 + int(np.argmax(areas))}
    else:
        keep = {i for i, a in enumerate(areas, start=1) if a >= min_area_mm2}
        if not keep:
            keep = {1 + int(np.argmax(areas))}
    return np.isin(lab, list(keep))


# ------------------------------------------------------------------ main ----

def find_tissue(mov, ft, start=None, n_frames=3000, bin_xy=4,
                um_per_pixel=None, expected_false=EXPECTED_FALSE,
                n_harmonics=N_HARMONICS, smooth_sigma=SMOOTH_SIGMA,
                chunk=250, verbose=True) -> TissueMask:
    """
    Build the evaluation mask from the cardiac pulse.

    mov          : open movie (ioi.movie())
    ft           : FrameTimes
    start        : first frame; default the middle of the recording
    n_frames     : length of the window.  Longer is better -- the noise floor
                   falls as 1/sqrt(n) while the pulse amplitude does not -- but
                   the heart rate must stay put over the window, so 30-60 s.
    bin_xy       : MUST match the binning used by the analysis this mask will
                   be applied to.
    """
    H, W = mov.shape[1], mov.shape[2]
    if um_per_pixel is None:
        um_per_pixel = 6150.0 / W * bin_xy
    n = int(min(n_frames, ft.n_frames))
    start = (ft.n_frames - n) // 2 if start is None else int(start)
    fs = (ft.n_frames - 1) / (ft.t[-1] - ft.t[0])
    notes = []

    if fs / 2.0 <= PULSE_BAND[0]:
        raise ValueError(
            f"camera runs at {fs:.1f} Hz, Nyquist {fs/2:.1f} Hz is below the "
            f"cardiac band {PULSE_BAND[0]}-{PULSE_BAND[1]} Hz -- this mask "
            f"needs a frame rate above ~{2*PULSE_BAND[1]:.0f} Hz")

    if verbose:
        print(f"  reading {n} frames from {start} "
              f"({n/fs:.1f} s at {fs:.3f} Hz), {bin_xy}x binned")
    stack = _binned_stack(mov, start, n, bin_xy, chunk=chunk)
    mean_img = stack.mean(axis=0, dtype=np.float64)

    lit = ((mean_img - OFFSET) > MIN_COUNTS) & (mean_img < SAT_LEVEL)
    if lit.sum() < 50:
        raise RuntimeError("almost nothing is illuminated -- check OFFSET / "
                           "MIN_COUNTS / SAT_LEVEL")

    # fractional reflectance change, so amplitudes are in dR/R and comparable
    # between a bright window and a dim thinned skull
    denom = np.where(lit, mean_img - OFFSET, np.nan)
    dr = (stack - mean_img[None]) / denom[None]
    del stack
    dr = np.nan_to_num(dr, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- spectrum, per pixel --------------------------------------------
    dr = _detrend(dr, order=4)
    X = np.fft.rfft(dr.reshape(n, -1), axis=0) * (2.0 / n)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    del dr
    A = np.abs(X).astype(np.float32)
    npx = A.shape[1]

    # ---- each pixel's own noise floor -----------------------------------
    # The median of |X| for complex Gaussian noise is sigma*sqrt(ln 2), so
    # dividing by sqrt(ln 2) turns a robust median into the Rayleigh sigma.
    nb = (f >= NOISE_LO) & (f <= f[-1])
    if nb.sum() < 20:
        nb = f > PULSE_BAND[1]
        notes.append(f"noise band is only {int(nb.sum())} bins wide -- the "
                     f"camera rate leaves little room above the cardiac band")
    sigma = np.median(A[nb], axis=0) / np.sqrt(np.log(2.0))
    sigma = np.where(sigma > 0, sigma, np.nan)

    # ---- the animal's rates, from the spatial mean over lit pixels -------
    # COHERENTLY: the spectrum of the spatial mean, not the mean of the
    # spectra.  |mean(X)| suppresses noise by sqrt(n_px) because the cardiac
    # phase is nearly common across cortex, while mean(|X|) suppresses nothing
    # at all -- every pixel contributes its own noise magnitude, so a pulse
    # confined to the surface vessels (a few percent of the field) cannot lift
    # the average and the detector never fires.  Getting this wrong is what
    # forced the band-maximum fallback on real data.
    gX = np.asarray(X[:, lit.ravel()].mean(axis=1))
    glob = np.abs(gX)
    gsig0 = float(np.median(glob[nb]) / np.sqrt(np.log(2.0)))

    # Respiration first, then the heartbeat with the respiratory comb notched
    # out -- otherwise a fast-breathing mouse returns a harmonic as its pulse.
    #
    # But notch ONLY a respiration we can actually see.  The tallest bin in
    # 0.7-4.0 Hz always exists, and if it is noise its comb is a comb of
    # nothing: blanking 2f then deletes a genuine heartbeat that happens to
    # sit near twice that arbitrary frequency, and the whole detector falls
    # over.  A real fast breather clears this check comfortably, so the
    # 180/min failure this notch was written for is still caught.
    f_resp = _fundamental(f, glob, *RESP_BAND)
    rb = int(((f >= RESP_BAND[0]) & (f < RESP_BAND[1])).sum())
    r_ratio = (float(_amp_at(f, glob, f_resp)) / gsig0) if gsig0 > 0 else 0.0
    r_thr = _rayleigh_threshold(rb, 1, expected_false=0.01)
    if r_ratio >= r_thr:
        glob_h = _notch(f, glob, f_resp)
    else:
        glob_h = glob
        notes.append(f"respiration not detectable in the spatial mean "
                     f"({r_ratio:.1f}x, needs {r_thr:.1f}x) -- its harmonic "
                     f"comb was NOT notched out of the cardiac band")
        f_resp = np.nan
    f_pulse = _fundamental(f, glob_h, *PULSE_BAND)

    gsig = gsig0
    g_ratio = float(_amp_at(f, glob_h, f_pulse)) / gsig if gsig > 0 else 0.0
    g_thr = _rayleigh_threshold(int(((f >= PULSE_BAND[0]) &
                                     (f < PULSE_BAND[1])).sum()), 1,
                                expected_false=0.01)

    # ---- the statistic ---------------------------------------------------
    if g_ratio >= g_thr:
        method = "cardiac"
        # The heart rate belongs to the ANIMAL, so every pixel is interrogated
        # at the same DFT bin.  No per-pixel search means no search penalty.
        i_hr = int(np.argmin(np.abs(f - f_pulse)))
        bins = [i_hr]
        for h in range(2, n_harmonics + 1):
            j = int(np.argmin(np.abs(f - f_pulse * h)))
            if f[j] < f[-1] and j != i_hr:
                bins.append(j)
        S, sig_s, k_eff = _coherent_statistic(X, sigma, bins, lit,
                                              mean_img.shape, smooth_sigma)
        n_eff = max(float(lit.sum()) * k_eff, 1.0)
        thr = _gamma_threshold(len(bins), n_eff, expected_false)
        ratio = np.sqrt(np.maximum(S, 0.0))
        pulse_amp = (np.abs(_smooth_complex(
            X[i_hr].reshape(mean_img.shape), lit, smooth_sigma))
            if np.isfinite(f_pulse) else np.zeros(mean_img.shape))
    else:
        method = "cardiac_band"
        notes.append(f"no cardiac peak in the spatial mean "
                     f"({g_ratio:.1f}x, needs {g_thr:.1f}x) -- falling back to "
                     f"the band maximum, which cannot be spatially averaged "
                     f"(each pixel's peak is in a different bin) and is "
                     f"therefore {1/np.sqrt(max(k_from_sigma(smooth_sigma),1e-9)):.0f}x "
                     f"less sensitive.  Treat this recording's physiology as "
                     f"suspect, not just its mask.")
        pb = (f >= PULSE_BAND[0]) & (f < PULSE_BAND[1])
        num = A[pb].max(axis=0).astype(float)
        thr = _rayleigh_threshold(int(pb.sum()), npx, expected_false)
        ratio = (num / sigma).reshape(mean_img.shape)
        pulse_amp = np.asarray(_amp_at(f, A, f_pulse)).reshape(mean_img.shape) \
            if np.isfinite(f_pulse) else np.zeros(mean_img.shape)
        n_eff = float(npx)

    # ---- excess variance, always computed, used only as a last resort ----
    # counts = GAIN * photoelectrons + OFFSET, so a shot-noise-limited pixel
    # has var = GAIN * (mean - OFFSET), reduced by the binning factor because
    # binning averages independent pixels.
    # Parseval for this normalisation: var = 0.5 * sum of squared amplitudes.
    var_meas = 0.5 * np.nansum(A.astype(np.float64) ** 2,
                               axis=0).reshape(mean_img.shape)
    shot = GAIN / np.maximum(mean_img - OFFSET, 1.0) / (bin_xy ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        excess_var = var_meas / np.where(shot > 0, shot, np.nan)

    raw = lit & np.isfinite(ratio) & (ratio > thr)
    if raw.sum() * (um_per_pixel / 1000.0) ** 2 < MIN_AREA_MM2:
        method = "excess_var"
        notes.append("no pixel anywhere has a detectable pulse.  The mask "
                     "below is ANATOMICAL (variance above shot noise) and "
                     "carries no evidence that the tissue it covers is "
                     "perfused or that the optical path is working.  Fix the "
                     "preparation before trusting anything downstream.")
        ev = gaussian_filter(np.nan_to_num(excess_var), 2.0)
        lo = np.nanpercentile(ev[lit], 50)
        raw = lit & (ev > max(lo, 2.0))

    bleed = _bleed_px(ratio, raw, thr, smooth_sigma)
    mask = _morphology(raw, um_per_pixel,
                       erode_iter=max(ERODE_ITER, bleed))
    if bleed > ERODE_ITER:
        notes.append(f"eroded {bleed} px to undo the dilation from "
                     f"{smooth_sigma:g}-px coherent smoothing (peak signal "
                     f"{np.nanpercentile(ratio[raw], 99):.0f}x threshold's "
                     f"worth of bleed)")

    # Is the mask plausibly a window, or has it traced the vasculature?
    # The illuminated, unsaturated field is an upper bound on how much of the
    # frame could be brain.  A mask that is a small fraction of it, and in
    # several thin pieces, is a sensitivity failure rather than a small
    # window -- say so, with the lever that fixes it.
    frac = mask.sum() / max(lit.sum(), 1)
    n_pieces = int(label(mask)[1])
    if method != "excess_var" and (frac < 0.15 or
                                   (n_pieces > 3 and frac < 0.3)):
        gain = 1.0 / np.sqrt(max(k_from_sigma(smooth_sigma), 1e-9))
        notes.append(
            f"the mask covers only {100*frac:.0f}% of the illuminated field, "
            f"in {n_pieces} piece(s).  That is the signature of thresholding "
            f"the vasculature rather than the tissue: the pulse clears "
            f"threshold over surface vessels and not over parenchyma.  "
            f"Currently averaging {1/max(k_from_sigma(smooth_sigma),1e-9):.0f} "
            f"px coherently ({gain:.0f}x).  Raise SMOOTH_SIGMA (3.0 gives "
            f"{1/np.sqrt(k_from_sigma(3.0)):.0f}x) or lengthen the window -- "
            f"noise falls as 1/sqrt(n_frames), so 4x the frames is 2x.")

    tm = TissueMask(mask=mask, raw=raw, ratio=ratio, threshold=float(thr),
                    f_pulse=float(f_pulse), f_resp=float(f_resp),
                    pulse_amp=pulse_amp, excess_var=excess_var,
                    mean_img=mean_img, method=method, n_frames=n, fs=float(fs),
                    bin_xy=int(bin_xy), um_per_pixel=float(um_per_pixel),
                    smooth_sigma=float(smooth_sigma), n_eff=float(n_eff),
                    global_ratio=float(g_ratio), lit=lit, notes=notes)
    return tm


# ------------------------------------------------------------ persistence ----

DEFAULT_PATH = "tissue_mask.npz"


def save(tm: TissueMask, path: str = DEFAULT_PATH):
    np.savez_compressed(
        path, mask=tm.mask, raw=tm.raw, ratio=tm.ratio,
        threshold=tm.threshold, f_pulse=tm.f_pulse, f_resp=tm.f_resp,
        pulse_amp=tm.pulse_amp, excess_var=tm.excess_var,
        mean_img=tm.mean_img, method=tm.method, n_frames=tm.n_frames,
        fs=tm.fs, bin_xy=tm.bin_xy, um_per_pixel=tm.um_per_pixel,
        smooth_sigma=tm.smooth_sigma, n_eff=tm.n_eff,
        global_ratio=tm.global_ratio,
        lit=(tm.lit if tm.lit is not None else np.zeros_like(tm.mask)),
        notes=np.array(tm.notes, dtype=object), source=np.array(ioi.DAT_PATH),
    )
    return path


def load(path: str = DEFAULT_PATH, shape=None, bin_xy=None,
         require_source=True) -> TissueMask | None:
    """
    Read a saved mask, or return None if there isn't a usable one.

    Refuses a mask built from a different recording or at a different binning
    rather than silently applying it -- the whole point of an automatic mask is
    that it belongs to the data it was measured from.
    """
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    if require_source and "source" in z.files \
            and str(z["source"]) != str(ioi.DAT_PATH):
        print(f"  !! {path} was built from a different recording:\n"
              f"       mask  : {str(z['source'])}\n"
              f"       config: {ioi.DAT_PATH}\n"
              f"     ignoring it -- re-run block0c_mask.py")
        return None
    if bin_xy is not None and int(z["bin_xy"]) != int(bin_xy):
        print(f"  !! {path} was built at bin_xy={int(z['bin_xy'])}, "
              f"this analysis uses {bin_xy} -- ignoring it")
        return None
    if shape is not None and tuple(z["mask"].shape) != tuple(shape):
        print(f"  !! {path} has shape {tuple(z['mask'].shape)}, "
              f"expected {tuple(shape)} -- ignoring it")
        return None
    return TissueMask(
        mask=z["mask"], raw=z["raw"], ratio=z["ratio"],
        threshold=float(z["threshold"]), f_pulse=float(z["f_pulse"]),
        f_resp=float(z["f_resp"]), pulse_amp=z["pulse_amp"],
        excess_var=z["excess_var"], mean_img=z["mean_img"],
        method=str(z["method"]), n_frames=int(z["n_frames"]),
        fs=float(z["fs"]), bin_xy=int(z["bin_xy"]),
        um_per_pixel=float(z["um_per_pixel"]),
        smooth_sigma=float(z["smooth_sigma"]) if "smooth_sigma" in z.files
        else 0.0,
        n_eff=float(z["n_eff"]) if "n_eff" in z.files else 0.0,
        global_ratio=float(z["global_ratio"]) if "global_ratio" in z.files
        else np.nan,
        lit=z["lit"] if "lit" in z.files else None,
        notes=[str(x) for x in np.atleast_1d(z["notes"])] if "notes" in z.files
        else [],
    )


def evaluation_mask(mean_image, bin_xy, um_per_pixel, offset=OFFSET,
                    min_counts=MIN_COUNTS, bright_lo=60.0, bright_hi=99.5,
                    roi=None, path: str = DEFAULT_PATH, verbose=True):
    """
    The one function the block scripts call.

    Prefers the measured tissue mask; falls back to the old brightness band if
    there isn't one, so a recording analysed before block0c_mask.py existed
    still runs.  `roi` is applied on top of whichever mask is used, and exists
    only as a manual override -- it should normally be None.

    Returns (mask, description).
    """
    shape = mean_image.shape
    tm = load(path, shape=shape, bin_xy=bin_xy)
    if tm is not None:
        m = tm.mask.copy()
        how = (f"measured tissue mask ({tm.method}, HR "
               f"{tm.f_pulse*60:.0f}/min, {tm.area_mm2:.2f} mm2)")
        if verbose:
            print(f"  mask: {how}")
            for nt in tm.notes:
                print(f"    NOTE: {nt}")
    else:
        I = mean_image - offset
        valid = I > min_counts
        lo, hi = np.percentile(I[valid], [bright_lo, bright_hi])
        m = valid & (I >= lo) & (I <= hi)
        how = f"brightness band {bright_lo:g}-{bright_hi:g} pct (NO pulse mask)"
        if verbose:
            print(f"  mask: {how}")
            print(f"    no usable {path} -- run block0c_mask.py to replace "
                  f"this with a measured one")
    if roi is not None:
        r0, r1, c0, c1 = roi
        keep = np.zeros(shape, dtype=bool)
        keep[r0:r1, c0:c1] = True
        before = int(m.sum())
        m = m & keep
        how += f" + manual ROI {roi}"
        if verbose:
            print(f"    manual ROI {roi}: {before} -> {int(m.sum())} px")
    if verbose:
        print(f"    {int(m.sum())} px = "
              f"{m.sum() * (um_per_pixel/1000.0)**2:.2f} mm2")
    return m, how
