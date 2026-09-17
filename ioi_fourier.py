"""
ioi_fourier.py -- reduce one stimulus block to a cycle-resolved time series and
extract the Fourier component at the stimulus frequency.

Design decisions that matter:

* Binning is by STIMULUS PHASE, not by frame index.  Each camera frame is
  assigned to a phase bin using its own timestamp, so dropped frames and clock
  jitter are handled exactly rather than assumed away.

* The number of bins is an exact multiple of the cycle count (n_bins =
  bins_per_cycle * 15).  The stimulus therefore lands precisely on DFT bin 15
  with zero leakage, and a nuisance oscillation one bin away is orthogonal to
  it -- provided no taper is applied.

* NO WINDOW FUNCTION.  A Hann or Hamming taper would smear the neighbouring
  bin into the signal bin, which is exactly the failure mode we are guarding
  against (ch4 carries a ~0.064 Hz oscillation ~1 bin from the elevation
  stimulus frequency).  Rectangular window is deliberate.

* dR/R uses (I - offset) in the denominator.  The camera pedestal (~1540
  counts here) carries no photons; dividing by raw counts understates every
  fractional signal by ~18%.  Phase is unaffected, amplitudes are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import ioi_core as ioi


# --------------------------------------------------------------------------
# Folding a block into phase bins
# --------------------------------------------------------------------------

@dataclass
class FoldedBlock:
    label: str
    cycles: np.ndarray      # (n_bins, h, w) float32, mean raw counts per bin
    n_per_bin: np.ndarray   # (n_bins,) how many camera frames landed in each
    mean_image: np.ndarray  # (h, w) float32, block mean
    bins_per_cycle: int
    n_cycles: int
    bin_xy: int
    duration: float
    f_stim: float
    n_empty: int = 0        # bins that had none, filled from a neighbour

    @property
    def n_bins(self) -> int:
        return self.cycles.shape[0]

    @property
    def fs(self) -> float:
        """Effective sampling rate of the folded series, Hz."""
        return self.n_bins / self.duration


def fold_block(mov, ft, block, bins_per_cycle: int = 80, bin_xy: int = 4,
               chunk: int = 256, progress=None) -> FoldedBlock:
    """
    Stream one stimulus block and average camera frames into phase bins.

    mov   : (n_frames, H, W) array-like -- a memmap view from ioi.open_movie()
    ft    : FrameTimes
    block : StimBlock
    bins_per_cycle : phase bins per stimulus cycle.  80 gives ~4.9 Hz
            effective sampling and a Nyquist of 40x the stimulus frequency.
    bin_xy : spatial binning factor (4 -> 135x160 from 540x640)
    """
    H, W = mov.shape[1], mov.shape[2]
    if H % bin_xy or W % bin_xy:
        raise ValueError(f"bin_xy={bin_xy} does not divide {H}x{W}")
    h, w = H // bin_xy, W // bin_xy

    n_bins = bins_per_cycle * block.n_cycles
    c0, c1 = block.cam0, block.cam1

    # phase-bin edges, uniform in time across the block
    t = ft.t[c0:c1]
    t0 = ft.t[c0]
    # The block ends where the NEXT frame begins, not at the last frame's
    # timestamp -- otherwise the window is short by one frame and the
    # stimulus no longer lands exactly on bin 15.
    if c1 < ft.n_frames:
        t1 = ft.t[c1]
    else:
        t1 = ft.t[-1] + float(np.median(np.diff(ft.t[-1000:])))
    idx = np.floor((t - t0) / (t1 - t0) * n_bins).astype(np.int64)
    np.clip(idx, 0, n_bins - 1, out=idx)

    sums = np.zeros((n_bins, h, w), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for a in range(c0, c1, chunk):
        b = min(a + chunk, c1)
        data = np.asarray(mov[a:b], dtype=np.float32)
        if bin_xy > 1:
            data = data.reshape(b - a, h, bin_xy, w, bin_xy).mean(axis=(2, 4))
        sub = idx[a - c0:b - c0]
        # frames are time-ordered, so each chunk spans a contiguous bin range
        for u in np.unique(sub):
            m = sub == u
            sums[u] += data[m].sum(axis=0)
            counts[u] += int(m.sum())
        if progress is not None:
            progress(b - c0, c1 - c0)

    # Isolated empty bins can occur if the camera dropped frames.  Fill them
    # by interpolating between neighbours rather than failing, but refuse if
    # too many are empty -- that means bins_per_cycle is simply too high.
    empty = counts == 0
    n_empty = int(empty.sum())
    if n_empty:
        frac = n_empty / n_bins
        if frac > 0.02:
            raise RuntimeError(
                f"{block.label}: {n_empty}/{n_bins} phase bins are empty "
                f"({frac:.1%}) -- lower bins_per_cycle")
        good = np.flatnonzero(~empty)
        filled = np.interp(np.flatnonzero(empty), good,
                           np.arange(good.size)).round().astype(int)
        sums[empty] = sums[good[filled]]
        counts[empty] = counts[good[filled]]

    cycles = (sums / counts[:, None, None]).astype(np.float32)
    mean_image = (sums.sum(axis=0) / counts.sum()).astype(np.float32)

    return FoldedBlock(
        label=block.label, cycles=cycles, n_per_bin=counts,
        n_empty=n_empty,
        mean_image=mean_image, bins_per_cycle=bins_per_cycle,
        n_cycles=block.n_cycles, bin_xy=bin_xy,
        duration=t1 - t0, f_stim=block.f_stim,
    )


# --------------------------------------------------------------------------
# Spectral estimation
# --------------------------------------------------------------------------

@dataclass
class BlockSpectrum:
    label: str
    component: np.ndarray   # (h, w) complex, Fourier coefficient at bin 15
    amplitude: np.ndarray   # (h, w) float, |component| in dR/R units
    phase: np.ndarray       # (h, w) float, radians in [-pi, pi)
    noise: np.ndarray       # (h, w) float, per-quadrature sd from neighbouring
                            #   bins; phase error ~ 1/snr radians
    snr: np.ndarray         # (h, w) float, amplitude / noise
    freqs: np.ndarray       # (n_bins//2+1,) Hz
    power_mean: np.ndarray  # mean power spectrum over valid pixels
    k_stim: int
    mask: np.ndarray        # (h, w) bool, pixels included


def analyse(fold: FoldedBlock, offset: float = 1540.0, detrend_order: int = 3,
            min_counts: float = 500.0,
            noise_bins=(11, 12, 13, 14, 16, 17, 18, 19)) -> BlockSpectrum:
    """
    Convert a folded block to dR/R, detrend, and read off DFT bin 15.

    offset     : camera pedestal in counts.  Amplitudes scale with this;
                 phase does not.
    min_counts : pixels whose block-mean is below offset + min_counts are
                 masked out.  Without this, vignetted / dental-cement pixels
                 divide by ~0 and produce enormous meaningless phases.
    """
    n_bins, h, w = fold.cycles.shape
    k_stim = fold.n_cycles  # exactly, by construction

    denom = fold.mean_image - offset
    mask = denom > min_counts

    x = (fold.cycles - fold.mean_image[None]) / np.where(mask, denom, np.nan)[None]
    x = x.reshape(n_bins, -1)

    # polynomial detrend across the block (removes lamp/physiological drift)
    if detrend_order is not None and detrend_order > 0:
        tt = np.linspace(-1, 1, n_bins)
        V = np.vander(tt, detrend_order + 1)
        good = np.isfinite(x).all(axis=0)
        coef = np.linalg.lstsq(V, np.nan_to_num(x[:, good]), rcond=None)[0]
        x[:, good] -= V @ coef

    X = np.fft.rfft(np.nan_to_num(x), axis=0)          # rectangular window
    scale = 2.0 / n_bins                                # -> amplitude units
    comp = X[k_stim] * scale

    # Noise is reported as the per-quadrature standard deviation, i.e. the
    # scatter of the real (or imaginary) part of the coefficient.  The rms
    # MAGNITUDE over the neighbouring bins is sqrt(2) larger, so divide it
    # out.  With this convention, white noise of sd s over N frames gives
    # noise = s*sqrt(2/N), and the single-pixel phase error is ~1/snr radians.
    nb = np.asarray(noise_bins)
    noise = np.sqrt((np.abs(X[nb] * scale) ** 2).mean(axis=0) / 2.0)

    freqs = np.fft.rfftfreq(n_bins, 1.0 / fold.fs)
    pw = (np.abs(X * scale) ** 2)
    power_mean = pw[:, mask.ravel()].mean(axis=1)

    amp = np.abs(comp).reshape(h, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = amp / noise.reshape(h, w)

    return BlockSpectrum(
        label=fold.label,
        component=comp.reshape(h, w),
        amplitude=amp,
        phase=np.angle(comp).reshape(h, w),
        noise=noise.reshape(h, w),
        snr=snr,
        freqs=freqs,
        power_mean=power_mean,
        k_stim=k_stim,
        mask=mask,
    )
