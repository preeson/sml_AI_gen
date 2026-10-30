"""
block0b_physio.py -- is the optical path seeing haemodynamics at all?

Completely model-free: no stimulus log, no folding, no assumption about when
anything happened.  It reads a window of raw frames at the full camera rate
and looks for the two signals that MUST be present if red light is reaching
perfused cortex and coming back:

    respiration   ~1-3 Hz in an anaesthetised mouse
    heartbeat     ~5-12 Hz

Both are ordinary blood-volume changes, one to two orders of magnitude larger
than a visually evoked intrinsic signal, and neither needs the animal to see
anything.  So this splits the failure cleanly:

  * both peaks present -> the optics, window, illumination and camera are
    delivering real haemodynamic signal.  A null at the stimulus frequency is
    then about visual drive reaching cortex, not about light.
  * neither present    -> the light is not sampling perfused tissue, and
    nothing downstream can work until that is fixed.

Run it on any recording; it does not care whether a stimulus was shown.
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ioi_core as ioi

N_FRAMES = 6000        # 60 s at 100 Hz
START = None           # default: the middle of the recording
BIN_XY = 10            # 540x640 -> 54x64.  Must divide BOTH 540
                       # and 640, so only 1, 2, 4, 5, 10 or 20.
CHUNK = 250
OFFSET = 1540.0
MIN_COUNTS = 500.0
SAT_LEVEL = 55000.0    # conservative: binned means near the ceiling are
                       # compressed, and this session runs bright
BANDS = {"respiration": (0.7, 4.0), "heartbeat": (4.5, 14.0)}
OUT_PNG = "block0b_physio.png"


def load_temp_log(path):
    """
    Parse a two-column body-temperature log: seconds, degrees C.

    Written by hand at the rig, so it is forgiving.  A title line, a column
    header, blanks and comments are all skipped; tabs, commas or spaces all
    separate.  Any line whose first two fields both parse as numbers is data,
    which means the file can grow a header or a note without breaking.
    """
    sec, deg = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            sec.append(a)
            deg.append(b)
    if not sec:
        raise ValueError(f"{path}: no rows of 'seconds temperature' found")
    return np.asarray(sec, float), np.asarray(deg, float)


def collect(mov, ft, start, n, bin_xy=BIN_XY, chunk=CHUNK):
    """Stream frames, bin spatially, return (t, binned stack)."""
    H, W = mov.shape[1], mov.shape[2]
    if H % bin_xy or W % bin_xy:
        raise ValueError(f"bin_xy={bin_xy} does not divide {H}x{W} "
                         f"(try 1, 2, 4, 5, 10 or 20)")
    h, w = H // bin_xy, W // bin_xy
    out = np.empty((n, h, w), dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d = np.asarray(mov[start + a:start + b], dtype=np.float32)
        out[a:b] = d.reshape(b - a, h, bin_xy, w, bin_xy).mean(axis=(2, 4))
    return ft.t[start:start + n].copy(), out


def spectrum(t, x, fs):
    """Interpolate onto a uniform grid (handles dropped frames), detrend, FFT."""
    tu = np.linspace(t[0], t[-1], x.shape[0])
    xu = np.interp(tu, t, x)
    v = np.vander(np.linspace(-1, 1, xu.size), 4)
    xu = xu - v @ np.linalg.lstsq(v, xu, rcond=None)[0]
    X = np.abs(np.fft.rfft(xu)) * 2.0 / xu.size
    f = np.fft.rfftfreq(xu.size, 1.0 / fs)
    return f, X


def band_threshold(n_bins, alpha=0.01):
    """
    How far above the local median the LARGEST bin in a band has to stand
    before it means anything.

    The amplitude of a pure-noise DFT bin is Rayleigh distributed, so
    P(one bin > x * median) = exp(-ln2 * x^2) and the largest of n_bins
    exceeds that with probability ~n_bins times as often.  Setting the
    false-positive rate per band to alpha gives

        x = sqrt( ln(n_bins / alpha) / ln 2 )

    which is about 3.8-4.0 for the band widths used here -- a fixed "3x"
    rule of thumb calls pure noise a heartbeat roughly one time in three.
    """
    return float(np.sqrt(np.log(max(n_bins, 1) / alpha) / np.log(2.0)))


def _at(f, X, target, halfwidth=1):
    """Amplitude near a frequency, max over +-halfwidth bins."""
    if not np.isfinite(target) or target <= 0 or target >= f[-1]:
        return 0.0
    i = int(np.searchsorted(f, target))
    a, b = max(0, i - halfwidth), min(X.size, i + halfwidth + 1)
    return float(X[a:b].max()) if b > a else 0.0


def fundamental(f, X, lo, hi, weights=(1.0, 0.5, 0.25)):
    """
    Pick the FUNDAMENTAL in [lo, hi), not simply the tallest bin.

    The tallest peak in a band is often not the fundamental.  A pulsatile
    waveform can carry more energy in its second harmonic than in its first,
    so argmax returns twice the true rate.  Scoring each candidate by its own
    amplitude plus a decaying contribution from its harmonics fixes that.

    It does NOT resolve the opposite case.  If the spectrum has peaks at
    f/2, f and 2f with f/2 the tallest, "fundamental f/2 with harmonics" and
    "fundamental f with a subharmonic" are both consistent with the data and
    no scoring rule can separate them without more information.  That case is
    reported rather than guessed: harmonics() prints the three amplitudes and
    main() warns when f/2 is the larger, so a halved heart rate shows up as a
    flagged ambiguity instead of a confident bradycardia.
    """
    b = np.flatnonzero((f >= lo) & (f < hi))
    if b.size == 0:
        return np.nan, np.nan
    score = np.zeros(b.size)
    for k, w in enumerate(weights, start=1):
        score += w * np.array([_at(f, X, fb * k) for fb in f[b]])
    j = int(b[int(np.argmax(score))])
    # The score at a candidate uses the best bin within +-1 (so that a peak
    # landing between bins still scores), which means the winning candidate
    # can sit one bin off the actual peak and have little amplitude of its
    # own.  Snap to the local maximum before reporting.
    a, c = max(0, j - 1), min(X.size, j + 2)
    j = a + int(np.argmax(X[a:c]))
    return float(f[j]), float(X[j])


def peak_in(f, X, lo, hi, alpha=0.01):
    """Fundamental in a band, its height above the local floor, the threshold."""
    b = (f >= lo) & (f < hi)
    if not b.any():
        return np.nan, np.nan, np.nan, np.nan
    fp, amp = fundamental(f, X, lo, hi)
    wide = (f >= max(0.3, lo - 2.0)) & (f < hi + 2.0)
    floor = np.median(X[wide])
    rel = amp / floor if floor > 0 else np.nan
    return fp, amp, rel, band_threshold(int(b.sum()), alpha)


def harmonics(f, X, f0):
    """Amplitudes at f0/2, f0 and 2*f0 -- makes sub/harmonic ambiguity visible."""
    return _at(f, X, f0 / 2.0), _at(f, X, f0), _at(f, X, 2.0 * f0)


def rate_warning(f, X, f0, hi=0.8, lo=1.3):
    """
    Say so when the reported rate could be double or half.

    A rate read off a spectrum is only unambiguous if there is nothing
    comparable at twice or half of it.  When there is, both readings fit the
    data and the difference is a factor of two in heart rate -- the
    difference between "normal" and "profoundly bradycardic".  Better to say
    which, than to report one and let it become a finding.
    """
    h_half, h_one, h_two = harmonics(f, X, f0)
    if h_one <= 0:
        return ""
    if h_two > hi * h_one:
        return f"could also be {2 * f0 * 60:.0f}/min -- 2f is nearly as large"
    if h_half > lo * h_one:
        return f"could also be {f0 * 30:.0f}/min -- f/2 is larger"
    return ""


def measure(mov, ft, start, n, fs):
    """HR, RR and their amplitudes from one window.  Returns a dict."""
    t, stack = collect(mov, ft, start, n)
    mean_img = stack.mean(axis=0)
    mask = ((mean_img - OFFSET) > MIN_COUNTS) & (mean_img < SAT_LEVEL)
    if mask.sum() < 20:
        raise RuntimeError("mask is empty -- check OFFSET / SAT_LEVEL")
    denom = np.where(mask, mean_img - OFFSET, np.nan)
    dr = (stack - mean_img[None]) / denom[None]
    glob = np.nanmean(dr[:, mask], axis=1)
    f, X = spectrum(t, glob, fs)
    per_frame = np.nanstd(dr[:, mask], axis=0)
    out = {"t0": t[0], "t": t, "glob": glob, "f": f, "X": X,
           "mask": mask, "mean_img": mean_img,
           "per_frame": float(np.median(per_frame)),
           "floor": float(np.median(per_frame) / np.sqrt(mask.sum())
                          * np.sqrt(2.0 / n))}
    for name, (lo, hi) in BANDS.items():
        out[name] = peak_in(f, X, lo, hi)
    return out


def track(mov, ft, fs, n_win, win, temp=None):
    """
    Heart and respiratory rate across the whole session.

    In a mouse, heart rate tracks core temperature closely, so this is the
    body-temperature record that a session without a rectal probe otherwise
    does not have -- a falling HR over a session is cooling, and a flat low
    HR is an animal that was cold from the start.
    """
    starts = np.linspace(0, ft.n_frames - win, n_win).astype(int)
    rows = []
    for i, st in enumerate(starts):
        r = measure(mov, ft, int(st), win, fs)
        rows.append(r)
        hr, ha, hrel, hthr = r["heartbeat"]
        rr, ra, rrel, rthr = r["respiration"]
        w = rate_warning(r["f"], r["X"], hr)
        print(f"  {r['t0']:7.1f} s   HR {hr * 60:5.0f}/min "
              f"({'ok ' if hrel >= hthr else 'weak'}, {100 * ha:.4f}%)   "
              f"RR {rr * 60:5.0f}/min "
              f"({'ok ' if rrel >= rthr else 'weak'}, {100 * ra:.4f}%)"
              + (f"  {np.interp(r['t0'], temp[0], temp[1]):5.2f} C"
                 if temp is not None else "")
              + (f"   [{w}]" if w else ""))
    return starts, rows


def main():
    args = [a for a in sys.argv[1:]]
    do_track = "--track" in args
    if do_track:
        args.remove("--track")
    temp = None
    if "--temp" in args:
        i = args.index("--temp")
        ts, td = load_temp_log(args[i + 1])
        temp = (ts, td)
        del args[i:i + 2]
        print(f"body temperature: {td.min():.1f}-{td.max():.1f} C over "
              f"{ts[-1] - ts[0]:.0f} s (mean {td.mean():.2f})")
    n = int(args[0]) if args else (3000 if do_track else N_FRAMES)
    ft = ioi.load_frame_times()
    start = (ft.n_frames - n) // 2 if START is None else START
    fs = (ft.n_frames - 1) / (ft.t[-1] - ft.t[0])
    print(f"camera {fs:.4f} Hz; window {n} frames ({n / fs:.1f} s, "
          f"{n * ioi.BYTES_PER_FRAME / 1024**3:.1f} GiB), {BIN_XY}x binned")
    if do_track:
        pass
    else:
        print(f"reading frames {start}-{start + n} "
              f"from the middle of the recording")

    if do_track:
        n_win = 8
        print(f"tracking {n_win} windows of {n / fs:.0f} s across the session\n")
        with ioi.movie() as mov:
            starts, rows = track(mov, ft, fs, n_win, n, temp)
        hr = np.array([r["heartbeat"][0] for r in rows]) * 60
        rr = np.array([r["respiration"][0] for r in rows]) * 60
        tt = np.array([r["t0"] for r in rows]) / 60.0
        ok = np.array([r["heartbeat"][2] >= r["heartbeat"][3] for r in rows])
        print(f"\n  heart rate   {hr.min():.0f} - {hr.max():.0f}/min "
              f"(drift {hr[-1] - hr[0]:+.0f} over {tt[-1] - tt[0]:.0f} min)")
        print(f"  respiration  {rr.min():.0f} - {rr.max():.0f}/min "
              f"(drift {rr[-1] - rr[0]:+.0f})")
        if ok.sum() >= 4 and hr[ok].max() > 0:
            slope = np.polyfit(tt[ok], hr[ok], 1)[0]
            print(f"  HR trend {slope:+.1f}/min per minute of session")
            if slope < -2:
                print("  -> falling heart rate: the animal is cooling or "
                      "going deeper during the recording.")
        if temp is not None:
            tw = np.interp([r["t0"] for r in rows], temp[0], temp[1])
            print(f"  body temperature at the windows: {tw.min():.2f}-"
                  f"{tw.max():.2f} C")
            if ok.sum() >= 4 and tw.max() - tw.min() >= 0.3:
                k = np.polyfit(tw[ok], hr[ok], 1)[0]
                print(f"  HR vs temperature: {k:+.0f} bpm per degree C")
            elif tw.max() - tw.min() < 0.3:
                print("  temperature held to better than 0.3 C -- no range "
                      "to fit HR against, which is what you want")
        ncol = 3 if temp is not None else 2
        fig, ax = plt.subplots(1, ncol, figsize=(5.5 * ncol, 4))
        ax[0].plot(tt, hr, "o-"); ax[0].set_ylabel("heart rate (/min)")
        ax[0].axhspan(500, 650, alpha=0.12, color="tab:green")
        ax[1].plot(tt, rr, "o-", color="tab:orange")
        ax[1].set_ylabel("respiration (/min)")
        ax[1].axhspan(60, 120, alpha=0.12, color="tab:green")
        if temp is not None:
            ax[2].plot(temp[0] / 60.0, temp[1], "-", color="tab:red")
            ax[2].set_ylabel("body temperature (C)")
            ax[2].axhspan(36.5, 37.5, alpha=0.12, color="tab:green")
        for a in ax:
            a.set_xlabel("session time (min)"); a.grid(alpha=0.3)
        fig.suptitle("vital signs across the session "
                     "(shaded = typical anaesthetised mouse)")
        fig.tight_layout(); fig.savefig("block0b_vitals.png", dpi=110)
        plt.close(fig)
        print("\nwrote block0b_vitals.png")
        return

    with ioi.movie() as mov:
        t, stack = collect(mov, ft, start, n)

    mean_img = stack.mean(axis=0)
    mask = ((mean_img - OFFSET) > MIN_COUNTS) & (mean_img < SAT_LEVEL)
    print(f"mask {mask.sum()}/{mask.size} binned px "
          f"({100 * mask.mean():.0f}%), mean {mean_img[mask].mean():.0f} counts")
    if mask.sum() < 20:
        raise RuntimeError("mask is empty -- check OFFSET / SAT_LEVEL")

    # dR/R per binned pixel, then the spatial mean over the mask
    denom = np.where(mask, mean_img - OFFSET, np.nan)
    dr = (stack - mean_img[None]) / denom[None]
    glob = np.nanmean(dr[:, mask], axis=1)

    f, X = spectrum(t, glob, fs)
    per_frame = np.nanstd(dr[:, mask], axis=0)
    floor = np.median(per_frame) / np.sqrt(mask.sum()) * np.sqrt(2.0 / n)
    print(f"per-pixel per-frame noise {100 * np.median(per_frame):.3f} %dR/R")
    print(f"expected spectral floor for the masked mean "
          f"{100 * floor:.5f} %dR/R\n")

    print(f"{'band':<12}{'peak Hz':>9}{'/min':>7}{'amplitude':>12}"
          f"{'vs floor':>10}{'needs':>8}{'verdict':>9}")
    found = {}
    for name, (lo, hi) in BANDS.items():
        fp, amp, rel, thr = peak_in(f, X, lo, hi)
        found[name] = (fp, amp, rel, thr)
        print(f"{name:<12}{fp:9.3f}{fp * 60:7.0f}{100 * amp:11.4f}%"
              f"{rel:9.1f}x{thr:7.1f}x{'  FOUND' if rel >= thr else '  --':>9}")
        h2, h1, h3 = harmonics(f, X, fp)
        warn = rate_warning(f, X, fp)
        print(f"{'':<12}harmonics  f/2 {100 * h2:.4f}%  f {100 * h1:.4f}%  "
              f"2f {100 * h3:.4f}%" + (f"   AMBIGUOUS: {warn}" if warn else ""))

    verdict = all(found[k][2] >= found[k][3] for k in BANDS)
    print()
    if verdict:
        print("  Both physiological rhythms are present.  Red light IS reaching")
        print("  perfused cortex and the detection chain works.  A null at the")
        print("  stimulus frequency is about visual drive, not about light.")
    else:
        weak = [k for k in BANDS if found[k][2] < found[k][3]]
        print(f"  NOT DETECTED: {', '.join(weak)}.")
        print("  The optical path is not reporting blood volume changes that")
        print("  must be there in perfused cortex.  Fix this before any")
        print("  stimulus work -- no visual signal can survive it.")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].imshow(np.where(mask, mean_img, np.nan), cmap="gray")
    ax[0].set_title(f"mean of {n} frames ({BIN_XY}x binned, masked)")
    ax[1].plot(t - t[0], 100 * glob, lw=0.5)
    ax[1].set_xlim(0, min(10, t[-1] - t[0]))
    ax[1].set_xlabel("s"); ax[1].set_ylabel("%dR/R")
    ax[1].set_title("masked-mean time course (first 10 s)")
    ax[2].loglog(f[1:], 100 * X[1:], lw=0.7)
    for name, (lo, hi) in BANDS.items():
        ax[2].axvspan(lo, hi, alpha=0.12,
                      color=("tab:green" if found[name][2] >= found[name][3]
                             else "tab:red"))
        if np.isfinite(found[name][0]):
            ax[2].axvline(found[name][0], ls="--", lw=0.8, color="k")
    ax[2].axhline(100 * floor, ls=":", color="gray", lw=0.8)
    ax[2].set_xlabel("Hz"); ax[2].set_ylabel("%dR/R")
    ax[2].set_title("spectrum of the masked mean (shaded = looked-for bands)")
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=110); plt.close(fig)
    print(f"\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
