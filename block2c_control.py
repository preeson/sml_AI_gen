"""
Block 2c -- the off-bin control.

THE PROBLEM THIS SOLVES
-----------------------
Block 2b reported a spatial-structure z of -121 for the residual phase, which
looks overwhelming.  It is not evidence of anything.  The shuffle test in
ioi_leak compares neighbouring pixels against a spatially shuffled control,
and REAL imaging noise is spatially correlated (4x binning, the optical PSF,
shared physiology).  Pure correlated noise with no signal at all gives:

    correlation length   neighbour diff   z
        0 px                 89.9 deg     -0.5
        0.5 px               75.3 deg    -69.9
        1.0 px               39.1 deg   -224.5
        2.0 px               20.2 deg   -325.8

Block 2b measured 44.1 deg, i.e. a correlation length under 1 px.  Fully
explained without any signal.

The forward/reverse agreement is no better as evidence: if both directions
share a direction-INDEPENDENT artefact, phi_F = phi_R, and subtracting a
smoothed delay makes P_F = P_R exactly.  Agreement collapses to zero for a
shared artefact just as it does for real retinotopy.  It discriminates
against independent noise only.

THE CONTROL
-----------
Run the ENTIRE pipeline -- common-mode removal, structure score, delay,
position, agreement -- at DFT bins either side of the stimulus bin.  Those
bins contain no stimulus but carry identical spatial noise correlation, the
same vasculature and the same optics.  Whatever the stimulus bin does, the
off-bins should do too, unless there is genuine signal.

This needs no model of the noise, which is exactly why it is trustworthy.
Read the final table: if bin 15 is not an outlier against its neighbours,
there is nothing at the stimulus frequency.

Also fixed here: the common mode is now FITTED on the full valid field (where
I spans ~150x) and only EVALUATED in the bright band.  Fitting inside the
bright band alone was ill-conditioned -- 1/I varies only 3.6x there, so
alpha/I and beta are nearly collinear, which is why Block 2b returned alphas
of 0.151, 1.707, 1.377 and 0.212 counts for what should be one stray-light
path.
"""

import os
import time

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ioi_core as ioi
import ioi_fourier as iof
import ioi_leak as iol
import ioi_maps as iom

BIN_XY = 4
BINS_PER_CYCLE = 80
CHUNK = 256
OFFSET = 1540.0
MIN_COUNTS = 500.0
UM_PER_PIXEL = 6150.0 / 640.0 * BIN_XY   # sensor FOV 6.15 mm wide
BRIGHT_LO_PCT = 60.0
BRIGHT_HI_PCT = 99.5

# Optional rectangle, in BINNED pixels, to restrict the evaluation mask:
#   ROI = (row0, row1, col0, col1)
# The brightness mask alone can pick up skull, dental cement and anything
# else that reflects well, and every metric here is a MEDIAN over the mask.
# Non-responsive pixels do not add noise to that median -- they move it
# toward the no-signal value, so a mask that is half off-brain roughly
# halves the measured effect.  Set this to the cranial window.
ROI = (26, 79, 33, 86)
ORDER = 0
PHASE_SIGMA = 2.0
DELAY_SIGMA = 4.0

K_STIM = 15
K_LIST = [11, 12, 13, 14, 15, 16, 17, 18, 19]
LABELS = ["B2U", "U2B", "L2R", "R2L"]
PAIRS = [("elevation", "B2U", "U2B"), ("azimuth", "L2R", "R2L")]
# Screen half-extents come from the stimulus log (see ioi_core), never from a
# constant here -- they change with viewing distance and a stale value fails
# silently.  They are carried in the cache alongside the components.

# ---------------------------------------------------------------------------
# Which pass through the four directions to analyse.  DisplaySequence's
# display_iter replays the whole B2U/U2B/L2R/R2L sequence that many times in
# one recording, so a display_iter=3 session holds three independent trials.
# No argument  -> rep 0, output names unchanged (old single-pass sessions).
# An argument  -> that rep, with the rep number in the output file names.
# ---------------------------------------------------------------------------
REP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SUF = f"_rep{REP}" if len(sys.argv) > 1 else ""

CACHE = f"components_bins{SUF}.npz"
OUT_PNG = f"block2c_control{SUF}.png"


def reduce_all_bins():
    if os.path.exists(CACHE):
        print(f"loading cached multi-bin reduction from {CACHE}")
        z = np.load(CACHE, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        missing = [f"{l}_vishalf" for l in LABELS if f"{l}_vishalf" not in d]
        stale = "_source_dat" not in d
        wrong = (not stale) and str(d["_source_dat"]) != str(ioi.DAT_PATH)
        if missing or stale:
            print("  cache predates a change in the pipeline -- redoing the "
                  "reduction")
        elif wrong:
            # A fixed cache filename in the working directory is a real hazard
            # when several recordings are analysed in the same folder: without
            # this check you would silently analyse the previous recording.
            print(f"  !! cache was built from a DIFFERENT recording:\n"
                  f"       cache : {str(d['_source_dat'])}\n"
                  f"       config: {ioi.DAT_PATH}\n"
                  f"     redoing the reduction")
        else:
            return d

    ft = ioi.load_frame_times()
    log = ioi.load_stim_log(ft=ft, rep=REP)
    out = {}
    t0 = time.time()
    with ioi.movie() as mov:
        for blk in log.blocks:
            print(f"  {blk.label} ...", flush=True)
            fold = iof.fold_block(mov, ft, blk, bins_per_cycle=BINS_PER_CYCLE,
                                  bin_xy=BIN_XY, chunk=CHUNK)
            for k in K_LIST:
                sp = iof.analyse(fold, offset=OFFSET, min_counts=MIN_COUNTS,
                                 k=k, exclude_bins=(K_STIM,))
                out[f"{blk.label}_k{k}_component"] = sp.component
                out[f"{blk.label}_k{k}_noise"] = sp.noise
            out[f"{blk.label}_mask"] = sp.mask
            out[f"{blk.label}_mean"] = fold.mean_image
            out[f"{blk.label}_duration"] = np.array(blk.duration)
            out[f"{blk.label}_p0"] = np.array(blk.bar_deg_first)
            out[f"{blk.label}_p1"] = np.array(blk.bar_deg_last)
            out[f"{blk.label}_vishalf"] = np.array(blk.visible_half)
            del fold
    out["_source_dat"] = np.array(ioi.DAT_PATH)
    out["_source_nframes"] = np.array(ft.n_frames)
    print(f"  reduction took {time.time()-t0:.0f} s")
    np.savez_compressed(CACHE, **out)
    return out


class _Spec:
    def __init__(self, label, component, noise, mask):
        self.label, self.component, self.noise = label, component, noise
        self.amplitude = np.abs(component)
        self.phase = np.angle(component)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.snr = self.amplitude / noise
        self.mask = mask


def _blk(label, d, k):
    """A StimBlock whose 'stimulus frequency' is bin k of this block."""
    dur = float(d[f"{label}_duration"])
    return ioi.StimBlock(
        label=label, axis="elevation" if label in ("B2U", "U2B") else "azimuth",
        k0=0, k1=0, frames_per_cycle=1, n_cycles=k, cam0=0, cam1=1,
        t0=0.0, t1=dur,
        bar_deg_first=float(d[f"{label}_p0"]),
        bar_deg_last=float(d[f"{label}_p1"]),
        visible_half=float(d[f"{label}_vishalf"]),
    )


def metrics_for_bin(d, k, bright, valid, I):
    resid = {}
    per_block = {}
    for label in LABELS:
        sp = _Spec(label, d[f"{label}_k{k}_component"],
                   d[f"{label}_k{k}_noise"], d[f"{label}_mask"])
        r, fit = iol.remove_common_mode(sp, I, bright, order=ORDER,
                                        fit_mask=valid)
        resid[label] = r
        per_block[label] = dict(
            z=iol.structure_score(r.phase, bright)[2],
            snr=float(np.nanmedian(r.snr[bright])),
            alpha=abs(fit.alpha),
        )

    axes = {}
    for axis, f, rv in PAIRS:
        am = iom.combine_axis(resid[f], _blk(f, d, k), resid[rv],
                              _blk(rv, d, k), phase_sigma=PHASE_SIGMA,
                              delay_sigma=DELAY_SIGMA, snr_min=0.0,
                              visible_half=float(d[f"{f}_vishalf"]))
        am.mask = bright
        lim = am.visible / 2.0
        axes[axis] = dict(
            delay=float(np.nanmedian(am.delay[bright])),
            delay_chance=am.period / 4.0,
            agree=float(np.nanmedian(np.abs(am.agreement[bright]))),
            inside=float(np.mean(np.abs(am.position[bright]) <= lim) * 100),
            maps=am,
        )
    return per_block, axes, resid


def main():
    d = reduce_all_bins()

    I = d["B2U_mean"] - OFFSET
    valid = I > MIN_COUNTS
    lo = np.percentile(I[valid], BRIGHT_LO_PCT)
    hi = np.percentile(I[valid], BRIGHT_HI_PCT)
    bright = valid & (I >= lo) & (I <= hi)
    if ROI is not None:
        r0, r1, c0, c1 = ROI
        keep = np.zeros_like(bright)
        keep[r0:r1, c0:c1] = True
        before = int(bright.sum())
        bright = bright & keep
        print(f"\nROI {ROI}: {before} -> {int(bright.sum())} evaluation px")
    print(f"\nfit on {valid.sum()} px (I spans "
          f"{I[valid].min():.0f}-{I[valid].max():.0f}, "
          f"{I[valid].max()/I[valid].min():.0f}x); "
          f"evaluate on {bright.sum()} bright px "
          f"({lo:.0f}-{hi:.0f} counts, {hi/lo:.1f}x)")

    results = {k: metrics_for_bin(d, k, bright, valid, I) for k in K_LIST}
    _component_report(results, bright)
    _excess_report(results, bright)

    print("\n" + "=" * 78)
    print("PER-BIN METRICS  (bin 15 is the stimulus; all others are controls)")
    print("=" * 78)
    hdr = (f"  {'bin':>4}{'Hz(el)':>9} | {'structure z (per block)':^30} | "
           f"{'resid SNR':>10}")
    print(hdr)
    dur_el = float(d["B2U_duration"])
    for k in K_LIST:
        pb, _, _ = results[k]
        zs = " ".join(f"{pb[l]['z']:+7.1f}" for l in LABELS)
        sn = np.mean([pb[l]["snr"] for l in LABELS])
        star = "  <== STIMULUS" if k == K_STIM else ""
        print(f"  {k:>4}{k/dur_el:9.5f} | {zs} | {sn:10.2f}{star}")

    print("\n" + "=" * 78)
    print("AXIS METRICS PER BIN")
    print("=" * 78)
    for axis in ("elevation", "azimuth"):
        print(f"\n  {axis}:")
        print(f"  {'bin':>4}{'delay(s)':>10}{'chance':>9}"
              f"{'agree(deg)':>12}{'% inside':>10}")
        for k in K_LIST:
            _, ax_, _ = results[k]
            a = ax_[axis]
            star = "  <== STIMULUS" if k == K_STIM else ""
            print(f"  {k:>4}{a['delay']:10.2f}{a['delay_chance']:9.2f}"
                  f"{a['agree']:12.2f}{a['inside']:10.0f}{star}")

    _verdict(results, d)
    _figure(results, d, I, bright)
    print(f"\nwrote {OUT_PNG}")


def _outlier_z(value, controls):
    controls = np.asarray([c for c in controls if np.isfinite(c)])
    if controls.size < 3 or controls.std() == 0:
        return np.nan
    return (value - controls.mean()) / controls.std()


def _verdict(results, d):
    print("\n" + "=" * 78)
    print("VERDICT -- is bin 15 an outlier against its own neighbours?")
    print("=" * 78)
    off = [k for k in K_LIST if k != K_STIM]
    tests = []
    block_z = []

    for label in LABELS:
        v = results[K_STIM][0][label]["snr"]
        c = [results[k][0][label]["snr"] for k in off]
        z = _outlier_z(v, c)
        print(f"  residual SNR  {label}: stimulus {v:.2f}  "
              f"controls {np.mean(c):.2f} +- {np.std(c):.2f}  ->  z = {z:+.2f}")
        tests.append(z)
        block_z.append(z)

    for axis in ("elevation", "azimuth"):
        v = results[K_STIM][1][axis]["agree"]
        c = [results[k][1][axis]["agree"] for k in off]
        z = _outlier_z(v, c)
        print(f"  agreement     {axis}: stimulus {v:.2f} deg  "
              f"controls {np.mean(c):.2f} +- {np.std(c):.2f}  ->  z = {z:+.2f}")
        tests.append(-z)  # smaller is better, so flip the sign

        v = results[K_STIM][1][axis]["delay"]
        c = [results[k][1][axis]["delay"] for k in off]
        print(f"  delay         {axis}: stimulus {v:.2f} s   "
              f"controls {np.mean(c):.2f} +- {np.std(c):.2f} s   "
              f"(chance {results[K_STIM][1][axis]['delay_chance']:.2f} s)")

    # ------------------------------------------------------------------
    # Combine the four per-block tests instead of judging them one at a
    # time.  Each block is a separate test of the same hypothesis, so four
    # modest agreeing z-scores are much stronger evidence than the largest
    # of them -- and judging them singly made this test miss a real response
    # whose per-block z were +1.52, +3.58, +1.38, +2.95.
    #
    # Consistency matters as much as size.  Every direction sweeps the same
    # cortex, so a genuine response has to appear in ALL FOUR, whereas one
    # block at z = +5 with another at z = -2 is the signature of a noisy
    # estimate rather than a signal.  Requiring agreement in sign is the
    # thing that separates them.
    #
    # One-sided throughout: only an EXCESS at bin 15 is evidence of signal.
    # The old test counted abs(z), so a residual SNR far BELOW the controls
    # counted in favour.
    # ------------------------------------------------------------------
    zb = [z for z in block_z if np.isfinite(z)]
    z_comb = float(np.mean(zb) * np.sqrt(len(zb))) if zb else np.nan
    consistent = bool(zb) and all(z > 0 for z in zb)
    best = max((t for t in tests if np.isfinite(t)), default=0.0)

    print()
    print(f"  combined over the {len(zb)} blocks (Stouffer): z = {z_comb:+.2f}"
          f"   all four same sign: {'yes' if consistent else 'NO'}")
    print()
    if z_comb > 3 and consistent:
        print("  Bin 15 stands clearly apart from its neighbours: there IS")
        print("  signal at the stimulus frequency.")
    elif (z_comb > 2 and consistent) or best > 3:
        print("  Bin 15 is marginally distinguishable. Weak but not nothing;")
        print("  any map from this session is provisional at best.")
    else:
        print("  Bin 15 behaves like every other bin. There is no detectable")
        print("  signal at the stimulus frequency -- the structure seen in")
        print("  Block 2b is spatially correlated noise, which every bin has.")
        print("  No analysis can recover a map from this session.")


def excess_map(results, k, labels=None, sigma=3.0, mask=None):
    """
    Where in the field does bin k respond, using AMPLITUDE only?

    The sign map needs a clean per-pixel PHASE, which is far harder to get
    than a clean amplitude -- a phase that is merely noisy rails against the
    travel limit and poisons the gradients, while an amplitude that is merely
    noisy still averages.  So when the phase maps are too noisy to delineate
    anything, this still localises the responsive tissue.

    For each direction it takes the residual SNR at bin k minus the mean over
    the other bins, then averages the four directions.  Running it for every
    bin gives its own null: the control bins show what this statistic does
    when there is nothing there.
    """
    from scipy.ndimage import gaussian_filter
    labels = LABELS if labels is None else labels
    off = [j for j in K_LIST if j != k]
    per = []
    for lb in labels:
        s_k = results[k][2][lb].snr
        s_c = np.nanmean([results[j][2][lb].snr for j in off], axis=0)
        per.append(s_k - s_c)
    ex = np.nanmean(per, axis=0)
    if mask is None:
        return ex
    m = mask & np.isfinite(ex)
    den = gaussian_filter(m.astype(float), sigma)
    num = gaussian_filter(np.where(m, np.nan_to_num(ex), 0.0), sigma)
    out = np.full_like(ex, np.nan)
    good = den > 1e-6
    out[good] = num[good] / den[good]
    return np.where(mask, out, np.nan)


def _excess_report(results, bright, sigma=3.0, pct=99.0):
    """Peak of the excess map at every bin -- the controls are the null."""
    peaks, maps = {}, {}
    for k in K_LIST:
        ex = excess_map(results, k, sigma=sigma, mask=bright)
        maps[k] = ex
        peaks[k] = float(np.nanpercentile(ex[bright], pct))
    ctrl = [peaks[k] for k in K_LIST if k != K_STIM]
    mu, sd = float(np.mean(ctrl)), float(np.std(ctrl))
    z = (peaks[K_STIM] - mu) / sd if sd > 0 else np.nan

    print(f"\n  AMPLITUDE LOCALISER -- {pct:.0f}th pct of the smoothed "
          f"excess SNR map (sigma {sigma:.0f} px)")
    for k in K_LIST:
        tag = "  <== STIMULUS" if k == K_STIM else ""
        print(f"    bin {k:3d}  {peaks[k]:+7.3f}{tag}")
    print(f"    stimulus {peaks[K_STIM]:+.3f}  controls {mu:+.3f} +- {sd:.3f}"
          f"  ->  z = {z:+.2f}")

    ex = maps[K_STIM]
    thr = mu + 3 * sd
    hot = bright & np.isfinite(ex) & (ex > thr)
    px_mm2 = (UM_PER_PIXEL / 1000.0) ** 2
    if hot.sum():
        rows, cols = np.nonzero(hot)
        print(f"    region above the control ceiling ({thr:+.3f}): "
              f"{hot.sum()} px = {hot.sum() * px_mm2:.2f} mm2, "
              f"centroid ({rows.mean():.0f}, {cols.mean():.0f})")
        print("    that centroid is where the responsive cortex is, from "
              "amplitude alone --\n    no phase, no gradients, nothing that "
              "can rail at a travel limit.")
    else:
        print("    nothing above the control ceiling")
    return maps, z


def _component_report(results, bright, min_px=100):
    """
    Residual SNR at the stimulus bin, per connected piece of the mask.

    A brightness threshold often returns several disconnected regions -- the
    window, and whatever else on the skull happens to reflect as well.  Since
    every metric here is a median over the whole mask, a large unresponsive
    region drags the result toward "no signal" even when one region responds
    perfectly well.  This says which piece carries what.
    """
    from scipy.ndimage import label as _label
    lab, n = _label(bright)
    if n < 2:
        return
    print("\n  mask is in {} pieces -- residual SNR at bin {} in each:"
          .format(n, K_STIM))
    px_mm2 = (UM_PER_PIXEL / 1000.0) ** 2
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < min_px:
            continue
        vals = [float(np.nanmedian(results[K_STIM][2][lb].snr[m]))
                for lb in LABELS]
        rows, cols = np.nonzero(m)
        print(f"    piece {i}: {m.sum():5d} px ({m.sum() * px_mm2:5.2f} mm2) "
              f"rows {rows.min():3d}-{rows.max():3d} cols {cols.min():3d}-"
              f"{cols.max():3d}   SNR " +
              "  ".join(f"{lb} {v:.2f}" for lb, v in zip(LABELS, vals)))
    print("    a piece that sits near 1.0 in every direction is contributing "
          "nothing\n    but dilution -- consider setting ROI to exclude it.")


def _figure(results, d, I, bright):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    off = [k for k in K_LIST if k != K_STIM]

    a = ax[0, 0]
    for label in LABELS:
        a.plot(K_LIST, [results[k][0][label]["z"] for k in K_LIST],
               "o-", label=label, ms=4)
    a.axvline(K_STIM, color="r", ls="--", lw=1)
    a.set_xlabel("DFT bin"); a.set_ylabel("structure z")
    a.legend(fontsize=8)
    a.set_title("spatial structure per bin\n(flat = correlated noise, not signal)")

    a = ax[0, 1]
    for label in LABELS:
        a.plot(K_LIST, [results[k][0][label]["snr"] for k in K_LIST],
               "o-", label=label, ms=4)
    a.axhline(np.sqrt(2*np.log(2)), color="k", ls=":", lw=1,
              label="pure-noise median")
    a.axvline(K_STIM, color="r", ls="--", lw=1)
    a.set_xlabel("DFT bin"); a.set_ylabel("median residual SNR")
    a.legend(fontsize=7); a.set_title("residual amplitude vs noise floor")

    a = ax[0, 2]
    for axis, mk in (("elevation", "o-"), ("azimuth", "s-")):
        a.plot(K_LIST, [results[k][1][axis]["agree"] for k in K_LIST], mk,
               label=axis, ms=4)
    a.axvline(K_STIM, color="r", ls="--", lw=1)
    a.set_xlabel("DFT bin"); a.set_ylabel("median |fwd-rev| (deg)")
    a.legend(fontsize=8); a.set_title("forward/reverse agreement per bin")

    a = ax[1, 0]
    for axis, mk in (("elevation", "o-"), ("azimuth", "s-")):
        a.plot(K_LIST, [results[k][1][axis]["delay"] for k in K_LIST], mk,
               label=axis, ms=4)
        a.axhline(results[K_STIM][1][axis]["delay_chance"], ls=":", lw=1,
                  color="gray")
    a.axvline(K_STIM, color="r", ls="--", lw=1)
    a.set_xlabel("DFT bin"); a.set_ylabel("median delay (s)")
    a.legend(fontsize=8); a.set_title("delay per bin (dotted = chance)")

    lim = results[K_STIM][1]["elevation"]["maps"].visible / 2.0
    for j, k in enumerate((K_STIM - 2, K_STIM)):
        a = ax[1, 1 + j]
        am = results[k][1]["elevation"]["maps"]
        im = a.imshow(np.where(bright, am.position, np.nan), cmap="jet",
                      vmin=-lim, vmax=lim)
        plt.colorbar(im, ax=a, label="deg")
        a.set_title(f"elevation 'position' from bin {k}"
                    + ("  (STIMULUS)" if k == K_STIM else "  (CONTROL)"))

    fig.suptitle("Block 2c -- off-bin control: does the stimulus bin differ "
                 "from bins with no stimulus?")
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=105); plt.close(fig)


if __name__ == "__main__":
    main()
