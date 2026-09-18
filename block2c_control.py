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
BRIGHT_LO_PCT = 60.0
BRIGHT_HI_PCT = 99.5
ORDER = 0
PHASE_SIGMA = 2.0
DELAY_SIGMA = 4.0

K_STIM = 15
K_LIST = [11, 12, 13, 14, 15, 16, 17, 18, 19]
LABELS = ["B2U", "U2B", "L2R", "R2L"]
PAIRS = [("elevation", "B2U", "U2B"), ("azimuth", "L2R", "R2L")]
VIS_HALF = {"elevation": 33.4, "azimuth": 49.7}

CACHE = "components_bins.npz"
OUT_PNG = "block2c_control.png"


def reduce_all_bins():
    if os.path.exists(CACHE):
        print(f"loading cached multi-bin reduction from {CACHE}")
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}

    ft = ioi.load_frame_times()
    log = ioi.load_stim_log(ft=ft)
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
            del fold
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
                              visible_half=VIS_HALF[axis])
        am.mask = bright
        lim = VIS_HALF[axis]
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
    print(f"\nfit on {valid.sum()} px (I spans "
          f"{I[valid].min():.0f}-{I[valid].max():.0f}, "
          f"{I[valid].max()/I[valid].min():.0f}x); "
          f"evaluate on {bright.sum()} bright px "
          f"({lo:.0f}-{hi:.0f} counts, {hi/lo:.1f}x)")

    results = {k: metrics_for_bin(d, k, bright, valid, I) for k in K_LIST}

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

    for label in LABELS:
        v = results[K_STIM][0][label]["snr"]
        c = [results[k][0][label]["snr"] for k in off]
        z = _outlier_z(v, c)
        print(f"  residual SNR  {label}: stimulus {v:.2f}  "
              f"controls {np.mean(c):.2f} +- {np.std(c):.2f}  ->  z = {z:+.2f}")
        tests.append(z)

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

    strong = sum(abs(t) > 3 for t in tests if np.isfinite(t))
    print()
    if strong >= 4:
        print("  Bin 15 stands clearly apart from its neighbours: there IS")
        print("  signal at the stimulus frequency.")
    elif strong >= 2:
        print("  Bin 15 is marginally distinguishable. Weak but not nothing;")
        print("  any map from this session is provisional at best.")
    else:
        print("  Bin 15 behaves like every other bin. There is no detectable")
        print("  signal at the stimulus frequency -- the structure seen in")
        print("  Block 2b is spatially correlated noise, which every bin has.")
        print("  No analysis can recover a map from this session.")


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

    lim = VIS_HALF["elevation"]
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
