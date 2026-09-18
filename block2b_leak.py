"""
Block 2b -- is there a real retinotopic signal underneath the contamination?

The stimulus-locked signal in this recording is dominated by light reaching
the detector rather than by tissue.  This script fits and removes the
spatially-common part of it (see ioi_leak) and asks whether what is left has
the properties a retinotopic map must have:

  1. phase more spatially organised than chance, on UNSMOOTHED data
  2. haemodynamic delay in the 1-3 s range, not ~0 s
  3. position inside the screen extent, not piled at the travel limits
  4. forward/reverse agreement

The analysis mask is now BRIGHTNESS-based, not SNR-based.  Selecting on SNR
is what produced the ring in Block 2: the contamination has the highest SNR
precisely where there is the least tissue, so an SNR mask selects against
cortex.

Reduction results are cached in components.npz, so re-running with different
settings costs seconds rather than the ~200 s first pass.
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

BRIGHT_LO_PCT = 60.0    # analysis mask: brightest 40% of the field
BRIGHT_HI_PCT = 99.5    # drop the specular highlights
ORDER = 0               # common-mode model; see the sensitivity check below
PHASE_SIGMA = 2.0
DELAY_SIGMA = 4.0

PAIRS = [("elevation", "B2U", "U2B"), ("azimuth", "L2R", "R2L")]
VIS_HALF = {"elevation": 33.4, "azimuth": 49.7}

CACHE = "components.npz"
OUT_PNG = "block2b_leak.png"


def reduce_blocks():
    if os.path.exists(CACHE):
        print(f"loading cached reduction from {CACHE}")
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
            sp = iof.analyse(fold, offset=OFFSET, min_counts=MIN_COUNTS)
            out[f"{blk.label}_component"] = sp.component
            out[f"{blk.label}_noise"] = sp.noise
            out[f"{blk.label}_mask"] = sp.mask
            out[f"{blk.label}_mean"] = fold.mean_image
            out[f"{blk.label}_fstim"] = np.array(blk.f_stim)
            out[f"{blk.label}_p0"] = np.array(blk.bar_deg_first)
            out[f"{blk.label}_p1"] = np.array(blk.bar_deg_last)
            del fold
    print(f"  reduction took {time.time()-t0:.0f} s")
    np.savez_compressed(CACHE, **out)
    return out


class _Spec:
    """Minimal stand-in for BlockSpectrum."""
    def __init__(self, label, component, noise, mask):
        self.label, self.component, self.noise = label, component, noise
        self.amplitude = np.abs(component)
        self.phase = np.angle(component)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.snr = self.amplitude / noise
        self.mask = mask


def _blk(label, d):
    return ioi.StimBlock(
        label=label, axis="elevation" if label in ("B2U", "U2B") else "azimuth",
        k0=0, k1=0, frames_per_cycle=1, n_cycles=15, cam0=0, cam1=1,
        t0=0.0, t1=float(1 / d[f"{label}_fstim"]) * 15,
        bar_deg_first=float(d[f"{label}_p0"]),
        bar_deg_last=float(d[f"{label}_p1"]),
    )


def main():
    d = reduce_blocks()

    I = d["B2U_mean"] - OFFSET
    valid = I > MIN_COUNTS
    lo = np.percentile(I[valid], BRIGHT_LO_PCT)
    hi = np.percentile(I[valid], BRIGHT_HI_PCT)
    bright = valid & (I >= lo) & (I <= hi)
    print(f"\nbrightness mask: {lo:.0f} <= I <= {hi:.0f} counts -> "
          f"{bright.sum()} px ({100*bright.mean():.1f}% of field)")

    print("\n" + "=" * 70)
    print("PER-BLOCK COMMON-MODE REMOVAL")
    print("=" * 70)
    resid, fits = {}, {}
    for label in ("B2U", "U2B", "L2R", "R2L"):
        sp = _Spec(label, d[f"{label}_component"], d[f"{label}_noise"],
                   d[f"{label}_mask"])
        r, fit = iol.remove_common_mode(sp, I, bright, order=ORDER)
        resid[label], fits[label] = r, fit

        pre_c = iol.phase_concentration(sp.component, bright)
        post_c = iol.phase_concentration(r.component, bright)
        pre = iol.structure_score(sp.phase, bright)
        post = iol.structure_score(r.phase, bright)
        print(f"\n{label}:")
        print(f"  leak  alpha {abs(fit.alpha):.3f} counts @ "
              f"{np.degrees(np.angle(fit.alpha)):+.1f} deg |  "
              f"beta {abs(fit.beta)*100:.5f}% @ "
              f"{np.degrees(np.angle(fit.beta)):+.1f} deg")
        print(f"  variance removed in mask: {fit.var_removed*100:.1f}%")
        print(f"  amplitude in mask: {np.median(sp.amplitude[bright])*100:.5f}%"
              f"  ->  {np.median(r.amplitude[bright])*100:.5f}%")
        print(f"  phase concentration: {pre_c:.3f}  ->  {post_c:.3f}")
        print(f"  spatial structure z: {pre[2]:+.2f}  ->  {post[2]:+.2f}   "
              f"(obs {post[0]:.1f} deg vs shuffled {post[1]:.1f} deg)")
        print(f"  residual SNR median: {np.nanmedian(r.snr[bright]):.2f}")

    print("\n" + "=" * 70)
    print("RETINOTOPY FROM THE RESIDUALS")
    print("=" * 70)
    maps = {}
    for axis, f, rv in PAIRS:
        am = iom.combine_axis(resid[f], _blk(f, d), resid[rv], _blk(rv, d),
                              phase_sigma=PHASE_SIGMA, delay_sigma=DELAY_SIGMA,
                              snr_min=0.0, visible_half=VIS_HALF[axis])
        am.mask = bright
        maps[axis] = am
        m = bright
        lim = VIS_HALF[axis]
        inside = np.mean(np.abs(am.position[m]) <= lim) * 100
        print(f"\n{axis}:")
        chance = am.period / 4.0   # random phase -> delay uniform on [0, T/2]
        print(f"  delay      median {np.nanmedian(am.delay[m]):.2f} s, "
              f"IQR {np.nanpercentile(am.delay[m],25):.2f}-"
              f"{np.nanpercentile(am.delay[m],75):.2f} s")
        print(f"             expect 1-3 s for tissue, ~0 s for light, "
              f"{chance:.1f} s if phase is pure noise")
        print(f"  position   {np.nanpercentile(am.position[m],2):+.1f} .. "
              f"{np.nanpercentile(am.position[m],98):+.1f} deg, "
              f"{inside:.0f}% inside the +-{lim:.1f} deg screen")
        print(f"  fwd/rev    median |diff| "
              f"{np.nanmedian(np.abs(am.agreement[m])):.2f} deg, "
              f"consistency {np.nanmedian(am.consistency[m]):.3f}")

    print("\n" + "=" * 70)
    print("SENSITIVITY: does the answer depend on the model order?")
    print("=" * 70)
    print("(order >0 can absorb a real retinotopic gradient -- if structure")
    print(" only appears at order 0 and vanishes at 1, trust order 0)")
    for order in (0, 1, 2):
        zs = []
        for label in ("B2U", "U2B", "L2R", "R2L"):
            sp = _Spec(label, d[f"{label}_component"], d[f"{label}_noise"],
                       d[f"{label}_mask"])
            r, _ = iol.remove_common_mode(sp, I, bright, order=order)
            zs.append(iol.structure_score(r.phase, bright)[2])
        print(f"  order {order}: structure z per block = "
              + ", ".join(f"{v:+.1f}" for v in zs))

    _verdict(resid, maps, bright)
    _figure(d, I, resid, fits, maps, bright)
    print(f"\nwrote {OUT_PNG}" + (f" and {CACHE}" if not os.path.exists(CACHE)
                                  else ""))


def _verdict(resid, maps, bright):
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    zs = [iol.structure_score(resid[l].phase, bright)[2]
          for l in ("B2U", "U2B", "L2R", "R2L")]
    dly = [np.nanmedian(maps[a].delay[bright]) for a in maps]
    agree = [np.nanmedian(np.abs(maps[a].agreement[bright])) for a in maps]
    checks = [
        ("phase organised beyond chance (z < -5 in >=3 blocks)",
         sum(z < -5 for z in zs) >= 3),
        ("haemodynamic delay in 0.5-4 s for both axes",
         all(0.5 < x < 4.0 for x in dly)),
        ("fwd/rev agreement under 10 deg for both axes",
         all(a < 10 for a in agree)),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
    n = sum(ok for _, ok in checks)
    print()
    if n == 3:
        print("  All three hold: a real retinotopic signal survives removal.")
    elif n >= 1:
        print("  Mixed. Some structure survives but it does not behave like")
        print("  retinotopy on every count -- treat any map as provisional.")
    else:
        print("  Nothing survives. The stimulus-locked signal in this session")
        print("  is contamination, and no analysis will recover a map from it.")
        print("  Sensitivity floor from test_leak.py is ~0.003 %dR/R, so a")
        print("  tissue signal below that would also be invisible here.")


def _figure(d, I, resid, fits, maps, bright):
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    a = ax[0, 0]
    a.imshow(I, cmap="gray")
    a.contour(bright.astype(float), levels=[0.5], colors="c", linewidths=1)
    a.set_title("brightness (cyan = analysis mask)")

    sp0 = d["B2U_component"]
    a = ax[0, 1]
    im = a.imshow(np.where(bright, np.abs(sp0)*100, np.nan), cmap="magma")
    plt.colorbar(im, ax=a); a.set_title("B2U amplitude BEFORE (%dR/R)")
    a = ax[0, 2]
    im = a.imshow(np.where(bright, resid["B2U"].amplitude*100, np.nan),
                  cmap="magma")
    plt.colorbar(im, ax=a); a.set_title("B2U amplitude AFTER removal")

    a = ax[1, 0]
    im = a.imshow(np.where(bright, np.angle(sp0), np.nan), cmap="hsv",
                  vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im, ax=a); a.set_title("B2U phase BEFORE")
    a = ax[1, 1]
    im = a.imshow(np.where(bright, resid["B2U"].phase, np.nan), cmap="hsv",
                  vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im, ax=a); a.set_title("B2U phase AFTER (unsmoothed)")
    a = ax[1, 2]
    im = a.imshow(np.where(bright, np.abs(fits["B2U"].predicted)*100, np.nan),
                  cmap="viridis")
    plt.colorbar(im, ax=a); a.set_title("fitted common mode (%dR/R)")

    for j, (axis, am) in enumerate(maps.items()):
        lim = VIS_HALF[axis]
        a = ax[2, j]
        im = a.imshow(np.where(bright, am.position, np.nan), cmap="jet",
                      vmin=-lim, vmax=lim)
        plt.colorbar(im, ax=a, label="deg")
        a.set_title(f"{axis} position after removal")
    a = ax[2, 2]
    el, az = maps["elevation"], maps["azimuth"]
    a.scatter(az.position[bright], el.position[bright], s=1, alpha=0.1)
    a.axvline(-VIS_HALF["azimuth"], color="k", lw=0.5)
    a.axvline(VIS_HALF["azimuth"], color="k", lw=0.5)
    a.axhline(-VIS_HALF["elevation"], color="k", lw=0.5)
    a.axhline(VIS_HALF["elevation"], color="k", lw=0.5)
    a.set_xlabel("azimuth (deg)"); a.set_ylabel("elevation (deg)")
    a.set_title("visual field coverage after removal")

    fig.suptitle("Block 2b -- common-mode removal and residual retinotopy")
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=105); plt.close(fig)


if __name__ == "__main__":
    main()
