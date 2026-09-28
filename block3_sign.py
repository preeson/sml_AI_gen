"""
Block 3 -- V1 borders.

Three changes from Block 2, each aimed at the sign map:

1. AVERAGING REPEAT RECORDINGS -- but of POSITION, not of raw components.
   This is not the obvious choice and the data forced it.  Averaging complex
   components across trials only works if the trials differ by a single
   constant rotation; measured on M2 the coherence of that rotation was
   0.24-0.72, i.e. the phase offset varies across the field.  That is what you
   expect if the two recordings have different delay maps (different
   physiological state), and it is consistent with cross-trial POSITION
   agreement being excellent (circular r ~0.95) while the components disagree:
   a per-pixel offset common to forward and reverse cancels out of position.
   So we average what reproduces.  Trials are aligned spatially first.

2. PER-AXIS DELAY, deliberately kept.  A joint delay is better motivated in
   principle -- delay is a property of tissue, not of sweep direction -- but on
   these data it makes azimuth worse (agreement 1.9 -> 15.5 deg) and the
   diagnostic residual is large and structured (+1.4 s for azimuth).  The
   likely reason is a residual common-mode phase bias specific to each axis:
   it adds to phi_F and phi_R alike, so it corrupts their SUM (the delay) while
   cancelling in their DIFFERENCE (the position).  A per-axis delay absorbs
   that bias and thereby protects position; forcing a shared delay re-injects
   it.  Set USE_JOINT_DELAY = True to see it for yourself -- the printed
   residual is the evidence either way.

3. A GATED SIGN MAP (ioi_sign).  VFS is a sine, so it flips on noise wherever
   the azimuth and elevation gradients are near-parallel.  Those pixels get no
   sign rather than an arbitrary one.

Point it at one or more components_bins.npz files for the SAME animal.
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ioi_core as ioi
import ioi_maps as iom
import ioi_sign as ios

OFFSET = 1540.0
MIN_COUNTS = 500.0
BRIGHT_LO_PCT = 60.0
BRIGHT_HI_PCT = 99.5
K_STIM = 15
PHASE_SIGMA = 2.0
DELAY_SIGMA = 4.0
GRAD_SIGMA = 3.0
USE_JOINT_DELAY = False   # see the module docstring before turning this on
SNR_MIN = 3.0
BIN_XY = 4
UM_PER_PIXEL = 6150.0 / 640.0 * BIN_XY

PAIRS = [("elevation", "B2U", "U2B"), ("azimuth", "L2R", "R2L")]
LABELS = ["B2U", "U2B", "L2R", "R2L"]
OUT_PNG = "block3_sign.png"
OUT_NPZ = "v1_sign.npz"


class _Spec:
    def __init__(self, label, c, noise, mask):
        self.label, self.component, self.noise = label, c, noise
        self.amplitude = np.abs(c)
        self.phase = np.angle(c)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.snr = self.amplitude / noise
        self.mask = mask


def _blk(label, d):
    axis = "elevation" if label in ("B2U", "U2B") else "azimuth"
    dur = float(d[f"{label}_duration"])
    return ioi.StimBlock(
        label=label, axis=axis, k0=0, k1=0, frames_per_cycle=1,
        n_cycles=K_STIM, cam0=0, cam1=1, t0=0.0, t1=dur,
        bar_deg_first=float(d[f"{label}_p0"]),
        bar_deg_last=float(d[f"{label}_p1"]),
        visible_half=float(d[f"{label}_vishalf"]),
    )


def circular_mean_positions(positions, travel, p0, weights=None):
    """
    Average position maps across trials on the circle of the bar travel.

    Position is periodic over the travel, so a plain mean would be wrong near
    the wrap.  The result is returned on the same interval the inputs live on,
    [p0, p0 + travel), rather than [0, travel).
    """
    zs = [np.exp(2j * np.pi * p / travel) for p in positions]
    if weights is None:
        weights = [np.ones_like(positions[0])] * len(positions)
    num = sum(w * z for w, z in zip(weights, zs))
    den = sum(weights)
    z = num / np.where(den > 0, den, np.nan)
    frac = np.mod(np.angle(z) / (2 * np.pi) - p0 / travel, 1.0)
    return p0 + frac * travel, np.abs(z)


def load_trials(paths):
    ds = [np.load(p) for p in paths]
    ref = ds[0]["B2U_mean"]
    shifts = [(0, 0)]
    for d in ds[1:]:
        sh, r = ios.estimate_shift(ref, d["B2U_mean"])
        shifts.append(sh)
        print(f"  alignment of {len(shifts)-1}: shift {sh} px "
              f"({np.hypot(*sh)*UM_PER_PIXEL:.0f} um), correlation {r:.3f}")
        if max(abs(sh[0]), abs(sh[1])) > 8:
            print("    WARNING: large shift -- check these are the same field")

    def roll(x, sh):
        return np.roll(np.roll(x, sh[0], axis=0), sh[1], axis=1)

    trials = []
    for d, sh in zip(ds, shifts):
        specs, blocks = {}, {}
        for lab in LABELS:
            c = roll(d[f"{lab}_k{K_STIM}_component"].astype(np.complex128), sh)
            n = roll(d[f"{lab}_k{K_STIM}_noise"].astype(np.float64), sh)
            m = roll(d[f"{lab}_mask"], sh)
            specs[lab] = _Spec(lab, c, n, m)
            blocks[lab] = _blk(lab, d)
        trials.append((specs, blocks, roll(d["B2U_mean"], sh)))

    if len(ds) > 1:
        common = np.ones_like(trials[0][0]["B2U"].mask)
        for sp, _, _ in trials:
            for lab in LABELS:
                common &= sp[lab].mask
        for lab in LABELS:
            comps = [t[0][lab].component for t in trials]
            _, ang, coh = ios.align_phase(comps, common)
            print(f"  {lab}: cross-trial phase offset "
                  f"{', '.join(f'{a:+.0f} deg' for a in ang[1:])}, coherence "
                  f"{', '.join(f'{x:.2f}' for x in coh[1:])}"
                  + ("  (low -- offset varies across the field, so raw "
                     "components are NOT averaged)" if min(coh[1:]) < 0.85
                     else ""))
    return trials


def main(paths):
    print(f"loading {len(paths)} recording(s)")
    trials = load_trials(paths)
    mean_img = np.mean([t[2] for t in trials], axis=0)

    I = mean_img - OFFSET
    valid = I > MIN_COUNTS
    lo, hi = np.percentile(I[valid], [BRIGHT_LO_PCT, BRIGHT_HI_PCT])
    bright = valid & (I >= lo) & (I <= hi)
    print(f"\nbright mask: {bright.sum()} px "
          f"({bright.sum()*(UM_PER_PIXEL/1000)**2:.2f} mm2)")

    # ---- per-trial maps -------------------------------------------------
    per_trial = []
    for i, (specs, blocks, _) in enumerate(trials):
        jmaps, jinfo = iom.combine_axes_joint(
            specs, blocks, pairs=PAIRS, phase_sigma=PHASE_SIGMA,
            delay_sigma=DELAY_SIGMA, mask=bright)
        smaps = {}
        for axis, f, r in PAIRS:
            smaps[axis] = iom.combine_axis(
                specs[f], blocks[f], specs[r], blocks[r],
                phase_sigma=PHASE_SIGMA, delay_sigma=DELAY_SIGMA,
                snr_min=0.0, visible_half=blocks[f].visible_half)
        per_trial.append((jmaps, jinfo, smaps))
        print(f"\n=== DELAY, recording {i+1} ===")
        print(f"  joint estimate median "
              f"{np.nanmedian(jinfo['delay_joint'][bright]):.2f} s")
        for axis, _, _ in PAIRS:
            rr = jinfo[f"delay_resid_{axis}"][bright]
            print(f"  {axis:10s} own {np.nanmedian(jinfo[f'delay_{axis}'][bright]):5.2f} s"
                  f"   residual vs joint {np.nanmedian(rr):+.2f} s "
                  f"(IQR {np.nanpercentile(rr,25):+.2f} to "
                  f"{np.nanpercentile(rr,75):+.2f})")
        print("  a large structured residual means delay is not shared across "
              "axes -- most likely an axis-specific phase bias, which a "
              "per-axis delay absorbs")

    use = 0 if USE_JOINT_DELAY else 2
    maps = {}
    print("\n=== POSITION (averaged across recordings) ===")
    for axis, f, _ in PAIRS:
        b0 = trials[0][1][f]
        travel = abs(b0.bar_deg_last - b0.bar_deg_first)
        p0 = min(b0.bar_deg_first, b0.bar_deg_last)
        poss = [pt[use][axis].position for pt in per_trial]
        wts = [np.where(bright & np.isfinite(pt[use][axis].snr),
                        np.nan_to_num(pt[use][axis].snr), 0.0)
               for pt in per_trial]
        pos, consist = circular_mean_positions(poss, travel, p0, wts)
        am = per_trial[0][use][axis]
        am.position = pos
        am.consistency = consist
        am.mask = bright
        maps[axis] = am
        if len(poss) > 1:
            d = np.angle(np.exp(2j*np.pi*(poss[0]-poss[1])/travel))/(2*np.pi)*travel
            print(f"  {axis:10s} cross-recording |difference| median "
                  f"{np.nanmedian(np.abs(d[bright])):.2f} deg, "
                  f"consistency {np.nanmedian(consist[bright]):.3f}")
    info = per_trial[0][1]

    for axis, am in maps.items():
        lim = am.visible / 2
        p = am.position[bright]
        print(f"  {axis:10s} range {np.nanpercentile(p,2):+.1f} .. "
              f"{np.nanpercentile(p,98):+.1f} deg, "
              f"{100*np.nanmean(np.abs(p)<=lim):.0f}% inside +-{lim:.1f}, "
              f"{100*np.nanmean(np.abs(p)>am.travel/2-5):.1f}% at the travel "
              f"limit")
        print(f"  {'':10s} fwd/rev agreement median "
              f"{np.nanmedian(np.abs(am.agreement[bright])):.2f} deg "
              f"(recording 1)")

    el, az = maps["elevation"], maps["azimuth"]
    snr = np.sqrt(el.snr * az.snr)
    sm = ios.visual_field_sign(el.position, az.position, bright,
                               um_per_pixel=UM_PER_PIXEL, sigma=GRAD_SIGMA,
                               snr=snr, snr_min=SNR_MIN)
    ps = ios.patches(sm)

    print("\n=== VISUAL FIELD SIGN ===")
    print(ios.report(sm, ps))

    _figure(mean_img, bright, maps, info, sm, ps)
    np.savez_compressed(
        OUT_NPZ, vfs=sm.vfs, gated=sm.gated, reliable=sm.reliable,
        reliability=sm.reliability, mask=bright,
        elevation_position=el.position, azimuth_position=az.position,
        delay=info["delay_joint"], snr=snr, mean_image=mean_img,
        um_per_pixel=UM_PER_PIXEL,
        patch_masks=np.stack([p.mask for p in ps]) if ps else np.zeros((0, 1, 1)),
        patch_signs=np.array([p.sign for p in ps]),
        patch_areas=np.array([p.area_mm2 for p in ps]),
    )
    print(f"\nwrote {OUT_PNG} and {OUT_NPZ}")


def _figure(mean_img, bright, maps, info, sm, ps):
    fig, ax = plt.subplots(2, 4, figsize=(19, 8.5))
    el, az = maps["elevation"], maps["azimuth"]

    a = ax[0, 0]
    a.imshow(mean_img, cmap="gray")
    a.contour(bright.astype(float), levels=[0.5], colors="c", linewidths=1)
    a.set_title("mean image (cyan = mask)")

    for j, (axis, am) in enumerate((("elevation", el), ("azimuth", az))):
        a = ax[0, 1 + j]
        lim = am.visible / 2
        im = a.imshow(np.where(bright, am.position, np.nan), cmap="jet",
                      vmin=-lim, vmax=lim)
        plt.colorbar(im, ax=a, label="deg")
        a.set_title(f"{axis} position (joint delay)")

    a = ax[0, 3]
    im = a.imshow(np.where(bright, info["delay_joint"], np.nan), cmap="viridis")
    plt.colorbar(im, ax=a, label="s")
    a.set_title("joint haemodynamic delay")

    a = ax[1, 0]
    im = a.imshow(np.where(bright, sm.vfs, np.nan), cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar(im, ax=a)
    a.set_title("sign map, ungated")

    a = ax[1, 1]
    im = a.imshow(np.where(bright, sm.reliability, np.nan), cmap="cividis",
                  vmin=0, vmax=1)
    plt.colorbar(im, ax=a)
    a.set_title("gradient reliability")

    a = ax[1, 2]
    im = a.imshow(sm.gated, cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar(im, ax=a)
    a.set_title("sign map, GATED")

    a = ax[1, 3]
    a.imshow(mean_img, cmap="gray")
    for p in ps[:6]:
        a.contour(p.mask.astype(float), levels=[0.5],
                  colors=["r" if p.sign > 0 else "b"], linewidths=1.5)
        a.text(p.centroid[1], p.centroid[0], f"{p.area_mm2:.1f}",
               color="y", fontsize=8, ha="center")
    a.set_title("patches on the anatomy (area in mm2)")

    fig.suptitle("Block 3 -- V1 delineation")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:] or ["components_bins.npz"]
    main(args)
