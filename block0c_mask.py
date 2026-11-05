"""
Block 0c -- measure the evaluation mask from the animal's own pulse.

Run this once per recording, after block0_verify.py and before block2c.  It
writes tissue_mask.npz, which block2c_control.py and block3_sign.py pick up
automatically.  Nothing has to be drawn by hand and nothing has to be changed
between a thinned skull and a cranial window: the criterion is each pixel's
cardiac modulation measured against its own noise floor, which is a ratio and
therefore does not care how much light the preparation returns.

    python block0c_mask.py              # 30 s from the middle of the recording
    python block0c_mask.py 6000         # use 6000 frames instead
    python block0c_mask.py 6000 12000   # ... starting at frame 12000

WHAT TO LOOK AT IN THE FIGURE
-----------------------------
Panel 2 is the one that matters.  The ratio map should look like cortex: a
vascular tree over a filled territory, stopping where the window stops.  If it
instead highlights the rim, the headplate or a bright reflection, the pulse is
not what is being measured there and the mask is picking up motion against a
high-contrast edge -- raise ERODE_ITER in ioi_tissue.py.

Panel 3 is the audit.  Two populations separated by the threshold is what a
working preparation looks like.  One population means either the whole field is
brain (fine -- check the mask covers it all) or none of it is pulsing (not fine
-- the method line in the printout will say so).
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ioi_core as ioi
import ioi_tissue as iot

BIN_XY = 4               # MUST match block2c_control.py / block3_sign.py
N_FRAMES = 3000          # 30 s at 100 Hz
OUT_PNG = "block0c_mask.png"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_FRAMES
    start = int(sys.argv[2]) if len(sys.argv) > 2 else None

    ft = ioi.load_frame_times()
    print(f"recording: {ft.n_frames} frames, {ft.t[-1]:.1f} s, "
          f"{ft.fs:.3f} Hz")
    with ioi.movie() as mov:
        tm = iot.find_tissue(mov, ft, start=start, n_frames=n, bin_xy=BIN_XY)

    print("\n" + "=" * 70)
    print("TISSUE MASK")
    print("=" * 70)
    print(tm.summary())

    # How different is this from what the brightness band would have given?
    I = tm.mean_img - iot.OFFSET
    valid = I > iot.MIN_COUNTS
    lo, hi = np.percentile(I[valid], [60.0, 99.5])
    bright = valid & (I >= lo) & (I <= hi)
    both = int((bright & tm.mask).sum())
    px_mm2 = (tm.um_per_pixel / 1000.0) ** 2
    print(f"\n  for comparison, the brightness band keeps "
          f"{int(bright.sum())} px ({bright.sum()*px_mm2:.2f} mm2)")
    print(f"    in both            : {both} px ({both*px_mm2:.2f} mm2)")
    print(f"    bright but no pulse: {int((bright & ~tm.mask).sum())} px "
          f"<- these were diluting every median")
    print(f"    pulsing but not bright: {int((~bright & tm.mask).sum())} px "
          f"<- these were being thrown away")
    if tm.mask.any() and bright.any():
        print(f"    median pulse ratio inside the mask "
              f"{np.nanmedian(tm.ratio[tm.mask]):.1f}x, in the bright-but-"
              f"unpulsing pixels {np.nanmedian(tm.ratio[bright & ~tm.mask]):.1f}x")

    _figure(tm, bright)
    path = iot.save(tm)
    print(f"\nwrote {OUT_PNG} and {path}")
    print("block2c_control.py and block3_sign.py will now use this mask; "
          "leave their ROI at None.")


def _figure(tm, bright):
    fig, ax = plt.subplots(1, 4, figsize=(21, 5))

    a = ax[0]
    a.imshow(tm.mean_img, cmap="gray")
    a.contour(tm.mask.astype(float), levels=[0.5], colors="lime", linewidths=1.4)
    a.contour(bright.astype(float), levels=[0.5], colors="orange",
              linewidths=0.8, linestyles="dashed")
    if tm.lit is not None:
        a.contour(tm.lit.astype(float), levels=[0.5], colors="deepskyblue",
                  linewidths=0.8, linestyles="dotted")
    frac = tm.mask.sum() / max(tm.lit.sum(), 1) if tm.lit is not None else \
        float("nan")
    a.set_title(f"mean image   green = pulse mask ({tm.area_mm2:.2f} mm2, "
                f"{100*frac:.0f}% of lit)\norange dashed = old brightness "
                f"band, blue dotted = illuminated field", fontsize=9)

    a = ax[1]
    v = np.nanpercentile(tm.ratio[np.isfinite(tm.ratio)], 99.5)
    im = a.imshow(tm.ratio, cmap="inferno", vmin=0, vmax=max(v, tm.threshold * 2))
    plt.colorbar(im, ax=a, label="x own noise floor")
    a.contour(tm.mask.astype(float), levels=[0.5], colors="lime", linewidths=1.0)
    what = (f"cardiac amplitude at {tm.f_pulse:.2f} Hz, {tm.smooth_sigma:g} px "
            f"coherent" if tm.method == "cardiac"
            else f"MAXIMUM over {iot.PULSE_BAND[0]:g}-{iot.PULSE_BAND[1]:g} Hz "
                 f"(fallback, unsmoothed)")
    a.set_title(f"{what},\nrelative to each pixel's own noise", fontsize=9)

    a = ax[2]
    r = tm.ratio[np.isfinite(tm.ratio) & (tm.ratio > 0)]
    a.hist(np.log10(r), bins=60, color="0.6")
    a.axvline(np.log10(tm.threshold), color="r", lw=1.5,
              label=f"threshold {tm.threshold:.2f}x")
    a.set_xlabel("log10(pulse / own noise)")
    a.set_ylabel("binned pixels")
    a.legend(fontsize=8)
    a.set_title("two populations = a working preparation", fontsize=9)

    a = ax[3]
    im = a.imshow(100 * tm.pulse_amp, cmap="viridis",
                  vmax=np.nanpercentile(100 * tm.pulse_amp, 99.5))
    plt.colorbar(im, ax=a, label="%dR/R")
    a.set_title(f"cardiac amplitude\n(HR {tm.f_pulse*60:.0f}/min, "
                f"RR {tm.f_resp*60:.0f}/min)", fontsize=9)

    fig.suptitle(f"Block 0c -- evaluation mask measured from the pulse "
                 f"[{tm.method}]")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
