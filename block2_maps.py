"""
Block 2 -- all four stimulus blocks, combined into delay and retinotopy.

Runs fold + Fourier on B2U, U2B, L2R, R2L (the whole 74 GB file, ~2.5 min at
0.54 GiB/s), then pairs opposite directions into:

  * haemodynamic delay   (from the phase SUM -- unambiguous here)
  * absolute position    (per direction, after subtracting a smoothed delay)
  * agreement            (fwd vs rev position; a real per-pixel validation,
                          not zero by construction -- see ioi_maps docstring)

Output feeds Block 3 (visual field sign map).

Sanity checks to read in the output:
  * delay should be 1-3 s and vary smoothly; a light leak shows ~0 s
  * agreement should be small (a few degrees) wherever SNR is decent
  * position range should roughly fill the screen extent (+-33.4 elevation,
    +-49.7 azimuth) and not pile up at the travel limits
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ioi_core as ioi
import ioi_fourier as iof
import ioi_maps as iom

BIN_XY = 4
BINS_PER_CYCLE = 80
CHUNK = 256
OFFSET = 1540.0
MIN_COUNTS = 500.0
PHASE_SIGMA = 2.0      # ~181 um FWHM at 4x binning
DELAY_SIGMA = 4.0      # ~362 um FWHM
SNR_MIN = 2.0

# screen half-extents, from the monitor geometry (33 x 60 cm at 25 cm)
VIS_HALF = {"elevation": 33.4, "azimuth": 49.7}

PAIRS = [("elevation", "B2U", "U2B"), ("azimuth", "L2R", "R2L")]
UM_PER_PIXEL = 6150.0 / 640.0 * BIN_XY   # 9.609 um * BIN_XY

CORNER = (slice(95, None), slice(0, 70))  # the bright region from the pilot
OUT_PNG = "block2_maps.png"
OUT_NPZ = "maps.npz"


def main():
    ft = ioi.load_frame_times()
    log = ioi.load_stim_log(ft=ft)

    specs, blocks = {}, {}
    t0 = time.time()
    with ioi.movie() as mov:
        for blk in log.blocks:
            t1 = time.time()
            print(f"\n{blk.label}: {blk.n_cam_frames} frames "
                  f"({blk.n_cam_frames*ioi.BYTES_PER_FRAME/1024**3:.1f} GiB), "
                  f"F = {blk.f_stim:.6f} Hz", flush=True)
            fold = iof.fold_block(mov, ft, blk,
                                  bins_per_cycle=BINS_PER_CYCLE,
                                  bin_xy=BIN_XY, chunk=CHUNK)
            sp = iof.analyse(fold, offset=OFFSET, min_counts=MIN_COUNTS)
            specs[blk.label] = sp
            blocks[blk.label] = blk
            m = sp.mask
            print(f"  {time.time()-t1:.0f} s | amplitude median "
                  f"{np.median(sp.amplitude[m])*100:.4f}% | "
                  f"SNR median {np.median(sp.snr[m]):.2f} | "
                  f"bin16/bin15 power "
                  f"{sp.power_mean[sp.k_stim+1]/sp.power_mean[sp.k_stim]:.3f}")
            if fold.n_empty:
                print(f"  note: {fold.n_empty} empty phase bins filled")
            del fold
    print(f"\ntotal fold+analyse time: {time.time()-t0:.0f} s")

    maps = {}
    for axis, fwd, rev in PAIRS:
        am = iom.combine_axis(specs[fwd], blocks[fwd], specs[rev], blocks[rev],
                              phase_sigma=PHASE_SIGMA, delay_sigma=DELAY_SIGMA,
                              snr_min=SNR_MIN, visible_half=VIS_HALF[axis])
        maps[axis] = am
        print("\n" + iom.report(am))

    _corner_check(maps)
    _figure(maps, specs)

    np.savez_compressed(
        OUT_NPZ,
        um_per_pixel=UM_PER_PIXEL, bin_xy=BIN_XY,
        **{f"{ax}_{k}": getattr(am, k)
           for ax, am in maps.items()
           for k in ("delay", "position", "pos_fwd", "pos_rev",
                     "agreement", "consistency", "amplitude", "snr", "mask")},
    )
    print(f"\nwrote {OUT_NPZ} and {OUT_PNG}")


def _corner_check(maps):
    print("\n" + "-" * 68)
    print("BRIGHT-CORNER DIAGNOSTIC")
    print("A stimulus light leak follows the screen with no haemodynamic lag,")
    print("so it shows delay ~0 s.  Real cortex shows 1-3 s.")
    print("-" * 68)
    for axis, am in maps.items():
        corner = np.zeros(am.mask.shape, bool)
        corner[CORNER] = True
        c = corner & am.mask
        r = am.mask & ~corner
        if not c.any() or not r.any():
            print(f"  {axis}: one region empty, skipping")
            continue
        print(f"  {axis:10s} delay  corner {np.nanmedian(am.delay[c]):5.2f} s"
              f"   cortex {np.nanmedian(am.delay[r]):5.2f} s")
        print(f"  {'':10s} amp    corner "
              f"{np.nanmedian(am.amplitude[c])*100:.4f}%"
              f"  cortex {np.nanmedian(am.amplitude[r])*100:.4f}%")
        print(f"  {'':10s} agree  corner "
              f"{np.nanmedian(np.abs(am.agreement[c])):5.2f} deg"
              f" cortex {np.nanmedian(np.abs(am.agreement[r])):5.2f} deg")


def _figure(maps, specs):
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    ext = None

    for j, (axis, am) in enumerate(maps.items()):
        m = am.mask
        lim = VIS_HALF[axis]

        a = ax[0, j]
        im = a.imshow(np.where(m, am.position, np.nan), cmap="jet",
                      vmin=-lim, vmax=lim)
        plt.colorbar(im, ax=a, label="deg")
        a.set_title(f"{axis} position ({am.label_fwd}/{am.label_rev})")

        a = ax[1, j]
        im = a.imshow(np.where(m, am.delay, np.nan), cmap="viridis",
                      vmin=0, vmax=np.nanpercentile(am.delay[m], 98))
        plt.colorbar(im, ax=a, label="s")
        a.set_title(f"{axis} haemodynamic delay")

        a = ax[2, j]
        v = np.where(m, am.agreement, np.nan)
        s = np.nanpercentile(np.abs(v), 95)
        im = a.imshow(v, cmap="RdBu_r", vmin=-s, vmax=s)
        plt.colorbar(im, ax=a, label="deg")
        a.set_title(f"{axis} fwd-rev agreement")

    # third column: SNR, consistency, position scatter
    a = ax[0, 2]
    el, az = maps["elevation"], maps["azimuth"]
    both = el.mask & az.mask
    im = a.imshow(np.where(both, np.sqrt(el.snr * az.snr), np.nan),
                  cmap="magma", vmax=np.nanpercentile(
                      np.sqrt(el.snr * az.snr)[both], 99))
    plt.colorbar(im, ax=a)
    a.set_title("combined SNR (geometric mean)")

    a = ax[1, 2]
    im = a.imshow(np.where(both, 0.5 * (el.consistency + az.consistency),
                           np.nan), cmap="cividis", vmin=0, vmax=1)
    plt.colorbar(im, ax=a)
    a.set_title("fwd/rev consistency (1 = perfect)")

    a = ax[2, 2]
    good = both & (np.abs(el.agreement) < 10) & (np.abs(az.agreement) < 10)
    a.scatter(az.position[good], el.position[good], s=1, alpha=0.15,
              c=np.sqrt(el.snr * az.snr)[good], cmap="magma")
    a.axvline(-VIS_HALF["azimuth"], color="k", lw=0.5)
    a.axvline(VIS_HALF["azimuth"], color="k", lw=0.5)
    a.axhline(-VIS_HALF["elevation"], color="k", lw=0.5)
    a.axhline(VIS_HALF["elevation"], color="k", lw=0.5)
    a.set_xlabel("azimuth (deg)")
    a.set_ylabel("elevation (deg)")
    a.set_title("visual field coverage\n(lines = screen extent)")

    fig.suptitle("Block 2 -- retinotopy from all four directions")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    main()
