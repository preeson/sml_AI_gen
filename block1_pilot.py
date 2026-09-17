"""
Block 1 pilot -- answer one question: is there enough SNR to map V1?

Processes ONE stimulus block (default B2U, 24,368 frames ~ 16.8 GB of the
74 GB file) and reports the Fourier component at the stimulus frequency
against the noise in neighbouring bins.

Read this output before committing to a full pass:

  * median SNR in the mask -- single-pixel phase error is ~1/SNR radians, so
    SNR 10 means ~6 deg of phase, ~2 deg of visual angle.  SNR below ~3 across
    the board means the protocol needs changing, not the analysis.
  * measured noise vs the shot-noise prediction -- if measured is much larger,
    you are 1/f limited, and more light will not help as much as you expect.
  * the neighbouring-bin profile -- if bin 16 rivals bin 15, the ~0.064 Hz
    oscillation seen on analog ch4 is in the imaging data too.
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ioi_core as ioi
import ioi_fourier as iof

BLOCK = "B2U"          # which block to pilot
BIN_XY = 4             # spatial binning: 4 -> 135 x 160
BINS_PER_CYCLE = 80    # -> 1200 samples, ~4.9 Hz, Nyquist 40x F
CHUNK = 256            # camera frames held in RAM at once (~350 MB float32)
OFFSET = 1540.0        # camera pedestal, counts
MIN_COUNTS = 500.0     # mask: block-mean must exceed OFFSET + this
FRAC_NOISE = 0.0174    # measured per-frame per-pixel noise from Block 0
OUT_PNG = "block1_pilot.png"


def run(mov, ft, block, bin_xy=BIN_XY, bins_per_cycle=BINS_PER_CYCLE,
        chunk=CHUNK, offset=OFFSET, min_counts=MIN_COUNTS, plot=True):
    n_frames = block.n_cam_frames
    gb = n_frames * ioi.BYTES_PER_FRAME / 1024 ** 3
    print(f"block {block.label}: camera frames {block.cam0}-{block.cam1} "
          f"({n_frames}, {gb:.2f} GiB)")
    print(f"  {block.n_cycles} cycles, F = {block.f_stim:.6f} Hz, "
          f"duration {block.duration:.2f} s")
    print(f"  spatial binning {bin_xy}x -> "
          f"{mov.shape[1]//bin_xy} x {mov.shape[2]//bin_xy}")

    t_start = time.time()
    last = [0.0]

    def progress(done, total):
        f = done / total
        if f - last[0] >= 0.1 or done == total:
            last[0] = f
            el = time.time() - t_start
            print(f"    {f*100:5.1f}%  {el:6.1f} s elapsed, "
                  f"{el/max(f,1e-9)*(1-f):6.1f} s left", flush=True)

    fold = iof.fold_block(mov, ft, block, bins_per_cycle=bins_per_cycle,
                          bin_xy=bin_xy, chunk=chunk, progress=progress)
    io_s = time.time() - t_start
    print(f"  folded in {io_s:.1f} s "
          f"({gb/io_s:.2f} GiB/s effective)")
    if fold.n_empty:
        print(f"  note: {fold.n_empty} empty phase bins filled from neighbours")

    sp = iof.analyse(fold, offset=offset, min_counts=min_counts)

    m = sp.mask
    print(f"\n  mask: {m.sum()} / {m.size} pixels "
          f"({100*m.mean():.1f}%) above offset + {min_counts:.0f}")

    amp = sp.amplitude[m]
    snr = sp.snr[m]
    noise = sp.noise[m]

    shot = FRAC_NOISE * np.sqrt(2.0 / fold.n_per_bin.sum()) / bin_xy
    print(f"\n  noise per bin (per-quadrature):")
    print(f"    measured  median {np.median(noise)*100:.5f} %dR/R")
    print(f"    shot-noise prediction {shot*100:.5f} %dR/R")
    print(f"    ratio {np.median(noise)/shot:.2f}   "
          f"(1.0 = shot limited; >>1 = 1/f limited)")

    print(f"\n  signal at bin {sp.k_stim} ({sp.freqs[sp.k_stim]:.6f} Hz):")
    print(f"    amplitude median {np.median(amp)*100:.5f} %dR/R, "
          f"90th pct {np.percentile(amp,90)*100:.5f} %")
    print(f"    SNR median {np.median(snr):.2f}, "
          f"90th pct {np.percentile(snr,90):.2f}, max {snr.max():.2f}")
    for thr in (2, 3, 5, 10):
        print(f"    pixels with SNR > {thr:2d}: {100*(snr>thr).mean():5.1f}%  "
              f"(phase error < {np.degrees(1/thr):.0f} deg)")

    print(f"\n  neighbouring bins (mean power over mask, relative to bin "
          f"{sp.k_stim}):")
    k = sp.k_stim
    for j in range(k - 4, k + 5):
        bar = "#" * int(40 * min(sp.power_mean[j] / sp.power_mean[k], 1.5))
        tag = "  <-- stimulus" if j == k else ""
        print(f"    bin {j:3d}  {sp.freqs[j]:.5f} Hz  "
              f"{sp.power_mean[j]/sp.power_mean[k]:6.3f}  {bar}{tag}")

    if plot:
        _figure(fold, sp)
    return fold, sp


def _figure(fold, sp):
    m = sp.mask
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    a = ax[0, 0]
    a.imshow(fold.mean_image, cmap="gray")
    a.contour(m.astype(float), levels=[0.5], colors="c", linewidths=1)
    a.set_title("block mean (cyan = analysis mask)")

    a = ax[0, 1]
    v = np.where(m, sp.amplitude * 100, np.nan)
    im = a.imshow(v, cmap="magma", vmax=np.nanpercentile(v, 99))
    plt.colorbar(im, ax=a, label="%dR/R")
    a.set_title(f"amplitude at F ({sp.freqs[sp.k_stim]:.5f} Hz)")

    a = ax[0, 2]
    v = np.where(m & (sp.snr > 2), sp.phase, np.nan)
    im = a.imshow(v, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    plt.colorbar(im, ax=a, label="rad")
    a.set_title("phase (SNR > 2 only)")

    a = ax[1, 0]
    v = np.where(m, sp.snr, np.nan)
    im = a.imshow(v, cmap="viridis", vmax=np.nanpercentile(v, 99))
    plt.colorbar(im, ax=a)
    a.set_title("SNR at F")

    a = ax[1, 1]
    f, p = sp.freqs[1:], sp.power_mean[1:]
    a.loglog(f, p, lw=0.8)
    a.axvline(sp.freqs[sp.k_stim], color="r", ls="--", lw=1,
              label=f"F = {sp.freqs[sp.k_stim]:.5f} Hz")
    a.axvline(sp.freqs[sp.k_stim + 1], color="orange", ls=":", lw=1,
              label="bin 16 (nuisance?)")
    a.set_xlabel("Hz")
    a.set_ylabel("mean power")
    a.legend(fontsize=8)
    a.set_title("mean power spectrum over mask")

    a = ax[1, 2]
    a.hist(sp.snr[m], bins=60, range=(0, max(6, np.percentile(sp.snr[m], 99))))
    a.axvline(3, color="r", ls="--", label="SNR 3")
    a.set_xlabel("SNR")
    a.legend(fontsize=8)
    a.set_title("SNR distribution")

    fig.suptitle(f"Block 1 pilot -- {sp.label}, {fold.bin_xy}x binned")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)
    print(f"\n  wrote {OUT_PNG}")


def main():
    ft = ioi.load_frame_times()
    log = ioi.load_stim_log(ft=ft)
    block = log.by_label(BLOCK)
    with ioi.movie() as mov:
        fold, sp = run(mov, ft, block)
    np.savez_compressed(f"pilot_{BLOCK}.npz",
                        component=sp.component, amplitude=sp.amplitude,
                        phase=sp.phase, noise=sp.noise, snr=sp.snr,
                        mask=sp.mask, mean_image=fold.mean_image,
                        freqs=sp.freqs, power_mean=sp.power_mean)
    print(f"  wrote pilot_{BLOCK}.npz")


if __name__ == "__main__":
    main()
