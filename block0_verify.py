"""
Block 0 -- verify the raw data before touching it.

Checks, in order:
  1. .dat byte count is a whole number of frames, and matches frameTimes.
  2. Camera clock: rate, jitter, dropped frames.
  3. Stimulus blocks land inside the recording and tile it without overlap.
  4. Pixel layout: read a frame both ways and look at it.
  5. Dynamic range: check for saturation or a dead sensor.

Run this first.  Everything downstream assumes it passed.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # comment out in PyCharm if you want interactive windows
import matplotlib.pyplot as plt

import ioi_core as ioi

OUT_PNG = "block0_qc.png"


def rule(title):
    print("\n" + "-" * 68)
    print(title)
    print("-" * 68)


def main():
    # ---- 1. file geometry ------------------------------------------------
    rule("1. File geometry")
    n_disk = ioi.frame_count_on_disk()
    ft = ioi.load_frame_times()
    print(f"  .dat frames        : {n_disk}")
    print(f"  frameTimes entries : {ft.n_frames}")
    print(f"  bytes per frame    : {ioi.BYTES_PER_FRAME} "
          f"({ioi.FRAME_H} x {ioi.FRAME_W} x {ioi.DTYPE.itemsize} B)")
    if n_disk != ft.n_frames:
        print(f"  !! MISMATCH of {n_disk - ft.n_frames} frames. "
              f"Do not proceed until this is understood.")
    else:
        print("  OK: camera frame index == frameTimes index.")

    # ---- 2. camera clock -------------------------------------------------
    rule("2. Camera clock")
    dt = np.diff(ft.t)
    med = np.median(dt)
    print(f"  duration           : {ft.t[-1]:.3f} s")
    print(f"  mean rate          : {ft.fs:.4f} Hz")
    print(f"  dt median          : {med*1000:.4f} ms   sd {dt.std()*1000:.4f} ms")
    print(f"  dt min / max       : {dt.min()*1000:.3f} / {dt.max()*1000:.3f} ms")
    print(f"  preStim / postStim : {ft.pre_stim} / {ft.post_stim}   "
          f"removedFrames {ft.removed}")
    print(f"  baseline frames    : {ft.pre_stim} before, "
          f"{ft.n_frames - ft.post_stim} after")

    gaps = np.flatnonzero(dt > 1.5 * med)
    missing = float((dt[gaps] / med - 1).sum()) if gaps.size else 0.0
    print(f"  intervals > 1.5x median : {gaps.size}  "
          f"(~{missing:.1f} frames' worth of time lost)")
    for g in gaps[:10]:
        print(f"      after frame {g:>7d}  dt = {dt[g]*1000:7.2f} ms  "
              f"t = {ft.t[g]:8.2f} s")

    # ---- 3. stimulus blocks ---------------------------------------------
    rule("3. Stimulus blocks on the camera clock")
    log = ioi.load_stim_log(ft=ft)
    print(f"  stimulus frames    : {log.n_stim_frames}")
    print(f"  nominal refresh    : {log.nominal_refresh:.2f} Hz")
    print(f"  effective refresh  : {log.effective_refresh:.4f} Hz  "
          f"(from the camera clock)")
    print(f"  sweep width {log.sweep_width:.1f} deg, "
          f"step {log.step_width:.4f} deg")
    print()
    hdr = (f"  {'blk':<5}{'axis':<11}{'cam frames':<20}{'dur (s)':>9}"
           f"{'cyc':>5}{'F (Hz)':>10}{'bar centre (deg)':>22}")
    print(hdr)
    for b in log.blocks:
        print(f"  {b.label:<5}{b.axis:<11}"
              f"{f'{b.cam0}-{b.cam1}':<20}{b.duration:9.2f}"
              f"{b.n_cycles:5d}{b.f_stim:10.6f}"
              f"{f'{b.bar_deg_first:+.1f} -> {b.bar_deg_last:+.1f}':>22}")

    # contiguity / containment
    ok = True
    for a, c in zip(log.blocks, log.blocks[1:]):
        if a.cam1 != c.cam0:
            print(f"  !! gap/overlap between {a.label} and {c.label}: "
                  f"{a.cam1} vs {c.cam0}")
            ok = False
    if log.blocks[0].cam0 != ft.pre_stim:
        print(f"  !! first block starts at {log.blocks[0].cam0}, "
              f"expected {ft.pre_stim}")
        ok = False
    if log.blocks[-1].cam1 != ft.post_stim:
        print(f"  !! last block ends at {log.blocks[-1].cam1}, "
              f"expected {ft.post_stim}")
        ok = False
    if ok:
        print("  OK: blocks tile the stimulus window exactly.")

    # ---- 4. pixel layout -------------------------------------------------
    rule("4. Pixel layout")
    mov = ioi.open_movie(n_frames=n_disk)
    print(f"  movie view shape   : {mov.shape}  (frame, row, col)")

    probe = [0, ft.pre_stim, n_disk // 2, n_disk - 1]
    raw = np.memmap(ioi.DAT_PATH, dtype=ioi.DTYPE, mode="r",
                    shape=(n_disk, ioi.FRAME_W, ioi.FRAME_H))

    f_correct = np.asarray(mov[probe[1]], dtype=np.float32)       # (540, 640)
    f_wrong = np.asarray(raw[probe[1]], dtype=np.float32)          # (640, 540)
    f_cwrong = np.asarray(raw[probe[1]]).reshape(ioi.FRAME_H, ioi.FRAME_W)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ax, img, ttl in zip(
        axes[0],
        [f_correct, f_wrong, f_cwrong],
        ["Fortran-order -> (540, 640)\nEXPECTED (imgSize says 540x640)",
         "as stored, (640, 540)\n(transpose of the above)",
         "naive C-order reshape (540, 640)\nWRONG -- should look sheared"],
    ):
        lo, hi = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel(f"{img.shape[1]} px")
        ax.set_ylabel(f"{img.shape[0]} px")

    for ax, i in zip(axes[1], probe[:3]):
        img = np.asarray(mov[i], dtype=np.float32)
        lo, hi = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"frame {i}   t = {ft.t[i]:.2f} s", fontsize=9)

    fig.suptitle("Block 0 QC -- top row: which layout looks like a brain?  "
                 "bottom row: stability over the session")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)
    ioi.close_movie(raw)
    print(f"  wrote {OUT_PNG} -- open it and confirm the first panel is the "
          f"one that looks like cortex.")

    # ---- 5. dynamic range ------------------------------------------------
    rule("5. Dynamic range")
    for i in probe:
        img = np.asarray(mov[i], dtype=np.float32)
        sat = float((img >= 65535).mean() * 100)
        zero = float((img == 0).mean() * 100)
        print(f"  frame {i:>7d}: min {img.min():6.0f}  "
              f"p1 {np.percentile(img,1):7.0f}  median {np.median(img):7.0f}  "
              f"p99 {np.percentile(img,99):7.0f}  max {img.max():6.0f}   "
              f"sat {sat:.3f}%  zero {zero:.3f}%")

    base = np.asarray(mov[:ft.pre_stim], dtype=np.float32)
    print(f"\n  baseline ({ft.pre_stim} frames): mean {base.mean():.1f}, "
          f"per-pixel temporal sd (median) "
          f"{np.median(base.std(axis=0)):.2f}")
    print(f"  photon-limited shot noise would be ~sqrt(mean) = "
          f"{np.sqrt(base.mean()):.2f} counts")

    ioi.close_movie(mov)
    print("\nBlock 0 complete.")


if __name__ == "__main__":
    main()
