"""
Tests for the automatic tissue mask.

The point of the mask is that ONE threshold works across preparations, so the
tests are built to attack exactly that: the same synthetic field is presented
at a bright cranial-window light level and at a dim thinned-skull level, and
the mask must come back the same.  A field with no pulse at all must NOT come
back as a confident cardiac mask.
"""

import numpy as np

import ioi_core as ioi
import ioi_tissue as iot

H, W, BIN = 120, 160, 2   # binned 60x80, window radius 20 px: the
                         # kernel and the erosion must be small next to the
                         # window, as they are on the real 135x160 fields
FS = 100.0
N = 2000
F_HR = 7.3          # Hz, 438/min
F_RR = 1.6          # Hz, 96/min


def _ft(n=N, fs=FS):
    t = np.arange(n) / fs
    return ioi.FrameTimes(t=t, n_frames=n, pre_stim=0, post_stim=n,
                          removed=0, datenum0=0.0)


def _disk(h, w, r0, c0, rad):
    r, c = np.ogrid[:h, :w]
    return (r - r0) ** 2 + (c - c0) ** 2 <= rad ** 2


def synth(level=20000.0, pulse=0.002, resp=0.004, seed=0, tissue=None,
          with_pulse=True):
    """
    A field in raw camera counts.

    `tissue` pixels carry a cardiac and a respiratory modulation; everything
    else carries photon shot noise only, which is what bone, cement and the
    headplate actually look like.  `level` is the illumination, and halving it
    is the thinned-skull case: the pulse amplitude in dR/R is unchanged but
    every pixel is noisier, so a mask that depends on absolute amplitude will
    fail this and a mask that depends on the ratio will not.
    """
    rng = np.random.default_rng(seed)
    if tissue is None:
        tissue = _disk(H, W, 56, 88, 40)
    t = np.arange(N) / FS
    base = np.full((H, W), level, dtype=np.float64)
    base[~tissue] *= 1.35          # bone is BRIGHTER than brain, as in the data
    frac = np.zeros((N, H, W))
    if with_pulse:
        wave = (pulse * np.sin(2 * np.pi * F_HR * t)
                + 0.45 * pulse * np.sin(2 * np.pi * 2 * F_HR * t)
                + resp * np.sin(2 * np.pi * F_RR * t))
        frac += wave[:, None, None] * tissue[None, :, :]
    clean = base[None] * (1.0 + frac)
    counts = iot.OFFSET + rng.normal(clean, np.sqrt(iot.GAIN * clean))
    return counts.astype(np.float32), tissue


def _score(got, truth, bin_xy=BIN):
    t = truth.reshape(H // bin_xy, bin_xy, W // bin_xy, bin_xy).mean(axis=(1, 3))
    t = t > 0.5
    tp = int((got & t).sum())
    fp = int((got & ~t).sum())
    fn = int((~got & t).sum())
    return tp, fp, fn, tp / max(tp + fn, 1), fp / max(int((~t).sum()), 1)


def test_recovers_tissue_and_rejects_bone():
    mov, truth = synth()
    tm = iot.find_tissue(mov, _ft(), n_frames=N, bin_xy=BIN, verbose=False)
    tp, fp, fn, recall, fpr = _score(tm.mask, truth)
    print(f"  method {tm.method}, HR {tm.f_pulse*60:.0f}/min "
          f"(true {F_HR*60:.0f}), RR {tm.f_resp*60:.0f}/min "
          f"(true {F_RR*60:.0f})")
    print(f"  threshold {tm.threshold:.2f}x, median ratio inside "
          f"{np.nanmedian(tm.ratio[tm.mask]):.0f}x, outside "
          f"{np.nanmedian(tm.ratio[~tm.mask]):.1f}x")
    print(f"  recall {100*recall:.0f}%, false-positive rate on bone "
          f"{100*fpr:.1f}%  (tp {tp}, fp {fp}, fn {fn})")
    assert tm.method == "cardiac", tm.method
    assert abs(tm.f_pulse - F_HR) < 0.1, tm.f_pulse
    assert recall > 0.85, recall
    assert fpr < 0.05, fpr
    print("  PASS  test_recovers_tissue_and_rejects_bone")


def test_invariant_to_light_level():
    """
    The thinned-skull case.  A quarter of the light, and bone brighter than
    brain, so any brightness-based mask gets this exactly backwards.
    """
    out = {}
    for name, level in (("window", 20000.0), ("thin skull", 5000.0)):
        mov, truth = synth(level=level, seed=1)
        tm = iot.find_tissue(mov, _ft(), n_frames=N, bin_xy=BIN, verbose=False)
        tp, fp, fn, recall, fpr = _score(tm.mask, truth)
        out[name] = (tm, recall, fpr)
        print(f"  {name:<11} {level:>6.0f} counts: median ratio inside "
              f"{np.nanmedian(tm.ratio[tm.mask]):5.0f}x, recall "
              f"{100*recall:3.0f}%, fpr {100*fpr:4.1f}%")

    # and what a brightness band would have done with the same field
    mov, truth = synth(level=5000.0, seed=1)
    mean_img = mov.mean(axis=0).reshape(H // BIN, BIN, W // BIN,
                                        BIN).mean(axis=(1, 3))
    I = mean_img - iot.OFFSET
    lo, hi = np.percentile(I, [60.0, 99.5])
    bright = (I >= lo) & (I <= hi)
    _, _, _, br, bfpr = _score(bright, truth)
    print(f"  brightness band on the same field: recall {100*br:.0f}%, "
          f"fpr {100*bfpr:.0f}%  <-- what we are replacing")

    dr = out["window"][1] - out["thin skull"][1]
    print(f"  recall changes by {100*dr:+.0f} points for a 4x change in light")
    for name, (tm, recall, fpr) in out.items():
        assert recall > 0.80, (name, recall)
        assert fpr < 0.05, (name, fpr)
    # The dim prep gives a slightly SMALLER mask, because the rim pixels, where
    # the pulse is weakest, drop below threshold first.  That is the right
    # direction to err and it must stay small.
    assert 0 <= dr < 0.15, dr
    assert bfpr > 0.5, bfpr      # the brightness band really is that bad here
    print("  PASS  test_invariant_to_light_level")


def test_no_pulse_is_not_silently_accepted():
    """A field with no cardiac signal must not come back as 'cardiac'."""
    mov, truth = synth(with_pulse=True, pulse=0.0, resp=0.0, seed=2)
    tm = iot.find_tissue(mov, _ft(), n_frames=N, bin_xy=BIN, verbose=False)
    print(f"  method {tm.method!r}, {int(tm.mask.sum())} px kept, "
          f"{len(tm.notes)} note(s)")
    for n in tm.notes:
        print(f"    {n[:72]}")
    assert tm.method != "cardiac", "pure noise was accepted as a pulse"
    assert tm.notes, "fell back without saying so"
    print("  PASS  test_no_pulse_is_not_silently_accepted")


def _vessels(h=H, w=W):
    """A couple of thin curved surface vessels inside the window."""
    v = np.zeros((h, w), dtype=bool)
    tt = np.linspace(0, np.pi, 400)
    for r0, c0, rad, thick in ((56, 88, 26, 1), (56, 88, 14, 1)):
        rr = np.clip((r0 + rad * np.sin(tt)).astype(int), 0, h - 1)
        cc = np.clip((c0 + rad * np.cos(tt)).astype(int), 0, w - 1)
        for dr in range(-thick, thick + 1):
            for dc in range(-thick, thick + 1):
                v[np.clip(rr + dr, 0, h - 1), np.clip(cc + dc, 0, w - 1)] = True
    return v


def test_parenchyma_is_recovered_not_just_the_vasculature():
    """
    The failure seen on the first real recording.

    Over a surface vessel the cardiac modulation is ~0.09 %dR/R; over
    parenchyma it is nearer 0.01 %, which at 4x binning and 30 s sits at about
    its own single-pixel noise floor.  Thresholding pixel by pixel therefore
    returns a thin crescent of vasculature and calls the rest of the window
    non-tissue.

    Coherent spatial averaging is what recovers it, and the test asserts the
    contrast between the two settings rather than just the final number.
    """
    rng = np.random.default_rng(11)
    window = _disk(H, W, 56, 88, 40)
    vessel = _vessels() & window
    t = np.arange(N) / FS

    amp = np.where(vessel, 0.0009, 0.00012) * window     # as measured
    wave = (np.sin(2 * np.pi * F_HR * t)[:, None, None]
            + 0.45 * np.sin(2 * np.pi * 2 * F_HR * t)[:, None, None])
    base = np.where(window, 20000.0, 27000.0)
    clean = base[None] * (1.0 + amp[None] * wave)
    mov = (iot.OFFSET + rng.normal(clean, np.sqrt(iot.GAIN * clean))
           ).astype(np.float32)

    out = {}
    for s in (0.0, 1.0, 2.0, 3.0):
        tm = iot.find_tissue(mov, _ft(), n_frames=N, bin_xy=BIN,
                             smooth_sigma=s, verbose=False)
        tp, fp, fn, recall, fpr = _score(tm.mask, window)
        out[s] = (recall, fpr, tm)
        print(f"  smooth sigma {s:3.1f} ({s*38.4:3.0f} um): "
              f"{int(tm.mask.sum()):4d} px, recall {100*recall:3.0f}%, "
              f"fpr {100*fpr:4.1f}%, method {tm.method}, "
              f"threshold {tm.threshold:.2f}x")

    d = iot.SMOOTH_SIGMA
    assert out[0.0][0] < 0.35, ("unsmoothed should find little more than the "
                                "vessels", out[0.0][0])
    # monotone in the smoothing scale, and the shipped default must work
    rec = [out[s][0] for s in (0.0, 1.0, 2.0, 3.0)]
    assert all(b >= a - 0.02 for a, b in zip(rec, rec[1:])), rec
    assert out[d][0] > 0.85, (d, out[d][0])
    for s, (r, fp, _) in out.items():
        assert fp < 0.05, (s, fp)
    print(f"  coherent averaging takes recall from {100*out[0.0][0]:.0f}% "
          f"to {100*out[d][0]:.0f}% at the shipped default (sigma {d:g})")
    print("  PASS  test_parenchyma_is_recovered_not_just_the_vasculature")


def test_global_detector_uses_the_coherent_mean():
    """
    The global pulse detector must survive a pulse confined to the vessels.

    mean(|X|) over pixels suppresses no noise whatever, so a signal present in
    a few percent of the field cannot lift it and the detector falls back to
    the band maximum -- which is what happened on the real recording.
    |mean(X)| suppresses noise by sqrt(n_px) because the cardiac phase is
    common across the field.
    """
    rng = np.random.default_rng(12)
    window = _disk(H, W, 56, 88, 40)
    vessel = _vessels() & window
    t = np.arange(N) / FS
    amp = np.where(vessel, 0.0009, 0.0) * window
    clean = np.where(window, 20000.0, 27000.0)[None] * (
        1.0 + amp[None] * np.sin(2 * np.pi * F_HR * t)[:, None, None])
    mov = (iot.OFFSET + rng.normal(clean, np.sqrt(iot.GAIN * clean))
           ).astype(np.float32)

    tm = iot.find_tissue(mov, _ft(), n_frames=N, bin_xy=BIN, verbose=False)
    print(f"  pulse on {100*vessel.mean():.1f}% of the field -> global ratio "
          f"{tm.global_ratio:.1f}x, method {tm.method!r}, "
          f"HR {tm.f_pulse*60:.0f}/min (true {F_HR*60:.0f})")
    assert tm.method == "cardiac", tm.method
    assert abs(tm.f_pulse - F_HR) < 0.1, tm.f_pulse
    print("  PASS  test_global_detector_uses_the_coherent_mean")


def test_threshold_is_not_a_free_parameter():
    """
    The threshold must come from the field size, not from taste, and it must
    actually control the false-positive count on a pulse-free field.
    """
    thr = iot._rayleigh_threshold(6, 21600, 1.0)
    print(f"  21600 binned px, 6 bins searched, 1 expected false px "
          f"-> {thr:.2f}x")
    assert 3.0 < thr < 3.7, thr
    # 100x more tolerance for false positives should barely move it
    thr2 = iot._rayleigh_threshold(6, 21600, 100.0)
    print(f"  tolerating 100 false px instead of 1 -> {thr2:.2f}x "
          f"(only {100*(thr-thr2)/thr:.0f}% lower: the test is insensitive "
          f"to this choice)")
    assert thr2 < thr and (thr - thr2) / thr < 0.25

    # End to end, the way find_tissue actually does it: each pixel's sigma is
    # estimated from the MEDIAN of its own noise band, and the median of a
    # Rayleigh variable is sigma*sqrt(2 ln 2), so the code divides by
    # sqrt(ln 2).  Testing the formula without that conversion tests nothing;
    # this simulates the whole chain.
    rng = np.random.default_rng(3)
    n_px, n_bins, n_noise = 20000, 6, 300
    noise = rng.rayleigh(1.0, size=(n_px, n_noise))
    sigma = np.median(noise, axis=1) / np.sqrt(np.log(2.0))
    peak = rng.rayleigh(1.0, size=(n_px, n_bins)).max(axis=1)
    thr3 = iot._rayleigh_threshold(n_bins, n_px, 1.0)
    got = int((peak / sigma > thr3).sum())
    print(f"  simulated {n_px} pulse-free px ({n_noise}-bin noise band, "
          f"{n_bins} bins searched): {got} exceed {thr3:.2f}x "
          f"(expected ~1)")
    assert got <= 10, got
    print("  PASS  test_threshold_is_not_a_free_parameter")


if __name__ == "__main__":
    print("Tissue-mask tests")
    test_recovers_tissue_and_rejects_bone()
    test_invariant_to_light_level()
    test_no_pulse_is_not_silently_accepted()
    test_parenchyma_is_recovered_not_just_the_vasculature()
    test_global_detector_uses_the_coherent_mean()
    test_threshold_is_not_a_free_parameter()
    print("all passed")
