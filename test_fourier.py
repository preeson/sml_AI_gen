"""
Tests for the Fourier core.  A synthetic block with a known dR/R amplitude,
a known phase gradient and known shot-like noise must come back with the right
amplitude, the right phase, and an SNR matching theory.
"""

import numpy as np

import ioi_core as ioi
import ioi_fourier as iof


def make_synthetic(H=40, W=48, n_cycles=15, period=2.0, fs=100.0,
                   amp=0.002, base=8618.0, offset=1540.0, frac_noise=0.0174,
                   drop=None, seed=0):
    """A fake block: sinusoidal dR/R with a left-to-right phase ramp."""
    rng = np.random.default_rng(seed)
    dur = n_cycles * period
    n = int(round(dur * fs))
    t = np.arange(n) / fs
    if drop:  # simulate dropped frames by deleting timestamps and frames
        keep = np.ones(n, bool)
        keep[rng.choice(n, drop, replace=False)] = False
        t, n = t[keep], int(keep.sum())

    phase = np.linspace(-np.pi, np.pi, W, endpoint=False)[None, :] * np.ones((H, 1))
    f = 1.0 / period
    sig = amp * np.cos(2 * np.pi * f * t[:, None, None] - phase[None])
    signal_counts = (base - offset) * (1 + sig) + offset
    noisy = signal_counts + rng.normal(
        0, frac_noise * (base - offset), size=(n, H, W))

    class FT:
        pass
    ft = FT()
    ft.t = t
    ft.n_frames = n

    blk = ioi.StimBlock(label="SYN", axis="test", k0=0, k1=0,
                        frames_per_cycle=1, n_cycles=n_cycles,
                        cam0=0, cam1=n, t0=t[0], t1=t[-1])
    return noisy.astype(np.float32), ft, blk, phase, f


def test_amplitude_and_phase():
    mov, ft, blk, true_phase, f = make_synthetic()
    fold = iof.fold_block(mov, ft, blk, bins_per_cycle=80, bin_xy=1)
    sp = iof.analyse(fold, offset=1540.0, min_counts=500.0)

    assert sp.k_stim == 15, sp.k_stim
    got_f = sp.freqs[sp.k_stim]
    assert abs(got_f - f) < 1e-6 * f, (got_f, f)

    amp = np.median(sp.amplitude[sp.mask])
    print(f"  amplitude: recovered {amp*100:.4f}%  expected 0.2000%  "
          f"({amp/0.002:.3f}x)")
    assert 0.9 < amp / 0.002 < 1.1, amp

    # phase should track the imposed ramp (sign convention: cos(wt - phi))
    err = np.angle(np.exp(1j * (sp.phase + true_phase)))
    err = err - np.median(err)
    err = np.angle(np.exp(1j * err))
    # expected single-pixel phase scatter is ~1/SNR radians
    n = mov.shape[0]
    snr_theory = 0.002 / (0.0174 * np.sqrt(2.0 / n))
    expect = np.degrees(1.0 / snr_theory)
    got = np.degrees(err[sp.mask].std())
    print(f"  phase: residual sd {got:.2f} deg, theory ~{expect:.2f} deg "
          f"(single-pixel SNR {snr_theory:.2f})")
    assert got < 1.6 * expect, (got, expect)
    print("  PASS  test_amplitude_and_phase")


def test_noise_matches_theory():
    mov, ft, blk, _, _ = make_synthetic(amp=0.0)  # no signal at all
    n = mov.shape[0]
    for bxy in (1, 4):
        fold = iof.fold_block(mov, ft, blk, bins_per_cycle=80, bin_xy=bxy)
        sp = iof.analyse(fold, offset=1540.0, min_counts=500.0)
        theory = 0.0174 * np.sqrt(2.0 / n) / bxy   # spatial binning: /bxy
        got = np.median(sp.noise.ravel())
        print(f"  bin_xy={bxy}: noise/bin {got*100:.5f}%  "
              f"theory {theory*100:.5f}%  ratio {got/theory:.3f}")
        assert 0.85 < got / theory < 1.15, (got, theory)
        # with no signal present, SNR must sit near 1
        med_snr = np.median(sp.snr[sp.mask])
        assert med_snr < 1.6, med_snr
    print("  PASS  test_noise_matches_theory")


def test_adjacent_bin_is_orthogonal():
    """A nuisance one bin away must not contaminate the stimulus bin."""
    mov, ft, blk, _, f = make_synthetic(amp=0.0, frac_noise=1e-6)
    t = ft.t
    base, offset = 8618.0, 1540.0
    f_nuis = f * 16.0 / 15.0          # exactly one DFT bin above
    nuis = 0.02 * np.cos(2 * np.pi * f_nuis * t + 0.7)
    mov = mov + (base - offset) * nuis[:, None, None]

    fold = iof.fold_block(mov, ft, blk, bins_per_cycle=80, bin_xy=4)
    sp = iof.analyse(fold, offset=offset, min_counts=500.0, detrend_order=None)
    leak = np.median(sp.amplitude[sp.mask])
    print(f"  nuisance 2.000% at bin 16 -> leakage into bin 15: "
          f"{leak*100:.6f}%  ({leak/0.02*100:.4f}% of it)")
    assert leak / 0.02 < 0.01, leak

    # and demonstrate why: the same signal WITH a Hann taper
    x = nuis - nuis.mean()
    N = x.size
    X_rect = np.abs(np.fft.rfft(x)[15]) * 2 / N
    xw = x * np.hanning(N)
    X_hann = np.abs(np.fft.rfft(xw)[15]) * 2 / N / (np.hanning(N).mean())
    print(f"  same nuisance, rectangular window leaks {X_rect/0.02*100:.4f}% "
          f"into bin 15; Hann window leaks {X_hann/0.02*100:.2f}%")
    assert X_hann > 50 * X_rect
    print("  PASS  test_adjacent_bin_is_orthogonal")


def test_dropped_frames_handled():
    """Phase binning by timestamp must survive dropped frames."""
    mov, ft, blk, true_phase, _ = make_synthetic(drop=300, seed=3)
    fold = iof.fold_block(mov, ft, blk, bins_per_cycle=80, bin_xy=1)
    sp = iof.analyse(fold, offset=1540.0, min_counts=500.0)
    amp = np.median(sp.amplitude[sp.mask])
    err = np.angle(np.exp(1j * (sp.phase + true_phase)))
    err = np.angle(np.exp(1j * (err - np.median(err))))
    n = mov.shape[0]
    snr_theory = 0.002 / (0.0174 * np.sqrt(2.0 / n))
    expect = np.degrees(1.0 / snr_theory)
    got = np.degrees(err[sp.mask].std())
    print(f"  300 frames dropped: amplitude {amp*100:.4f}% "
          f"(expected 0.2000%), phase sd {got:.2f} deg (theory ~{expect:.2f})")
    assert 0.9 < amp / 0.002 < 1.1
    assert got < 1.6 * expect, (got, expect)
    print("  PASS  test_dropped_frames_handled")


def test_mask_excludes_dark_pixels():
    mov, ft, blk, _, _ = make_synthetic()
    mov[:, :, :10] = 1545.0  # a strip of "dental cement" just above pedestal
    fold = iof.fold_block(mov, ft, blk, bins_per_cycle=80, bin_xy=1)
    sp = iof.analyse(fold, offset=1540.0, min_counts=500.0)
    assert not sp.mask[:, :10].any(), "dark pixels were not masked"
    assert sp.mask[:, 12:].all(), "bright pixels were wrongly masked"
    print("  PASS  test_mask_excludes_dark_pixels")


if __name__ == "__main__":
    print("Fourier core tests")
    test_amplitude_and_phase()
    test_noise_matches_theory()
    test_adjacent_bin_is_orthogonal()
    test_dropped_frames_handled()
    test_mask_excludes_dark_pixels()
    print("all passed")
