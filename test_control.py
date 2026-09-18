"""
Tests for the off-bin control logic.

The point of Block 2c is that it needs no model of the noise.  These tests
confirm that:

  1. spatially correlated noise -- which fools the shuffle test completely --
     gives the SAME structure z at every bin, so the off-bin comparison
     correctly returns "nothing here";
  2. a signal injected at one bin only makes that bin an outlier;
  3. analyse(k=...) really does read the bin it is asked for.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

import ioi_fourier as iof
import ioi_leak as iol
import test_fourier as tf


def correlated_noise(h, w, sigma, rng):
    c = rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w))
    if sigma > 0:
        c = gaussian_filter(c.real, sigma) + 1j * gaussian_filter(c.imag, sigma)
        c /= np.std(np.abs(c))
    return c


def test_shuffle_test_is_fooled_by_correlated_noise():
    """Documents the failure mode Block 2c exists to catch."""
    rng = np.random.default_rng(0)
    m = np.ones((135, 160), bool)
    print("  correlation  neighbour diff   z   (NO signal present)")
    zs = {}
    for s in (0.0, 0.5, 1.0, 2.0):
        c = correlated_noise(135, 160, s, rng)
        obs, shuf, z = iol.structure_score(np.angle(c), m, n_shuffle=10)
        zs[s] = z
        print(f"    {s:4.1f} px      {obs:6.1f} deg    {z:+8.1f}")
    assert abs(zs[0.0]) < 4, zs[0.0]
    assert zs[1.0] < -100, zs[1.0]
    print("  PASS  test_shuffle_test_is_fooled_by_correlated_noise")


def test_offbin_control_returns_null_on_noise():
    """Correlated noise at every bin -> no bin stands out."""
    rng = np.random.default_rng(1)
    h, w = 135, 160
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w * 0.5) / (w * 0.45), (yy - h * 0.5) / (h * 0.5))
    I = 400 + 24000 * np.exp(-(r ** 2))
    bright = I > np.percentile(I, 60)
    valid = I > 500

    zs, snrs = [], []
    for k in range(11, 20):
        c = 6e-5 * correlated_noise(h, w, 1.0, rng)

        class S:
            pass
        sp = S()
        sp.label = f"k{k}"
        sp.component = c
        sp.noise = np.full((h, w), 5.5e-5)
        sp.amplitude = np.abs(c)
        sp.phase = np.angle(c)
        sp.snr = sp.amplitude / sp.noise
        sp.mask = valid
        res, _ = iol.remove_common_mode(sp, I, bright, order=0, fit_mask=valid)
        zs.append(iol.structure_score(res.phase, bright)[2])
        snrs.append(np.nanmedian(res.snr[bright]))

    zs, snrs = np.asarray(zs), np.asarray(snrs)
    k15 = 15 - 11
    off = np.delete(np.arange(9), k15)
    z_out = (zs[k15] - zs[off].mean()) / zs[off].std()
    s_out = (snrs[k15] - snrs[off].mean()) / snrs[off].std()
    print(f"  structure z across bins: {np.round(zs, 0)}")
    print(f"  bin 15 outlier score: structure {z_out:+.2f}, SNR {s_out:+.2f}")
    assert abs(z_out) < 3 and abs(s_out) < 3, (z_out, s_out)
    print("  PASS  test_offbin_control_returns_null_on_noise "
          "(every bin equally 'structured')")


def test_offbin_control_detects_injected_signal():
    rng = np.random.default_rng(2)
    h, w = 135, 160
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w * 0.5) / (w * 0.45), (yy - h * 0.5) / (h * 0.5))
    I = 400 + 24000 * np.exp(-(r ** 2))
    bright = I > np.percentile(I, 60)
    valid = I > 500
    sig = 2.5e-4 * np.exp(1j * 2 * np.pi * (0.15 + 0.7 * xx / w))

    snrs = []
    for k in range(11, 20):
        c = 6e-5 * correlated_noise(h, w, 1.0, rng)
        if k == 15:
            c = c + sig

        class S:
            pass
        sp = S()
        sp.label = f"k{k}"
        sp.component = c
        sp.noise = np.full((h, w), 5.5e-5)
        sp.amplitude = np.abs(c)
        sp.phase = np.angle(c)
        sp.snr = sp.amplitude / sp.noise
        sp.mask = valid
        res, _ = iol.remove_common_mode(sp, I, bright, order=0, fit_mask=valid)
        snrs.append(np.nanmedian(res.snr[bright]))

    snrs = np.asarray(snrs)
    off = np.delete(np.arange(9), 4)
    s_out = (snrs[4] - snrs[off].mean()) / snrs[off].std()
    print(f"  residual SNR across bins: {np.round(snrs, 2)}")
    print(f"  bin 15 outlier score: {s_out:+.2f}")
    assert s_out > 5, s_out
    print("  PASS  test_offbin_control_detects_injected_signal")


def test_analyse_reads_the_requested_bin():
    """A signal at 15 cycles must appear at k=15 and not at k=13."""
    mov, ft, blk, _, _ = tf.make_synthetic(amp=0.002, frac_noise=0.002)
    fold = iof.fold_block(mov, ft, blk, bins_per_cycle=80, bin_xy=4)
    a15 = iof.analyse(fold, offset=1540.0, min_counts=500.0, k=15)
    a13 = iof.analyse(fold, offset=1540.0, min_counts=500.0, k=13,
                      exclude_bins=(15,))
    m = a15.mask
    print(f"  k=15 amplitude {np.median(a15.amplitude[m])*100:.4f}%  "
          f"SNR {np.median(a15.snr[m]):.2f}")
    print(f"  k=13 amplitude {np.median(a13.amplitude[m])*100:.4f}%  "
          f"SNR {np.median(a13.snr[m]):.2f}")
    assert np.median(a15.snr[m]) > 4 * np.median(a13.snr[m])
    print("  PASS  test_analyse_reads_the_requested_bin")


def test_fit_mask_conditioning():
    """Fitting in a narrow brightness band recovers alpha badly; full field is fine."""
    import test_leak as tl
    sp, I, _ = tl.scene(noise=3e-5, seed=4)
    bright = I > np.percentile(I, 60)
    valid = I > 400
    a_true = 0.84
    narrow = iol.fit_common_mode(sp.component, I, sp.noise, bright, order=0)
    full = iol.fit_common_mode(sp.component, I, sp.noise, bright, order=0,
                               fit_mask=valid)
    e_n = abs(abs(narrow.alpha) - a_true) / a_true
    e_f = abs(abs(full.alpha) - a_true) / a_true
    print(f"  alpha fitted in bright band only: {abs(narrow.alpha):.3f} "
          f"counts ({e_n*100:.0f}% error)")
    print(f"  alpha fitted on the full field  : {abs(full.alpha):.3f} "
          f"counts ({e_f*100:.0f}% error)")
    assert e_f < e_n, (e_f, e_n)
    print("  PASS  test_fit_mask_conditioning")


if __name__ == "__main__":
    print("Off-bin control tests")
    test_shuffle_test_is_fooled_by_correlated_noise()
    test_offbin_control_returns_null_on_noise()
    test_offbin_control_detects_injected_signal()
    test_analyse_reads_the_requested_bin()
    test_fit_mask_conditioning()
    print("all passed")
