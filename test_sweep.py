"""
Tests for the patch-stability sweep in block3_sign.

Smoothing is a free parameter, and turning it until a map appears is not
evidence.  The sweep is only worth printing if it actually separates a real
cortical area from noise, so that is what these tests check: a planted disc
of smooth orthogonal retinotopy must grow toward its true area and settle,
while pure noise under the same mask must not.
"""
import numpy as np

import block3_sign as b3
import ioi_sign as ios

H, W = 135, 160
PX_MM2 = (b3.UM_PER_PIXEL / 1000.0) ** 2


class _P:
    def __init__(self, position):
        self.position = position


def _mask():
    m = np.zeros((H, W), bool)
    m[20:100, 15:85] = True
    return m


def _areas(el, az, bright, sigma, snr_val=5.0):
    sm = ios.visual_field_sign(el, az, bright, um_per_pixel=b3.UM_PER_PIXEL,
                               sigma=sigma, snr=np.full((H, W), snr_val),
                               snr_min=b3.SNR_MIN)
    px = ios.patches(sm)
    return sorted((p.area_mm2 for p in px), reverse=True)


def _planted(seed=0, radius=23, noise=3.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    disc = ((yy - 60) ** 2 + (xx - 50) ** 2) < radius ** 2
    el = np.where(disc, (yy - 60) * 0.8, rng.normal(0, 25, (H, W)))
    az = np.where(disc, (xx - 50) * 0.8, rng.normal(0, 25, (H, W)))
    el += rng.normal(0, noise, (H, W))
    az += rng.normal(0, noise, (H, W))
    return disc, el, az


def test_real_area_grows_and_settles():
    bright = _mask()
    disc, el, az = _planted()
    true_mm2 = (disc & bright).sum() * PX_MM2
    big = [(_areas(el, az, bright, s) or [0.0])[0] for s in (2.0, 4.0, 8.0)]
    print(f"  planted disc {true_mm2:.2f} mm2 -> largest patch at sigma "
          f"2/4/8: {big[0]:.2f}, {big[1]:.2f}, {big[2]:.2f} mm2")
    assert big[2] > big[0], "a real area should grow with smoothing"
    assert 0.6 * true_mm2 < big[2] < 1.6 * true_mm2, (big[2], true_mm2)
    print("  PASS  test_real_area_grows_and_settles")


def test_noise_never_reaches_cortical_size():
    bright = _mask()
    worst = 0.0
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        el = rng.normal(0, 25, (H, W))
        az = rng.normal(0, 25, (H, W))
        for s in (2.0, 4.0, 8.0):
            a = _areas(el, az, bright, s)
            worst = max(worst, a[0] if a else 0.0)
    print(f"  pure noise, 5 seeds x 3 sigmas: largest patch ever "
          f"{worst:.2f} mm2 (mouse V1 is 2-3)")
    assert worst < 1.5, worst
    print("  PASS  test_noise_never_reaches_cortical_size")


def test_noise_fragments_while_signal_does_not():
    """At heavy smoothing, noise breaks into many patches; an area does not."""
    bright = _mask()
    _, el, az = _planted(seed=1)
    rng = np.random.default_rng(7)
    eln, azn = rng.normal(0, 25, (H, W)), rng.normal(0, 25, (H, W))
    n_sig = len(_areas(el, az, bright, 8.0))
    n_noise = len(_areas(eln, azn, bright, 8.0))
    print(f"  at sigma 8: planted disc gives {n_sig} patch(es), "
          f"noise gives {n_noise}")
    assert n_noise > n_sig, (n_sig, n_noise)
    print("  PASS  test_noise_fragments_while_signal_does_not")


if __name__ == "__main__":
    print("Patch-stability sweep tests")
    test_real_area_grows_and_settles()
    test_noise_never_reaches_cortical_size()
    test_noise_fragments_while_signal_does_not()
    print("all passed")
