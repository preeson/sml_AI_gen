"""
Tests for ioi_leak.  Two matter most:

  test_recovers_buried_retinotopy -- a real signal 10x weaker than the leak
      must survive removal and be detectable.
  test_does_not_manufacture_structure -- leak plus noise and NOTHING else must
      come out unstructured.  Without this, a "signal" found in the real data
      would be worthless.
"""

import numpy as np

import ioi_leak as iol

TWO_PI = 2 * np.pi


def scene(h=135, w=160, leak_counts=0.84, leak_frac=3.7e-5, leak_phase=2.2,
          sig_amp=0.0, noise=5.5e-5, seed=0):
    """
    Brightness falls off towards the edges (vignetting + cement), so the
    additive leak dominates there -- as in the real data.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w * 0.4) / (w * 0.42), (yy - h * 0.5) / (h * 0.5))
    I = 400 + 24000 * np.exp(-(r ** 2))

    c = (leak_counts / I + leak_frac) * np.exp(1j * leak_phase)

    true_pos = None
    if sig_amp:
        # retinotopy: phase ramps diagonally across the bright region
        u = 0.15 + 0.7 * (xx / w)
        c = c + sig_amp * np.exp(1j * TWO_PI * u)
        true_pos = u
    c = c + noise * (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w)))

    class S:
        pass
    sp = S()
    sp.component = c
    sp.amplitude = np.abs(c)
    sp.phase = np.angle(c)
    sp.noise = np.full((h, w), noise)
    sp.snr = sp.amplitude / noise
    sp.label = "SYN"
    sp.mask = I > np.percentile(I, 40)
    return sp, I, true_pos


def test_fit_recovers_leak_parameters():
    sp, I, _ = scene(noise=1e-8)
    fit = iol.fit_common_mode(sp.component, I, sp.noise, sp.mask, order=0)
    a_true = 0.84 * np.exp(1j * 2.2)
    b_true = 3.7e-5 * np.exp(1j * 2.2)
    print(f"  alpha fitted {abs(fit.alpha):.4f} counts @ "
          f"{np.degrees(np.angle(fit.alpha)):.1f} deg "
          f"(true {abs(a_true):.4f} @ {np.degrees(np.angle(a_true)):.1f})")
    print(f"  beta  fitted {abs(fit.beta)*100:.5f}% @ "
          f"{np.degrees(np.angle(fit.beta)):.1f} deg "
          f"(true {abs(b_true)*100:.5f}% @ {np.degrees(np.angle(b_true)):.1f})")
    assert abs(fit.alpha - a_true) / abs(a_true) < 0.02
    assert abs(fit.beta - b_true) / abs(b_true) < 0.05
    print("  PASS  test_fit_recovers_leak_parameters")


def test_does_not_manufacture_structure():
    """THE CONTROL: leak + noise only.  Residual must be unstructured."""
    for seed in (0, 1, 2):
        sp, I, _ = scene(sig_amp=0.0, seed=seed)
        bright = I > np.percentile(I, 60)
        res, fit = iol.remove_common_mode(sp, I, bright, order=0)
        obs, shuf, z = iol.structure_score(res.phase, bright, seed=seed)
        conc = iol.phase_concentration(res.component, bright)
        print(f"  seed {seed}: var removed {fit.var_removed*100:5.1f}%, "
              f"residual phase {obs:.1f} deg vs shuffled {shuf:.1f} deg "
              f"(z = {z:+.2f}), concentration {conc:.3f}")
        assert abs(z) < 4.0, f"structure invented from pure leak (z={z})"
        assert conc < 0.15, conc
    print("  PASS  test_does_not_manufacture_structure")


def test_recovers_buried_retinotopy():
    """A signal well below the leak must survive and be detectable."""
    leak_typical = 0.84 / 12000 + 3.7e-5      # ~1.1e-4 in bright tissue
    for ratio in (1.0, 0.3, 0.1):
        sig = leak_typical * ratio
        sp, I, _ = scene(sig_amp=sig, seed=5)
        bright = I > np.percentile(I, 60)

        pre = iol.structure_score(sp.phase, bright, seed=1)
        res, fit = iol.remove_common_mode(sp, I, bright, order=0)
        post = iol.structure_score(res.phase, bright, seed=1)
        print(f"  signal/leak {ratio:4.2f} (amp {sig*100:.5f}%): "
              f"z before {pre[2]:+6.2f} -> after {post[2]:+6.2f}, "
              f"residual SNR median {np.median(res.snr[bright]):.2f}")
        if ratio >= 0.3:
            assert post[2] < -5, f"buried signal not recovered (z={post[2]})"
    print("  PASS  test_recovers_buried_retinotopy")


def test_order_sensitivity():
    """Higher polynomial order must not silently eat a real signal."""
    sig = (0.84 / 12000 + 3.7e-5) * 0.5
    sp, I, _ = scene(sig_amp=sig, seed=7)
    bright = I > np.percentile(I, 60)
    for order in (0, 1, 2):
        res, fit = iol.remove_common_mode(sp, I, bright, order=order)
        _, _, z = iol.structure_score(res.phase, bright, seed=2)
        print(f"  order {order}: var removed {fit.var_removed*100:5.1f}%, "
              f"residual structure z = {z:+.2f}")
    print("  PASS  test_order_sensitivity (inspect: z should stay negative)")


if __name__ == "__main__":
    print("Leak removal tests")
    test_fit_recovers_leak_parameters()
    test_does_not_manufacture_structure()
    test_recovers_buried_retinotopy()
    test_order_sensitivity()
    print("all passed")
