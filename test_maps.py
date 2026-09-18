"""
Tests for ioi_maps.  Builds Fourier components analytically from a known
position map and a known delay map, then checks that combine_axis() recovers
both -- including in the region where the textbook half-difference aliases.
"""

import numpy as np

import ioi_core as ioi
import ioi_maps as iom

TWO_PI = 2 * np.pi


def make_pair(h=60, w=80, travel=86.87, visible_half=33.4, period=16.26,
              delay_amp=1.8, amp=0.03, noise=0.0, seed=0):
    """
    Ground truth: position ramps across the full VISIBLE screen extent;
    delay is a smooth gradient.  Returns fake BlockSpectrum/StimBlock pairs.
    """
    rng = np.random.default_rng(seed)
    p0F, p1F = -travel / 2, travel / 2

    yy, xx = np.mgrid[0:h, 0:w]
    true_pos = np.linspace(-visible_half, visible_half, w)[None, :] * np.ones((h, 1))
    true_delay = delay_amp * (0.6 + 0.4 * yy / h)

    u = (true_pos - p0F) / (p1F - p0F)             # forward fractional travel
    phiF = np.mod(true_delay / period + u, 1.0)
    phiR = np.mod(true_delay / period + 1 - u, 1.0)

    def spec(phi, label):
        c = amp * np.exp(-1j * TWO_PI * phi)       # rfft sign convention
        if noise:
            c = c + noise * (rng.normal(size=c.shape)
                             + 1j * rng.normal(size=c.shape))

        class S:
            pass
        s = S()
        s.component = c
        s.amplitude = np.abs(c)
        s.phase = np.angle(c)
        s.noise = np.full(c.shape, max(noise, 1e-9))
        s.snr = s.amplitude / s.noise
        s.mask = np.ones(c.shape, bool)
        s.label = label
        return s

    def blk(label, pa, pb):
        return ioi.StimBlock(label=label, axis="elevation", k0=0, k1=0,
                             frames_per_cycle=1, n_cycles=15,
                             cam0=0, cam1=1, t0=0.0, t1=period * 15,
                             bar_deg_first=pa, bar_deg_last=pb)

    return (spec(phiF, "FWD"), blk("FWD", p0F, p1F),
            spec(phiR, "REV"), blk("REV", p1F, p0F),
            true_pos, true_delay)


def test_recovers_position_and_delay():
    sF, bF, sR, bR, true_pos, true_delay = make_pair()
    am = iom.combine_axis(sF, bF, sR, bR, phase_sigma=0, delay_sigma=0,
                          snr_min=0, visible_half=33.4)
    m = am.mask
    ep = np.abs(am.position - true_pos)[m]
    ed = np.abs(am.delay - true_delay)[m]
    print(f"  position error: max {ep.max():.4f} deg")
    print(f"  delay error   : max {ed.max():.5f} s")
    assert ep.max() < 1e-6, ep.max()
    assert ed.max() < 1e-9, ed.max()
    print("  PASS  test_recovers_position_and_delay")


def test_half_difference_would_alias():
    """The method we did NOT use must demonstrably fail on this geometry."""
    sF, bF, sR, bR, true_pos, true_delay = make_pair()
    travel = bF.bar_deg_last - bF.bar_deg_first
    phiF = np.mod(-np.angle(sF.component) / TWO_PI, 1.0)
    phiR = np.mod(-np.angle(sR.component) / TWO_PI, 1.0)

    # textbook: position from the half-difference
    naive_u = iom.wrap_pi(TWO_PI * (phiF - phiR)) / TWO_PI / 2 + 0.5
    naive_pos = bF.bar_deg_first + naive_u * travel
    err = np.abs(naive_pos - true_pos)

    ours = iom.combine_axis(sF, bF, sR, bR, phase_sigma=0, delay_sigma=0,
                            snr_min=0).position
    print(f"  half-difference: max error {err.max():.1f} deg, "
          f"{100*(err>5).mean():.0f}% of pixels wrong by >5 deg")
    print(f"  our method    : max error "
          f"{np.abs(ours-true_pos).max():.2e} deg")
    assert err.max() > 20, "aliasing should be gross here"
    assert (err > 5).mean() > 0.2
    print("  PASS  test_half_difference_would_alias")


def test_delay_unambiguous_up_to_half_period():
    for d in (0.5, 2.0, 5.0, 7.5):
        sF, bF, sR, bR, _, td = make_pair(delay_amp=d, h=20, w=30)
        am = iom.combine_axis(sF, bF, sR, bR, phase_sigma=0, delay_sigma=0,
                              snr_min=0)
        err = np.abs(am.delay - td).max()
        ok = "OK " if err < 1e-6 else "WRAP"
        print(f"  delay up to {d*1.0:4.1f} s (T/2 = {am.period/2:.2f}): "
              f"max error {err:.2e} s  {ok}")
        if d < am.period / 2 * 0.95:
            assert err < 1e-6, (d, err)
    print("  PASS  test_delay_unambiguous_up_to_half_period")


def test_agreement_is_informative_with_noise():
    """
    With a smoothed delay the two directions are independent, so the
    agreement map must track the actual error rather than being zero by
    construction.
    """
    sF, bF, sR, bR, true_pos, _ = make_pair(noise=0.004, seed=1)
    am = iom.combine_axis(sF, bF, sR, bR, phase_sigma=1.0, delay_sigma=4.0,
                          snr_min=2.0, visible_half=33.4)
    m = am.mask & np.isfinite(am.agreement) & np.isfinite(am.position)
    spread = np.nanstd(am.agreement[m])
    err = np.abs(am.position - true_pos)[m]
    print(f"  agreement sd {spread:.3f} deg, "
          f"median |position error| {np.nanmedian(err):.3f} deg")
    assert spread > 1e-3, "agreement is zero by construction -- delay not smoothed?"
    # the two should be the same order of magnitude
    assert 0.2 < spread / (np.nanmedian(err) + 1e-9) < 20
    print("  PASS  test_agreement_is_informative_with_noise")


def test_per_pixel_delay_makes_agreement_vacuous():
    """The control: without smoothing, agreement is identically zero."""
    sF, bF, sR, bR, _, _ = make_pair(noise=0.004, seed=2)
    am = iom.combine_axis(sF, bF, sR, bR, phase_sigma=0, delay_sigma=0,
                          snr_min=0)
    a = np.abs(am.agreement[am.mask])
    print(f"  unsmoothed delay -> max |agreement| {np.nanmax(a):.2e} deg "
          f"(vacuous, as expected)")
    assert np.nanmax(a) < 1e-6
    print("  PASS  test_per_pixel_delay_makes_agreement_vacuous")


def test_azimuth_geometry():
    sF, bF, sR, bR, true_pos, _ = make_pair(travel=119.53, visible_half=49.7,
                                            period=22.37, delay_amp=1.5)
    am = iom.combine_axis(sF, bF, sR, bR, phase_sigma=0, delay_sigma=0,
                          snr_min=0, visible_half=49.7)
    err = np.abs(am.position - true_pos)[am.mask].max()
    print(f"  azimuth: travel {am.travel:.1f} deg, max position error "
          f"{err:.2e} deg")
    assert err < 1e-6
    print("  PASS  test_azimuth_geometry")


if __name__ == "__main__":
    print("Map combination tests")
    test_recovers_position_and_delay()
    test_half_difference_would_alias()
    test_delay_unambiguous_up_to_half_period()
    test_agreement_is_informative_with_noise()
    test_per_pixel_delay_makes_agreement_vacuous()
    test_azimuth_geometry()
    print("all passed")
