"""
Tests for the amplitude-only localiser in block2c_control.

It exists because a phase map can be unusable while an amplitude map is
still perfectly informative -- noisy phase rails at the travel limit and
destroys the gradients, noisy amplitude just averages.  So it has to do two
things: find a localised response that the sign map would miss, and report
nothing when every bin is equivalent.
"""
import numpy as np

import block2c_control as b2c

H, W = 135, 160
PX_MM2 = (b2c.UM_PER_PIXEL / 1000.0) ** 2


class _S:
    def __init__(self, snr):
        self.snr = snr


def _results(hot_mask=None, hot_gain=0.8, seed=0):
    """A results dict shaped like metrics_for_bin's, with noise in every bin."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in b2c.K_LIST:
        per = {}
        for lb in b2c.LABELS:
            snr = rng.normal(1.0, 0.35, (H, W))
            if hot_mask is not None and k == b2c.K_STIM:
                snr[hot_mask] += hot_gain
            per[lb] = _S(snr)
        out[k] = (None, None, per)
    return out


def _mask():
    m = np.zeros((H, W), bool)
    m[20:100, 15:85] = True
    return m


def test_finds_a_localised_response():
    bright = _mask()
    yy, xx = np.mgrid[0:H, 0:W]
    hot = (((yy - 62) ** 2 + (xx - 48) ** 2) < 20 ** 2) & bright
    res = _results(hot_mask=hot, hot_gain=0.8, seed=1)
    maps, z = b2c._excess_report(res, bright)
    ex = maps[b2c.K_STIM]
    rows, cols = np.nonzero(hot)
    inside = np.nanmean(ex[hot])
    outside = np.nanmean(ex[bright & ~hot])
    print(f"  planted hot spot {hot.sum() * PX_MM2:.2f} mm2 at "
          f"({rows.mean():.0f}, {cols.mean():.0f}); excess inside "
          f"{inside:+.3f} vs outside {outside:+.3f}; z = {z:+.2f}")
    assert z > 4, z
    assert inside > outside + 0.3, (inside, outside)
    # the peak of the excess map must land inside the planted spot
    pk = np.unravel_index(np.nanargmax(np.where(bright, ex, -np.inf)), ex.shape)
    assert hot[pk], pk
    print("  PASS  test_finds_a_localised_response")


def test_reports_nothing_when_no_bin_is_special():
    bright = _mask()
    zs = []
    for seed in range(6):
        _, z = b2c._excess_report(_results(seed=seed), bright)
        zs.append(z)
    print(f"  six pure-noise datasets: z = "
          + ", ".join(f"{z:+.2f}" for z in zs))
    assert max(zs) < 3.0, zs
    print("  PASS  test_reports_nothing_when_no_bin_is_special")


def test_survives_unusable_phase():
    """
    The point of the thing: amplitude localisation must work even when the
    phase is pure noise, which is the situation the sign map fails in.
    """
    bright = _mask()
    yy, xx = np.mgrid[0:H, 0:W]
    hot = (((yy - 50) ** 2 + (xx - 60) ** 2) < 18 ** 2) & bright
    res = _results(hot_mask=hot, hot_gain=0.5, seed=5)   # weak
    _, z = b2c._excess_report(res, bright)
    print(f"  weak localised response (gain 0.5, no phase information at "
          f"all): z = {z:+.2f}")
    assert z > 3, z
    print("  PASS  test_survives_unusable_phase")


if __name__ == "__main__":
    print("Amplitude localiser tests")
    test_finds_a_localised_response()
    test_reports_nothing_when_no_bin_is_special()
    test_survives_unusable_phase()
    print("all passed")
