"""
Tests for the circular, weighted, wrap-safe sign map.

Both tests are built around the two failure mechanisms actually seen in the
data, and both compare the new path against the old one on the SAME synthetic
field, so the printed numbers say how much the change is worth rather than just
that it runs.
"""

import numpy as np

import ioi_sign as ios

H, W = 60, 80
UM = 38.4375
TRAVEL_EL, TRAVEL_AZ = 86.9, 119.5


def _disk(r0, c0, rad, h=H, w=W):
    r, c = np.ogrid[:h, :w]
    return (r - r0) ** 2 + (c - c0) ** 2 <= rad ** 2


def _largest(sm):
    ps = ios.patches(sm, min_area_mm2=0.0)
    return (ps[0].area_mm2, ps[0].sign, len(ps)) if ps else (0.0, 0, 0)


def test_wrap_does_not_flip_the_sign():
    """
    A map that wraps must not grow a sign reversal at the wrap.

    Elevation runs with the row and azimuth with the column, so the two
    gradients are perpendicular everywhere and the true sign is a single
    constant over the whole field.  Azimuth is made to cover two and a half
    bar travels, so it wraps twice.  np.gradient on the degrees reports a
    gradient the size of the whole travel at each wrap, which rotates the
    gradient by 180 deg and flips the sine -- a stripe.
    """
    rows, cols = np.mgrid[0:H, 0:W]
    el = rows / (H - 1) * TRAVEL_EL * 0.4
    az = np.mod(cols / (W - 1) * 2.5 * TRAVEL_AZ, TRAVEL_AZ)
    mask = np.ones((H, W), dtype=bool)
    per = {"elevation": TRAVEL_EL, "azimuth": TRAVEL_AZ}

    old = ios.visual_field_sign(el, az, mask, um_per_pixel=UM, sigma=2.0,
                                snr_min=0.0)
    new = ios.visual_field_sign(el, az, mask, um_per_pixel=UM, sigma=2.0,
                                snr_min=0.0, period=per)

    truth = -1          # d(el)/drow > 0, d(az)/dcol > 0  ->  sin(-pi/2)
    for name, sm in (("linear  ", old), ("phasor  ", new)):
        agree = float(np.nanmean(np.sign(sm.vfs[sm.mask]) == truth))
        area, sign, n = _largest(sm)
        print(f"  {name} correct sign on {100*agree:5.1f}% of px, "
              f"{n} patch(es), largest {area:.2f} mm2 (sign {sign:+d})")
        sm._agree = agree

    assert old._agree < 0.9, ("the legacy path should fail this", old._agree)
    assert new._agree > 0.98, new._agree
    # and the wrap must not fragment the field
    assert _largest(new)[2] <= 2, _largest(new)
    print(f"  wrap lines cost the legacy path "
          f"{100*(new._agree - old._agree):.0f} points of correct sign")
    print("  PASS  test_wrap_does_not_flip_the_sign")


def test_weighting_stops_the_surround_leaking_in():
    """
    The real geometry: a responsive region of about V1's size inside a much
    larger mask of unresponsive pixels.

    The surround has no retinotopy, only noise, and the existing SNR gate
    removes it from the final map -- but NOT from the smoothing that precedes
    the gradients.  So the gradients at genuinely responsive pixels are
    computed from a neighbourhood that is partly noise, and the damage reaches
    a kernel's width inside the border.  Weighting the circular mean by SNR
    stops it, because a noise pixel's vote is scaled by its own small
    amplitude.
    """
    rng = np.random.default_rng(0)
    resp = _disk(30, 40, 12)
    rows, cols = np.mgrid[0:H, 0:W]

    el_true = (rows - 30) / 12.0 * 20.0
    az_true = (cols - 40) / 12.0 * 25.0
    el = np.where(resp, el_true, rng.uniform(0, TRAVEL_EL, (H, W)))
    az = np.where(resp, az_true, rng.uniform(0, TRAVEL_AZ, (H, W)))
    # the responsive pixels are not perfect either
    el = el + np.where(resp, rng.normal(0, 1.5, (H, W)), 0.0)
    az = az + np.where(resp, rng.normal(0, 2.3, (H, W)), 0.0)

    snr = np.where(resp, 5.9, 1.2)
    mask = np.ones((H, W), dtype=bool)
    per = {"elevation": TRAVEL_EL, "azimuth": TRAVEL_AZ}
    truth = -1          # same orientation as above

    out = {}
    for name, kw in (("linear  ", {}),
                     ("phasor  ", {"period": per}),
                     ("weighted", {"period": per, "weight": snr})):
        sm = ios.visual_field_sign(el, az, mask, um_per_pixel=UM, sigma=3.0,
                                   snr=snr, snr_min=3.0, **kw)
        inside = resp & sm.mask
        agree = float(np.nanmean(np.sign(sm.vfs[inside]) == truth))
        area, sign, n = _largest(sm)
        out[name.strip()] = (agree, area, n)
        print(f"  {name} correct sign on {100*agree:5.1f}% of the responsive "
              f"region, largest patch {area:.2f} mm2, {n} patch(es)")

    true_area = resp.sum() * (UM / 1000.0) ** 2
    print(f"  true responsive area {true_area:.2f} mm2")

    a_lin = out["linear"][0]
    a_w = out["weighted"][0]
    print(f"  weighting recovers {100*(a_w - a_lin):+.0f} points of correct "
          f"sign inside the responsive region")
    assert a_w > a_lin, (a_lin, a_w)
    assert a_w > 0.9, a_w
    # the weighted patch must be about the right size, not a sliver
    assert out["weighted"][1] > 0.5 * true_area, (out["weighted"][1], true_area)
    print("  PASS  test_weighting_stops_the_surround_leaking_in")


def test_coherence_marks_the_unresponsive_surround():
    """
    The modulus of the smoothed phasor should separate signal from noise on its
    own, with no SNR input -- it is a measurement of whether neighbouring
    pixels agree about position.
    """
    rng = np.random.default_rng(1)
    resp = _disk(30, 40, 14)
    rows, cols = np.mgrid[0:H, 0:W]
    el = np.where(resp, (rows - 30) / 14.0 * 20.0,
                  rng.uniform(0, TRAVEL_EL, (H, W)))
    az = np.where(resp, (cols - 40) / 14.0 * 25.0,
                  rng.uniform(0, TRAVEL_AZ, (H, W)))
    mask = np.ones((H, W), dtype=bool)
    sm = ios.visual_field_sign(
        el, az, mask, um_per_pixel=UM, sigma=3.0, snr_min=0.0,
        period={"elevation": TRAVEL_EL, "azimuth": TRAVEL_AZ})
    ci = float(np.nanmedian(sm.coherence[resp]))
    co = float(np.nanmedian(sm.coherence[~resp]))
    print(f"  coherence: {ci:.3f} inside the responsive region, "
          f"{co:.3f} outside  ({ci/max(co,1e-9):.1f}x)")
    assert sm.coherence is not None
    assert ci > 0.8, ci
    assert co < 0.5, co
    print("  PASS  test_coherence_marks_the_unresponsive_surround")


def test_flatness_gate_costs_area_without_buying_rejection():
    """
    Pin down the gate settings block3_sign.py uses.

    The flatness gate (min_grad_pct) exists to reject maps with no direction.
    Once the coherence gate is present it is no longer what rejects noise, and
    it then costs real area: this measures both sides of that trade on the same
    pair of fields -- one with a responsive region of known size, one pure
    noise -- so the choice is a measurement and not a preference.
    """
    rng = np.random.default_rng(5)
    resp = _disk(30, 40, 14)
    rows, cols = np.mgrid[0:H, 0:W]
    el = np.where(resp, (rows - 30) / 14.0 * 22.0,
                  rng.uniform(0, TRAVEL_EL, (H, W)))
    az = np.where(resp, (cols - 40) / 14.0 * 28.0,
                  rng.uniform(0, TRAVEL_AZ, (H, W)))
    snr = np.where(resp, 5.9, 1.2)
    # a pure-noise field whose SNR is high enough to pass the SNR gate, so the
    # rejection has to come from somewhere else
    el_n = rng.uniform(0, TRAVEL_EL, (H, W))
    az_n = rng.uniform(0, TRAVEL_AZ, (H, W))
    snr_n = np.abs(rng.normal(1.0, 0.3, (H, W))) * 3.5

    mask = np.ones((H, W), dtype=bool)
    per = {"elevation": TRAVEL_EL, "azimuth": TRAVEL_AZ}
    true_area = resp.sum() * (UM / 1000.0) ** 2

    def run(el_, az_, snr_, gp):
        sm = ios.visual_field_sign(
            el_, az_, mask, um_per_pixel=UM, sigma=3.0, snr=snr_, snr_min=3.0,
            period=per, weight=snr_, min_coherence=0.3, min_grad_pct=gp)
        return _largest(sm)[0]

    print(f"  true responsive area {true_area:.2f} mm2")
    print(f"  {'min_grad_pct':>13}{'signal mm2':>12}{'recovered':>11}"
          f"{'noise mm2':>11}")
    rows_out = {}
    for gp in (25.0, 15.0, 10.0, 5.0):
        s = run(el, az, snr, gp)
        n = run(el_n, az_n, snr_n, gp)
        rows_out[gp] = (s, n)
        print(f"  {gp:13.0f}{s:12.2f}{s/true_area:10.0%}{n:11.2f}")

    # loosening the gate must buy area ...
    assert rows_out[10.0][0] > rows_out[25.0][0], rows_out
    assert rows_out[10.0][0] > 0.7 * true_area, rows_out[10.0]
    # ... and must NOT let a pure-noise field produce a cortex-sized patch
    for gp, (s, n) in rows_out.items():
        assert n < 0.3, (gp, n)
    print("  PASS  test_flatness_gate_costs_area_without_buying_rejection")


if __name__ == "__main__":
    print("Weighted / wrap-safe sign-map tests")
    test_wrap_does_not_flip_the_sign()
    test_weighting_stops_the_surround_leaking_in()
    test_coherence_marks_the_unresponsive_surround()
    test_flatness_gate_costs_area_without_buying_rejection()
    print("all passed")
