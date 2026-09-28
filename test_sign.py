"""
Tests for the joint delay estimate and the visual field sign map.

The ones that matter:
  test_joint_delay_beats_the_noisy_axis -- the slower axis must actually get
      better, which is the whole point of estimating delay jointly.
  test_sign_flips_between_mirrored_areas -- a synthetic V1 plus a mirror-image
      neighbour must come out as two patches of opposite sign.
  test_gating_removes_degenerate_pixels -- where gradients are parallel, the
      gate must refuse to assign a sign rather than assigning a random one.
"""

import numpy as np

import ioi_core as ioi
import ioi_maps as iom
import ioi_sign as ios

TWO_PI = 2 * np.pi


class S:
    def __init__(self, label, c, noise, mask):
        self.label, self.component, self.noise = label, c, noise
        self.amplitude = np.abs(c)
        self.phase = np.angle(c)
        self.snr = self.amplitude / noise
        self.mask = mask


def scene(h=120, w=150, T_el=14.40, T_az=22.37, amp=3e-4,
          noise_el=5e-5, noise_az=5e-5, delay_amp=2.0, seed=0,
          mirror=False):
    """
    Ground truth: elevation ramps down the image, azimuth across it, plus a
    smooth delay gradient.  With mirror=True the right half has a reversed
    azimuth map, so the sign must flip there.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    el_true = -22.0 + 44.0 * (yy / h)
    az_true = -35.0 + 70.0 * (xx / w)
    if mirror:
        half = w // 2
        ref = az_true[:, half - 1][:, None]
        az_true[:, half:] = 2.0 * ref - az_true[:, half:]
    delay = delay_amp * (0.7 + 0.3 * xx / w)

    travel = {"elevation": 76.87, "azimuth": 108.10}
    p0 = {"elevation": -38.435, "azimuth": -54.05}
    specs, blocks = {}, {}
    for axis, f, r, T, nz in (("elevation", "B2U", "U2B", T_el, noise_el),
                              ("azimuth", "L2R", "R2L", T_az, noise_az)):
        true = el_true if axis == "elevation" else az_true
        u = (true - p0[axis]) / travel[axis]
        for lab, uu in ((f, u), (r, 1 - u)):
            phi = np.mod(delay / T + uu, 1.0)
            c = amp * np.exp(-1j * TWO_PI * phi)
            c = c + nz * (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w)))
            specs[lab] = S(lab, c, np.full((h, w), nz), np.ones((h, w), bool))
            blocks[lab] = ioi.StimBlock(
                label=lab, axis=axis, k0=0, k1=0, frames_per_cycle=1,
                n_cycles=15, cam0=0, cam1=1, t0=0.0, t1=T * 15,
                bar_deg_first=p0[axis] if lab == f else -p0[axis],
                bar_deg_last=-p0[axis] if lab == f else p0[axis],
                visible_half=28.41 if axis == "elevation" else 44.05)
    return specs, blocks, el_true, az_true, delay


def test_joint_delay_beats_the_noisy_axis():
    """Azimuth is slower, so its own delay is noisier; jointly it improves."""
    specs, blocks, el_t, az_t, delay_t = scene(noise_az=9e-5, seed=1)

    sep = {}
    for axis, f, r in (("elevation", "B2U", "U2B"), ("azimuth", "L2R", "R2L")):
        sep[axis] = iom.combine_axis(specs[f], blocks[f], specs[r], blocks[r],
                                     phase_sigma=2.0, delay_sigma=4.0,
                                     snr_min=0.0)
    joint, info = iom.combine_axes_joint(specs, blocks, phase_sigma=2.0,
                                         delay_sigma=4.0)
    m = np.ones_like(delay_t, dtype=bool)
    m[:6] = m[-6:] = False
    m[:, :6] = m[:, -6:] = False

    travel = {"elevation": 76.87, "azimuth": 108.10}
    print("              delay RMS (s)        single-direction pos RMS (deg)")
    for axis, truth in (("elevation", el_t), ("azimuth", az_t)):
        ds = np.sqrt(np.nanmean((sep[axis].delay[m] - delay_t[m]) ** 2))
        dj = np.sqrt(np.nanmean((joint[axis].delay[m] - delay_t[m]) ** 2))
        fs = np.sqrt(np.nanmean((sep[axis].pos_fwd[m] - truth[m]) ** 2))
        fj = np.sqrt(np.nanmean((joint[axis].pos_fwd[m] - truth[m]) ** 2))
        print(f"  {axis:10s} sep {ds:7.4f} joint {dj:7.4f}   |   "
              f"sep {fs:6.3f} joint {fj:6.3f}")
        if axis == "azimuth":
            assert dj < ds, (dj, ds)
    print("  note: single-direction position is NOT expected to improve.  A")
    print("        per-axis delay is built from that axis' own phases, so its")
    print("        error partly cancels in pos_fwd; an independent delay does")
    print("        not.  What the joint delay buys is branch correctness --")
    print("        see the next test.")
    print("  PASS  test_joint_delay_beats_the_noisy_axis")


def test_delay_fixes_branch_errors_not_small_offsets():
    """
    The combined position is algebraically independent of tau within a branch;
    tau decides WHICH branch.  So the delay's job is to stop pixels wrapping by
    half the bar travel, not to shave degrees off.  Verify both halves of that.
    """
    specs, blocks, el_t, az_t, _ = scene(noise_az=3.0e-4, seed=11)
    sep = {}
    for axis, f, r in (("elevation", "B2U", "U2B"), ("azimuth", "L2R", "R2L")):
        sep[axis] = iom.combine_axis(specs[f], blocks[f], specs[r], blocks[r],
                                     phase_sigma=2.0, delay_sigma=4.0,
                                     snr_min=0.0)
    joint, _ = iom.combine_axes_joint(specs, blocks, phase_sigma=2.0,
                                      delay_sigma=4.0)
    m = np.ones(el_t.shape, bool)
    m[:6] = m[-6:] = False
    m[:, :6] = m[:, -6:] = False
    travel = 108.10
    bs = np.mean(np.abs(sep["azimuth"].position[m] - az_t[m]) > travel / 4) * 100
    bj = np.mean(np.abs(joint["azimuth"].position[m] - az_t[m]) > travel / 4) * 100
    print(f"  azimuth pixels wrapped by ~half the travel: "
          f"separate delay {bs:.1f}%  ->  joint delay {bj:.1f}%")
    assert bj <= bs, (bj, bs)
    print("  PASS  test_delay_fixes_branch_errors_not_small_offsets")


def test_joint_delay_is_one_map():
    specs, blocks, *_ = scene(seed=2)
    joint, info = iom.combine_axes_joint(specs, blocks)
    assert np.allclose(joint["elevation"].delay, joint["azimuth"].delay,
                       equal_nan=True)
    r = info["delay_resid_elevation"]
    print(f"  one delay map shared by both axes; elevation residual "
          f"median {np.nanmedian(np.abs(r)):.4f} s")
    print("  PASS  test_joint_delay_is_one_map")


def test_sign_flips_between_mirrored_areas():
    specs, blocks, el_t, az_t, _ = scene(mirror=True, seed=3, noise_el=2e-5,
                                         noise_az=2e-5)
    joint, _ = iom.combine_axes_joint(specs, blocks)
    m = np.ones(el_t.shape, bool)
    m[:8] = m[-8:] = False
    m[:, :8] = m[:, -8:] = False
    sm = ios.visual_field_sign(joint["elevation"].position,
                               joint["azimuth"].position, m,
                               um_per_pixel=38.44, sigma=3.0)
    w = el_t.shape[1]
    left = m.copy(); left[:, w//2 - 10:] = False
    right = m.copy(); right[:, :w//2 + 10] = False
    sl = np.nanmedian(sm.vfs[left]); sr = np.nanmedian(sm.vfs[right])
    print(f"  median VFS left half {sl:+.2f}, right (mirrored) half {sr:+.2f}")
    assert sl * sr < 0, (sl, sr)
    assert abs(sl) > 0.5 and abs(sr) > 0.5
    ps = ios.patches(sm, min_area_mm2=0.1)
    print(f"  patches found: {len(ps)}; two largest "
          f"{[f'{p.area_mm2:.2f} mm2 sign {p.sign:+d}' for p in ps[:2]]}")
    assert len(ps) >= 2 and ps[0].sign * ps[1].sign < 0
    print("  PASS  test_sign_flips_between_mirrored_areas")


def test_gating_removes_degenerate_pixels():
    """Where azimuth and elevation ramp in the SAME direction, refuse a sign."""
    h, w = 100, 120
    yy, xx = np.mgrid[0:h, 0:w]
    el = -20 + 40 * yy / h
    az = -30 + 60 * yy / h          # deliberately parallel to elevation
    m = np.ones((h, w), bool)
    m[:6] = m[-6:] = False
    m[:, :6] = m[:, -6:] = False
    rng = np.random.default_rng(0)
    el = el + rng.normal(0, 0.3, el.shape)
    az = az + rng.normal(0, 0.3, az.shape)
    sm = ios.visual_field_sign(el, az, m, um_per_pixel=38.44, sigma=3.0)
    frac = sm.reliable.sum() / m.sum()
    print(f"  parallel-gradient scene: only {100*frac:.1f}% of pixels pass "
          f"the gate (should be near 0)")
    assert frac < 0.15, frac
    print("  PASS  test_gating_removes_degenerate_pixels")


def test_align_phase_recovers_a_constant_rotation():
    rng = np.random.default_rng(9)
    n = 3000
    sig = 2e-4 * np.exp(1j * np.linspace(0, 8, n))
    rot = np.deg2rad(57.0)
    a = sig + 2e-5*(rng.normal(size=n)+1j*rng.normal(size=n))
    b = sig*np.exp(1j*rot) + 2e-5*(rng.normal(size=n)+1j*rng.normal(size=n))
    m = np.ones(n, bool)
    (aa, bb), ang, coh = ios.align_phase([a, b], m)
    print(f"  true rotation +57.0 deg -> estimated {ang[1]:+.1f} deg, "
          f"coherence {coh[1]:.3f}")
    assert abs(ang[1] - 57.0) < 2.0
    assert coh[1] > 0.95
    naive = np.abs(0.5*(a+b)).mean()
    fixed = np.abs(0.5*(aa+bb)).mean()
    print(f"  mean |average|: unaligned {naive:.3e}, aligned {fixed:.3e} "
          f"({fixed/naive:.2f}x)")
    assert fixed > naive
    print("  PASS  test_align_phase_recovers_a_constant_rotation")


def test_average_components_adds_coherently():
    rng = np.random.default_rng(4)
    sig = 2e-4 * np.exp(1j * np.linspace(0, 6, 4000))
    noise = 4e-4
    reps = [sig + noise * (rng.normal(size=4000) + 1j*rng.normal(size=4000))
            for _ in range(4)]
    one = np.abs(reps[0] - sig).std()
    avg = ios.average_components(reps, mask=np.ones(4000, bool))[0]
    four = np.abs(avg - sig).std()
    print(f"  noise sd: 1 repeat {one:.2e}, 4 averaged {four:.2e}  "
          f"(ratio {one/four:.2f}, expected ~2.0)")
    assert 1.6 < one / four < 2.5
    print("  PASS  test_average_components_adds_coherently")


def test_shift_estimator():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(135, 160))
    from scipy.ndimage import gaussian_filter
    base = gaussian_filter(base, 3)
    for truth in ((0, 0), (3, -4), (-6, 2)):
        moved = np.roll(np.roll(base, truth[0], axis=0), truth[1], axis=1)
        got, r = ios.estimate_shift(base, moved)
        want = (-truth[0], -truth[1])   # the correction, not the displacement
        print(f"  displaced by {truth} -> correction {got} "
              f"(expected {want}, r={r:.3f})")
        assert got == want, (got, want)
    print("  PASS  test_shift_estimator")


if __name__ == "__main__":
    print("Joint-delay and sign-map tests")
    test_joint_delay_beats_the_noisy_axis()
    test_delay_fixes_branch_errors_not_small_offsets()
    test_joint_delay_is_one_map()
    test_sign_flips_between_mirrored_areas()
    test_gating_removes_degenerate_pixels()
    test_align_phase_recovers_a_constant_rotation()
    test_average_components_adds_coherently()
    test_shift_estimator()
    print("all passed")
