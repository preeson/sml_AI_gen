"""
Tests for the Block 2c verdict rule, using the per-block z-scores actually
produced by every session in this project.

The rule has to do two things at once: not resurrect the sessions that were
genuinely empty, and not miss a response that is spread evenly across the
four directions instead of concentrated in one of them.
"""
import numpy as np


def verdict(block_z, other_tests=()):
    """Mirror of the rule in block2c_control.main()."""
    zb = [z for z in block_z if np.isfinite(z)]
    z_comb = float(np.mean(zb) * np.sqrt(len(zb))) if zb else np.nan
    consistent = bool(zb) and all(z > 0 for z in zb)
    best = max([t for t in list(block_z) + list(other_tests)
                if np.isfinite(t)], default=0.0)
    if z_comb > 3 and consistent:
        return "signal", z_comb, consistent
    if (z_comb > 2 and consistent) or best > 3:
        return "marginal", z_comb, consistent
    return "none", z_comb, consistent


# label: (per-block z for B2U/U2B/L2R/R2L, extra tests, what it should say)
SESSIONS = {
    "G1 rep0":            ([-0.03, -0.36, +0.68, +0.84], [+0.45, -0.10], "none"),
    "G1 rep1":            ([+0.69, +1.43, -0.30, -1.38], [-1.41, +0.52], "none"),
    "G1 rep2":            ([+5.38, +1.21, +2.77, -1.66], [+0.44, -0.78], "marginal"),
    "G3 Oct rep0":        ([+1.49, +0.45, -0.20, +0.75], [+1.77, +0.88], "none"),
    "G3 Oct rep1":        ([-0.30, -0.79, +1.58, -0.93], [-0.08, +0.60], "none"),
    "G3 Oct rep2":        ([-0.79, +0.57, -0.39, -0.66], [+0.06, -0.07], "none"),
    "G3 29/10 unwrapped": ([+3.41, +0.99, +4.47, +10.55], [+0.45, +3.11], "signal"),
    "G3 29/10 wrapped":   ([+1.52, +3.58, +1.38, +2.95], [+2.46, +1.27], "signal"),
}


def test_every_session_gets_the_right_verdict():
    print(f"  {'session':<20}{'blocks':<30}{'comb z':>8}{'same sign':>11}"
          f"{'verdict':>11}")
    bad = []
    for name, (zb, extra, want) in SESSIONS.items():
        got, zc, cons = verdict(zb, extra)
        flag = "" if got == want else f"  <-- expected {want}"
        if got != want:
            bad.append(name)
        print(f"  {name:<20}{str([round(z, 2) for z in zb]):<30}{zc:>8.2f}"
              f"{('yes' if cons else 'no'):>11}{got:>11}{flag}")
    assert not bad, bad
    print("  PASS  test_every_session_gets_the_right_verdict")


def test_one_sided():
    """A residual SNR far BELOW the controls is not evidence of signal."""
    got, zc, _ = verdict([-4.5, -3.2, -5.1, -2.8])
    print(f"  all four strongly negative -> comb z {zc:+.2f}, verdict '{got}'")
    assert got == "none", got
    print("  PASS  test_one_sided")


def test_consistency_is_required():
    """Four blocks averaging well above 3 but with one negative stay marginal."""
    got, zc, cons = verdict([+8.0, +4.0, +3.0, -0.5])
    print(f"  [+8, +4, +3, -0.5] -> comb z {zc:+.2f}, same sign {cons}, "
          f"verdict '{got}'")
    assert got == "marginal" and not cons
    got2, zc2, cons2 = verdict([+2.2, +2.0, +1.9, +2.1])
    print(f"  [+2.2, +2.0, +1.9, +2.1] -> comb z {zc2:+.2f}, same sign "
          f"{cons2}, verdict '{got2}'  (smaller but unanimous)")
    assert got2 == "signal" and cons2
    print("  PASS  test_consistency_is_required")


if __name__ == "__main__":
    print("Block 2c verdict tests")
    test_every_session_gets_the_right_verdict()
    test_one_sided()
    test_consistency_is_required()
    print("all passed")
