"""
Tests for load_stim_log under DisplaySequence.display_iter > 1 and the
wrapped sweep.  The failure these guard against is silent: with display_iter
= 3 every block comes out three times too long and block 1 swallows the
whole recording, while block0_verify's "last block ends at post_stim" check
still passes.
"""
import os, pickle, tempfile
import numpy as np
import ioi_core as ioi

REFRESH, CAM_FPS = 60.0, 100.0
FPC = {"B2U": 640, "U2B": 640, "L2R": 990, "R2L": 990}   # wrapped sweep
N_CYC = 15


def make_pickle(n_reps, wrap=True, center=(0., 0.)):
    h, w = 216, 384
    y = np.linspace(28.41, -28.41, h)[:, None] * np.ones((1, w)) + center[0]
    x = np.linspace(-44.05, 44.05, w)[None, :] * np.ones((h, 1)) + center[1]

    frames, sweep_table = [], []
    for lab in ("B2U", "U2B", "L2R", "R2L"):
        n = FPC[lab]
        lo, hi = (-28.41, 28.41) if lab in ("B2U", "U2B") else (-44.05, 44.05)
        span = hi - lo
        steps = lo + np.arange(n) * (span / n if wrap else (span + 20.) / n)
        if lab in ("U2B", "R2L"):
            steps = steps[::-1]
        # KSstimAllDir rewrites the orientation field to the direction label
        sweep_table += [(lab, float(s), float(s + 20.)) for s in steps]
        frames += [(1, 1, i, 1, lab) for i in range(n)] * N_CYC

    d = {"stimulation": {"frames": frames, "sweep_table": sweep_table,
                         "direction": ["B2U", "U2B", "L2R", "R2L"],
                         "sweep_width": 20.0, "step_width": 0.089},
         "monitor": {"deg_coord_x": x, "deg_coord_y": y,
                     "refresh_rate": REFRESH},
         "presentation": {"display_iter": n_reps, "keep_display": True,
                          "displayed_frames": [0] * (len(frames) * n_reps)}}
    fh, p = tempfile.mkstemp(suffix=".pkl")
    os.close(fh)
    with open(p, "wb") as f:
        pickle.dump(d, f)
    return p, len(frames)


def make_ft(n_pass, n_reps, pre=600):
    n_stim = n_pass * n_reps
    n_cam = pre + int(round(n_stim * CAM_FPS / REFRESH)) + 400
    ft = ioi.FrameTimes(t=np.arange(n_cam) / CAM_FPS, n_frames=n_cam,
                        pre_stim=pre,
                        post_stim=pre + int(round(n_stim * CAM_FPS / REFRESH)),
                        removed=0, datenum0=0.0)
    return ft


def check(n_reps):
    p, n_pass = make_pickle(n_reps)
    ft = make_ft(n_pass, n_reps)
    seen = []
    for rep in range(n_reps):
        log = ioi.load_stim_log(path=p, ft=ft, rep=rep)
        assert log.n_reps == n_reps, (log.n_reps, n_reps)
        assert len(log.blocks) == 4
        assert abs(log.effective_refresh - REFRESH) < 0.01, log.effective_refresh
        for b in log.blocks:
            assert b.n_cycles == N_CYC, (b.label, b.n_cycles)
            assert b.frames_per_cycle == FPC[b.label]
            expect = FPC[b.label] * N_CYC / REFRESH
            assert abs(b.duration - expect) < 0.05, (b.label, b.duration, expect)
            assert abs(b.f_stim - N_CYC / expect) < 1e-3
            seen.append((b.label, b.cam0, b.cam1))
    # the 4 x n_reps blocks must tile the stimulus window with no gaps
    seen.sort(key=lambda r: r[1])
    assert seen[0][1] == ft.pre_stim, seen[0]
    assert seen[-1][2] == ft.post_stim, seen[-1]
    for a, c in zip(seen, seen[1:]):
        assert a[2] == c[1], (a, c)
    covered = sum(c - b for _, b, c in seen)
    print(f"  display_iter={n_reps}: {len(seen)} blocks tile "
          f"{covered} camera frames ({covered/CAM_FPS:.1f} s), "
          f"effective refresh {log.effective_refresh:.3f} Hz")
    os.unlink(p)


def test_display_iter():
    for n in (1, 3):
        check(n)
    print("  PASS  test_display_iter")


def test_rep_out_of_range():
    p, n_pass = make_pickle(3)
    ft = make_ft(n_pass, 3)
    try:
        ioi.load_stim_log(path=p, ft=ft, rep=3)
    except IndexError as e:
        print(f"  rep=3 of 3 rejected: {e}")
    else:
        raise AssertionError("rep out of range was not rejected")
    os.unlink(p)
    print("  PASS  test_rep_out_of_range")


def test_interrupted_display_is_caught():
    p, n_pass = make_pickle(3)
    with open(p, "rb") as f:
        d = pickle.load(f)
    d["presentation"]["displayed_frames"] = [0] * (2 * n_pass + 137)
    d["presentation"]["keep_display"] = False
    with open(p, "wb") as f:
        pickle.dump(d, f)
    try:
        ioi.load_stim_log(path=p, ft=make_ft(n_pass, 3))
    except ValueError as e:
        print(f"  partial display rejected: {str(e)[:70]}...")
    else:
        raise AssertionError("interrupted display was not caught")
    os.unlink(p)
    print("  PASS  test_interrupted_display_is_caught")


def test_wrapped_sweep_is_not_aliased():
    for wrap, want in ((False, "ALIASES"), (True, "is safe")):
        p, n_pass = make_pickle(1, wrap=wrap)
        log = ioi.load_stim_log(path=p, ft=make_ft(n_pass, 1))
        for line in log.travel_vs_visible.split("\n"):
            assert want in line, line
            print(f"  {'wrapped' if wrap else 'current'}: {line}")
        os.unlink(p)
    print("  PASS  test_wrapped_sweep_is_not_aliased")


def test_visible_half_with_offset_monitor():
    p, n_pass = make_pickle(1, center=(0., 60.))
    log = ioi.load_stim_log(path=p, ft=make_ft(n_pass, 1))
    got = log.visible_half["azimuth"]
    assert abs(got - 44.05) < 0.01, got
    print(f"  monitor centred on azimuth 60 deg: visible_half {got:.2f} "
          f"(max|coord| would give {104.05:.2f})")
    print("  PASS  test_visible_half_with_offset_monitor")


if __name__ == "__main__":
    print("Stimulus log tests")
    test_display_iter()
    test_rep_out_of_range()
    test_interrupted_display_is_caught()
    test_wrapped_sweep_is_not_aliased()
    test_visible_half_with_offset_monitor()
    print("all passed")
