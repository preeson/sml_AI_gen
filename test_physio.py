"""
Tests for block0b_physio: a synthetic recording with a known heartbeat and
respiration must be detected, and one with neither must be rejected.  The
failure this guards against is a check that says "yes" to pure noise.
"""
import os, shutil, tempfile
import numpy as np
import ioi_core as ioi
import block0b_physio as ph

FS, N, H, W = 100.0, 6000, ioi.FRAME_H, ioi.FRAME_W


def make_session(d, heart=0.0020, resp=0.0008, f_heart=8.3, f_resp=2.1,
                 base=28000.0, offset=1540.0, frac_noise=0.0093, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    gain = (base - offset)
    sig = (heart * np.sin(2 * np.pi * f_heart * t)
           + resp * np.sin(2 * np.pi * f_resp * t + 0.5))
    path = os.path.join(d, "Frames_1_640_540_uint16_0001.dat")
    with open(path, "wb") as fh:
        for i in range(N):
            fr = base + gain * sig[i] + rng.normal(0, frac_noise * gain, (H, W))
            fr[:, :60] = 1600.0                      # a dark strip, masked out
            # MATLAB writes column-major: element (row,col) at col*H + row
            fh.write(np.clip(fr, 0, 65535).astype("<u2").T.tobytes(order="C"))
    import scipy.io as sio
    sio.savemat(os.path.join(d, "frameTimes_0001.mat"),
                {"frameTimes": 737000.0 + t / 86400.0,
                 "preStim": 0, "postStim": N, "removedFrames": 0,
                 "imgSize": np.array([H, W, 1, N])})
    return path


def run(d):
    ioi.DATA_DIR = d
    ioi.DAT_PATH = os.path.join(d, "Frames_1_640_540_uint16_0001.dat")
    ioi.FRAMETIMES_PATH = os.path.join(d, "frameTimes_0001.mat")
    ft = ioi.load_frame_times()
    fs = (ft.n_frames - 1) / (ft.t[-1] - ft.t[0])
    with ioi.movie() as mov:
        t, stack = ph.collect(mov, ft, 0, N)
    m = stack.mean(axis=0)
    mask = ((m - ph.OFFSET) > ph.MIN_COUNTS) & (m < ph.SAT_LEVEL)
    dr = (stack - m[None]) / np.where(mask, m - ph.OFFSET, np.nan)[None]
    g = np.nanmean(dr[:, mask], axis=1)
    f, X = ph.spectrum(t, g, fs)
    return mask, {k: ph.peak_in(f, X, *v) for k, v in ph.BANDS.items()}


def test_detects_known_rhythms():
    d = tempfile.mkdtemp()
    try:
        make_session(d)
        mask, got = run(d)
        assert not mask[:, :6].any(), "dark strip was not masked"
        fh, ah, rh, th = got["heartbeat"]
        fr, ar, rr, tr = got["respiration"]
        print(f"  heartbeat   found {fh:.2f} Hz (true 8.30), "
              f"amp {100*ah:.4f}% (true 0.2000%), {rh:.0f}x floor")
        print(f"  respiration found {fr:.2f} Hz (true 2.10), "
              f"amp {100*ar:.4f}% (true 0.0800%), {rr:.0f}x floor")
        assert abs(fh - 8.3) < 0.1 and abs(fr - 2.1) < 0.1
        assert 0.8 < ah / 0.0020 < 1.2 and 0.8 < ar / 0.0008 < 1.2
        assert rh > th and rr > tr
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  PASS  test_detects_known_rhythms")


def test_rejects_pure_noise():
    d = tempfile.mkdtemp()
    try:
        make_session(d, heart=0.0, resp=0.0, seed=7)
        _, got = run(d)
        for k, (fp, amp, rel, thr) in got.items():
            print(f"  {k:12s} best peak {fp:.2f} Hz, {rel:.1f}x floor "
                  f"(threshold {thr:.1f}x)")
            assert rel < thr, (k, rel, thr)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  PASS  test_rejects_pure_noise")


def test_survives_dropped_frames():
    """Timestamps are non-uniform; the spectrum must still find the peak."""
    d = tempfile.mkdtemp()
    try:
        make_session(d, seed=3)
        import scipy.io as sio
        t = np.arange(N) / FS
        t[3000:] += 0.040                     # a 40 ms hiccup mid-recording
        sio.savemat(os.path.join(d, "frameTimes_0001.mat"),
                    {"frameTimes": 737000.0 + t / 86400.0, "preStim": 0,
                     "postStim": N, "removedFrames": 0,
                     "imgSize": np.array([H, W, 1, N])})
        _, got = run(d)
        fh, ah, rh, th = got["heartbeat"]
        print(f"  with a 40 ms gap: heartbeat {fh:.2f} Hz, {rh:.0f}x floor")
        assert abs(fh - 8.3) < 0.15 and rh > th
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("  PASS  test_survives_dropped_frames")



# ---------------------------------------------------------------------------
# Fundamental picking.  These run on synthetic spectra, no file I/O.
# ---------------------------------------------------------------------------

def _spec(fs, n, comps):
    t = np.arange(n) / fs
    x = sum(a * np.sin(2 * np.pi * fr * t + p) for fr, a, p in comps)
    X = np.abs(np.fft.rfft(x)) * 2.0 / n
    return np.fft.rfftfreq(n, 1.0 / fs), X


def test_fundamental_beats_argmax():
    """A pulsatile beat with a 2nd harmonic taller than its fundamental."""
    fs, n, f0 = 100.0, 6000, 5.5      # 2*f0 = 11 Hz, inside the band
    f, X = _spec(fs, n, [(f0, 1.0, 0), (2 * f0, 1.8, 0.3), (3 * f0, 0.6, 1.0)])
    lo, hi = 4.5, 14.0
    b = (f >= lo) & (f < hi)
    naive = f[b][np.argmax(X[b])]
    got, _ = ph.fundamental(f, X, lo, hi)
    print(f"  2nd harmonic dominant: argmax says {naive:.2f} Hz "
          f"({naive*60:.0f}/min), fundamental() says {got:.2f} Hz "
          f"({got*60:.0f}/min), true {f0:.2f}")
    assert abs(naive - 2 * f0) < 0.1, naive      # argmax really does fail
    assert abs(got - f0) < 0.1, got
    print("  PASS  test_fundamental_beats_argmax")


def test_subharmonic_ambiguity_is_flagged():
    """
    Peaks at f/2, f and 2f are consistent with two different rates.  The code
    must not report one of them confidently -- it must raise the flag that
    makes a human look.
    """
    fs, n, f0 = 100.0, 6000, 10.0
    f, X = _spec(fs, n, [(f0 / 2, 1.1, 0.2), (f0, 1.0, 0), (2 * f0, 0.5, 0.7)])
    got, _ = ph.fundamental(f, X, 4.5, 14.0)
    warn = ph.rate_warning(f, X, got)
    print(f"  peaks at 5/10/20 Hz -> reported {got * 60:.0f}/min, "
          f"warning: {warn!r}")
    assert warn, "ambiguous spectrum was reported without a warning"
    assert "600" in warn, warn
    print("  PASS  test_subharmonic_ambiguity_is_flagged")


def test_plain_sinusoid_unchanged():
    fs, n, f0 = 100.0, 6000, 7.3
    f, X = _spec(fs, n, [(f0, 1.0, 0)])
    got, amp = ph.fundamental(f, X, 4.5, 14.0)
    warn = ph.rate_warning(f, X, got)
    print(f"  pure tone {f0} Hz -> {got:.3f} Hz, amplitude {amp:.3f} "
          f"(true 1.0), warning {warn!r}")
    assert abs(got - f0) < 0.05 and abs(amp - 1.0) < 0.02
    assert not warn, "an unambiguous tone was flagged"

    # a realistic pulsatile beat (2nd harmonic at 50%) must NOT be flagged
    f, X = _spec(fs, n, [(f0, 1.0, 0), (2 * f0, 0.5, 0.4)])
    got, _ = ph.fundamental(f, X, 4.5, 14.0)
    assert abs(got - f0) < 0.05 and not ph.rate_warning(f, X, got)
    print(f"  pulsatile beat (2f at 50%) -> {got:.3f} Hz, not flagged")
    print("  PASS  test_plain_sinusoid_unchanged")

def test_temp_log_parser():
    """The log is hand-written: a title, a header, and ragged separators."""
    import tempfile, os
    txt = ("SST-Ai9 G3 29.10.2026 Temp log\n"
           "sec\ttemp_c\n"
           "0\t36.7\n"
           "10\t36.7\n"
           "\n"
           "# probe re-seated\n"
           "20, 36.9\n"
           "30 37.0\n")
    fh, p = tempfile.mkstemp(suffix=".txt"); os.close(fh)
    with open(p, "w") as f:
        f.write(txt)
    t, c = ph.load_temp_log(p)
    print(f"  parsed {t.size} rows from a file with a title, a header, "
          f"a blank line, a comment, a comma row and a space row")
    assert list(t) == [0, 10, 20, 30], t
    assert list(c) == [36.7, 36.7, 36.9, 37.0], c
    os.unlink(p)
    # a file with no numeric rows must fail loudly, not silently return nothing
    fh, p = tempfile.mkstemp(suffix=".txt"); os.close(fh)
    with open(p, "w") as f:
        f.write("sec\ttemp_c\nno data here\n")
    try:
        ph.load_temp_log(p)
    except ValueError as e:
        print(f"  empty log rejected: {str(e)[-30:]}")
    else:
        raise AssertionError("a log with no data was accepted")
    os.unlink(p)
    print("  PASS  test_temp_log_parser")


if __name__ == "__main__":
    print("Physiology check tests")
    test_detects_known_rhythms()
    test_rejects_pure_noise()
    test_survives_dropped_frames()
    test_fundamental_beats_argmax()
    test_subharmonic_ambiguity_is_flagged()
    test_plain_sinusoid_unchanged()
    test_temp_log_parser()
    print("all passed")
