"""
Tests for Block 0.  Test A proves the pixel-layout logic on a tiny synthetic
file written the way MATLAB writes one.  Test B runs the real timing code on
the real metadata (no .dat needed).

Windows note: an open np.memmap locks the file, so every test here releases
its handle via ioi.close_movie() / ioi.movie() before deleting anything.
"""

import os
import shutil
import tempfile

import numpy as np

import ioi_core as ioi


def _scratch_dat(truth: np.ndarray) -> tuple[str, str]:
    """Write `truth` (N, H, W) the way MATLAB writes a column-major file."""
    d = tempfile.mkdtemp(prefix="ioi_test_")
    path = os.path.join(d, "synth.dat")
    with open(path, "wb") as f:
        # MATLAB puts element (r, c) at offset c*H + r within each frame.
        # transpose(0, 2, 1) then .tobytes() reproduces exactly that order.
        f.write(np.ascontiguousarray(truth.transpose(0, 2, 1)).tobytes())
    return d, path


def test_layout():
    """A MATLAB (H, W, 1, N) column-major write must read back exactly."""
    H, W, N = 7, 5, 4
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 65535, size=(N, H, W), dtype=np.uint16)
    tmpdir, path = _scratch_dat(truth)

    H0, W0, B0 = ioi.FRAME_H, ioi.FRAME_W, ioi.BYTES_PER_FRAME
    try:
        ioi.FRAME_H, ioi.FRAME_W = H, W
        ioi.BYTES_PER_FRAME = H * W * ioi.DTYPE.itemsize

        assert ioi.frame_count_on_disk(path) == N, "frame count wrong"

        with ioi.movie(path) as mov:
            assert mov.shape == (N, H, W), f"shape {mov.shape}"
            got = np.array(mov)  # copy out before the handle closes
        assert np.array_equal(got, truth), "pixel values misplaced"

        # the naive C-order read must NOT match, or the test is vacuous
        naive = np.fromfile(path, dtype=ioi.DTYPE).reshape(N, H, W)
        assert not np.array_equal(naive, truth), "test is not discriminating"
    finally:
        ioi.FRAME_H, ioi.FRAME_W, ioi.BYTES_PER_FRAME = H0, W0, B0
        shutil.rmtree(tmpdir, ignore_errors=True)
    print("  PASS  test_layout: Fortran-order round-trip exact, "
          "C-order read correctly rejected")


def test_handle_released():
    """close_movie() must actually let Windows delete the file."""
    H, W, N = 4, 3, 2
    truth = np.zeros((N, H, W), dtype=np.uint16)
    tmpdir, path = _scratch_dat(truth)

    H0, W0, B0 = ioi.FRAME_H, ioi.FRAME_W, ioi.BYTES_PER_FRAME
    try:
        ioi.FRAME_H, ioi.FRAME_W = H, W
        ioi.BYTES_PER_FRAME = H * W * ioi.DTYPE.itemsize

        mov = ioi.open_movie(path)
        assert ioi.close_movie(mov) is True, "close_movie did not find the mmap"
        os.remove(path)  # this is the line that failed on Windows before
        assert not os.path.exists(path)
    finally:
        ioi.FRAME_H, ioi.FRAME_W, ioi.BYTES_PER_FRAME = H0, W0, B0
        shutil.rmtree(tmpdir, ignore_errors=True)
    print("  PASS  test_handle_released: file deletable after close_movie()")


def test_short_file_rejected():
    d = tempfile.mkdtemp(prefix="ioi_test_")
    path = os.path.join(d, "trunc.dat")
    with open(path, "wb") as f:
        f.write(b"\x00" * (ioi.BYTES_PER_FRAME + 17))
    try:
        ioi.frame_count_on_disk(path)
    except ValueError as e:
        print(f"  PASS  test_short_file_rejected: {str(e)[-48:]}")
    else:
        raise AssertionError("truncated file was accepted")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_timing():
    """Blocks must tile the stimulus window exactly, with 15 cycles each."""
    ft = ioi.load_frame_times()
    log = ioi.load_stim_log(ft=ft)

    assert len(log.blocks) == 4, len(log.blocks)
    assert log.blocks[0].cam0 == ft.pre_stim
    assert log.blocks[-1].cam1 == ft.post_stim
    for a, c in zip(log.blocks, log.blocks[1:]):
        assert a.cam1 == c.cam0, f"{a.label}->{c.label}: {a.cam1} != {c.cam0}"

    covered = sum(b.n_cam_frames for b in log.blocks)
    assert covered == ft.post_stim - ft.pre_stim, covered

    for b in log.blocks:
        assert b.n_cycles == 15, (b.label, b.n_cycles)
        assert b.frames_per_cycle * b.n_cycles == b.k1 - b.k0 + 1

    # elevation and azimuth must differ, and the ratio must match the
    # frames-per-cycle ratio to within rounding
    el = log.by_label("B2U")
    az = log.by_label("L2R")
    ratio_f = el.f_stim / az.f_stim
    ratio_n = az.frames_per_cycle / el.frames_per_cycle
    assert abs(ratio_f - ratio_n) < 1e-3, (ratio_f, ratio_n)

    print(f"  PASS  test_timing: 4 blocks tile {covered} camera frames, "
          f"15 cycles each")
    print(f"        elevation F = {el.f_stim:.6f} Hz, "
          f"azimuth F = {az.f_stim:.6f} Hz, ratio {ratio_f:.4f} "
          f"(expected {ratio_n:.4f})")
    print(f"        effective display refresh = {log.effective_refresh:.4f} Hz")


def test_analog():
    """Analog file must decode to 5 channels covering the camera window."""
    if not os.path.exists(ioi.ANALOG_PATH):
        print("  SKIP  test_analog: no Analog_1.dat in DATA_DIR")
        return
    ft = ioi.load_frame_times()
    a = ioi.load_analog(ft=ft)
    ns, nch = a.data.shape
    assert nch == 5, nch
    # the DAQ must start before, and end near, the camera recording
    assert a.t_offset < 0, a.t_offset
    assert a.t[-1] > 0.99 * ft.t[-1], (a.t[-1], ft.t[-1])

    # ch1 is the camera exposure trigger: a clean 100 Hz pulse train
    x = a.channel(1).astype(np.float64)
    hi, lo = 70.0, 30.0
    state = np.zeros(x.size, dtype=bool)
    s_ = False
    for i in range(x.size):
        if x[i] > hi:
            s_ = True
        elif x[i] < lo:
            s_ = False
        state[i] = s_
    on = np.flatnonzero(np.diff(state.astype(np.int8)) == 1) + 1
    rate = 1.0 / (np.median(np.diff(on)) / a.fs)
    assert abs(rate - 100.0) < 0.5, rate
    print(f"  PASS  test_analog: {nch} ch x {ns} samples, "
          f"t = {a.t[0]:.2f}..{a.t[-1]:.2f} s")
    print(f"        ch1 exposure trigger: {on.size} pulses at "
          f"{rate:.4f} Hz")


if __name__ == "__main__":
    print("Block 0 tests")
    test_layout()
    test_handle_released()
    test_short_file_rejected()
    test_timing()
    test_analog()
    print("all passed")
