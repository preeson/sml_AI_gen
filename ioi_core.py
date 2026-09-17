"""
ioi_core.py -- shared configuration and I/O for intrinsic optical imaging analysis.

Data model established in Block 0:

  * The .dat is raw uint16, little-endian, no header.
  * MATLAB wrote it with imgSize = [540 640 1 N], column-major.  So each frame
    occupies 540*640*2 bytes laid out column-by-column: element (row, col) sits
    at offset col*540 + row.  Reading a frame as C-order (640, 540) therefore
    gives you [col][row]; transpose to get (row, col) = (540, 640).
  * frameTimes has exactly one entry per frame in the .dat (verified by byte
    count), so camera frame index == frameTimes index.  No resampling needed to
    align them.
  * The stimulus log's *timestamps* are corrupt (inflated ~1.98x).  Only its
    design parameters and frame/sweep tables are trustworthy.  All timing comes
    from the camera clock.
"""

from __future__ import annotations

import contextlib
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import scipy.io as sio

# --------------------------------------------------------------------------
# Configuration -- edit these paths for your machine
# --------------------------------------------------------------------------

DATA_DIR = r"C:\Users\gbouvier\Desktop\09-Sep-2026"

DAT_PATH = os.path.join(DATA_DIR, "Frames_1_640_540_uint16_0001.dat")
FRAMETIMES_PATH = os.path.join(DATA_DIR, "frameTimes_0001.mat")
ANALOG_PATH = os.path.join(DATA_DIR, "Analog_1.dat")
STIMLOG_PATH = os.path.join(
    DATA_DIR,
    "260909145607-KSstimAllDir-Mc57-M1_trial1-plr-plr-notTriggered-complete.pkl",
)

FRAME_H = 540           # rows,    = imgSize(1)
FRAME_W = 640           # columns, = imgSize(2)
DTYPE = np.dtype("<u2")  # uint16 little-endian

BYTES_PER_FRAME = FRAME_H * FRAME_W * DTYPE.itemsize  # 691200


# --------------------------------------------------------------------------
# Camera frame times
# --------------------------------------------------------------------------

@dataclass
class FrameTimes:
    t: np.ndarray          # seconds, relative to first frame
    n_frames: int
    pre_stim: int          # number of baseline frames before stimulus onset
    post_stim: int         # first camera frame index *after* the stimulus
    removed: int
    datenum0: float        # original MATLAB datenum of frame 0

    @property
    def stim_slice(self) -> slice:
        """Camera frames covered by the stimulus, as a Python slice."""
        return slice(self.pre_stim, self.post_stim)

    @property
    def fs(self) -> float:
        """Mean camera sampling rate in Hz."""
        return (self.n_frames - 1) / (self.t[-1] - self.t[0])


def load_frame_times(path: str | None = None) -> FrameTimes:
    m = sio.loadmat(_require(FRAMETIMES_PATH if path is None else path))
    ft = np.asarray(m["frameTimes"], dtype=np.float64).ravel()
    t = (ft - ft[0]) * 86400.0  # MATLAB datenum is in days
    return FrameTimes(
        t=t,
        n_frames=t.size,
        pre_stim=int(np.asarray(m["preStim"]).ravel()[0]),
        post_stim=int(np.asarray(m["postStim"]).ravel()[0]),
        removed=int(np.asarray(m["removedFrames"]).ravel()[0]),
        datenum0=float(ft[0]),
    )


# --------------------------------------------------------------------------
# Movie access
# --------------------------------------------------------------------------

def _require(path: str) -> str:
    """Fail with a useful message instead of a bare FileNotFoundError."""
    if os.path.exists(path):
        return path
    folder = os.path.dirname(path) or "."
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    near = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in (".dat", ".mat", ".pkl")
    )
    listing = "\n    ".join(near) if near else "(no .dat/.mat/.pkl files here)"
    raise FileNotFoundError(
        f"Not found: {path}\n"
        f"  Data files present in {folder}:\n    {listing}\n"
        f"  Fix the path constants at the top of ioi_core.py."
    )


def frame_count_on_disk(path: str | None = None) -> int:
    path = _require(DAT_PATH if path is None else path)
    size = os.path.getsize(path)
    if size % BYTES_PER_FRAME:
        raise ValueError(
            f"{path}: size {size} is not a whole number of "
            f"{FRAME_H}x{FRAME_W} uint16 frames ({BYTES_PER_FRAME} B each); "
            f"remainder {size % BYTES_PER_FRAME} B"
        )
    return size // BYTES_PER_FRAME


def open_movie(path: str | None = None, n_frames: int | None = None) -> np.memmap:
    """
    Memory-map the .dat and return a view of shape (n_frames, FRAME_H, FRAME_W).

    The transpose is a view, not a copy -- no data is read until you index.
    Slicing along axis 0 (e.g. mov[a:b]) reads a contiguous byte range, which
    is what makes chunked processing fast.  Avoid fancy-indexing the whole
    array; it will try to materialise 74 GB.
    """
    path = _require(DAT_PATH if path is None else path)
    if n_frames is None:
        n_frames = frame_count_on_disk(path)
    mm = np.memmap(path, dtype=DTYPE, mode="r", shape=(n_frames, FRAME_W, FRAME_H))
    return mm.transpose(0, 2, 1)  # -> (n, FRAME_H, FRAME_W)


def close_movie(mov) -> bool:
    """
    Release the OS file handle behind a memmap (or any view of one).

    On Windows an open memmap locks the file: you cannot delete, rename or
    move it, and a second process cannot open it for writing.  Call this when
    you are done, or use the `movie()` context manager below.

    After calling this, any further access to `mov` -- or to any view or slice
    derived from it -- is undefined and will likely crash the interpreter.
    Copy out anything you still need first.
    """
    obj = mov
    for _ in range(8):  # walk the .base chain to the underlying np.memmap
        if obj is None:
            break
        mm = getattr(obj, "_mmap", None)
        if mm is not None:
            if not mm.closed:
                mm.close()
            return True
        obj = getattr(obj, "base", None)
    return False


@contextlib.contextmanager
def movie(path: str | None = None, n_frames: int | None = None):
    """
    Context manager around open_movie() that always releases the handle:

        with ioi.movie() as mov:
            chunk = np.asarray(mov[0:2000], dtype=np.float32)   # copy out
    """
    mov = open_movie(path, n_frames)
    try:
        yield mov
    finally:
        close_movie(mov)


# --------------------------------------------------------------------------
# Stimulus log
# --------------------------------------------------------------------------

@dataclass
class StimBlock:
    label: str            # 'B2U', 'U2B', 'L2R', 'R2L'
    axis: str             # 'elevation' or 'azimuth'
    k0: int               # first stimulus frame index (inclusive)
    k1: int               # last stimulus frame index (inclusive)
    frames_per_cycle: int
    n_cycles: int
    cam0: int = 0         # first camera frame (inclusive)
    cam1: int = 0         # last camera frame (exclusive)
    t0: float = 0.0       # seconds on the camera clock
    t1: float = 0.0
    bar_deg_first: float = 0.0   # bar CENTRE at the first frame of a cycle
    bar_deg_last: float = 0.0    # bar CENTRE at the last frame of a cycle

    @property
    def n_cam_frames(self) -> int:
        return self.cam1 - self.cam0

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def f_stim(self) -> float:
        """Stimulus frequency in Hz, measured on the camera clock."""
        return self.n_cycles / self.duration

    @property
    def cam_slice(self) -> slice:
        return slice(self.cam0, self.cam1)


@dataclass
class StimLog:
    raw: dict = field(repr=False)
    n_stim_frames: int = 0
    blocks: list = field(default_factory=list)
    sweep_width: float = 0.0
    step_width: float = 0.0
    nominal_refresh: float = 60.0
    effective_refresh: float = 0.0

    def by_label(self, label: str) -> StimBlock:
        for b in self.blocks:
            if b.label == label:
                return b
        raise KeyError(label)


_AXIS_OF = {"B2U": "elevation", "U2B": "elevation",
            "L2R": "azimuth", "R2L": "azimuth"}


def load_stim_log(path: str | None = None, ft: FrameTimes | None = None) -> StimLog:
    """
    Parse the WarpedVisualStim pickle and place every block on the camera clock.

    The pickle's own timestamps are discarded.  Instead we assume the 69630
    stimulus frames were presented uniformly across the camera's stimulus
    window (frames pre_stim .. post_stim).  End-to-end this assumption agrees
    with the nominal 60 Hz design to 0.13%, which bounds any cumulative drift.
    """
    path = _require(STIMLOG_PATH if path is None else path)
    if ft is None:
        ft = load_frame_times()

    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    stim = d["stimulation"]
    frames = stim["frames"]
    # frame_config = ('is_display', 'squarePolarity', 'sweep_index', 'indicator_color')
    # plus a 5th element carrying the direction label.
    direction = np.array([fr[4] for fr in frames])
    n_stim = len(frames)

    sweep_table = stim["sweep_table"]  # (direction, start_deg, end_deg)

    # camera frames per stimulus frame
    span = ft.post_stim - ft.pre_stim
    scale = span / n_stim

    blocks = []
    for label in stim["direction"]:
        idx = np.flatnonzero(direction == label)
        if idx.size == 0:
            continue
        # sanity: the label must occupy one contiguous run
        if idx[-1] - idx[0] + 1 != idx.size:
            raise ValueError(f"direction {label!r} is not contiguous in the log")

        rows = [r for r in sweep_table if r[0] == label]
        fpc = len(rows)
        n_cyc, rem = divmod(idx.size, fpc)
        if rem:
            raise ValueError(
                f"{label}: {idx.size} frames is not a whole number of "
                f"{fpc}-frame cycles (remainder {rem})"
            )

        k0, k1 = int(idx[0]), int(idx[-1])
        cam0 = int(round(ft.pre_stim + k0 * scale))
        cam1 = int(round(ft.pre_stim + (k1 + 1) * scale))

        blocks.append(StimBlock(
            label=str(label),
            axis=_AXIS_OF.get(str(label), "?"),
            k0=k0, k1=k1,
            frames_per_cycle=fpc,
            n_cycles=n_cyc,
            cam0=cam0, cam1=cam1,
            t0=float(ft.t[cam0]),
            t1=float(ft.t[min(cam1, ft.n_frames - 1)]),
            bar_deg_first=float(rows[0][1] + rows[0][2]) / 2.0,
            bar_deg_last=float(rows[-1][1] + rows[-1][2]) / 2.0,
        ))

    log = StimLog(
        raw=d,
        n_stim_frames=n_stim,
        blocks=blocks,
        sweep_width=float(stim["sweep_width"]),
        step_width=float(stim["step_width"]),
        nominal_refresh=float(d["monitor"]["refresh_rate"]),
    )
    total = ft.t[min(ft.post_stim, ft.n_frames - 1)] - ft.t[ft.pre_stim]
    log.effective_refresh = n_stim / total
    return log


# --------------------------------------------------------------------------
# Analog side-channels (Analog_N.dat)
# --------------------------------------------------------------------------

ANALOG_FS = 1000.0  # Hz, DAQ sampling rate


@dataclass
class Analog:
    data: np.ndarray       # (n_samples, n_channels) uint16
    fs: float              # Hz
    datenum0: float        # MATLAB datenum of the first sample
    t_offset: float        # first sample's time on the CAMERA clock (negative
                           # if the DAQ started before camera frame 0)

    @property
    def t(self) -> np.ndarray:
        """Sample times on the camera clock, in seconds."""
        return self.t_offset + np.arange(self.data.shape[0]) / self.fs

    def channel(self, c: int) -> np.ndarray:
        return self.data[:, c]


def load_analog(path: str | None = None, ft: FrameTimes | None = None,
                fs: float = ANALOG_FS) -> Analog:
    """
    Read a WidefieldImager Analog_N.dat.

    Layout: one float64 giving the header length, then that many float64s
    (datenum, n_channels, n_samples -- the last is Inf when the recording was
    streamed), then uint16 samples.  MATLAB stores analogData as
    (n_channels x n_samples) column-major, so the byte stream is
    channel-interleaved and reshapes to (n_samples, n_channels).
    """
    path = _require(ANALOG_PATH if path is None else path)
    if ft is None:
        ft = load_frame_times()

    with open(path, "rb") as f:
        hlen = int(np.fromfile(f, dtype="<f8", count=1)[0])
        header = np.fromfile(f, dtype="<f8", count=hlen)
        raw = np.fromfile(f, dtype="<u2")

    nch = int(header[1])
    ns, rem = divmod(raw.size, nch)
    if rem:
        raise ValueError(f"{path}: {raw.size} samples is not divisible by "
                         f"{nch} channels (remainder {rem})")

    return Analog(
        data=raw.reshape(ns, nch),
        fs=fs,
        datenum0=float(header[0]),
        t_offset=(float(header[0]) - ft.datenum0) * 86400.0,
    )
