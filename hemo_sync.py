"""
hemo_sync.py - step 1 of the hemodynamic sanity-check analysis.

Reads
  * the WarpedVisualStim .pkl log from stim_hemo_check.py  -> block schedule
  * the MATLAB Analog_*.dat file                            -> 1 kHz analog channels
and writes a CSV with the onset of every block (stimulus or blank).

Timing reference used for onsets ("t_sync"):
  DisplaySequence._display() sets start_time immediately before raising the
  sync pin (which starts acquisition), so `frame_ts_start` in the log is
  already "seconds since the sync pulse".

Analog_*.dat format (inferred from the test file, confirm with the MATLAB code):
  header : float64 [n_hdr, then n_hdr values: MATLAB datenum of start, n_columns, inf]
  data   : uint16, n_columns interleaved, 1 kHz
           col 0 = sample counter (saturates at 65535), cols 1..4 = analog inputs
"""

import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ANALOG_FS = 1000.0       # Hz


# --------------------------------------------------------------------------- stim log
def load_stim_log(pkl_path):
    with open(pkl_path, 'rb') as f:
        log = pickle.load(f, encoding='latin1')
    return log


def block_schedule(log):
    """Return a list of dicts, one per block (stimulus or blank), in display order."""
    pres = log['presentation']
    stim = log['stimulation']
    frames = pres['displayed_frames']          # per displayed frame: tuple, see frame_config
    ts = np.asarray(pres['frame_ts_start'])    # s since sync pulse
    cfg = stim['frame_config']
    i_disp = cfg.index('is_display')
    i_sf = cfg.index('spatial frequency (cycle/deg)')
    i_dir = cfg.index('direction (deg)')

    is_disp = np.array([fr[i_disp] for fr in frames], dtype=float)
    onsets = np.where(np.diff(np.r_[0., is_disp]) == 1)[0]
    offsets = np.where(np.diff(np.r_[is_disp, 0.]) == -1)[0]   # last ON frame
    if len(onsets) != len(offsets):
        raise RuntimeError('Unmatched block onsets/offsets in log')

    t_first = ts[onsets[0]]
    blocks = []
    for k, (on, off) in enumerate(zip(onsets, offsets)):
        fr = frames[on]
        is_blank = (fr[i_sf] == 0.0)
        blocks.append(dict(
            block=k,
            onset_frame=int(on),
            t_sync=float(ts[on]),                    # s since sync pulse
            t_rel_first=float(ts[on] - t_first),     # s since first block onset
            duration=float(ts[off] - ts[on] + 1. / 60.),
            is_blank=bool(is_blank),
            direction=None if is_blank else float(fr[i_dir]),
        ))
    return blocks


# --------------------------------------------------------------------------- analog
def read_analog_dat(dat_path):
    n_hdr = int(np.fromfile(dat_path, dtype='<f8', count=1)[0])
    header = np.fromfile(dat_path, dtype='<f8', count=1 + n_hdr)[1:]
    n_cols = int(header[1])
    data = np.fromfile(dat_path, dtype='<u2', offset=8 * (1 + n_hdr))
    if data.size % n_cols:
        raise ValueError(f'File size not a multiple of {n_cols} int16 columns')
    data = data.reshape(-1, n_cols)
    return header, data


def find_ttl_onsets(trace, fs=ANALOG_FS, min_gap_s=0.5):
    """Rising-edge onsets (s) of a TTL recorded on an analog channel."""
    trace = trace.astype(float)
    thr = (np.percentile(trace, 1) + np.percentile(trace, 99.9)) / 2.
    hi = trace > thr
    edges = np.where(np.diff(hi.astype(int)) == 1)[0] + 1
    if edges.size:
        keep = np.r_[True, np.diff(edges) > min_gap_s * fs]
        edges = edges[keep]
    return edges / fs


# --------------------------------------------------------------------------- camera
def load_frametimes(mat_path):
    """
    frameTimes_XXXX.mat from WidefieldImager.
    Returns dict with
      t        : frame times in s, relative to the first saved frame
      period   : camera frame period (s) from a linear fit (timestamps jitter by
                 up to ~50 ms, the fit is the better clock)
      pre_stim : number of saved frames before the stim-pin trigger
                 -> 0-based index of the first frame after the first block onset
      img_size : (H, W, channels, n_frames)
    """
    from scipy.io import loadmat
    m = loadmat(mat_path, squeeze_me=True)
    ft = np.asarray(m['frameTimes'], dtype=float)          # MATLAB datenum
    t = (ft - ft[0]) * 86400.
    idx = np.arange(t.size)
    period, offset = np.polyfit(idx, t, 1)
    return dict(t=t, t_fit=offset + period * idx, period=period,
                pre_stim=int(m['preStim']), post_stim=int(m['postStim']),
                removed_frames=int(m['removedFrames']),
                img_size=tuple(int(v) for v in np.atleast_1d(m['imgSize'])),
                datenum_first=ft[0])


def align_blocks_to_frames(blocks, ftimes):
    """
    Add camera-frame onsets to each block.

    Anchor: the stim-pin pulse (first block onset) is camera frame `pre_stim`
    (0-based). Later blocks: anchor time + t_rel_first from the stimulus log,
    converted to frames with the fitted camera period.
    """
    t0 = ftimes['t_fit'][ftimes['pre_stim']]
    n = ftimes['t_fit'].size
    for b in blocks:
        t_cam = t0 + b['t_rel_first']
        f = int(round((t_cam - ftimes['t_fit'][0]) / ftimes['period']))
        b['cam_frame'] = f if f < n else None    # 0-based frame index of block onset
        b['t_cam'] = float(t_cam)                 # s since first saved frame
    return blocks


def open_frames(frames_dat, img_size, dtype=np.uint16):
    """
    Memory-map a WidefieldImager Frames_*.dat file without loading it.

    img_size = (H, W, channels, n_frames) from frameTimes.mat.
    Any bytes before the pixel data are treated as a float64 header
    (WidefieldImager style: first value = header length).
    Data are MATLAB column-major, so each frame is stored W-major and is
    transposed here. Returns (header, frames) with frames shape (n, H, W)
    for one channel or (n, C, H, W) for several.
    """
    H, W, C, N = img_size
    itemsize = np.dtype(dtype).itemsize
    file_bytes = Path(frames_dat).stat().st_size
    n_frames = None
    header = np.array([])
    data_bytes = H * W * C * N * itemsize
    header_bytes = file_bytes - data_bytes
    if header_bytes < 0 or header_bytes % 8:
        # size disagrees with imgSize: assume a header like the analog file
        # (first float64 = number of header values) and infer the frame count.
        n_hdr = int(np.fromfile(frames_dat, dtype='<f8', count=1)[0])
        header_bytes = 8 * (1 + n_hdr)
        n_frames = (file_bytes - header_bytes) // (H * W * C * itemsize)
        print(f'[open_frames] size does not match imgSize; using {n_frames} frames')
    if header_bytes:
        header = np.fromfile(frames_dat, dtype='<f8', count=header_bytes // 8)
    n_frames = n_frames or N
    mm = np.memmap(frames_dat, dtype=dtype, mode='r', offset=header_bytes,
                   shape=(n_frames, C, W, H))          # column-major frame layout
    frames = mm.transpose(0, 1, 3, 2)                  # -> (n, C, H, W), still lazy
    if C == 1:
        frames = frames[:, 0]
    return header, frames


# --------------------------------------------------------------------------- main
def main(pkl_path, dat_path=None, out_csv=None, frametimes_mat=None):
    log = load_stim_log(pkl_path)
    blocks = block_schedule(log)
    n_blank = sum(b['is_blank'] for b in blocks)
    print(f'{len(blocks)} blocks: {len(blocks) - n_blank} stimulus, {n_blank} blank')
    print(f"First block onset: {blocks[0]['t_sync']:.3f} s after sync pulse "
          f"(frame {blocks[0]['onset_frame']}, = stim_pin pulse)")
    iti = np.diff([b['t_sync'] for b in blocks])
    print(f'Onset-to-onset: {iti.mean():.3f} s (min {iti.min():.3f}, max {iti.max():.3f})')

    if dat_path is not None:
        header, data = read_analog_dat(dat_path)
        dur = data.shape[0] / ANALOG_FS
        t_last = log['presentation']['frame_ts_end'][-1]
        print(f'\nAnalog: {data.shape[0]} samples x {data.shape[1]} cols = {dur:.2f} s at 1 kHz')
        print(f'Stimulus log: sync pulse -> last frame = {t_last:.2f} s '
              f'(difference {t_last - dur:+.2f} s)')
        for c in range(1, data.shape[1]):
            on = find_ttl_onsets(data[:, c])
            span = np.ptp(data[:, c])
            print(f'  col {c}: range {span:5.0f} counts, {len(on)} rising edges')

    if frametimes_mat is not None:
        ftimes = load_frametimes(frametimes_mat)
        align_blocks_to_frames(blocks, ftimes)
        n = ftimes['t'].size
        print(f"\nCamera: {n} frames, period {ftimes['period'] * 1e3:.3f} ms "
              f"({1 / ftimes['period']:.2f} Hz), preStim = {ftimes['pre_stim']} frames "
              f"({ftimes['t_fit'][ftimes['pre_stim']]:.2f} s baseline before block 0)")
        stim_end = log['presentation']['frame_ts_end'][-1] - blocks[0]['t_sync']
        cam_end = ftimes['t_fit'][-1] - ftimes['t_fit'][ftimes['pre_stim']]
        print(f'First block -> end: stimulus {stim_end:.2f} s, camera {cam_end:.2f} s')
        last = blocks[-1]
        print(f"Last block onset at frame {last['cam_frame']} "
              f"(+{(n - 1 - (last['cam_frame'] or n)) * ftimes['period']:.1f} s of frames after it)")

    out_csv = out_csv or str(Path(pkl_path).with_suffix('')) + '_blocks.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(blocks[0].keys()))
        w.writeheader()
        w.writerows(blocks)
    print(f'\nBlock table written to {out_csv}')
    return blocks


if __name__ == '__main__':
    # python hemo_sync.py <stim.pkl> <Analog_1.dat> <out.csv> <frameTimes_0001.mat>
    main(*sys.argv[1:])
