"""What the soft X-ray camera sees: for a lost shot and its twin, a slow-motion video of the last 300 ms before the
loss and a still picture of the same. Runs where the store answers (the fetch_sxr job), for a few example pairs that
scripts/sxr_audit.py picks from the development campaigns only (M5, M6).

Top: the camera's 18 chords as a strip of cells, brightness = signal over that chord's own peak, lost shot on the
left, twin on the right, both at the same moment. Bottom: three central chords over time, with a cursor.

    python scripts/sxr_view.py --examples "30001:29998:312;..."   # lost:twin:loss time in ms; writes DATA_DIR/sxr_audit/examples/
"""
import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import programme_signals, sxr_fetch

WINDOW_S, BIN_S, SECONDS, FPS = 0.30, 2e-4, 12, 30
BG, INK, DIM, LOST, KEPT = "#05060a", "#eef1f5", "#8a93a3", "#ff6fa3", "#3cc7bf"
PREFER = ("horizontal_cam_upper", "horizontal_cam_lower", "tangential_cam")


def parse_examples(text):
    out = []
    for item in filter(None, (text or "").split(";")):
        lost, kept, ms = item.split(":")
        out.append((int(lost), int(kept), float(ms) / 1000))
    return out


def camera_series(ds, camera, t0, t1):
    """(time bins, chords x bins) of one camera between t0 and t1, as means over BIN_S, NaN where empty."""
    arr = ds[camera]
    axis = [d for d in arr.dims if d != "time"][0]
    t = np.asarray(ds["time"].values, dtype=float)
    keep = (t >= t0) & (t <= t1)
    x = np.asarray(arr.transpose(axis, "time").values[:, keep], dtype=float)
    tk = t[keep]
    edges = np.arange(t0, t1 + BIN_S, BIN_S)
    idx = np.clip(np.searchsorted(edges, tk) - 1, 0, len(edges) - 2)
    sums = np.zeros((x.shape[0], len(edges) - 1))
    counts = np.zeros(len(edges) - 1)
    ok = np.isfinite(x)
    for c in range(x.shape[0]):
        np.add.at(sums[c], idx[ok[c]], x[c, ok[c]])
    np.add.at(counts, idx, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums / np.where(counts > 0, counts, np.nan)
    return edges[:-1] + BIN_S / 2, out


def normalise(series):
    """Each chord divided by its own 99th percentile over the window: 0 is dark, 1 is that chord's brightest."""
    top = np.nanpercentile(series, 99, axis=1, keepdims=True) if np.isfinite(series).any() else np.ones((len(series), 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(series / np.where(np.abs(top) > 0, top, np.nan), 0, 1.2)


def pick_camera(a, b):
    """The preferred camera both shots have, else any two-dimensional one they share, else None."""
    cams = lambda ds: [v for v in ds.data_vars if ds[v].ndim == 2 and "time" in ds[v].dims]  # noqa: E731
    shared = [c for c in cams(a) if c in cams(b)]
    preferred = [c for c in PREFER if c in shared]
    return (preferred or shared or [None])[0]


def render(lost, kept, t_loss, out_dir, open_group=programme_signals.open_group, fps=FPS, seconds=SECONDS):
    """Write <lost>_<kept>.mp4 and .png into out_dir; returns their paths, or [] when the pair cannot be shown."""
    try:
        a, b = open_group(lost, "soft_x_rays"), open_group(kept, "soft_x_rays")
    except Exception:
        return []
    cam = pick_camera(a, b)
    if cam is None:
        return []
    t0, t1 = t_loss - WINDOW_S, t_loss + 0.01
    tb, sa = camera_series(a, cam, t0, t1)
    _, sb = camera_series(b, cam, t0, t1)
    na, nb = normalise(sa), normalise(sb)
    mid = np.arange(na.shape[0] // 2 - 1, na.shape[0] // 2 + 2)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{lost}_{kept}")

    # the still: both shots, chord against time
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), dpi=100, facecolor=BG, sharex=True)
    for ax, n, sid, col in ((axes[0], na, lost, LOST), (axes[1], nb, kept, KEPT)):
        ax.imshow(n, aspect="auto", origin="lower", cmap="inferno", vmin=0.0, vmax=1.05,
                  extent=[(t0 - t_loss) * 1e3, (t1 - t_loss) * 1e3, 0, n.shape[0]])
        ax.axvline(0, color=INK, lw=1, ls=":")
        ax.set_facecolor(BG)
        ax.set_ylabel("chord", color=DIM)
        ax.set_title(f"shot {sid} ({'lost' if sid == lost else 'twin, kept'})", color=col, fontsize=12, loc="left")
        ax.tick_params(colors=DIM)
    axes[1].set_xlabel("ms from the lost shot's loss", color=DIM)
    fig.suptitle(f"soft X-ray camera {cam}: same programme, different fate", color=INK, fontsize=12)
    fig.savefig(base + ".png", facecolor=BG, bbox_inches="tight")
    plt.close(fig)

    # the video: slow motion through the window
    W, H = 720, 1280
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                             "-movflags", "+faststart", base + ".mp4"], stdin=subprocess.PIPE)
    n_frames = int(seconds * fps)
    for f in range(n_frames):
        t = t0 + (t1 - t0) * f / (n_frames - 1)
        k = int(np.clip(np.searchsorted(tb, t), 0, len(tb) - 1))
        fig.clear()
        fig.text(0.5, 0.955, f"{(t - t_loss) * 1e3:+.0f} ms", color=INK, fontsize=34, ha="center", family="DejaVu Sans Mono")
        fig.text(0.5, 0.925, "from the loss" if t < t_loss else "after the loss", color=DIM, fontsize=13, ha="center")
        for x0, n, sid, col, label in ((0.10, na, lost, LOST, "lost"), (0.56, nb, kept, KEPT, "twin, kept")):
            ax = fig.add_axes([x0, 0.47, 0.34, 0.42])
            ax.imshow(n[:, max(k - 2, 0):k + 1].mean(axis=1, keepdims=True), aspect="auto", origin="lower", cmap="inferno",
                      vmin=0.0, vmax=1.05)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(col)
            ax.set_title(f"{label}  #{sid}", color=col, fontsize=14)
        ax = fig.add_axes([0.10, 0.08, 0.80, 0.30], facecolor=BG)
        for n, col in ((na, LOST), (nb, KEPT)):
            ax.plot((tb - t_loss) * 1e3, np.nanmean(n[mid], axis=0), color=col, lw=1.2)
        ax.axvline((t - t_loss) * 1e3, color=INK, lw=1)
        ax.axvline(0, color=DIM, lw=1, ls=":")
        ax.set_ylim(0.0, 1.2)
        ax.tick_params(colors=DIM)
        for s in ax.spines.values():
            s.set_color(DIM)
        ax.set_xlabel("ms from the loss", color=DIM)
        ax.set_ylabel("central chords / own peak", color=DIM)
        fig.canvas.draw()
        proc.stdin.write(fig.canvas.buffer_rgba().tobytes())
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    return [base + ".mp4", base + ".png"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", default="")
    args = parser.parse_args()
    out = os.path.join(sxr_fetch.folder(), "examples")
    for lost, kept, t_loss in parse_examples(args.examples):
        paths = render(lost, kept, t_loss, out)
        print(f"{lost} vs {kept}: " + (", ".join(os.path.basename(p) for p in paths) if paths else "no shared camera"), flush=True)


if __name__ == "__main__":
    main()
