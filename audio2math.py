#!/usr/bin/env python3
"""
audio2math.py - turn music into a mathematical function, then play and plot that function.

Model (additive sinusoidal synthesis, McAulay-Quatieri style):

    x(t) = sum_k  A_k(t) * cos( phi_k + 2*pi * integral_0^t f_k(u) du )

Each partial k is a "track" with piecewise-linear amplitude A_k(t) and frequency f_k(t).
The sound you hear is computed ONLY from these tracks (the original samples are not used).

Install:   pip install numpy scipy matplotlib sounddevice soundfile
           (ffmpeg on PATH is used as a fallback decoder, e.g. for mp3/m4a)

Usage:     python audio2math.py song.mp3
           python audio2math.py song.mp3 --partials 96 --save song.npz
           python audio2math.py song.npz                 # play a saved model, no audio file needed
           python audio2math.py song.mp3 --formula 12.5  # print the formula around t = 12.5 s

Keys:      SPACE = pause/resume     Click on the bottom plot = seek
"""
import argparse
import shutil
import subprocess
import sys
import time
from math import gcd

import numpy as np
from scipy.signal import get_window, resample_poly


# ----------------------------------------------------------------------------- I/O
def load_audio(path, sr):
    try:
        import soundfile as sf
        x, fs = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if fs != sr:
            g = gcd(fs, sr)
            x = resample_poly(x, sr // g, fs // g).astype(np.float32)
        return x
    except Exception:
        if not shutil.which("ffmpeg"):
            raise SystemExit("Cannot decode this file: install soundfile or ffmpeg.")
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
            capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.float32).copy()


# ----------------------------------------------------------------------------- analysis
def analyze(x, sr, n_fft=2048, hop=512, K=48, floor_db=-65.0, tol_cents=60.0, min_len=3):
    """STFT peak picking + frequency-continuity tracking -> list of sinusoidal tracks."""
    win = get_window("hann", n_fft).astype(np.float32)
    norm = 2.0 / win.sum()
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad + n_fft))
    nfr = len(x) // hop + 1
    floor = max(float(np.abs(x).max()), 1e-9) * 10 ** (floor_db / 20)

    active, done = [], []
    for i in range(nfr):
        seg = xp[i * hop:i * hop + n_fft]
        X = np.fft.rfft(seg * win)
        m = np.abs(X) * norm
        pk = np.flatnonzero((m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]) & (m[1:-1] > floor)) + 1
        if len(pk) > K:
            pk = pk[np.argsort(m[pk])[-K:]]
        pk = pk[np.argsort(-m[pk])]
        if len(pk):
            la, lb, lc = (np.log(m[pk + d] + 1e-12) for d in (-1, 0, 1))
            den = la - 2 * lb + lc
            p = np.clip(np.where(np.abs(den) > 1e-12, 0.5 * (la - lc) / np.where(den == 0, 1, den), 0.0), -0.5, 0.5)
            freq = (pk + p) * sr / n_fft
            amp = np.exp(lb - 0.25 * (la - lc) * p)
            ph = np.angle(X[pk]) + np.pi * (pk + p)  # phase at frame centre (cos convention)
        else:
            freq = amp = ph = []

        lastf = np.array([t["f"][-1] for t in active]) if active else np.empty(0)
        claimed = np.zeros(len(active), bool)
        new_active = []
        for j in range(len(freq)):
            best = -1
            if len(lastf):
                d = np.abs(1200 * np.log2(freq[j] / lastf))
                d[claimed] = np.inf
                b = int(np.argmin(d))
                if d[b] < tol_cents:
                    best = b
            if best < 0:
                t = {"start": i, "f": [freq[j]], "a": [amp[j]], "ph": float(ph[j])}
            else:
                claimed[best] = True
                t = active[best]
                t["f"].append(freq[j])
                t["a"].append(amp[j])
            new_active.append(t)
        done += [t for t, c in zip(active, claimed) if not c]
        active = new_active
    done += active

    tr = sorted((t for t in done if len(t["f"]) >= min_len), key=lambda t: t["start"])
    if not tr:
        raise SystemExit("No partials found (silent input?).")
    length = np.array([len(t["f"]) for t in tr], np.int64)
    return {
        "sr": sr, "hop": hop, "n": len(x),
        "start": np.array([t["start"] for t in tr], np.int64),
        "length": length,
        "offs": np.r_[0, np.cumsum(length)[:-1]].astype(np.int64),
        "phase0": np.array([t["ph"] for t in tr], np.float32),
        "freq": np.concatenate([t["f"] for t in tr]).astype(np.float32),
        "amp": np.concatenate([t["a"] for t in tr]).astype(np.float32),
    }


# ----------------------------------------------------------------------------- synthesis
def render(m):
    """Evaluate the function x(t) from the model."""
    sr, hop, n = int(m["sr"]), int(m["hop"]), int(m["n"])
    out = np.zeros(n, np.float64)
    for s, L, o, ph0 in zip(m["start"], m["length"], m["offs"], m["phase0"]):
        f = m["freq"][o:o + L].astype(np.float64)
        a = m["amp"][o:o + L].astype(np.float64)
        fr = np.arange(s - 1, s + L + 1) * hop                      # knot positions (samples)
        a_ext = np.r_[0.0, a, 0.0]                                    # fade in / out
        f_ext = np.r_[f[0], f, f[-1]]
        s0, s1 = max(fr[0], 0), min(fr[-1], n - 1) + 1
        if s1 <= s0:
            continue
        t = np.arange(s0, s1)
        fi, ai = np.interp(t, fr, f_ext), np.interp(t, fr, a_ext)
        c = 2 * np.pi * np.cumsum(fi) / sr
        pivot = int(np.clip(s * hop - s0, 0, len(t) - 1))
        out[s0:s1] += ai * np.cos(ph0 + c - c[pivot])
    peak = np.abs(out).max()
    return (out * (0.9 / peak) if peak > 0 else out).astype(np.float32)


# ----------------------------------------------------------------------------- formula helpers
def active_at(m, pos):
    i = int(round(pos / int(m["hop"])))
    idx = np.flatnonzero((m["start"] <= i) & (i < m["start"] + m["length"]))
    k = i - m["start"][idx] + m["offs"][idx]
    return m["freq"][k], m["amp"][k]


def formula_text(m, pos, top=6):
    f, a = active_at(m, pos)
    order = np.argsort(-a)[:top]
    t = pos / int(m["sr"])
    lines = [f"t = {t:8.3f} s   x(t) = sum_k A_k(t)*cos(phi_k + 2*pi*int f_k dt)   [{len(f)} active partials]"]
    for j in order:
        lines.append(f"   {a[j]:9.4f} * cos(2*pi * {f[j]:8.2f} Hz * t + phi)")
    if len(f) > top:
        lines.append(f"   ... + {len(f) - top} more terms")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- live viewer
def run_viewer(m, synth, orig, play):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    sr, hop, n = int(m["sr"]), int(m["hop"]), len(synth)
    state = {"pos": 0, "paused": False, "last": time.perf_counter()}
    stream = None
    if play:
        import sounddevice as sd

        def cb(out, frames, _t, _s):
            if state["paused"]:
                out[:] = 0
                return
            p = state["pos"]
            ch = synth[p:p + frames]
            out[:len(ch), 0] = ch
            out[len(ch):, 0] = 0
            state["pos"] = min(p + frames, n)

        stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32", callback=cb)
        stream.start()

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 3], hspace=0.38, bottom=0.2, top=0.95)
    ax_w, ax_s, ax_r = (fig.add_subplot(gs[i]) for i in range(3))

    W = int(0.06 * sr)
    tms = np.arange(W) / sr * 1000
    ax_w.set_title("Waveform around playhead (blue = computed from function, grey = original)", fontsize=9)
    ax_w.set_xlim(0, tms[-1]); ax_w.set_ylim(-1, 1); ax_w.set_xlabel("ms")
    ln_o, = ax_w.plot(tms, np.zeros(W), color="0.65", lw=1)
    ln_s, = ax_w.plot(tms, np.zeros(W), color="C0", lw=1.2)

    amax = float(m["amp"].max())
    ax_s.set_title("Partials active now (frequency vs level)", fontsize=9)
    ax_s.set_xscale("log"); ax_s.set_xlim(30, sr / 2); ax_s.set_ylim(-70, 3)
    ax_s.set_xlabel("Hz"); ax_s.set_ylabel("dB")
    ln_p, = ax_s.plot([], [], color="C1", lw=1.5, marker="o", ms=3)

    segs, cols = [], []
    for s, L, o in zip(m["start"], m["length"], m["offs"]):
        tt = (s + np.arange(L)) * hop / sr
        segs.append(np.c_[tt, m["freq"][o:o + L]])
        cols.append(20 * np.log10(float(m["amp"][o:o + L].mean()) / amax + 1e-9))
    lc = LineCollection(segs, cmap="magma", linewidths=0.8)
    lc.set_array(np.clip(np.array(cols), -70, 0)); lc.set_clim(-70, 0)
    ax_r.add_collection(lc)
    ax_r.set_xlim(0, n / sr); ax_r.set_yscale("log"); ax_r.set_ylim(40, sr / 2)
    ax_r.set_xlabel("time (s)  -  click to seek"); ax_r.set_ylabel("Hz")
    ax_r.set_title("Frequency tracks of the model", fontsize=9)
    head = ax_r.axvline(0, color="cyan", lw=1.2)

    txt = fig.text(0.01, 0.01, "", family="monospace", fontsize=8, va="bottom")

    def on_key(e):
        if e.key == " ":
            state["paused"] = not state["paused"]

    def on_click(e):
        if e.inaxes is ax_r and e.xdata is not None:
            state["pos"] = int(np.clip(e.xdata, 0, n / sr) * sr)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    def update(_):
        now = time.perf_counter()
        if not play and not state["paused"]:
            state["pos"] = min(n, state["pos"] + int((now - state["last"]) * sr))
        state["last"] = now
        p = min(state["pos"], max(n - W, 0))
        ln_s.set_ydata(np.pad(synth[p:p + W], (0, max(0, W - len(synth[p:p + W])))))
        if orig is not None:
            ln_o.set_ydata(np.pad(orig[p:p + W], (0, max(0, W - len(orig[p:p + W])))))
        f, a = active_at(m, state["pos"])
        if len(f):
            xs = np.repeat(f, 3).astype(float); xs[2::3] = np.nan
            db = 20 * np.log10(a / amax + 1e-9)
            ys = np.empty(len(f) * 3); ys[0::3] = -70; ys[1::3] = db; ys[2::3] = np.nan
            ln_p.set_data(xs, ys)
        head.set_xdata([state["pos"] / sr] * 2)
        txt.set_text(formula_text(m, state["pos"]))

    anim = FuncAnimation(fig, update, interval=40, cache_frame_data=False)  # noqa: F841
    try:
        plt.show()
    finally:
        if stream:
            stream.stop(); stream.close()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Convert music into a sum-of-sinusoids function, plot and play it.")
    ap.add_argument("input", help="audio file (wav/flac/mp3/...) or a saved model .npz")
    ap.add_argument("--sr", type=int, default=22050, help="analysis sample rate")
    ap.add_argument("--partials", type=int, default=48, help="max partials kept per frame (quality vs size)")
    ap.add_argument("--nfft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--floor-db", type=float, default=-65.0, help="ignore peaks below this level")
    ap.add_argument("--save", help="save the model to .npz")
    ap.add_argument("--formula", type=float, metavar="SEC", help="print the formula at this time and exit")
    ap.add_argument("--no-play", action="store_true", help="plot only, no sound")
    a = ap.parse_args()

    orig = None
    if a.input.lower().endswith(".npz"):
        z = np.load(a.input)
        m = {k: z[k] for k in z.files}
    else:
        x = load_audio(a.input, a.sr)
        print(f"Loaded {len(x) / a.sr:.1f} s, analysing...")
        m = analyze(x, a.sr, a.nfft, a.hop, a.partials, a.floor_db)
        pk = np.abs(x).max()
        orig = x * (0.9 / pk) if pk > 0 else x
    n_params = len(m["freq"]) * 2 + len(m["start"]) * 3
    print(f"Model: {len(m['start'])} partial tracks, {n_params:,} parameters "
          f"(~{n_params * 4 / 1024:.0f} KiB) for {int(m['n']) / int(m['sr']):.1f} s of audio")

    if a.save:
        np.savez_compressed(a.save, **m)
        print("Saved", a.save)

    if a.formula is not None:
        print(formula_text(m, int(a.formula * int(m["sr"])), top=12))
        return

    synth = render(m)
    run_viewer(m, synth, orig, play=not a.no_play)


if __name__ == "__main__":
    sys.exit(main())
