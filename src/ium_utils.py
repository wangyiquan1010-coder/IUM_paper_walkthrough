# -*- coding: utf-8 -*-
"""
ium_utils.py — teaching utilities for the IUM (In-situ Ultrasonic Monitoring) course
package. Refactored (read-only) from the code of:

    Wang, Y., & Zhao, X. (2026). Machine learning-aided in-situ ultrasonic
    characterization for multi-parametric monitoring of vat photopolymerization.
    Additive Manufacturing, 105300. https://doi.org/10.1016/j.addma.2026.105300

Sections
--------
1. Data loading            : sample waveforms (.npz) + feature/label CSVs
2. Echo physics            : B1/B2 peak finding, ToF
3. Feature extraction      : the paper's 11 layer-wise descriptors (subset impl.)
4. Part-scale statistics
5. Dataset assembly        : 344-d representation, LOIO splits
6. Model                   : three-branch attention-fusion network (deployed Case 5)
7. Training / evaluation   : compact leak-free trainer for classroom budgets
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

FS = 2.5e9                     # oscilloscope sampling rate [Hz]
DT_NS = 1e9 / FS               # 0.4 ns per sample
N_POINTS = 62509
LABEL_COLS = ["thickness", "modulus", "DoC"]
PROCESS_COLS = ["layer", "intensity"]
SEQ_LEN = 30                   # max layer count (zero-padded)
FEATS_PER_LAYER = 11
INTENSITY_MW = {10: 10.70, 20: 12.57, 30: 14.35, 40: 15.93, 50: 17.48,
                60: 18.86, 70: 20.16, 80: 21.41, 90: 22.57, 100: 23.78}

# ----------------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------------
def load_sample(npz_path):
    """Load one printed sample's waveforms from a compressed .npz archive.

    The archive holds one array per printed layer (extracted read-only from the
    paper's raw oscilloscope CSV files) plus a JSON metadata record. The first
    layer of every print is exposed for 21 s instead of 15 s and therefore has
    7 frames; we keep its reference frame plus the last five so that every
    layer has the same 6 comparable frames.

    Returns
    -------
    wf   : np.ndarray (n_layers, 6 frames, N_POINTS)  raw amplitudes [a.u.]
           frame 0 = reference (before exposure), frames 1-5 = during the 15 s
           exposure, 3 s apart.
    meta : dict  (n_layers, intensity_pct, intensity_mw_cm2, fs_hz, ...)
    """
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    keys = sorted([k for k in z.files if k.startswith("layer_")])
    layers = []
    for k in keys:
        w = z[k]
        if w.shape[0] > 6:          # first layer (21 s) has an extra frame:
            w = np.vstack([w[:1], w[-5:]])   # keep reference + last 5
        layers.append(w)
    wf = np.stack(layers)
    return wf, meta


def time_axis_us(n=N_POINTS):
    """Time axis in microseconds."""
    return np.arange(n) * DT_NS / 1000.0


def load_features(csv_path):
    df = pd.read_csv(csv_path).dropna(how="all").reset_index(drop=True)
    df["sample_id"] = np.arange(len(df), dtype=int)
    return df


def load_labels(csv_path):
    return pd.read_csv(csv_path).dropna(how="all").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Echo physics: B1/B2 finding + ToF
# ----------------------------------------------------------------------------
# Windows (in samples) chosen for this dataset's geometry: B1 (printhead bottom
# echo) sits near ~5.7 us (idx ~14300); B2 (vat-bottom echo) follows it.
B1_WIN = (14000, 14800)


def _env(x):
    """Cheap envelope: |x - median| smoothed over a 101-sample (40 ns) window.

    An ultrasonic echo is an oscillating burst, so its individual peaks are
    ambiguous; the envelope has a single clear maximum that we can call "the
    arrival". The median subtraction removes the DC offset of the digitizer.
    """
    a = np.abs(x - np.median(x))
    k = 101
    ker = np.ones(k) / k
    return np.convolve(a, ker, mode="same")


def find_b1(wf_1d):
    """Index of the B1 peak (printhead internal echo) — max envelope in window."""
    e = _env(wf_1d)
    i0, i1 = B1_WIN
    return i0 + int(np.argmax(e[i0:i1]))


def find_b2(wf_1d, b1_idx, prev_b2=None, search_after=500, search_span=1600):
    """B2 peak via the paper's regression-guided idea, simplified for teaching.

    Tracking mode (prev_b2 given): B2 must advance, but by at most ~800 samples
    per layer (a 100-um layer adds ~330 samples of round-trip in liquid resin;
    the cap keeps the tracker from jumping to the later B3 echo).

    Why the cap matters: B3 is the same round trip made twice, so it sits a
    whole ToF (>3000 samples here) further right. Without an advance limit a
    greedy peak search lands on it and the reported ToF doubles.

    All indices and windows are in SAMPLES (0.4 ns each).
    """
    e = _env(wf_1d)
    if prev_b2 is None:
        lo = b1_idx + search_after
        hi = min(len(e), lo + search_span)
    else:
        lo = prev_b2 + 60
        hi = min(len(e), prev_b2 + 800)
    return lo + int(np.argmax(e[lo:hi]))


def track_sample(wf, frame=-1):
    """Track B1/B2 across all layers of one sample (using one frame per layer).

    Returns DataFrame with b1_idx, b2_idx, tof_us per layer.
    """
    rows = []
    prev = None
    for li in range(wf.shape[0]):
        x = wf[li, frame].astype(float)
        b1 = find_b1(x)
        b2 = find_b2(x, b1, prev_b2=prev)
        prev = b2
        rows.append(dict(layer=li + 1, b1_idx=b1, b2_idx=b2,
                         tof_us=(b2 - b1) * DT_NS / 1000.0))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. Layer-wise feature extraction (teaching subset, mirrors the paper)
# ----------------------------------------------------------------------------
def spectral_features(x, fs=FS, band=(2e6, 30e6)):
    """Center frequency and bandwidth of the spectrum inside `band`."""
    X = np.abs(np.fft.rfft(x - x.mean()))
    f = np.fft.rfftfreq(len(x), d=1 / fs)
    m = (f >= band[0]) & (f <= band[1])
    P = X[m] ** 2
    if P.sum() == 0:
        return 0.0, 0.0
    fc = float((f[m] * P).sum() / P.sum())
    bw = float(np.sqrt(((f[m] - fc) ** 2 * P).sum() / P.sum()))
    return fc, bw


def layer_features(wf_layer, b1_idx, b2_idx, win=800):
    """Compute the teaching subset of layer-wise features for one layer.

    wf_layer : (n_frames, N)   all frames of this layer
    b1_idx, b2_idx : tracked echo positions for this layer [sample index]
    win      : half-width of the analysis window around B2 [samples]; 800
               samples = 0.32 us, wide enough to hold the whole echo burst.

    Two scales, as in the paper's framework:
      * across-layer features use the FINAL frame only (end of exposure) ->
        ToF [us], B2/B1 envelope amplitude ratio [-], RMS energy [a.u.],
        center frequency and bandwidth [Hz];
      * within-layer features use ALL frames of the same, frozen window ->
        frame_diff_rms and within_layer_std [a.u.]. The acoustic path cannot
        change during one exposure, so these are geometry-immune cure probes.
    """
    x = wf_layer[-1].astype(float)
    e = _env(x)
    b1_amp = e[b1_idx]
    b2_amp = e[b2_idx]
    seg = x[b2_idx - win: b2_idx + win]
    fc, bw = spectral_features(seg)
    rms_energy = float(np.sqrt(np.mean(seg ** 2)))
    # within-layer: frame-to-frame differences in a fixed window around B2
    frames = wf_layer[:, b2_idx - win: b2_idx + win].astype(float)
    diffs = np.diff(frames, axis=0)
    within_rms = float(np.sqrt(np.mean(diffs ** 2)))
    within_std = float(np.std(frames, axis=0).mean())
    return dict(ToF=(b2_idx - b1_idx) * DT_NS / 1000.0,
                amplitude=float(b2_amp / (b1_amp + 1e-9)),
                RMS_Energy=rms_energy,
                Center_Freq=fc, Bandwidth=bw,
                frame_diff_rms=within_rms, within_layer_std=within_std)


# ----------------------------------------------------------------------------
# 4. Part-scale statistics (mirrors build_stats_func.compute_part_scale_stats)
# ----------------------------------------------------------------------------
def part_scale_stats(df):
    """12 global descriptors from the layer-wise ToF and amplitude columns.

    df : the feature11-style wide table (Layer_{i}_{name} columns).
    Returns DataFrame aligned with df rows.
    """
    out = {}
    for base in ["ToF", "amplitude"]:
        cols = [f"Layer_{i}_{base}" for i in range(1, SEQ_LEN + 1)
                if f"Layer_{i}_{base}" in df.columns]
        V = df[cols].values
        n_real = df["layer"].values.astype(int)
        mean, mx, sd, slope, early_late, final = [], [], [], [], [], []
        for row, n in zip(V, n_real):
            v = row[:n]
            k = np.arange(1, n + 1)
            mean.append(v.mean()); mx.append(v.max()); sd.append(v.std())
            slope.append(np.polyfit(k, v, 1)[0])
            h = max(1, n // 3)
            early_late.append(v[:h].mean() - v[-h:].mean())
            final.append(v[-1])
        p = base if base == "ToF" else "Amp"
        out.update({f"{p}_mean": mean, f"{p}_max": mx, f"{p}_std": sd,
                    f"{p}_slope": slope, f"{p}_early_late": early_late,
                    f"{p}_final": final})
    return pd.DataFrame(out, index=df.index)


# ----------------------------------------------------------------------------
# 5. Dataset assembly + LOIO
# ----------------------------------------------------------------------------
def seq_columns(df):
    import re
    pat = re.compile(r"^Layer_(\d+)_(.+)$")
    cols = [c for c in df.columns
            if c not in LABEL_COLS + PROCESS_COLS + ["sample_id"] and pat.match(c)]
    return sorted(cols, key=lambda c: (int(pat.match(c).group(1)),
                                       pat.match(c).group(2)))


def assemble_arrays(train_df, test_df):
    """Pack one LOIO fold into the three branch inputs of the network.

    LEAK-FREE RULE: every StandardScaler is fitted on the TRAINING partition
    only and then applied to both partitions (mirrors FourBranch_DataIO in the
    paper's code). Fitting a scaler on all data would let test statistics
    influence training and quietly inflate the reported scores.

    Returns (train_pack, test_pack, sc_y) where each pack holds
        stats : (n, 12)              part-scale descriptors, standardised
        proc  : (n, 2)               layer count + intensity, standardised
        seq   : (n, 30, 11)          layer-wise features, zero-padded
        y     : (n, 3)               labels, standardised
    and sc_y is the label scaler, needed to convert predictions back into
    physical units (see `predict`).
    """
    from sklearn.preprocessing import StandardScaler
    stats_tr = part_scale_stats(train_df)
    stats_te = part_scale_stats(test_df)
    seq_cols = seq_columns(train_df)
    feat_dim = len(seq_cols) // SEQ_LEN

    sc_stats = StandardScaler().fit(stats_tr.values)
    sc_proc = StandardScaler().fit(train_df[PROCESS_COLS].values)
    sc_y = StandardScaler().fit(train_df[LABEL_COLS].values)

    def pack(df, stats):
        return dict(
            stats=sc_stats.transform(stats.values).astype(np.float32),
            proc=sc_proc.transform(df[PROCESS_COLS].values).astype(np.float32),
            seq=df[seq_cols].values.reshape(len(df), SEQ_LEN, feat_dim
                                            ).astype(np.float32),
            y=sc_y.transform(df[LABEL_COLS].values).astype(np.float32))
    return pack(train_df, stats_tr), pack(test_df, stats_te), sc_y


def loio_folds(df):
    """Yield (train_df, test_df, held_out_intensity) for each of the 10 folds."""
    for inten in sorted(df["intensity"].unique()):
        yield (df[df.intensity != inten].reset_index(drop=True),
               df[df.intensity == inten].reset_index(drop=True), inten)


# ----------------------------------------------------------------------------
# 6. Model — three-branch attention fusion (deployed Case 5 of the paper)
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn


class Attn(nn.Module):
    """Element-wise sigmoid gate over the fusion vector (readable weights)."""

    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(),
                               nn.Linear(d // 2, d))

    def forward(self, x):
        w = torch.sigmoid(self.f(x))
        return x * w, w


class ThreeBranch(nn.Module):
    """Case-5 configuration: part-scale stats + layer-wise sequence + process."""

    D_STATS_OUT, D_SEQ_OUT, D_PROC_OUT = 32, 128, 16

    def __init__(self, d_stats, d_seq, d_proc):
        super().__init__()
        self.mlp_s = nn.Sequential(nn.Linear(d_stats, 64), nn.ReLU(),
                                   nn.Linear(64, 32))
        self.cnn = nn.Conv1d(d_seq, 32, 3, padding=1)
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.mlp_p = nn.Sequential(nn.Linear(d_proc, 16), nn.ReLU())
        D = self.D_STATS_OUT + self.D_SEQ_OUT + self.D_PROC_OUT
        self.fuse = Attn(D)
        self.head = nn.Sequential(nn.Linear(D, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, stats, seq, proc, return_attn=False):
        a = self.mlp_s(stats)
        z = self.cnn(seq.permute(0, 2, 1)).permute(0, 2, 1)
        z, _ = self.lstm(z)
        b = z.max(1).values
        c = self.mlp_p(proc)
        x = torch.cat([a, b, c], 1)
        x, w = self.fuse(x)
        y = self.head(x)
        return (y, w) if return_attn else y


# ----------------------------------------------------------------------------
# 7. Compact trainer + metrics
# ----------------------------------------------------------------------------
def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train_model(tr, epochs=60, lr=1e-3, batch=32, device=None, verbose=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = ThreeBranch(tr["stats"].shape[1], tr["seq"].shape[2],
                      tr["proc"].shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = [torch.tensor(tr[k]).to(device) for k in ("stats", "seq", "proc")]
    Y = torch.tensor(tr["y"]).to(device)
    n = len(Y)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(net(X[0][idx], X[1][idx], X[2][idx]), Y[idx])
            loss.backward(); opt.step()
        if verbose and (ep + 1) % 20 == 0:
            print(f"  epoch {ep+1:3d}  train MSE {loss.item():.4f}")
    return net


def predict(net, te, sc_y, device=None, return_attn=False):
    device = device or next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        args = [torch.tensor(te[k]).to(device) for k in ("stats", "seq", "proc")]
        out = net(*args, return_attn=return_attn)
    if return_attn:
        y, w = out
        return sc_y.inverse_transform(y.cpu().numpy()), w.cpu().numpy()
    return sc_y.inverse_transform(out.cpu().numpy())


def metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, r2_score
    rows = {}
    for i, l in enumerate(LABEL_COLS):
        rows[l] = dict(
            RMSE=float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            R2=float(r2_score(y_true[:, i], y_pred[:, i])))
    return pd.DataFrame(rows).T
