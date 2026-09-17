# -*- coding: utf-8 -*-
"""Generate the four teaching notebooks (nbformat). Run from repo root."""
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import nbformat as nbf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NBDIR = os.path.join(ROOT, "notebooks")
os.makedirs(NBDIR, exist_ok=True)

BOOT = '''\
# --- Setup (works both locally and on Google Colab) ---------------------------
import os, sys

if "google.colab" in sys.modules and not os.path.exists("src"):
    !git clone -q https://github.com/wangyiquan1010-coder/IUM_teaching_colab.git
    %cd IUM_teaching_colab

# find the repo root (folder that contains src/) from wherever we run
_here = os.getcwd()
while not os.path.isdir(os.path.join(_here, "src")):
    _parent = os.path.dirname(_here)
    if _parent == _here:
        raise FileNotFoundError("repo root with src/ not found")
    _here = _parent
os.chdir(_here)
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc("animation", html="jshtml")     # render animations as interactive players
import ium_utils as iu
print("setup ok — repo root:", _here)
'''


def nb(cells, title):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md"
               else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


# ==============================================================================
# Notebook 1
# ==============================================================================
nb1 = [
("md", """\
# 01 · Ultrasound Meets 3D Printing — Listening to a Part Being Born

**IUM teaching series, Notebook 1/4** · based on: Wang, Y., & Zhao, X. (2026).
*Machine learning-aided in-situ ultrasonic characterization for multi-parametric
monitoring of vat photopolymerization.* **Additive Manufacturing**, 105300.
[doi:10.1016/j.addma.2026.105300](https://doi.org/10.1016/j.addma.2026.105300)

## Why this matters

Vat photopolymerization (VPP/DLP) prints polymer parts layer by layer with
projected light — fast and precise, but **blind**: the three states that decide
part quality (degree of conversion, storage modulus, thickness) are invisible
during the build and are normally measured only *after* printing, destructively.

This series shows how a **10 MHz ultrasonic transducer embedded in the printhead**
turns the printer into its own sensor. In this notebook you will:

1. load real ultrasonic A-scans recorded *during* printing;
2. learn to read the echoes (B1 / B2 / B3);
3. watch the part grow — literally — through waveform animations;
4. extract your first physical measurement: the **time of flight (ToF)**.

![system](figs/system_overview.png)
*The DLP-IUM platform: the Rexolite printhead doubles as the transducer's delay
line, so the part is monitored continuously without interrupting printing.*
"""),
("code", BOOT),
("md", """\
## 1. The data

We use three printed samples (a small teaching subset of the paper's 50-sample
dataset). Each `.npz` holds all layers of one print; each layer has **6 frames**:
one reference waveform plus five captured during the 15 s exposure (3 s apart).

| file | layers | exposure intensity |
|---|---|---|
| `waveforms_15L_I1.npz` | 15 | I1 = 10.70 mW/cm² (weak cure) |
| `waveforms_15L_I7.npz` | 15 | I7 = 20.16 mW/cm² (strong cure) |
| `waveforms_30L_I7.npz` | 30 | I7 = 20.16 mW/cm² |
"""),
("code", """\
wf_i1, meta_i1 = iu.load_sample("data/waveforms_15L_I1.npz")
wf_i7, meta_i7 = iu.load_sample("data/waveforms_15L_I7.npz")
wf_30, meta_30 = iu.load_sample("data/waveforms_30L_I7.npz")
t_us = iu.time_axis_us()
print("shape (layers, frames, points):", wf_i1.shape)
print("sampling: 2.5 GHz  ->  dt = 0.4 ns;  window =", round(t_us[-1], 1), "us")
"""),
("md", """\
## 2. Anatomy of one A-scan

The pulse travels: transducer → printhead (delay line) → growing sample → resin
vat bottom, and back. Three echo families matter:

- **B1** — the printhead's internal round trip. It never moves: a built-in reference.
- **B2** — first round trip through the growing sample: **the information carrier**.
- **B3** — a second, weaker round trip (arrives ≈ 2×ToF after B1): redundant.
"""),
("code", """\
# Track B1/B2 through the whole 30-layer print (details of the tracker in NB2),
# then look at one A-scan with the tracked picks.
track30 = iu.track_sample(wf_30)

x = wf_30[14, -1]                       # layer 15 of the 30-layer print
b1 = int(track30.b1_idx[14])
b2 = int(track30.b2_idx[14])
b3_approx = b1 + 2 * (b2 - b1)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_us, x, lw=0.5, color="k")
for idx, name, c in [(b1, "B1 (printhead)", "tab:red"),
                     (b2, "B2 (through sample)", "tab:green"),
                     (b3_approx, "≈B3 (2nd round trip)", "tab:blue")]:
    ax.axvline(t_us[idx], color=c, ls="--", lw=1.2)
    ax.text(t_us[idx], ax.get_ylim()[1]*0.9, "  " + name, color=c, fontsize=10)
ax.set(xlim=(5, 10), xlabel="time (µs)", ylabel="amplitude (a.u.)",
       title="One A-scan during printing (30-layer sample, layer 15)")
plt.tight_layout(); plt.show()

tof_us = (b2 - b1) * iu.DT_NS / 1000
print(f"ToF(B1→B2) = {tof_us:.3f} µs")
"""),
("md", """\
## 3. The classic waterfall — 30 layers stacked

Plot each layer's final waveform with a vertical offset. **B1 stays put; B2
marches right** as the part grows ~100 µm per layer.
"""),
("code", """\
fig, ax = plt.subplots(figsize=(11, 8))
sl = slice(13800, 21500, 3)                      # zoom to the echo region
for li in range(wf_30.shape[0]):
    x = wf_30[li, -1, sl].astype(float)
    x = x - np.median(x)                         # remove baseline offset
    ax.plot(t_us[sl], x / 3.2e4 + li + 1, lw=0.6,
            color=plt.cm.viridis(li / (wf_30.shape[0] - 1)))
ax.plot(t_us[track30.b1_idx], track30.layer, "r.-", ms=5, lw=1.2, label="B1 (fixed)")
ax.plot(t_us[track30.b2_idx], track30.layer, "g.-", ms=5, lw=1.2, label="B2 (advancing)")
ax.set(xlim=(t_us[sl][0], t_us[sl][-1]), ylim=(-0.8, wf_30.shape[0] + 2),
       xlabel="time (µs)", ylabel="layer index",
       title="Waterfall: the part growing, one echo at a time (color = layer)")
ax.legend(loc="upper left"); plt.tight_layout(); plt.show()
"""),
("md", """\
### Interactive 3D waterfall (rotate me!)

The same data as a rotatable 3D figure (plotly). Drag to rotate, scroll to zoom.
"""),
("code", """\
import plotly.graph_objects as go

figly = go.Figure()
sl = slice(12000, 24000, 8)                      # zoom to the echo region
for li in range(wf_30.shape[0]):
    x = wf_30[li, -1, sl]
    figly.add_trace(go.Scatter3d(
        x=t_us[sl], y=np.full(len(x), li + 1), z=x,
        mode="lines", line=dict(color="black", width=1.2), showlegend=False))
figly.add_trace(go.Scatter3d(
    x=t_us[track30.b2_idx], y=track30.layer,
    z=[wf_30[li, -1, b] for li, b in enumerate(track30.b2_idx)],
    mode="lines+markers", line=dict(color="green", width=4),
    marker=dict(size=3), name="B2 track"))
figly.update_layout(scene=dict(xaxis_title="time (µs)", yaxis_title="layer",
                               zaxis_title="amplitude"),
                    height=560, margin=dict(l=0, r=0, t=30, b=0),
                    title="30-layer print — interactive waterfall")
figly.show()
"""),
("md", """\
## 4. Watch the part grow — waveform animation

Now the same story as a movie: one frame per layer, with the tracked B1/B2
markers and a live ToF readout. (This is the visual proof that the acoustic
path grows layer by layer.)
"""),
("code", """\
from matplotlib.animation import FuncAnimation

sl = slice(12000, 24000, 6)
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], lw=0.6, color="k")
vb1 = ax.axvline(0, color="tab:red", ls="--", lw=1.2)
vb2 = ax.axvline(0, color="tab:green", ls="--", lw=1.2)
txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=12)
ax.set(xlim=(t_us[sl][0], t_us[sl][-1]), ylim=(-4e4, 4e4),
       xlabel="time (µs)", ylabel="amplitude",
       title="30-layer print — layer-by-layer echo evolution")

def update(li):
    line.set_data(t_us[sl], wf_30[li, -1, sl])
    r = track30.iloc[li]
    vb1.set_xdata([t_us[int(r.b1_idx)]]); vb2.set_xdata([t_us[int(r.b2_idx)]])
    txt.set_text(f"layer {li+1:2d}/30    ToF = {r.tof_us:.3f} µs")
    return line, vb1, vb2, txt

ani = FuncAnimation(fig, update, frames=wf_30.shape[0], interval=280, blit=False)
plt.close(fig)
ani
"""),
("md", """\
### The race: weak cure (I1) vs strong cure (I7)

Two 15-layer prints, same geometry, different exposure intensity — side by side.
Watch the **green marker (B2)**: at the same layer index the strongly cured
sample returns its echo *earlier*, because curing raises the wave velocity.

> **Thickness moves B2 right; cure moves it left.** Two causes, one echo — this
> entanglement is exactly what the machine-learning part of the series must
> untangle.
"""),
("code", """\
track_i1 = iu.track_sample(wf_i1)
track_i7 = iu.track_sample(wf_i7)

sl = slice(13500, 20000, 6)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
arts = []
for ax, name in zip(axes, ["I1 = 10.70 mW/cm² (weak)", "I7 = 20.16 mW/cm² (strong)"]):
    ln, = ax.plot([], [], lw=0.6, color="k")
    v1 = ax.axvline(0, color="tab:red", ls="--", lw=1.1)
    v2 = ax.axvline(0, color="tab:green", ls="--", lw=1.4)
    tx = ax.text(0.03, 0.9, "", transform=ax.transAxes, fontsize=11)
    ax.set(xlim=(t_us[sl][0], t_us[sl][-1]), ylim=(-4e4, 4e4),
           xlabel="time (µs)", title=name)
    arts.append((ln, v1, v2, tx))
axes[0].set_ylabel("amplitude")

def update2(li):
    out = []
    for (ln, v1, v2, tx), wf, tr in [(arts[0], wf_i1, track_i1),
                                     (arts[1], wf_i7, track_i7)]:
        ln.set_data(t_us[sl], wf[li, -1, sl])
        r = tr.iloc[li]
        v1.set_xdata([t_us[int(r.b1_idx)]]); v2.set_xdata([t_us[int(r.b2_idx)]])
        tx.set_text(f"layer {li+1}   ToF {r.tof_us:.3f} µs")
        out += [ln, v1, v2, tx]
    return out

ani2 = FuncAnimation(fig, update2, frames=wf_i1.shape[0], interval=380, blit=False)
plt.close(fig)
ani2
"""),
("md", """\
## 5. Explore for yourself — interactive waveform browser

Use the sliders to pick a sample, a layer, and a frame. (On Colab the widgets
are live; in a static render you see one snapshot.)
"""),
("code", """\
import ipywidgets as w

SAMPLES = {"15 layers, I1": (wf_i1, track_i1),
           "15 layers, I7": (wf_i7, track_i7),
           "30 layers, I7": (wf_30, track30)}

def browse(sample="30 layers, I7", layer=1, frame=6):
    wf, tr = SAMPLES[sample]
    layer = min(layer, wf.shape[0])
    x = wf[layer - 1, frame - 1]
    r = tr.iloc[layer - 1]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(t_us[::4], x[::4], lw=0.5, color="k")
    ax.axvline(t_us[int(r.b1_idx)], color="tab:red", ls="--", label="B1")
    ax.axvline(t_us[int(r.b2_idx)], color="tab:green", ls="--", label="B2")
    ax.set(xlim=(5, 10), xlabel="time (µs)", ylabel="amplitude",
           title=f"{sample} — layer {layer}, frame {frame}  |  ToF = {r.tof_us:.3f} µs")
    ax.legend(loc="upper right"); plt.tight_layout(); plt.show()

w.interact(browse, sample=list(SAMPLES), layer=(1, 30, 1), frame=(1, 6, 1));
"""),
("md", """\
## 6. First quantitative result: ToF vs layer

The simplest possible "monitor": track ToF layer by layer. It is nearly linear
(path grows ~100 µm/layer), and the *slope difference* between I1 and I7 already
carries material information (velocity).
"""),
("code", """\
fig, ax = plt.subplots(figsize=(7, 4.2))
for tr, name, c in [(track_i1, "15L @ I1 (weak cure)", "tab:blue"),
                    (track_i7, "15L @ I7 (strong cure)", "tab:orange"),
                    (track30, "30L @ I7", "tab:green")]:
    ax.plot(tr.layer, tr.tof_us, "o-", ms=4, color=c, label=name)
ax.set(xlabel="layer index", ylabel="ToF(B1→B2)  (µs)",
       title="Time of flight grows with the part — and bends with cure")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
"""),
("md", """\
## Exercises

1. Zoom the waterfall into layers 1–3 of the I1 sample. Why is B2 hard to
   separate from B1 for very thin parts? What does this imply for monitoring
   the first layers of any print?
2. Using `track30`, estimate the average ToF increment per layer, and — assuming
   a 100 µm layer thickness — compute the effective sound speed in the part.
   Compare it with water (~1.48 mm/µs). *(hint: ToF is a round trip.)*
3. In the race animation, the I1 sample's B2 amplitude stays higher than I7's
   at late layers. Offer a physical explanation (think attenuation vs cure).

**Next notebook:** we turn these observations into *features* — designed on a
two-scale physical framework.
"""),
]

# ==============================================================================
# Notebook 2
# ==============================================================================
nb2 = [
("md", """\
# 02 · Physics-Informed Features — Telling the Model What to Listen For

**IUM teaching series, Notebook 2/4** · Wang & Zhao, *Additive Manufacturing*
(2026) 105300.

Raw waveforms have 62,509 points; we have only 50 printed samples. Learning
end-to-end from raw signals is hopeless at this scale (the paper proves it —
Case 1 fails). Instead, the paper *designs* features on a physical map:

![two-scale](figs/two_scale_framework.png)

- **Across layers**: each layer lengthens the acoustic path *and* stiffens the
  material → ToF, amplitude ratio, spectral features encode geometry + material
  state, entangled.
- **Within a layer**: during one 15 s exposure the printhead does not move — the
  path is frozen. The six frames differ *only* through crosslinking → frame-to-
  frame change is a **geometry-immune** probe of curing (→ DoC).

This notebook rebuilds the key features from raw data and shows they carry
process signatures *before any learning*.
"""),
("code", BOOT),
("code", """\
wf_i1, _ = iu.load_sample("data/waveforms_15L_I1.npz")
wf_i7, _ = iu.load_sample("data/waveforms_15L_I7.npz")
t_us = iu.time_axis_us()
track_i1 = iu.track_sample(wf_i1)
track_i7 = iu.track_sample(wf_i7)
"""),
("md", """\
## 1. The within-layer scale, animated

Fix one layer. Play its six frames (reference + five during exposure), zoomed
to the B2 echo. The path cannot change — everything you see moving is
**chemistry**.
"""),
("code", """\
from matplotlib.animation import FuncAnimation

LAYER = 8
r1, r7 = track_i1.iloc[LAYER-1], track_i7.iloc[LAYER-1]
w1 = 500

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
arts = []
for ax, (wfS, r, name) in zip(axes, [(wf_i1, r1, "I1 (weak cure)"),
                                     (wf_i7, r7, "I7 (strong cure)")]):
    b2 = int(r.b2_idx)
    seg_t = t_us[b2-w1:b2+w1]
    ln, = ax.plot([], [], lw=1.2)
    tx = ax.text(0.03, 0.9, "", transform=ax.transAxes)
    ax.set(xlim=(seg_t[0], seg_t[-1]), xlabel="time (µs)",
           title=f"{name} — layer {LAYER}, B2 window")
    ax.set_ylim(wfS[LAYER-1, :, b2-w1:b2+w1].min()*1.1,
                wfS[LAYER-1, :, b2-w1:b2+w1].max()*1.1)
    arts.append((ln, tx, wfS, b2))
axes[0].set_ylabel("amplitude")

def upd(f):
    out = []
    for ln, tx, wfS, b2 in arts:
        ln.set_data(t_us[b2-w1:b2+w1], wfS[LAYER-1, f, b2-w1:b2+w1])
        tx.set_text(f"frame {f+1}/6  (t = {max(0,(f-1))*3} s)" if f else "reference")
        out += [ln, tx]
    return out

ani = FuncAnimation(fig, upd, frames=6, interval=600, blit=False)
plt.close(fig)
ani
"""),
("md", """\
The change is subtle by eye — which is precisely why we quantify it with
**within-layer RMS/STD** (frame-to-frame differences in the fixed window).

## 2. Rebuilding the layer-wise features

`ium_utils.layer_features` computes the teaching subset of the paper's 11
descriptors: ToF, amplitude ratio, RMS energy, center frequency, bandwidth,
within-layer RMS and STD.
"""),
("code", """\
def features_for(wf, track):
    rows = []
    for li in range(wf.shape[0]):
        r = track.iloc[li]
        f = iu.layer_features(wf[li], int(r.b1_idx), int(r.b2_idx))
        f["layer"] = li + 1
        rows.append(f)
    return pd.DataFrame(rows).set_index("layer")

F1 = features_for(wf_i1, track_i1)
F7 = features_for(wf_i7, track_i7)
F1.head().round(4)
"""),
("code", """\
names = ["ToF", "amplitude", "Center_Freq", "frame_diff_rms"]
titles = ["ToF (µs) — path + velocity", "B2/B1 amplitude ratio — attenuation",
          "center frequency (Hz) — dispersion", "within-layer RMS — curing dynamics"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, n, ti in zip(axes.ravel(), names, titles):
    ax.plot(F1.index, F1[n], "o-", label="I1 weak", color="tab:blue")
    ax.plot(F7.index, F7[n], "s-", label="I7 strong", color="tab:orange")
    ax.set(title=ti, xlabel="layer"); ax.grid(alpha=0.3)
axes[0, 0].legend()
plt.suptitle("Layer-wise features separate the two intensities — before any ML")
plt.tight_layout(); plt.show()
"""),
("md", """\
**Read the four panels like a physicist:**

- *ToF* rises for both (path), but slower for I7 (faster waves in cured material).
- *Amplitude ratio* decays faster for I7 (stiffer network → more attenuation).
- *Center frequency* separates the materials (frequency-dependent loss).
- *Within-layer RMS* is higher for the weak cure — unstable, incomplete curing.

## 3. Sanity check against the paper's stored features

The repository ships the paper's full 50-sample feature table
(`feature11.csv`, 11 features × 30 layer slots). Compare our recomputed ToF
with the stored one for the same condition (15 layers, I1 = 10%).
"""),
("code", """\
df = iu.load_features("data/feature11.csv")
row = df[(df.layer == 15) & (df.intensity == 10)].iloc[0]
stored_tof = [row[f"Layer_{i}_ToF"] for i in range(1, 16)]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, 16), stored_tof, "k.-", label="stored (paper pipeline)")
ax.plot(F1.index, F1.ToF, "ro--", ms=4, label="ours (this notebook)")
ax.set(xlabel="layer", ylabel="ToF (µs)", title="Recomputed vs stored ToF — 15L @ I1")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
corr = np.corrcoef(stored_tof, F1.ToF)[0, 1]
print(f"correlation: {corr:.4f}")
"""),
("md", """\
## 4. The whole design grid at a glance — feature heatmaps

Now use the full 50-sample table: layer × intensity heatmaps for one layer-count
group (the paper's Fig. 9/11 style). The structure is visible to the naked eye.
"""),
("code", """\
def heat(ax, base, group_layers, title, cmap="viridis"):
    g = df[df.layer == group_layers].sort_values("intensity")
    M = np.array([[r[f"Layer_{i}_{base}"] for i in range(1, group_layers + 1)]
                  for _, r in g.iterrows()]).T          # layers × intensities
    im = ax.imshow(M, aspect="auto", origin="upper", cmap=cmap)
    ax.set(xlabel="intensity I1→I10", ylabel="layer", title=title)
    return im

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, base, ti in zip(axes,
        ["ToF", "amplitude", "frame_diff_rms"],
        ["ToF — grows with depth", "amplitude ratio — falls with dose",
         "within-layer RMS — spikes at weak cure"]):
    im = heat(ax, base, 30, ti)
    plt.colorbar(im, ax=ax, shrink=0.85)
plt.suptitle("30-layer group, all 10 intensities")
plt.tight_layout(); plt.show()
"""),
("md", """\
## Exercises

1. Recreate the heatmaps for the 10-layer group. Where is within-layer RMS
   largest, and why does that match the physics of incomplete curing?
2. `iu.layer_features` uses the **last** frame of each layer for the
   across-layer features. Re-run section 2 using the *reference* frame
   (index 0) instead. Which features change most, and why?
3. Design ONE new feature you believe tracks curing but not geometry. Compute
   it for I1 vs I7 and defend your design in two sentences. *(You will test it
   properly — leak-free — after Notebook 3.)*

**Next notebook:** we assemble the 344-dimensional representation and learn —
without fooling ourselves.
"""),
]

# ==============================================================================
# Notebook 3
# ==============================================================================
nb3 = [
("md", """\
# 03 · Learning Without Fooling Yourself — Attention Fusion and the Leakage Trap

**IUM teaching series, Notebook 3/4** · Wang & Zhao, *Additive Manufacturing*
(2026) 105300.

This is the methodological heart of the series. You will:

1. assemble the paper's **344-dimensional** sample representation;
2. *see* why random cross-validation lies for this dataset (a true story from
   this paper's peer review!);
3. run the honest protocol — **Leave-One-Intensity-Out (LOIO)**;
4. train the paper's deployed **three-branch attention-fusion network**.
"""),
("code", BOOT),
("md", """\
## 1. Assemble the dataset

Each printed sample becomes: 330 layer-wise entries (11 features × 30 layer
slots, zero-padded) + 12 part-scale statistics + 2 printing conditions,
with three labels (thickness, modulus, DoC) from Keyence / rheometer / Raman.
"""),
("code", """\
feats = iu.load_features("data/feature11.csv")
labels = iu.load_labels("data/condition_labels.csv")
assert (feats.layer.values == labels.layer.values).all()
full = feats.copy()
for c in iu.LABEL_COLS:
    full[c] = labels[c]
print(f"{len(full)} samples;  layer counts {sorted(full.layer.unique())};  "
      f"intensities {sorted(full.intensity.unique())} (% of max power)")
full[["layer", "intensity"] + iu.LABEL_COLS].head()
"""),
("md", """\
## 2. Look at the feature space first

PCA of the 330 layer-wise features. **Color = intensity, size = layer count.**
Samples cluster by their process condition — remember this picture: it is the
geometric reason why a random train/test split leaks.
"""),
("code", """\
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

seq_cols = iu.seq_columns(full)
Z = StandardScaler().fit_transform(full[seq_cols].values)
P = PCA(n_components=2).fit(Z)
XY = P.transform(Z)

pdf = pd.DataFrame(dict(PC1=XY[:, 0], PC2=XY[:, 1],
                        intensity=full.intensity.astype(str),
                        layers=full.layer,
                        thickness=full.thickness.round(2),
                        modulus=full.modulus.round(0),
                        DoC=full.DoC.round(3)))
fig = px.scatter(pdf, x="PC1", y="PC2", color="intensity", size="layers",
                 hover_data=["layers", "thickness", "modulus", "DoC"],
                 title=f"Feature space (PCA of layer-wise features) — "
                       f"{P.explained_variance_ratio_[:2].sum():.0%} variance shown")
fig.update_layout(height=520)
fig.show()
"""),
("md", """\
## 3. The leakage trap — a true peer-review story

The paper's first submission used random K-fold CV. A reviewer objected:
*a randomly held-out sample always has same-intensity neighbors in training, so
a model can interpolate from the nominal settings — the ultrasonic features are
never actually tested.* The authors rebuilt the whole evaluation. Let's
reproduce both worlds with a fast Random Forest:
"""),
("code", """\
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import r2_score

X = full[seq_cols + ["layer", "intensity"]].values
X_stats = iu.part_scale_stats(full).values
X_all = np.hstack([X, X_stats])
Y = full[iu.LABEL_COLS].values

def cv_r2(splitter, groups=None):
    preds = np.zeros_like(Y)
    for tr_idx, te_idx in splitter.split(X_all, Y, groups):
        rf = RandomForestRegressor(300, random_state=0, n_jobs=-1)
        rf.fit(X_all[tr_idx], Y[tr_idx])
        preds[te_idx] = rf.predict(X_all[te_idx])
    return [r2_score(Y[:, i], preds[:, i]) for i in range(3)]

r2_random = cv_r2(KFold(5, shuffle=True, random_state=0))
r2_loio = cv_r2(LeaveOneGroupOut(), groups=full.intensity.values)

comp = pd.DataFrame({"random 5-fold": r2_random, "LOIO (honest)": r2_loio},
                    index=iu.LABEL_COLS).round(3)
ax = comp.plot.bar(figsize=(7, 4), rot=0, color=["tab:red", "tab:green"])
ax.set_ylabel("R²"); ax.set_title("The same model, two evaluation protocols")
ax.axhline(0, color="k", lw=0.8); ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.show()
comp
"""),
("md", """\
**The gap you just plotted is the leakage.** Random K-fold flatters every
metric because the test intensity was always seen in training. LOIO forces
*extrapolation to unseen process settings* — which is exactly the deployment
scenario for a real sensor.

> Three leak-free rules used from here on (and in the paper): ① the test
> intensity is absent from training; ② every scaler is fitted on the training
> partition only; ③ any augmentation would be applied to training folds only.

## 4. The deployed model — three-branch attention fusion

Branches: part-scale stats → MLP(32) · layer-wise sequences → CNN+Bi-LSTM(128)
· printing conditions → MLP(16). A sigmoid attention gate weighs the 176-d
fusion vector (readable — Notebook 4), and a shared head predicts all three
targets jointly.
"""),
("code", """\
import torch
print("device:", "cuda" if torch.cuda.is_available() else "cpu",
      " (Colab: Runtime → Change runtime type → GPU makes this ~5× faster)")

EPOCHS = 60      # classroom budget; the paper uses 100 epochs × 5 seeds
iu.set_seed(42)

results, all_pred, all_true = [], [], []
for tr_df, te_df, inten in iu.loio_folds(full):
    a_tr, a_te, sc_y = iu.assemble_arrays(tr_df, te_df)
    net = iu.train_model(a_tr, epochs=EPOCHS)
    yp = iu.predict(net, a_te, sc_y)
    all_pred.append(yp); all_true.append(te_df[iu.LABEL_COLS].values)
    fold_r2 = iu.metrics(te_df[iu.LABEL_COLS].values, yp)["R2"]
    results.append(dict(intensity=inten, **fold_r2))
    print(f"held-out intensity {inten:3d}%:  "
          + "  ".join(f"{l} R²={fold_r2[l]:+.2f}" for l in iu.LABEL_COLS))

YP, YT = np.vstack(all_pred), np.vstack(all_true)
print("\\n=== LOIO aggregate (all 50 out-of-fold predictions) ===")
agg = iu.metrics(YT, YP).round(3)
agg
"""),
("md", """\
Notes on what you should see (numbers vary a little with hardware/seed):

- **Thickness** R² ≈ 0.97+ — geometry lives in ToF; easy.
- **DoC** R² ≈ 0.6–0.75 — the within-layer probes carry it.
- **Modulus** is hardest, and the *edge folds* (I1, I10) are the worst — the
  model must extrapolate beyond the calibrated intensity range. The paper's
  full protocol (100 epochs, 5 seeds, mild noise augmentation) reports
  0.985 / 0.832 / 0.757.
"""),
("code", """\
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for i, (ax, l, unit) in enumerate(zip(axes, iu.LABEL_COLS, ["mm", "Pa", ""])):
    ax.scatter(YT[:, i], YP[:, i], s=28, alpha=0.7,
               c=full.intensity.values, cmap="viridis")
    lo, hi = YT[:, i].min(), YT[:, i].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlabel=f"measured {l} ({unit})", ylabel=f"predicted {l}",
           title=f"{l}:  R² = {agg.loc[l, 'R2']:.3f}")
plt.suptitle("LOIO out-of-fold predictions (color = exposure intensity)")
plt.tight_layout(); plt.show()
"""),
("md", """\
## Exercises

1. Re-run the leakage comparison with `groups=full.layer.values`
   (leave-one-layer-count-out). Which protocol is harder, LOIO or LOLO, and
   what does each simulate in deployment?
2. Drop the two printing-condition columns from `X_all` in section 3 and
   re-run both protocols. How much of the "random-split" performance was the
   recipe alone? *(This is the paper's Case-2-vs-Case-5 question.)*
3. Train the network with `EPOCHS=20` and `EPOCHS=100`. Which target benefits
   most from longer training, and why might that be?

**Next notebook:** opening the black box — attention gates, feature importance,
and what it takes to deploy.
"""),
]

# ==============================================================================
# Notebook 4
# ==============================================================================
nb4 = [
("md", """\
# 04 · Opening the Black Box — Interpretation and the Road to Deployment

**IUM teaching series, Notebook 4/4** · Wang & Zhao, *Additive Manufacturing*
(2026) 105300.

A sensor you cannot interpret is a sensor you cannot trust. In this notebook:

1. read the network's **attention gates** — does ultrasound actually contribute?
2. rank features per target and check the ranking **matches the physics**;
3. probe **noise robustness**;
4. play with a **virtual sensor dashboard** — and see why this model is the
   feedback signal for closed-loop printing.
"""),
("code", BOOT),
("code", """\
feats = iu.load_features("data/feature11.csv")
labels = iu.load_labels("data/condition_labels.csv")
full = feats.copy()
for c in iu.LABEL_COLS:
    full[c] = labels[c]
seq_cols = iu.seq_columns(full)
"""),
("md", """\
## 1. What do the attention gates say?

Train on one LOIO fold and read the sigmoid gates of the fusion layer.
Gate positions 0–31 = part-scale stats branch, 32–159 = layer-wise branch,
160–175 = printing-condition branch.
"""),
("code", """\
iu.set_seed(42)
folds = list(iu.loio_folds(full))
tr_df, te_df, inten = folds[4]              # hold out a mid-range intensity
a_tr, a_te, sc_y = iu.assemble_arrays(tr_df, te_df)
net = iu.train_model(a_tr, epochs=60)
yp, gates = iu.predict(net, a_te, sc_y, return_attn=True)
print("gates shape (test samples × fusion dims):", gates.shape)

seg = dict(stats=(0, 32), layerwise=(32, 160), process=(160, 176))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8),
                             gridspec_kw={"width_ratios": [2.4, 1]})
im = a1.imshow(gates, aspect="auto", cmap="viridis", vmin=0, vmax=1)
for name, (lo, hi) in seg.items():
    a1.axvline(hi - 0.5, color="w", lw=1)
    a1.text((lo + hi) / 2, -0.8, name, ha="center", fontsize=10)
a1.set(xlabel="fusion-vector channel", ylabel="test sample",
       title=f"Sigmoid gate activations (held-out intensity {inten}%)")
plt.colorbar(im, ax=a1, shrink=0.8)

means = {n: float(gates[:, lo:hi].mean()) for n, (lo, hi) in seg.items()}
tot = sum(means.values())
a2.bar(means.keys(), [v / tot * 100 for v in means.values()],
       color=["tab:blue", "tab:orange", "tab:green"])
a2.set(ylabel="relative contribution (%)", title="Branch-level mean gate")
a2.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.show()
print({k: f"{v/tot:.1%}" for k, v in means.items()})
"""),
("md", """\
In the paper (Fig. 16), the two **IUM-derived branches together carry ≈57%** of
the fusion contribution — more than the nominal printing conditions. That is
the quantitative answer to *"does sensing add information beyond the recipe?"*

> Caveat you should teach: sigmoid gates are per-element, not a softmax — these
> percentages are a relative ranking, not probability mass.

## 2. Feature importance, target by target

A gradient-boosting surrogate gives fast per-target rankings (the paper's
Fig. 17 uses XGBoost; sklearn's HistGradientBoosting behaves similarly).
"""),
("code", """\
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

# compact, physical feature set: per-feature layer-averages + process + stats
BASES = ["ToF", "amplitude", "RMS_Energy", "Center_Freq", "Bandwidth",
         "Wavelet_Energy_L0", "Wavelet_Energy_L1", "Wavelet_Energy_L2",
         "Wavelet_Energy_L3", "frame_diff_rms", "within_layer_std"]

def collapsed(df):
    out = pd.DataFrame(index=df.index)
    n_real = df.layer.values.astype(int)
    for b in BASES:
        cols = [f"Layer_{i}_{b}" for i in range(1, 31)]
        V = df[cols].values
        out[b] = [row[:n].mean() for row, n in zip(V, n_real)]
    out["layer_count"] = df.layer.values
    out["intensity"] = df.intensity.values
    return out

Xc = collapsed(full)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, target in zip(axes, iu.LABEL_COLS):
    m = HistGradientBoostingRegressor(random_state=0).fit(Xc, full[target])
    imp = permutation_importance(m, Xc, full[target], n_repeats=10,
                                 random_state=0)
    order = np.argsort(imp.importances_mean)[-8:]
    ax.barh(np.array(Xc.columns)[order], imp.importances_mean[order],
            color="tab:blue")
    ax.set_title(target)
plt.suptitle("Permutation importance (top 8) — compare with the physics!")
plt.tight_layout(); plt.show()
"""),
("md", """\
**Check against the two-scale framework:**

- *thickness* ← layer_count and the ToF family (geometry lives in arrival times);
- *modulus* ← spectral features (center frequency / bandwidth — dispersion);
- *DoC* ← within-layer dynamics + intensity (the geometry-immune cure probes).

The model recovered the division of labor the features were designed for.

## 3. How fragile is it? Noise robustness
"""),
("code", """\
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score

X_all = np.hstack([full[seq_cols + ["layer", "intensity"]].values,
                   iu.part_scale_stats(full).values])
Y = full[iu.LABEL_COLS].values
scale = np.abs(X_all).mean(axis=0)

def loio_r2_with_noise(noise_pct, n_rep=5, seed0=0):
    r2s = []
    for rep in range(n_rep):
        rng = np.random.default_rng(seed0 + rep)
        preds = np.zeros_like(Y)
        for tr_idx, te_idx in LeaveOneGroupOut().split(
                X_all, Y, full.intensity.values):
            rf = RandomForestRegressor(200, random_state=0, n_jobs=-1)
            rf.fit(X_all[tr_idx], Y[tr_idx])
            Xte = X_all[te_idx] + rng.normal(
                0, noise_pct / 100 * scale, X_all[te_idx].shape)
            preds[te_idx] = rf.predict(Xte)
        r2s.append([r2_score(Y[:, i], preds[:, i]) for i in range(3)])
    return np.array(r2s)

levels = [0, 1, 2, 5, 10, 20]
curves = {l: loio_r2_with_noise(l) for l in levels}

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for i, l in enumerate(iu.LABEL_COLS):
    mean = [curves[nv][:, i].mean() for nv in levels]
    sd = [curves[nv][:, i].std() for nv in levels]
    ax.errorbar(levels, mean, yerr=sd, marker="o", capsize=3, label=l)
ax.set(xlabel="per-feature noise level (%)", ylabel="LOIO R²",
       title="Graceful degradation — no cliff below ~10% noise")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),
("md", """\
## 4. The virtual sensor dashboard

Honest out-of-fold predictions for every sample (each was predicted by a model
that never saw its intensity). Pick a condition and compare sensor vs truth —
this is what a *soft sensor* delivers, in real time (2.3–4.7 ms/layer in the
paper's deployment test).
"""),
("code", """\
# cache out-of-fold RF predictions for all 50 samples (fast)
preds = np.zeros_like(Y)
for tr_idx, te_idx in LeaveOneGroupOut().split(X_all, Y, full.intensity.values):
    rf = RandomForestRegressor(300, random_state=0, n_jobs=-1)
    rf.fit(X_all[tr_idx], Y[tr_idx])
    preds[te_idx] = rf.predict(X_all[te_idx])

import ipywidgets as w

def dashboard(layers=15, intensity_pct=70):
    m = (full.layer == layers) & (full.intensity == intensity_pct)
    if not m.any():
        print("no such sample"); return
    i = int(np.where(m)[0][0])
    meas, pred = Y[i], preds[i]
    names = ["thickness (mm)", "modulus (Pa)", "DoC (–)"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3))
    for ax, n, mv, pv in zip(axes, names, meas, pred):
        ax.bar(["measured", "predicted"], [mv, pv],
               color=["0.4", "tab:green"])
        err = abs(pv - mv) / (abs(mv) + 1e-9) * 100
        ax.set_title(f"{n}\\nerror {err:.1f}%")
    plt.suptitle(f"{layers} layers @ {intensity_pct}% intensity "
                 f"(I = {iu.INTENSITY_MW[intensity_pct]} mW/cm²) — "
                 "prediction from a model that never saw this intensity")
    plt.tight_layout(); plt.show()

w.interact(dashboard, layers=sorted(full.layer.unique()),
           intensity_pct=sorted(full.intensity.unique()));
"""),
("md", """\
## 5. Where this goes: closing the loop

A validated, fast, interpretable estimate of DoC and thickness **per layer** is
exactly a feedback signal. The authors' ongoing work uses this sensor family
with grayscale exposure as the actuator to close the control loop on the same
printer — toward autonomous, defect-free photopolymer additive manufacturing.

## Exercises

1. In section 1, hold out an *edge* intensity (fold 0 or 9) instead of a
   mid-range one. How do the gates and the errors change, and why?
2. The dashboard uses out-of-fold predictions. Explain, in three sentences,
   why re-using a model trained on *all* data would overstate the sensor.
3. **Mini-capstone:** take your feature from Notebook 2's exercise 3, add it to
   `X_all`, and report the LOIO ΔR² per target with error bars over 5 seeds.
   Does it beat the recipe?

---
*Citation: Wang, Y., & Zhao, X. (2026). Machine learning-aided in-situ
ultrasonic characterization for multi-parametric monitoring of vat
photopolymerization. Additive Manufacturing, 105300.*
"""),
]

for fname, cells, title in [
        ("01_ultrasound_meets_3dprinting.ipynb", nb1, "NB1"),
        ("02_physics_informed_features.ipynb", nb2, "NB2"),
        ("03_learning_and_leakage.ipynb", nb3, "NB3"),
        ("04_interpretation_and_deployment.ipynb", nb4, "NB4")]:
    path = os.path.join(NBDIR, fname)
    nbf.write(nb(cells, title), path)
    print("wrote", fname, f"({len(cells)} cells)")
