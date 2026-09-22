# -*- coding: utf-8 -*-
"""Generate the four teaching notebooks (nbformat). Run from repo root."""
import io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import nbformat as nbf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NBDIR = os.path.join(ROOT, "notebooks")
os.makedirs(NBDIR, exist_ok=True)

# Figures shown in markdown cells use absolute raw-GitHub URLs on purpose: the
# notebooks live in notebooks/, so a relative "figs/..." path would resolve to
# notebooks/figs/... , and on Colab a GitHub-opened notebook has no local
# filesystem context at all, so relative paths never render.

BOOT = '''\
# --- Setup (works both locally and on Google Colab) ---------------------------
# On Colab the notebook starts in an empty machine, so we first download
# ("clone") the whole course repository, which contains the data, the helper
# library and the figures. Locally the files are already next to the notebook,
# so the clone is skipped.
import os, sys

if "google.colab" in sys.modules and not os.path.exists("src"):
    !git clone -q https://github.com/wangyiquan1010-coder/IUM_teaching_colab.git
    %cd IUM_teaching_colab

# Walk up the folder tree until we find the repository root (the folder that
# contains src/). This makes every later path such as "data/feature11.csv"
# work no matter where the notebook was launched from.
_here = os.getcwd()
while not os.path.isdir(os.path.join(_here, "src")):
    _parent = os.path.dirname(_here)
    if _parent == _here:
        raise FileNotFoundError("repo root with src/ not found")
    _here = _parent
os.chdir(_here)
sys.path.insert(0, "src")            # so that "import ium_utils" works

import numpy as np                   # arrays / signal math
import pandas as pd                  # tables (features, labels, results)
import matplotlib.pyplot as plt      # static plots
from matplotlib import rc
rc("animation", html="jshtml")       # render animations as interactive players
import ium_utils as iu               # OUR course library (see src/ium_utils.py)
print("setup ok — repo root:", _here)
'''

SETUP_MD = """\
### Before we run anything: the setup cell

**Input** — nothing from your side. On Google Colab this cell downloads the
course repository (`github.com/wangyiquan1010-coder/IUM_teaching_colab`), which
ships *all* the data used in this series; running locally, it just finds the
repository folder you already have.

**What this cell does** — clones the repo if needed, moves the working
directory to the repository root (so that every later path like
`data/feature11.csv` resolves), and imports the four things we use everywhere:
`numpy` (signal math), `pandas` (tables), `matplotlib` (plots) and
`ium_utils` — **our own teaching library**, a clean, read-only refactor of the
paper's code that lives in `src/ium_utils.py`. Feel free to open that file at
any point: every function used in the notebooks is defined there, with comments.

**Output** — the imported modules plus a confirmation line printing the
repository root. Nothing is computed yet.
"""


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

![system](https://raw.githubusercontent.com/wangyiquan1010-coder/IUM_teaching_colab/main/figs/system_overview.png)
*The DLP-IUM platform: the Rexolite printhead doubles as the transducer's delay
line, so the part is monitored continuously without interrupting printing.*
"""),
("md", """\
## Key terms before we start

If you have never worked with ultrasound, these five words are all you need:

| term | meaning in this notebook |
|---|---|
| **A-scan** | one recorded waveform: amplitude versus time, from a single ultrasonic pulse. Here each A-scan has 62,509 points sampled at 2.5 GHz, i.e. one point every **0.4 ns**, covering ~25 µs. |
| **frame** | one A-scan captured at one moment. We record **6 frames per printed layer**: a reference before the light turns on, then five during the 15 s exposure (3 s apart). |
| **echo (B1, B2, B3)** | a reflected pulse returning from an interface. B1 comes back from the printhead's own bottom face, B2 from the bottom of the resin vat after travelling through the part, B3 is the same trip made twice. |
| **ToF (time of flight)** | the time between two echoes, here B1 → B2, in microseconds. It is our first physical measurement: it grows as the part grows and shrinks as the material cures and becomes faster. |
| **round trip** | the pulse goes *down and back*, so a ToF of 1 µs corresponds to **twice** the physical path. Always divide by 2 before converting ToF into a distance. |
"""),
("md", SETUP_MD),
("code", BOOT),
("md", """\
## 1. The data — where the waveforms come from

The paper's dataset contains 50 printed cylinders (5 layer counts × 10 exposure
intensities). Shipping every raw waveform would be hundreds of megabytes, so
this course repository carries **three representative prints**, already
converted from the original oscilloscope CSV files into compressed `.npz`
archives (see `tools/extract_teaching_data.py` and `DATA_CARD.md` for the exact
provenance — the original dataset was never modified).

| file | layers | exposure intensity |
|---|---|---|
| `waveforms_15L_I1.npz` | 15 | I1 = 10.70 mW/cm² (weak cure) |
| `waveforms_15L_I7.npz` | 15 | I7 = 20.16 mW/cm² (strong cure) |
| `waveforms_30L_I7.npz` | 30 | I7 = 20.16 mW/cm² |

Two of them are printed with the **same geometry but different light intensity**
(15L @ I1 vs 15L @ I7) — that pair is our controlled comparison throughout the
notebook. The third is a taller print used for the growth animations.

---

**Input** — the three `.npz` files in `data/`. Each archive stores one array per
printed layer plus a small JSON metadata record (layer count, intensity in % of
maximum lamp power and in mW/cm², sampling rate).

**What this cell does** — `iu.load_sample()` reads an archive and stacks the
layers into one array of shape **(layers, frames, points)**. It also fixes a
quirk of the dataset: the *first* layer of every print is exposed for 21 s
instead of 15 s and therefore has 7 frames, so the loader keeps the reference
frame plus the last five to make every layer directly comparable.
`iu.time_axis_us()` builds the matching time axis in microseconds.

**Output** — `wf_i1`, `wf_i7`, `wf_30` (raw amplitudes, arbitrary oscilloscope
units), their metadata dictionaries, and `t_us` — the shared time axis. These
variables are used by every remaining cell of this notebook.
"""),
("code", """\
# Each call returns (waveforms, metadata).
#   waveforms : float array of shape (n_layers, 6 frames, 62509 points)
#   metadata  : dict with n_layers, intensity_pct, intensity_mw_cm2, fs_hz ...
wf_i1, meta_i1 = iu.load_sample("data/waveforms_15L_I1.npz")   # 15 layers, weak cure
wf_i7, meta_i7 = iu.load_sample("data/waveforms_15L_I7.npz")   # 15 layers, strong cure
wf_30, meta_30 = iu.load_sample("data/waveforms_30L_I7.npz")   # 30 layers, strong cure

# Time axis shared by all waveforms: point index -> microseconds.
# 2.5 GHz sampling means one point every 1/2.5e9 s = 0.4 ns.
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

---

**Input** — `wf_30`, the 30-layer print loaded above. We will look at **one**
waveform out of its 180: layer 15, final frame (`wf_30[14, -1]`, remembering
that Python counts from 0).

**What this cell does** — first it runs the echo tracker over the *whole* print,
because B2 can only be located reliably by following it from layer to layer
(picking it blindly in a single waveform is what makes beginners land on B3 by
mistake — the tracker is dissected in Notebook 2). Then it plots the chosen
A-scan and marks the tracked B1 and B2 positions, plus the expected position of
B3, and converts the B1→B2 index difference into a time of flight.

**Output** — `track30`, a small table with `layer`, `b1_idx`, `b2_idx` and
`tof_us` for each of the 30 layers (reused by the waterfall, the animation and
the widget), the annotated A-scan figure, and a printed ToF of ≈1.29 µs.
"""),
("code", """\
# Track B1/B2 through the whole 30-layer print. The tracker locates B1 inside a
# fixed index window (the printhead round trip never changes) and then follows
# B2 forward layer by layer, allowing it to advance by at most ~800 samples per
# layer so that it can never jump onto the later, weaker B3 echo.
track30 = iu.track_sample(wf_30)

x = wf_30[14, -1]                       # layer 15 (0-based index 14), final frame
b1 = int(track30.b1_idx[14])            # tracked B1 position, in sample index
b2 = int(track30.b2_idx[14])            # tracked B2 position, in sample index
b3_approx = b1 + 2 * (b2 - b1)          # B3 = the same trip made twice

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_us, x, lw=0.5, color="k")
for idx, name, c in [(b1, "B1 (printhead)", "tab:red"),
                     (b2, "B2 (through sample)", "tab:green"),
                     (b3_approx, "≈B3 (2nd round trip)", "tab:blue")]:
    ax.axvline(t_us[idx], color=c, ls="--", lw=1.2)
    ax.text(t_us[idx], ax.get_ylim()[1]*0.9, "  " + name, color=c, fontsize=10)
# Zoom to 5-10 us: everything before 5 us is the transmit pulse ringing down.
ax.set(xlim=(5, 10), xlabel="time (µs)", ylabel="amplitude (a.u.)",
       title="One A-scan during printing (30-layer sample, layer 15)")
plt.tight_layout(); plt.show()

# Index difference -> nanoseconds (0.4 ns per sample) -> microseconds.
tof_us = (b2 - b1) * iu.DT_NS / 1000
print(f"ToF(B1→B2) = {tof_us:.3f} µs")
"""),
("md", """\
## 3. The classic waterfall — 30 layers stacked

Plot each layer's final waveform with a vertical offset. **B1 stays put; B2
marches right** as the part grows ~100 µm per layer.

---

**Input** — `wf_30` again (one trace per layer: the *final* frame, i.e. the end
of that layer's exposure) together with the tracked positions in `track30`.

**What this cell does** — draws all 30 traces in one axes, each shifted upward
by its layer number so they do not overlap, and overlays the two tracked echo
positions as connected markers. Two small tricks make the plot readable: we zoom
to the index range that actually contains the echoes, and we subtract each
trace's baseline before scaling it.

**Output** — one static figure. Read it as a picture of the whole print: the
vertical red line is the unchanged printhead echo, the green diagonal is the
part growing.
"""),
("code", """\
fig, ax = plt.subplots(figsize=(11, 8))
# Zoom to the echo region (index 13800-21500 ≈ 5.5-8.6 µs) and keep every 3rd
# point — plotting all 62509 points × 30 layers would be slow and unreadable.
sl = slice(13800, 21500, 3)
for li in range(wf_30.shape[0]):
    x = wf_30[li, -1, sl].astype(float)
    x = x - np.median(x)                         # remove the DC baseline offset
    # Divide by a typical peak amplitude (~3.2e4) so one trace spans about one
    # unit, then shift it up by its layer number: that is the "waterfall" trick.
    ax.plot(t_us[sl], x / 3.2e4 + li + 1, lw=0.6,
            color=plt.cm.viridis(li / (wf_30.shape[0] - 1)))   # color = layer
ax.plot(t_us[track30.b1_idx], track30.layer, "r.-", ms=5, lw=1.2, label="B1 (fixed)")
ax.plot(t_us[track30.b2_idx], track30.layer, "g.-", ms=5, lw=1.2, label="B2 (advancing)")
ax.set(xlim=(t_us[sl][0], t_us[sl][-1]), ylim=(-0.8, wf_30.shape[0] + 2),
       xlabel="time (µs)", ylabel="layer index",
       title="Waterfall: the part growing, one echo at a time (color = layer)")
ax.legend(loc="upper left"); plt.tight_layout(); plt.show()
"""),
("md", """\
### Interactive 3D waterfall (rotate me!)

The same data as a rotatable 3D figure (plotly). Drag to rotate, scroll to zoom
— and **double-click to snap back to the starting view** at any time.

The starting camera is fixed on purpose, so that everyone in the class begins
from the same, readable orientation:

- **x = time (µs)**, increasing **left → right**, same direction as every other
  plot in this notebook;
- **y = layer index**, increasing into the depth of the scene — the front row is
  layer 1, the back row is layer 30;
- **z = amplitude**, vertical.

---

**Input** — identical to the previous cell (`wf_30` + `track30`); only the
rendering changes.

**What this cell does** — builds one 3D line per layer with plotly, using time
as x, layer index as y and amplitude as z, and adds the tracked B2 path as a
green 3D curve so you can see the echo front sweeping through the stack. It
then pins the camera: plotly's default viewpoint sits at `eye = (1.25, 1.25,
1.25)`, which makes the time axis run *right to left* on screen; a negative
`eye.y` flips it back to the natural left-to-right reading order.

**Output** — an interactive plotly figure. It works in Colab and in a live
Jupyter kernel; in a plain static export of the notebook it may appear blank,
which is normal.
"""),
("code", """\
import plotly.graph_objects as go

figly = go.Figure()
# Coarser decimation than the 2D plot (every 8th point) — 3D rendering in the
# browser is much heavier than a static image.
sl = slice(12000, 24000, 8)
for li in range(wf_30.shape[0]):
    x = wf_30[li, -1, sl]
    figly.add_trace(go.Scatter3d(
        x=t_us[sl], y=np.full(len(x), li + 1), z=x,      # time, layer, amplitude
        mode="lines", line=dict(color="black", width=1.2), showlegend=False))
# The tracked B2 path, drawn at the amplitude the waveform actually has there.
figly.add_trace(go.Scatter3d(
    x=t_us[track30.b2_idx], y=track30.layer,
    z=[wf_30[li, -1, b] for li, b in enumerate(track30.b2_idx)],
    mode="lines+markers", line=dict(color="green", width=4),
    marker=dict(size=3), name="B2 track"))
# Fix the starting viewpoint so the figure always opens the same way.
# Screen-right direction is proportional to (-eye.y, eye.x, 0): plotly's default
# eye = (1.25, 1.25, 1.25) therefore makes TIME run right-to-left. A negative
# eye.y restores the normal left-to-right reading order, and the explicit
# ascending ranges make the axis directions unambiguous.
figly.update_layout(
    scene=dict(
        xaxis=dict(title="time (µs)  →", range=[t_us[sl][0], t_us[sl][-1]]),
        yaxis=dict(title="layer (1 = front)", range=[0, wf_30.shape[0] + 1]),
        zaxis=dict(title="amplitude (a.u.)"),
        camera=dict(eye=dict(x=-0.6, y=-2.0, z=0.8),   # viewer in front, slightly left
                    up=dict(x=0, y=0, z=1)),           # keep z vertical
        aspectmode="manual", aspectratio=dict(x=1.7, y=1.1, z=0.5)),
    height=560, margin=dict(l=0, r=0, t=30, b=0),
    title="30-layer print — interactive waterfall (double-click to reset the view)")
figly.show()
"""),
("md", """\
## 4. Watch the part grow — waveform animation

Now the same story as a movie: one frame per layer, with the tracked B1/B2
markers and a live ToF readout. (This is the visual proof that the acoustic
path grows layer by layer.)

---

**Input** — `wf_30` (final frame of each layer) and `track30`. The animation has
one movie frame per **printed layer**, so it runs for 30 frames.

**What this cell does** — creates an empty axes once, then a small `update()`
function redraws the waveform and moves the two markers for each layer.
Matplotlib's `FuncAnimation` calls that function repeatedly; because we set
`rc("animation", html="jshtml")` in the setup cell, the result is embedded as a
video player with play/pause controls.

**Output** — an interactive animation player. `plt.close(fig)` before returning
`ani` simply prevents Jupyter from *also* showing the static first frame.
"""),
("code", """\
from matplotlib.animation import FuncAnimation

sl = slice(12000, 24000, 6)               # echo region, decimated for speed
fig, ax = plt.subplots(figsize=(10, 4))
# Create the artists ONCE; the animation only updates their data (much faster
# than redrawing a whole figure per frame).
line, = ax.plot([], [], lw=0.6, color="k")
vb1 = ax.axvline(0, color="tab:red", ls="--", lw=1.2)      # B1 marker
vb2 = ax.axvline(0, color="tab:green", ls="--", lw=1.2)    # B2 marker
txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=12)
# Fixed axis limits: if they moved from frame to frame the growth would be
# impossible to see.
ax.set(xlim=(t_us[sl][0], t_us[sl][-1]), ylim=(-4e4, 4e4),
       xlabel="time (µs)", ylabel="amplitude",
       title="30-layer print — layer-by-layer echo evolution")

def update(li):
    \"\"\"Draw layer li+1: waveform, tracked markers, and the live ToF readout.\"\"\"
    line.set_data(t_us[sl], wf_30[li, -1, sl])
    r = track30.iloc[li]
    vb1.set_xdata([t_us[int(r.b1_idx)]]); vb2.set_xdata([t_us[int(r.b2_idx)]])
    txt.set_text(f"layer {li+1:2d}/30    ToF = {r.tof_us:.3f} µs")
    return line, vb1, vb2, txt

ani = FuncAnimation(fig, update, frames=wf_30.shape[0], interval=280, blit=False)
plt.close(fig)     # hide the static figure; show only the player below
ani
"""),
("md", """\
### The race: weak cure (I1) vs strong cure (I7)

Two 15-layer prints, same geometry, different exposure intensity — side by side.
Because the two prints are identical in every respect except the light dose,
**everything that differs between the two panels is caused by curing alone.**
There are two distinct effects to watch, and they are the two independent ways
in which the echo carries material information:

**① The echo arrives earlier — B2 shifts left.** Curing crosslinks the resin
into a stiffer network, and a stiffer material carries sound *faster*. At the
same layer index the acoustic path has the same length in both panels, so the
strongly cured sample (I7) returns its echo sooner: its green marker sits to
the left of the weakly cured one. This is a **velocity** effect, and it is what
the ToF feature measures.

**② The echo comes back weaker — the peak amplitude decays more.** Watch the
*height* of the burst at the green marker, not just its position. The higher
exposure intensity produces a denser, more crosslinked network, which absorbs
and scatters more ultrasonic energy. The weakly cured print returns the stronger
echo at *every* layer, and — this is the part to watch — **the gap widens as the
part grows**: the two bursts differ by roughly 1.5× at the first layer but by
more than 5× around layer 9, because every added layer of strongly cured
material takes another bite out of the echo. Both panels use the same vertical
scale, so this comparison is a fair one. It is an **attenuation** effect, and it
is what the B2/B1 amplitude-ratio feature measures in Notebook 2.

> So the same echo carries at least three competing influences: **thickness
> moves B2 right, cure moves it left, and cure also pulls its amplitude down.**
> One measurement, several causes — untangling them is exactly what the
> machine-learning part of this series has to do, and it is why Notebook 2
> designs *several* complementary features instead of relying on ToF alone.

---

**Input** — the controlled pair `wf_i1` (I1, weak cure) and `wf_i7` (I7, strong
cure): same 15-layer geometry, same resin, only the exposure intensity differs.

**What this cell does** — tracks the echoes of both prints (giving `track_i1`
and `track_i7`, reused in section 6 and in Notebook 2), then animates them in
two side-by-side panels that advance layer-synchronously, so any difference you
see is caused by the light dose alone.

**Output** — a two-panel animation plus the two tracking tables.
"""),
("code", """\
# Track the echoes of the controlled pair (same geometry, different intensity).
track_i1 = iu.track_sample(wf_i1)
track_i7 = iu.track_sample(wf_i7)

sl = slice(13500, 20000, 6)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
arts = []
# Build the same set of artists (line + two markers + text) for each panel and
# remember them, so that update2() can refresh both panels together.
for ax, name in zip(axes, ["I1 = 10.70 mW/cm² (weak)", "I7 = 20.16 mW/cm² (strong)"]):
    ln, = ax.plot([], [], lw=0.6, color="k")
    v1 = ax.axvline(0, color="tab:red", ls="--", lw=1.1)
    v2 = ax.axvline(0, color="tab:green", ls="--", lw=1.4)
    tx = ax.text(0.03, 0.9, "", transform=ax.transAxes, fontsize=11)
    # Identical limits on both panels — otherwise the comparison would be unfair.
    ax.set(xlim=(t_us[sl][0], t_us[sl][-1]), ylim=(-4e4, 4e4),
           xlabel="time (µs)", title=name)
    arts.append((ln, v1, v2, tx))
axes[0].set_ylabel("amplitude")

def update2(li):
    \"\"\"Advance BOTH prints to the same layer index li+1.\"\"\"
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

**The two sliders are not interchangeable — they move along the two different
time scales of this experiment, and this is the single most important idea in
the whole series.** It is worth spending a few minutes here before moving on.

**Moving the `layer` slider = the across-layer scale (minutes).** Each step is
a whole new layer: the part is ~100 µm taller, and the acoustic path is
physically longer. Watch the **green B2 marker travel to
the right** — that is the part growing. Its amplitude also drifts as the beam
crosses more and more cured material. Because both geometry *and* material
change from one layer to the next, whatever you see here mixes the two.

**Moving the `frame` slider = the within-layer scale (seconds).** Now stay on
one layer and step through its 6 frames: frame 1 is the reference taken before
the light turns on, frames 2–6 are taken every 3 s during the 15 s exposure.
The printhead has not moved and no resin has been added, so **the acoustic path
is frozen** — the geometry is literally constant. Anything that changes here
can only come from the resin crosslinking: mostly a **decay of the peak
amplitude** and a slight change of the burst's shape as the network stiffens
and absorbs more. Notice that the red and green markers stay put while you do
this, which is the visual statement that the window is fixed.

The change within a layer is subtle to the eye — much smaller than what the
layer slider does. That is expected, and it is exactly why Notebook 2 stops
looking and starts *measuring* it, with frame-to-frame RMS and standard
deviation. The payoff is worth the effort: because the geometry cannot change
within a layer, those within-layer numbers are a **geometry-immune probe of
curing** — the cleanest route we have to the degree of conversion.

---

**Input** — all three prints and their tracking tables, collected into the
`SAMPLES` dictionary, so you can switch freely between the weak-cure and
strong-cure prints at the same layer.

**What this cell does** — defines a plotting function `browse()` whose arguments
become widget controls, and hands it to `ipywidgets.interact`, which builds a
dropdown plus two sliders automatically and re-runs the function on every change.
The B1/B2 markers it draws come from the tracking table, which was computed on
each layer's final frame — that is why they do not move when you change frames.

**Output** — an interactive browser. Two things worth trying: hold the layer
fixed and switch between I1 and I7 to see the cure difference, then hold the
sample fixed and sweep the layer slider to see the growth.
"""),
("code", """\
import ipywidgets as w

# One entry per print: (waveform array, tracking table).
SAMPLES = {"15 layers, I1": (wf_i1, track_i1),
           "15 layers, I7": (wf_i7, track_i7),
           "30 layers, I7": (wf_30, track30)}

def browse(sample="30 layers, I7", layer=1, frame=6):
    \"\"\"Plot one A-scan chosen by the widgets (layer and frame are 1-based).\"\"\"
    wf, tr = SAMPLES[sample]
    layer = min(layer, wf.shape[0])     # the 15-layer prints have no layer 16-30
    x = wf[layer - 1, frame - 1]        # widgets count from 1, Python from 0
    r = tr.iloc[layer - 1]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(t_us[::4], x[::4], lw=0.5, color="k")   # every 4th point: fast redraw
    ax.axvline(t_us[int(r.b1_idx)], color="tab:red", ls="--", label="B1")
    ax.axvline(t_us[int(r.b2_idx)], color="tab:green", ls="--", label="B2")
    ax.set(xlim=(5, 10), xlabel="time (µs)", ylabel="amplitude",
           title=f"{sample} — layer {layer}, frame {frame}  |  ToF = {r.tof_us:.3f} µs")
    ax.legend(loc="upper right"); plt.tight_layout(); plt.show()

# interact() inspects browse()'s arguments and builds the matching controls:
# a dropdown for `sample`, sliders for `layer` and `frame`.
w.interact(browse, sample=list(SAMPLES), layer=(1, 30, 1), frame=(1, 6, 1));
"""),
("md", """\
## 6. First quantitative result: ToF vs layer

The simplest possible "monitor": track ToF layer by layer. It is nearly linear
(path grows ~100 µm/layer), and the *slope difference* between I1 and I7 already
carries material information (velocity).

---

**Input** — the two tracking tables of the controlled pair, `track_i1` and
`track_i7`. The raw waveforms are no longer needed: we have compressed 62,509
points per frame down to **one number per layer**.

We deliberately plot *only* the two 15-layer prints here. The 30-layer sample
is a different build — more layers means more accumulated exposure, a taller
part and a longer print time — so putting its curve on the same axes would
compare two things at once and nothing could be attributed cleanly. The I1/I7
pair, in contrast, differs in **exactly one** variable, the exposure intensity,
so any gap between the two curves is caused by the light dose alone.

**What this cell does** — plots `tof_us` against layer index for the two prints
on common axes.

**Output** — the first real "process monitoring" curve of the series: a physical
quantity measured *during* printing, without touching the part. Both curves rise
with the same overall geometry-driven trend, and the strongly cured print sits
slightly lower because its waves travel faster.
"""),
("code", """\
fig, ax = plt.subplots(figsize=(7, 4.2))
# Each tracking table already contains tof_us per layer — one number per layer
# distilled from 6 x 62509 raw points.
# Only the controlled pair is plotted: same 15-layer geometry, only the exposure
# intensity differs, so the gap between the curves is attributable to cure alone.
for tr, name, c in [(track_i1, "15L @ I1 (weak cure)", "tab:blue"),
                    (track_i7, "15L @ I7 (strong cure)", "tab:orange")]:
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
   throughout the print, and the gap widens with depth. Offer a physical
   explanation (think attenuation per unit of cured path).

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

![two-scale](https://raw.githubusercontent.com/wangyiquan1010-coder/IUM_teaching_colab/main/figs/two_scale_framework.png)

- **Across layers**: each layer lengthens the acoustic path *and* stiffens the
  material → ToF, amplitude ratio, spectral features encode geometry + material
  state, entangled.
- **Within a layer**: during one 15 s exposure the printhead does not move — the
  path is frozen. The six frames differ *only* through crosslinking → frame-to-
  frame change is a **geometry-immune** probe of curing (→ DoC).

This notebook rebuilds the key features from raw data and then asks, honestly,
how much process information they carry *before any learning* — which turns out
to be a great deal for some of them and very little for others.
"""),
("md", SETUP_MD),
("code", BOOT),
("md", """\
### Loading the controlled pair again

**Input** — the same two `.npz` archives as in Notebook 1: `15L @ I1` (weak
cure) and `15L @ I7` (strong cure). Same geometry, same resin, only the light
intensity differs, so every difference we measure below is a **material**
difference.

**What this cell does** — reloads the waveforms, rebuilds the time axis, and
re-runs the B1/B2 tracker. Each notebook stands on its own, so nothing carries
over from Notebook 1's kernel.

**Output** — `wf_i1`, `wf_i7` (arrays of shape 15 × 6 × 62509), `t_us`, and the
tracking tables `track_i1`, `track_i7` (`b1_idx`, `b2_idx`, `tof_us` per layer)
that every feature below is computed *relative to*.
"""),
("code", """\
# Same controlled pair as Notebook 1: identical geometry, different light dose.
wf_i1, _ = iu.load_sample("data/waveforms_15L_I1.npz")   # weak cure
wf_i7, _ = iu.load_sample("data/waveforms_15L_I7.npz")   # strong cure
t_us = iu.time_axis_us()

# Echo positions are the anchor for every feature: all windows below are placed
# relative to the tracked B2 index, never at a fixed absolute time.
track_i1 = iu.track_sample(wf_i1)
track_i7 = iu.track_sample(wf_i7)
"""),
("md", """\
## 1. The within-layer scale, animated

Fix one layer. Play its six frames (reference + five during exposure), zoomed
to the B2 echo. The path cannot change — everything you see moving is
**chemistry**.

---

Both panels are **normalised to their own first frame**, so they share one
vertical scale and can be compared directly: every trace starts at ±1, and what
you watch is the *fraction* of the echo that survives as curing proceeds.

> The strongly cured echo is about five times weaker in raw counts, so fewer
> digitizer steps describe it and its trace looks coarser. Nothing is smoothed —
> that is quantization, not filtering.

---

**Input** — one single layer (layer 8) of each print, i.e. `wf_i1[7]` and
`wf_i7[7]`: **6 frames × 62,509 points** each. This time the animation's frames
are the *within-layer* snapshots, not layers.

**What this cell does** — cuts a ±500-sample window (±0.2 µs) centred on that
layer's tracked B2 position, normalises it by the reference frame, and animates
the six frames. Because the printhead does not move during an exposure, the
window content can only change through curing.

**Output** — a two-panel animation plus the per-frame amplitude. Both bursts
weaken during the 15 s exposure, the strongly cured one more steeply (about
−19% against −8%). That is one layer of one print, not a rule — section 4 puts
the question to all 50.
"""),
("code", """\
from matplotlib.animation import FuncAnimation

LAYER = 8                                    # any mid-print layer works
r1, r7 = track_i1.iloc[LAYER-1], track_i7.iloc[LAYER-1]
w1 = 500                                     # half-window: 500 samples = 0.2 µs

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
arts = []
for ax, (wfS, r, name) in zip(axes, [(wf_i1, r1, "I1 (weak cure)"),
                                     (wf_i7, r7, "I7 (strong cure)")]):
    b2 = int(r.b2_idx)                       # window follows the echo, not the clock
    seg_t = t_us[b2-w1:b2+w1]
    block = wfS[LAYER-1, :, b2-w1:b2+w1].astype(float)   # (6 frames, 2*w1)
    # Remove ONE common baseline (the digitizer's DC offset) from all six frames
    # so the burst is centred on zero. A single offset for the whole layer keeps
    # the frame-to-frame differences - the thing we came to see - intact.
    block = block - np.median(block)
    # Normalise by the REFERENCE frame, again one number for the whole layer.
    # Both panels then start at +-1 and share the axis below, so the amount of
    # motion you see is directly comparable between weak and strong cure.
    block = block / np.abs(block[0]).max()
    ln, = ax.plot([], [], lw=1.2)
    tx = ax.text(0.03, 0.9, "", transform=ax.transAxes)
    ax.set(xlim=(seg_t[0], seg_t[-1]), ylim=(-1.25, 1.25), xlabel="time (µs)",
           title=f"{name} — layer {LAYER}")
    arts.append((ln, tx, seg_t, block))
axes[0].set_ylabel("amplitude / reference frame")

def upd(f):
    \"\"\"Show frame f of the same layer in both panels (f=0 is the reference).\"\"\"
    out = []
    for ln, tx, seg_t, block in arts:
        ln.set_data(seg_t, block[f])
        tx.set_text(f"frame {f+1}/6  (t = {max(0,(f-1))*3} s)" if f else "reference")
        out += [ln, tx]
    return out

# Amplitude per frame, as a fraction of the reference frame — the same numbers
# the animation shows, now directly comparable between the two panels.
for name, (_, _, _, block) in zip(["I1 weak  ", "I7 strong"], arts):
    p2p = block.max(axis=1) - block.min(axis=1)
    print(f"{name} amplitude per frame: {np.round(p2p / p2p[0], 3)}   "
          f"({(p2p[-1] / p2p[0] - 1) * 100:+.0f}% over the exposure)")

ani = FuncAnimation(fig, upd, frames=6, interval=600, blit=False)
plt.close(fig)
ani
"""),
("md", """\
The change is subtle by eye — which is precisely why we quantify it with
**within-layer RMS/STD** (frame-to-frame differences in the fixed window).

## 2. Rebuilding the layer-wise features

The paper describes **11 descriptors per layer**, and the dataset shipped with
this repository stores exactly that: 11 features × 30 layer slots = 330 numbers
per printed sample. Here is what each one measures and which scale it belongs
to.

**Across-layer features** — computed from the **final frame** of the layer (the
end of its exposure), all inside a ±800-sample window placed around the tracked
B2 echo:

<table style="border-collapse:collapse">
<thead><tr>
<th style="border:1px solid #999;padding:5px 9px;text-align:left">feature</th>
<th style="border:1px solid #999;padding:5px 9px;text-align:left">unit</th>
<th style="border:1px solid #999;padding:5px 9px;text-align:left">what it measures</th>
</tr></thead>
<tbody>
<tr><td style="border:1px solid #999;padding:5px 9px"><code>ToF</code></td>
<td style="border:1px solid #999;padding:5px 9px">µs</td>
<td style="border:1px solid #999;padding:5px 9px">arrival-time difference B1 → B2. The acoustic path length divided by the wave speed, so it responds both to the part growing and to the material getting faster as it cures.</td></tr>
<tr><td style="border:1px solid #999;padding:5px 9px"><code>amplitude</code></td>
<td style="border:1px solid #999;padding:5px 9px">–</td>
<td style="border:1px solid #999;padding:5px 9px">the B2/B1 envelope ratio, i.e. how much of the echo survives the trip through the part. Dividing by B1 is deliberate: B1 never leaves the printhead, so the ratio cancels drifts in transducer coupling or pulse energy and leaves attenuation.</td></tr>
<tr><td style="border:1px solid #999;padding:5px 9px"><code>RMS_Energy</code></td>
<td style="border:1px solid #999;padding:5px 9px">a.u.</td>
<td style="border:1px solid #999;padding:5px 9px">root-mean-square amplitude of the B2 window — the total energy coming back, without normalising by B1.</td></tr>
<tr><td style="border:1px solid #999;padding:5px 9px"><code>Center_Freq</code></td>
<td style="border:1px solid #999;padding:5px 9px">Hz</td>
<td style="border:1px solid #999;padding:5px 9px">the spectral centroid of the echo (2–30 MHz band). Cured polymer absorbs high frequencies more strongly than low ones, so the centroid moves as the material changes.</td></tr>
<tr><td style="border:1px solid #999;padding:5px 9px"><code>Bandwidth</code></td>
<td style="border:1px solid #999;padding:5px 9px">Hz</td>
<td style="border:1px solid #999;padding:5px 9px">the spectral spread about that centroid — how much the pulse has been broadened and reshaped on its way through the part.</td></tr>
<tr><td style="border:1px solid #999;padding:5px 9px"><code>Wavelet_Energy_L0</code> … <code>L3</code></td>
<td style="border:1px solid #999;padding:5px 9px">a.u.</td>
<td style="border:1px solid #999;padding:5px 9px">energy in four wavelet decomposition bands. Unlike the FFT-based pair above, a wavelet split keeps <em>where in the burst</em> the energy sits, so these four capture changes in the echo's shape that a single centroid would average away.</td></tr>
</tbody>
</table>

**Within-layer features** — computed from **all six frames** of the same, frozen
window, and therefore immune to geometry:

<table style="border-collapse:collapse">
<thead><tr>
<th style="border:1px solid #999;padding:5px 9px;text-align:left">feature</th>
<th style="border:1px solid #999;padding:5px 9px;text-align:left">unit</th>
<th style="border:1px solid #999;padding:5px 9px;text-align:left">what it measures</th>
</tr></thead>
<tbody>
<tr><td style="border:1px solid #999;padding:5px 9px"><code>frame_diff_rms</code></td>
<td style="border:1px solid #999;padding:5px 9px">a.u.</td>
<td style="border:1px solid #999;padding:5px 9px">root-mean-square of the frame-to-frame differences: how much the echo moves from one 3 s snapshot to the next while the light is on.</td></tr>
<tr><td style="border:1px solid #999;padding:5px 9px"><code>within_layer_std</code></td>
<td style="border:1px solid #999;padding:5px 9px">a.u.</td>
<td style="border:1px solid #999;padding:5px 9px">the per-sample standard deviation across the six frames, averaged over the window: the same idea, measured as spread instead of as step-to-step change.</td></tr>
</tbody>
</table>

---

**Input** — the raw frames of one layer (`wf[li]`, shape 6 × 62509) plus that
layer's tracked `b1_idx` and `b2_idx`. Nothing else: every window below is
positioned relative to the echoes, never at a fixed absolute time.

**What this cell does** — computes all eleven descriptors from the raw samples,
written out step by step with the formula for each one, so you can see exactly
what every number in the dataset means. Nothing is read from a precomputed
file here.

**Output** — `F1` and `F7`: one row per layer, eleven columns, indexed by layer
number. The `.head()` call prints the first five rows so you can see the actual
values and units.
"""),
("code", """\
import pywt          # PyWavelets — preinstalled on Colab

WIN = 800            # half-width of the analysis window around B2 [samples]
                     # 800 samples = 0.32 µs, wide enough to hold the whole burst


def layer_features_explicit(frames, b1_idx, b2_idx):
    \"\"\"The 11 layer-wise descriptors for ONE layer, computed from scratch.

    frames         : (6, 62509) all frames of this layer, raw amplitudes
    b1_idx, b2_idx : tracked echo positions for this layer [sample index]
    \"\"\"
    f = {}
    x = frames[-1].astype(float)          # final frame = end of the exposure

    # --- envelope ----------------------------------------------------------
    # An echo is an oscillating burst whose individual peaks all look alike, so
    # we work with its envelope: the rectified signal, smoothed with a boxcar.
    #     env[n] = (1/K) * SUM_k |x[n-k] - median(x)| ,  K = 101 (= 40 ns)
    rect = np.abs(x - np.median(x))       # median removes the digitizer's DC offset
    env = np.convolve(rect, np.ones(101) / 101, mode="same")

    # --- 1. ToF ------------------------------------------------------------
    #     ToF = (b2 - b1) * dt ,   dt = 1 / 2.5 GHz = 0.4 ns
    f["ToF"] = (b2_idx - b1_idx) * iu.DT_NS / 1000.0                   # µs

    # --- 2. amplitude ratio -------------------------------------------------
    #     amplitude = env[b2] / env[b1]
    # Dividing by B1 is what makes this a *material* measurement: B1 never
    # leaves the printhead, so any drift in coupling or pulse energy scales
    # both echoes and cancels in the ratio.
    f["amplitude"] = float(env[b2_idx] / (env[b1_idx] + 1e-9))         # -

    # Every remaining across-layer feature reads the same window around B2.
    seg = x[b2_idx - WIN : b2_idx + WIN]               # 1600 samples = 0.64 µs

    # --- 3. RMS energy ------------------------------------------------------
    #     RMS = sqrt( mean( seg^2 ) )   — how much energy came back in total
    f["RMS_Energy"] = float(np.sqrt(np.mean(seg ** 2)))                # a.u.

    # --- 4 & 5. spectral centroid and bandwidth -----------------------------
    # Power spectrum of the echo (mean removed so the DC bin carries no weight):
    #     P(f) = |FFT(seg - mean(seg))|^2
    # Centroid  fc = SUM(f * P) / SUM(P)                   — "average" frequency
    # Bandwidth bw = sqrt( SUM((f - fc)^2 * P) / SUM(P) )  — its spread
    # Restricted to 2-30 MHz, the band where this 10 MHz transducer has gain.
    P = np.abs(np.fft.rfft(seg - seg.mean())) ** 2
    freqs = np.fft.rfftfreq(len(seg), d=1 / iu.FS)
    keep = (freqs >= 2e6) & (freqs <= 30e6)
    Pb, fb = P[keep], freqs[keep]
    fc = float((fb * Pb).sum() / Pb.sum())
    f["Center_Freq"] = fc                                              # Hz
    f["Bandwidth"] = float(np.sqrt(((fb - fc) ** 2 * Pb).sum() / Pb.sum()))

    # --- 6-9. wavelet-band energies -----------------------------------------
    # A Daubechies-4 transform with 3 levels splits the echo into four bands,
    # returned coarse-to-fine as [cA3, cD3, cD2, cD1]. The feature is each
    # band's energy:
    #     Wavelet_Energy_Lk = SUM( c_k^2 )
    # Why bother, when we already have a centroid? The FFT pair above averages
    # over the whole window and forgets WHEN each frequency arrived; wavelet
    # bands keep that, so they see changes in the burst's shape, not just its
    # average pitch.
    for k, c in enumerate(pywt.wavedec(seg, "db4", level=3)):
        f[f"Wavelet_Energy_L{k}"] = float(np.sum(c ** 2))              # a.u.

    # --- 10 & 11. within-layer features -------------------------------------
    # Same window, but now all six frames. The printhead does not move during
    # an exposure, so these two see chemistry and nothing else.
    block = frames[:, b2_idx - WIN : b2_idx + WIN].astype(float)
    diffs = np.diff(block, axis=0)              # 5 consecutive frame differences
    #     frame_diff_rms = mean_i( sqrt( mean( diff_i^2 ) ) )
    f["frame_diff_rms"] = float(np.mean([np.sqrt(np.mean(d ** 2)) for d in diffs]))
    #     within_layer_std = mean_n( std over the 6 frames at sample n )
    f["within_layer_std"] = float(np.std(block, axis=0).mean())
    return f


def features_for(wf, track):
    \"\"\"Apply the above to every layer of one print -> one row per layer.\"\"\"
    rows = []
    for li in range(wf.shape[0]):
        r = track.iloc[li]
        f = layer_features_explicit(wf[li], int(r.b1_idx), int(r.b2_idx))
        f["layer"] = li + 1
        rows.append(f)
    return pd.DataFrame(rows).set_index("layer")

F1 = features_for(wf_i1, track_i1)      # weak cure
F7 = features_for(wf_i7, track_i7)      # strong cure
print("features per layer:", F1.shape[1])
F1.head()
"""),
("md", """\
### Do the features separate the two intensities?

**Input** — the two freshly computed tables `F1` and `F7` (15 rows each).

**What this cell does** — plots four representative features against layer
index, one panel each, with both intensities overlaid. Each panel is titled with
the *physical* quantity the feature is meant to probe, so you can judge the
design rather than just the curve.

**Output** — a 2×2 figure. If the two colors separate here, the features already
carry process information **before any machine learning** — which is the whole
point of physics-informed design.
"""),
("code", """\
# Four features, one per physical mechanism (see the panel titles).
names = ["ToF", "amplitude", "Center_Freq", "frame_diff_rms"]
titles = ["ToF (µs) — path + velocity", "B2/B1 amplitude ratio — attenuation",
          "center frequency (Hz) — dispersion", "within-layer RMS — curing dynamics"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, n, ti in zip(axes.ravel(), names, titles):
    ax.plot(F1.index, F1[n], "o-", label="I1 weak", color="tab:blue")
    ax.plot(F7.index, F7[n], "s-", label="I7 strong", color="tab:orange")
    ax.set(title=ti, xlabel="layer"); ax.grid(alpha=0.3)
axes[0, 0].legend()
plt.suptitle("Layer-wise features for the two intensities — before any ML")
plt.tight_layout(); plt.show()
"""),
("md", """\
**Reading the four panels:**

- *ToF* rises in both prints, because the path keeps growing, and it rises more
  slowly for I7 (about 0.057 versus 0.074 µs per layer): the strongly cured
  material carries sound faster. This is the cleanest signal we have.
- *Amplitude ratio* falls in both prints, and over the first ten layers it
  falls much faster for I7 — more crosslinking, more attenuation. The jump back
  up around layer 11 is almost certainly the tracker latching onto a
  neighbouring peak in a weak signal, a reminder that every feature inherits
  the tracker's mistakes.
- *Center frequency* and *within-layer RMS* have no clear physical reading
  here. The curves differ, but not in a way this pair of prints lets you
  attribute to a cause.

## 3. Where the dataset's features live

Everything so far was recomputed in front of you from three sample prints.
Notebooks 3 and 4 instead use the paper's own tables, covering all 50 printed
cylinders:

- **`data/feature11.csv`** — 50 rows, one per print. Columns: `layer` (10–30),
  `intensity` (percent of lamp power, 10 = I1 … 100 = I10), and
  `Layer_{i}_{feature}` for i = 1…30 and the 11 features above, i.e. **330
  columns**. The 30 slots are fixed width, so a 15-layer print fills slots 1–15
  and leaves the rest at **zero**; any statistic over the slots must therefore
  be restricted to a print's real layers.
- **`data/condition_labels.csv`** — the same 50 rows in the same order, with
  the measured `thickness` (mm, Keyence), `modulus` (Pa, rheometer) and `DoC`
  (–, Raman).

---

**Input** — `data/feature11.csv`.

**What this cell does** — loads the table and prints its shape, the features
stored per layer slot, and one print's padding.

**Output** — `df`, used by the heatmaps below and by both remaining notebooks.
"""),
("code", """\
import re

# The paper's own feature table: 50 rows (one per printed sample), wide format
# with columns Layer_1_ToF ... Layer_30_within_layer_std.
df = iu.load_features("data/feature11.csv")
print("table shape (rows = printed samples, columns):", df.shape)
print("non-layer columns:", [c for c in df.columns if not c.startswith("Layer_")])

# Which features are stored for each of the 30 layer slots?
bases = sorted({re.match(r"Layer_(\\d+)_(.+)", c).group(2)
                for c in df.columns if c.startswith("Layer_")})
print(f"\\n{len(bases)} features per layer slot:")
for b in bases:
    print("   ", b)

# Zero padding, made concrete: this print has 15 layers, so slots 16-30 are 0.
row = df[(df.layer == 15) & (df.intensity == 10)].iloc[0]
print("\\n15-layer print at 10% lamp power — stored ToF per layer slot (µs):")
print("   slots  1- 3:", [round(float(row[f"Layer_{i}_ToF"]), 3) for i in (1, 2, 3)])
print("   slots 14-17:", [round(float(row[f"Layer_{i}_ToF"]), 3) for i in (14, 15, 16, 17)],
      "<- slots 16+ are zero padding")
"""),
("md", """\
## 4. The whole design grid at a glance — feature heatmaps

Now the full 50-sample table, so the same question can be asked of the whole
design instead of one pair of prints: does a feature respond to exposure?

---

**Input** — the **30-layer group** of `df`: 10 prints, one per intensity.

**What this cell does** — arranges each feature into a (30 layers × 10
intensities) matrix, draws it as an image (vertical = depth, horizontal =
dose), and prints the column averages behind the colours.

**Output** — three heatmaps and their numbers. ToF has a strong depth gradient;
the other two show no clean trend with exposure. No single feature reads out
the process state on its own — which is why Notebook 3 builds a model over all
of them at once.
"""),
("code", """\
def heat(ax, base, group_layers, title, cmap="viridis"):
    \"\"\"Draw a (layer × intensity) heatmap of one feature for one layer group.\"\"\"
    g = df[df.layer == group_layers].sort_values("intensity")   # 10 prints, I1..I10
    # Build the matrix column by column (one column per print), then transpose
    # so that rows = layers (depth) and columns = intensity (dose).
    M = np.array([[r[f"Layer_{i}_{base}"] for i in range(1, group_layers + 1)]
                  for _, r in g.iterrows()]).T          # layers × intensities
    im = ax.imshow(M, aspect="auto", origin="upper", cmap=cmap)
    ax.set(xlabel="intensity I1→I10", ylabel="layer", title=title)
    return im

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
BASES_SHOWN = ["ToF", "amplitude", "frame_diff_rms"]
for ax, base, ti in zip(axes, BASES_SHOWN,
        ["ToF — strong depth gradient", "amplitude ratio — mostly depth",
         "within-layer RMS — little dose structure"]):
    im = heat(ax, base, 30, ti)
    plt.colorbar(im, ax=ax, shrink=0.85)
plt.suptitle("30-layer group, all 10 intensities")
plt.tight_layout(); plt.show()

# Check the colours as numbers: average each feature down the 30 layers, so what
# is left is its response to exposure alone.
g = df[df.layer == 30].sort_values("intensity")
print("column average over the 30 layers, for intensities I1 -> I10:")
for base in BASES_SHOWN:
    M = np.array([[r[f"Layer_{i}_{base}"] for i in range(1, 31)]
                  for _, r in g.iterrows()])
    print(f"  {base:15s}", np.round(M.mean(axis=1), 3))
"""),
("md", """\
**Next notebook:** we assemble the 344-dimensional representation and learn —
without fooling ourselves.
"""),
]

# ==============================================================================
# Notebook 3
# ==============================================================================
nb3 = [
("md", """\
# 03 · Building and Training the Model

**IUM teaching series, Notebook 3/4** · Wang & Zhao, *Additive Manufacturing*
(2026) 105300.

We now have features. This notebook turns them into a working soft sensor:

1. assemble the **344-dimensional** sample representation;
2. choose how to split the data for training and testing — and see why that
   choice decides whether the reported numbers mean anything;
3. build and train the **three-branch attention-fusion network**.
"""),
("md", """\
## Machine-learning vocabulary used in this notebook

| term | what it means here |
|---|---|
| **sample** | one printed cylinder. We have exactly **50** of them — a small dataset, which is why every methodological detail matters. |
| **feature / representation** | the numbers describing one sample. Ours is 344-dimensional: 330 layer-wise values + 12 part-scale statistics + 2 printing conditions. |
| **label / target** | the ground truth we want to predict: thickness (Keyence profilometer, mm), storage modulus (rheometer, Pa), degree of conversion (Raman, 0–1). |
| **cross-validation (CV), fold** | to estimate performance honestly you split the data, train on one part and test on the other, and repeat. Each repetition is a *fold*. |
| **data leakage** | when information about the test samples reaches the model during training. The reported score then measures memorisation, not generalisation. This notebook shows a real example. |
| **LOIO** | *Leave-One-Intensity-Out*: each fold holds out **all** samples printed at one exposure intensity, so the model must predict a light dose it has never seen. |
| **StandardScaler** | rescales each column to zero mean / unit variance. Critically, it must be **fitted on the training fold only** — fitting it on all data is itself a form of leakage. |
| **epoch** | one pass of the training algorithm over the whole training set. |
| **R²** | fraction of the label's variance the model explains: 1.0 is perfect, 0 is no better than predicting the mean, and negative values are worse than that. |
"""),
("md", SETUP_MD),
("code", BOOT),
("md", """\
## 1. Assemble the dataset

Each printed sample becomes: 330 layer-wise entries (11 features × 30 layer
slots, zero-padded) + 12 part-scale statistics + 2 printing conditions,
with three labels (thickness, modulus, DoC) from Keyence / rheometer / Raman.

---

**Input** — two CSV files that together *are* the paper's dataset:

- `data/feature11.csv` — the feature table produced by the paper's extraction
  pipeline from all 50 prints (the same file we sanity-checked against our own
  recomputation in Notebook 2);
- `data/condition_labels.csv` — the measured ground truth for the same 50
  prints, in the same row order: `thickness` (mm, Keyence laser profilometer),
  `modulus` (Pa, rheometer) and `DoC` (dimensionless 0–1, Raman spectroscopy).

**What this cell does** — loads both tables, asserts that their rows really do
line up (the `assert` is a cheap guard against the single most common and most
embarrassing data bug: silently mismatched features and labels), and merges the
three label columns into one wide table called `full`.

**Output** — `full`: 50 rows × (330 layer-wise features + `layer` + `intensity`
+ 3 labels). Every remaining cell in this notebook reads from it.
"""),
("code", """\
# Features (from raw ultrasound) and labels (from destructive/offline metrology)
# live in two files with matching row order — one row per printed sample.
feats = iu.load_features("data/feature11.csv")
labels = iu.load_labels("data/condition_labels.csv")

# Guard against the classic silent bug: features and labels out of order.
assert (feats.layer.values == labels.layer.values).all()

full = feats.copy()
for c in iu.LABEL_COLS:                 # thickness, modulus, DoC
    full[c] = labels[c]

print(f"{len(full)} samples;  layer counts {sorted(full.layer.unique())};  "
      f"intensities {sorted(full.intensity.unique())} (% of max power)")
full[["layer", "intensity"] + iu.LABEL_COLS].head()
"""),
("md", """\
## 2. How we split the data — and why it decides everything

Before training anything, we have to choose which samples the model is tested
on. The obvious choice is a random split: shuffle the 50 prints, train on 40,
test on 10. For this dataset that choice is wrong, and badly so.

The reason is the design of the experiment. Each exposure intensity was printed
five times, at five different layer counts. So if a randomly held-out print was
made at 70% lamp power, four other prints at exactly 70% are still in the
training set. The model does not have to understand the ultrasound at all: it
can look up `intensity = 70`, recall what those neighbours looked like, and
interpolate. The score that comes out measures memorisation of the recipe, not
sensing.

What a deployed sensor actually faces is a print whose settings it has never
seen. So we hold out **one whole intensity at a time**: all five prints made at
that lamp power go into the test set together, and the model must extrapolate
to it. Ten intensities, ten folds — **Leave-One-Intensity-Out (LOIO)**. Let's
measure how much the choice is worth, using a fast Random Forest so the
comparison is about the protocol and not the model.

---

**Input** — the full 344-dimensional representation built from `full`: the 330
layer-wise features, the 2 process columns (`layer`, `intensity`) and the 12
part-scale statistics computed by `iu.part_scale_stats()` (mean, max, std,
slope, early-vs-late difference and final value of the ToF and amplitude
sequences, using only each print's *real* layers, not the zero padding).

**What this cell does** — trains the *same* Random Forest under two evaluation
protocols and nothing else changes: (a) random 5-fold, which scatters the 50
samples arbitrarily, and (b) Leave-One-Intensity-Out, where `groups=intensity`
forces every fold to hold out all five prints made at one light dose. A Random
Forest is used here instead of the neural network purely because it trains in
seconds, so the comparison is about the *protocol*, not the model.

**Output** — the bar chart and the `comp` table of R² per target under both
protocols. The distance between the two bars is the leakage.
"""),
("code", """\
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import r2_score

# Build the sample representation: 330 layer-wise + 2 process + 12 part-scale.
seq_cols = iu.seq_columns(full)                   # the 330 Layer_{i}_{feature} columns
X = full[seq_cols + ["layer", "intensity"]].values
X_stats = iu.part_scale_stats(full).values        # 12 global descriptors
X_all = np.hstack([X, X_stats])
Y = full[iu.LABEL_COLS].values                    # 50 × 3 labels

def cv_r2(splitter, groups=None):
    \"\"\"Run any CV splitter and return out-of-fold R² for the three targets.

    Every sample is predicted exactly once, by a model that did not see it
    during training — that is what makes the score 'out-of-fold'.
    \"\"\"
    preds = np.zeros_like(Y)
    for tr_idx, te_idx in splitter.split(X_all, Y, groups):
        rf = RandomForestRegressor(300, random_state=0, n_jobs=-1)
        rf.fit(X_all[tr_idx], Y[tr_idx])
        preds[te_idx] = rf.predict(X_all[te_idx])
    return [r2_score(Y[:, i], preds[:, i]) for i in range(3)]

# (a) Optimistic: random folds — a held-out print's own intensity is still in training.
r2_random = cv_r2(KFold(5, shuffle=True, random_state=0))
# (b) Honest: whole intensity groups held out — the model must extrapolate.
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

## 3. The model — three-branch attention fusion

<img src="https://raw.githubusercontent.com/wangyiquan1010-coder/IUM_teaching_colab/main/figs/model_architecture.png"
     alt="model architecture" style="max-width:100%;height:auto">

*The paper's full architecture (Fig. 7). Each input group gets its own branch, the
branch outputs are concatenated, an attention gate reweights that vector, and a
shared head predicts all three targets at once.*

Reading the diagram from left to right:

- **part-scale statistics** (12 numbers per print) → a small MLP → 32 channels;
- **layer-wise features** (30 layer slots × 11 features) → a 1-D convolution
  followed by a bidirectional LSTM and global max pooling → 128 channels. The
  convolution looks at neighbouring layers, the LSTM at the whole build history;
- **printing conditions** (layer count, intensity) → a tiny MLP → 16 channels;
- **raw waveforms** → a deeper CNN → 128 channels.

The **attention gate** is the part worth understanding. It takes the
concatenated vector, passes it through a small MLP ending in a sigmoid, and
multiplies the vector by the result. Each channel therefore gets its own weight
between 0 and 1, and those weights are readable — Notebook 4 opens them up to
ask whether the ultrasound really contributes more than the nominal recipe.

This notebook trains the **deployed configuration (the paper's Case 5)**, which
is the diagram *without* the raw-waveform branch: 32 + 128 + 16 = **176**
channels instead of 304. Feeding 62,509 raw points per layer to a network
trained on 50 prints overfits badly; the designed features are what make the
problem learnable at this sample size.

---

**Input** — the same `full` table, but now repacked per fold by
`iu.assemble_arrays()`, which delivers the three branch inputs in the shapes the
network expects: `stats` (n × 12), `seq` (n × 30 layers × 11 features) and
`proc` (n × 2), with the labels standardised as well. **All four scalers are
fitted on the training fold only** — rule ② above, implemented in one place so
you can read it in `src/ium_utils.py`.

**What this cell does** — loops over the 10 LOIO folds; in each one it packs the
arrays, trains a fresh network from scratch, predicts the held-out intensity,
converts the predictions back to physical units (`sc_y.inverse_transform`,
inside `iu.predict`) and stores them. Afterwards it aggregates all 50
out-of-fold predictions into one honest score table.

**Output** — per-fold RMSE printed live (in mm, Pa and DoC units), and `agg`:
the aggregate RMSE and R² over all 50 out-of-fold predictions. On a Colab GPU
this takes ~4 minutes; on CPU roughly 20–30.

> Classroom budget: 60 epochs, single seed. The paper uses 100 epochs × 5 seeds,
> so expect numbers that are close but slightly lower — and a little different
> on every machine.
"""),
("code", """\
import torch
print("device:", "cuda" if torch.cuda.is_available() else "cpu",
      " (Colab: Runtime → Change runtime type → GPU makes this ~5× faster)")

EPOCHS = 60      # classroom budget; the paper uses 100 epochs × 5 seeds
iu.set_seed(42)  # make this run reproducible (python, numpy and torch RNGs)

results, all_pred, all_true = [], [], []
# One fold per exposure intensity: 45 prints train, the 5 prints of the held-out
# intensity are tested. The model has never seen that light dose.
for tr_df, te_df, inten in iu.loio_folds(full):
    # Pack the three branch inputs; scalers are fitted on tr_df ONLY (leak-free).
    a_tr, a_te, sc_y = iu.assemble_arrays(tr_df, te_df)
    # A fresh network per fold — reusing one would leak across folds.
    net = iu.train_model(a_tr, epochs=EPOCHS)
    # predict() also undoes the label scaling, so yp is in physical units.
    yp = iu.predict(net, a_te, sc_y)
    all_pred.append(yp); all_true.append(te_df[iu.LABEL_COLS].values)
    # Per fold we print RMSE, not R². A fold holds only 5 prints made at one
    # intensity, so their labels barely vary; R² divides by that tiny variance
    # and becomes wild. RMSE is in physical units and stays meaningful.
    fold_rmse = iu.metrics(te_df[iu.LABEL_COLS].values, yp)["RMSE"]
    results.append(dict(intensity=inten, **fold_rmse))
    print(f"held-out intensity {inten:3d}%:   "
          + "   ".join(f"{l} RMSE={fold_rmse[l]:8.4g}" for l in iu.LABEL_COLS))

# Stack the 10 folds: every one of the 50 samples now has exactly one prediction
# made by a model that never saw its intensity.
YP, YT = np.vstack(all_pred), np.vstack(all_true)
print("\\n=== LOIO aggregate (all 50 out-of-fold predictions) ===")
agg = iu.metrics(YT, YP).round(3)
agg
"""),
("md", """\
Notes on what you should see (numbers vary a little with hardware/seed):

- **Thickness** R² ≈ 0.97+ — geometry lives in ToF; easy.
- **DoC** R² ≈ 0.6–0.75 — the within-layer probes carry it.
- **Modulus** is the hardest of the three. The paper's full protocol
  (100 epochs, 5 seeds, mild noise augmentation) reports 0.985 / 0.832 / 0.757.

**A note on R².** R² compares the model's error with the variance of the labels
themselves: 1 means perfect, 0 means no better than always predicting the mean,
and negative means worse than that. It can never exceed 1 — if you see `1.00`
printed anywhere, it is a rounded 0.995-or-so, not a number above one.

This is also why we print RMSE per fold and keep R² for the aggregate. A single
fold contains only 5 prints, all made at the same intensity, so their labels are
nearly identical and the variance R² divides by is tiny. Dividing by it makes
per-fold R² swing wildly — a modest error can come out as a large negative
number — even when the predictions are perfectly reasonable. The aggregate over
all 50 out-of-fold predictions spans the real spread of the dataset, so its R²
means what you expect it to mean.
"""),
("md", """\
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
("md", SETUP_MD),
("code", BOOT),
("md", """\
### Rebuilding the dataset table

**Input** — the same two files as Notebook 3: `data/feature11.csv` (the paper's
feature table for all 50 prints) and `data/condition_labels.csv` (the measured
thickness / modulus / DoC). No raw waveforms are needed any more — everything in
this notebook operates on the 344-dimensional representation.

**What this cell does** — reloads and merges the tables exactly as in Notebook 3
(each notebook is self-contained), and stores the list of the 330 layer-wise
column names in `seq_cols` for later reuse.

**Output** — `full` (50 × 344 + labels) and `seq_cols`.
"""),
("code", """\
# Same dataset table as Notebook 3 — each notebook runs standalone.
feats = iu.load_features("data/feature11.csv")
labels = iu.load_labels("data/condition_labels.csv")
full = feats.copy()
for c in iu.LABEL_COLS:
    full[c] = labels[c]
seq_cols = iu.seq_columns(full)        # the 330 Layer_{i}_{feature} columns
"""),
("md", """\
## 1. What do the attention gates say?

Train on one LOIO fold and read the sigmoid gates of the fusion layer.
Gate positions 0–31 = part-scale stats branch, 32–159 = layer-wise branch,
160–175 = printing-condition branch.

---

**Input** — `full`, split by `iu.loio_folds()`; we take fold index 4, i.e. a
**mid-range** exposure intensity is held out (edge folds are harder — exercise 1
asks you to try one).

**What this cell does** — trains one network for that fold, then calls
`iu.predict(..., return_attn=True)` to retrieve not just the predictions but the
**gate vector** the attention layer applied to each test sample. The three
branch outputs are concatenated in a fixed order before gating — stats (32
dimensions) + layer-wise sequence (128) + process conditions (16) = 176 — which
is where the segment boundaries 0–32–160–176 come from.

**Output** — a heatmap of all 176 gate values for each held-out print, plus a
bar chart of the per-branch means: a direct, quantitative answer to *"does the
ultrasound contribute more than the nominal recipe?"*
"""),
("code", """\
iu.set_seed(42)
folds = list(iu.loio_folds(full))
tr_df, te_df, inten = folds[4]              # hold out a mid-range intensity
a_tr, a_te, sc_y = iu.assemble_arrays(tr_df, te_df)
net = iu.train_model(a_tr, epochs=60)
# return_attn=True also gives the per-sample sigmoid gate vector (0...1 each).
yp, gates = iu.predict(net, a_te, sc_y, return_attn=True)
print("gates shape (test samples × fusion dims):", gates.shape)

# The fusion vector is the concatenation of the three branch outputs, in this
# fixed order — hence these index ranges.
seg = dict(stats=(0, 32), layerwise=(32, 160), process=(160, 176))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8),
                             gridspec_kw={"width_ratios": [2.4, 1]})
im = a1.imshow(gates, aspect="auto", cmap="viridis", vmin=0, vmax=1)
for name, (lo, hi) in seg.items():
    a1.axvline(hi - 0.5, color="w", lw=1)          # branch boundary
    a1.text((lo + hi) / 2, -0.8, name, ha="center", fontsize=10)
a1.set(xlabel="fusion-vector channel", ylabel="test sample",
       title=f"Sigmoid gate activations (held-out intensity {inten}%)")
plt.colorbar(im, ax=a1, shrink=0.8)

# Average gate per branch, normalised to 100% for a readable relative ranking.
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

---

**Input** — a *collapsed* version of the feature table: instead of 330 columns,
one average per physical descriptor (11 values) plus `layer_count` and
`intensity`. Averaging makes the ranking readable; the crucial detail is that
each print is averaged over its **own** number of layers only, because the
unused layer slots are zero padding and would drag the averages toward zero.

**What this cell does** — fits a gradient-boosting model per target and computes
**permutation importance**: shuffle one column, see how much the prediction
degrades. A feature the model truly relies on hurts a lot when scrambled.

**Output** — three horizontal bar charts (top 8 features per target). Compare
them with the two-scale design in Notebook 2 — the ranking should reproduce the
physics we built in.

> This section is about *interpretation*, so it fits on all 50 samples on
> purpose; it is not a performance claim. Performance numbers only ever come
> from the LOIO protocol.
"""),
("code", """\
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

# compact, physical feature set: per-feature layer-averages + process + stats
BASES = ["ToF", "amplitude", "RMS_Energy", "Center_Freq", "Bandwidth",
         "Wavelet_Energy_L0", "Wavelet_Energy_L1", "Wavelet_Energy_L2",
         "Wavelet_Energy_L3", "frame_diff_rms", "within_layer_std"]

def collapsed(df):
    \"\"\"330 layer-wise columns -> 11 per-feature averages (+ the 2 conditions).\"\"\"
    out = pd.DataFrame(index=df.index)
    n_real = df.layer.values.astype(int)      # this print's REAL number of layers
    for b in BASES:
        cols = [f"Layer_{i}_{b}" for i in range(1, 31)]
        V = df[cols].values
        # Average over real layers only — slots beyond n are zero padding and
        # would otherwise bias short prints toward zero.
        out[b] = [row[:n].mean() for row, n in zip(V, n_real)]
    out["layer_count"] = df.layer.values
    out["intensity"] = df.intensity.values
    return out

Xc = collapsed(full)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, target in zip(axes, iu.LABEL_COLS):
    m = HistGradientBoostingRegressor(random_state=0).fit(Xc, full[target])
    # Permutation importance: shuffle one column, measure the loss of skill.
    imp = permutation_importance(m, Xc, full[target], n_repeats=10,
                                 random_state=0)
    order = np.argsort(imp.importances_mean)[-8:]        # top 8, ascending
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

---

**Input** — the full 344-dimensional `X_all` and the labels `Y`, plus a
per-column magnitude scale so that "5% noise" means the same thing for a ToF in
microseconds and a frequency in hertz.

**What this cell does** — repeats the LOIO evaluation while adding Gaussian
noise to the **test** features only, at several levels, five repetitions each.
Training stays clean on purpose: we are asking "how does a model trained on good
data survive a degraded sensor in the field?", not "does noise help training?".

**Output** — R² versus noise level with error bars. A gentle slope means the
model leans on broad physical trends; a cliff would mean it depends on fragile
details.
"""),
("code", """\
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score

X_all = np.hstack([full[seq_cols + ["layer", "intensity"]].values,
                   iu.part_scale_stats(full).values])
Y = full[iu.LABEL_COLS].values
# Per-column magnitude, so a "5% noise" level is comparable across features
# whose units differ by many orders of magnitude (µs vs Hz vs ratios).
scale = np.abs(X_all).mean(axis=0)

def loio_r2_with_noise(noise_pct, n_rep=5, seed0=0):
    \"\"\"LOIO R² when the TEST features are corrupted by noise_pct % noise.\"\"\"
    r2s = []
    for rep in range(n_rep):                      # repeat: noise is random
        rng = np.random.default_rng(seed0 + rep)
        preds = np.zeros_like(Y)
        for tr_idx, te_idx in LeaveOneGroupOut().split(
                X_all, Y, full.intensity.values):
            rf = RandomForestRegressor(200, random_state=0, n_jobs=-1)
            rf.fit(X_all[tr_idx], Y[tr_idx])      # training data stays clean
            Xte = X_all[te_idx] + rng.normal(
                0, noise_pct / 100 * scale, X_all[te_idx].shape)
            preds[te_idx] = rf.predict(Xte)
        r2s.append([r2_score(Y[:, i], preds[:, i]) for i in range(3)])
    return np.array(r2s)                          # (n_rep, 3)

levels = [0, 1, 2, 5, 10, 20]
curves = {l: loio_r2_with_noise(l) for l in levels}

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for i, l in enumerate(iu.LABEL_COLS):
    mean = [curves[nv][:, i].mean() for nv in levels]
    sd = [curves[nv][:, i].std() for nv in levels]     # spread over repetitions
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

---

**Input** — `X_all` and `Y` again. The cell first computes and caches the
**out-of-fold** predictions for all 50 prints: 10 LOIO folds, each model
predicting only the five prints whose intensity it never saw.

**What this cell does** — after caching, `dashboard()` looks up one print by
(layer count, intensity), and plots measured versus predicted for the three
targets with the relative error. `ipywidgets.interact` turns its two arguments
into dropdowns over the values that actually exist in the design grid.

**Output** — an interactive dashboard. This is the honest version of a sensor
read-out: every number shown comes from a model that had never seen that
exposure intensity, exactly as it would be on a new print.
"""),
("code", """\
# Cache out-of-fold RF predictions for all 50 samples (fast enough for a widget).
# Out-of-fold = each sample is predicted by the one model that did NOT train on
# its intensity, so the dashboard cannot flatter itself.
preds = np.zeros_like(Y)
for tr_idx, te_idx in LeaveOneGroupOut().split(X_all, Y, full.intensity.values):
    rf = RandomForestRegressor(300, random_state=0, n_jobs=-1)
    rf.fit(X_all[tr_idx], Y[tr_idx])
    preds[te_idx] = rf.predict(X_all[te_idx])

import ipywidgets as w

def dashboard(layers=15, intensity_pct=70):
    \"\"\"Show measured vs predicted labels for one print of the design grid.\"\"\"
    m = (full.layer == layers) & (full.intensity == intensity_pct)
    if not m.any():
        print("no such sample"); return
    i = int(np.where(m)[0][0])
    meas, pred = Y[i], preds[i]              # truth and out-of-fold prediction
    names = ["thickness (mm)", "modulus (Pa)", "DoC (–)"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3))
    for ax, n, mv, pv in zip(axes, names, meas, pred):
        ax.bar(["measured", "predicted"], [mv, pv],
               color=["0.4", "tab:green"])
        err = abs(pv - mv) / (abs(mv) + 1e-9) * 100      # relative error, %
        ax.set_title(f"{n}\\nerror {err:.1f}%")
    plt.suptitle(f"{layers} layers @ {intensity_pct}% intensity "
                 f"(I = {iu.INTENSITY_MW[intensity_pct]} mW/cm²) — "
                 "prediction from a model that never saw this intensity")
    plt.tight_layout(); plt.show()

# Dropdowns are built from the values that actually exist in the design grid.
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
