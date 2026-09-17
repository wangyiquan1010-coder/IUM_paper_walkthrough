# Data Card — IUM Teaching Subset

**Source**: In-situ ultrasonic monitoring experiments for Wang & Zhao,
*Additive Manufacturing* (2026) 105300, ZIP-AM Laboratory, University of
Pittsburgh. This is a curated teaching subset; originals remain with the lab.

## Files

### `feature11.csv` — 50 samples × 336 columns
The paper's final layer-wise feature table (un-augmented, one row per printed
sample; row order defines `sample_id` 0–49).

| column | meaning |
|---|---|
| `layer` | total printed layers ∈ {10, 15, 20, 25, 30} |
| `intensity` | exposure intensity in % of max power ∈ {10 … 100}; mW/cm² mapping below |
| `Layer_{i}_{feat}` | feature `feat` of layer *i* (i = 1…30; zero-padded past the real build height) |

The 11 per-layer features: `ToF` (µs, B1→B2), `amplitude` (B2/B1 envelope
ratio), `RMS_Energy`, `Center_Freq` (Hz), `Bandwidth` (Hz),
`Wavelet_Energy_L0..L3`, `frame_diff_rms`, `within_layer_std`
(the last two computed across the 6 within-layer frames).

Intensity mapping (% → mW/cm²): 10→10.70, 20→12.57, 30→14.35, 40→15.93,
50→17.48, 60→18.86, 70→20.16, 80→21.41, 90→22.57, 100→23.78.

### `condition_labels.csv` — 50 samples × 5 columns
`layer`, `intensity`, plus ex-situ ground truth (same row order as
`feature11.csv`):

| label | instrument | unit |
|---|---|---|
| `thickness` | Keyence VR-3200 profilometer | mm |
| `modulus` | Anton Paar MCR 302e rheometer, G′ @ 0.1 rad/s | Pa |
| `DoC` | Raman spectroscopy (1640/1610 cm⁻¹ band ratio) | – (0–1) |

### `waveforms_15L_I1.npz`, `waveforms_15L_I7.npz`, `waveforms_30L_I7.npz`
Raw A-scans for three representative prints. Arrays `layer_01 … layer_NN`,
each `(6, 62509)` float32: frame 0 = reference (after stage settling),
frames 1–5 during the 15 s exposure at 3 s intervals. `meta` = JSON string
(layer count, intensity, fs). Sampling 2.5 GHz (0.4 ns/point, 25 µs window
starting at trigger). The first layer's 21 s exposure originally produced one
extra frame; loading code keeps reference + last five for a uniform shape.

## Known caveats (teach them!)

- Labels are **part-level** (one value per printed sample), not per layer.
- Surface-sensitive labels: Raman DoC probes the sample surface region.
- `modulus` spans ~3 orders of magnitude across the design grid — percentage
  errors are dominated by a few stiff samples (see paper §4.3).
- Sample count is small (n = 50) by ML standards — that is *the point* of the
  leak-free evaluation lesson in Notebook 3.

## Ethics / license

Experimental materials data; no personal data. Provided for teaching with
attribution; for other uses contact the authors.
