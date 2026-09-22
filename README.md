# Listening to a Part Being Born — Ultrasonic Monitoring of 3D Printing, Hands-On

A four-notebook teaching series (graduate level) built on the dataset and code of:

> Wang, Y., & Zhao, X. (2026). *Machine learning–aided in-situ ultrasonic
> characterization for multi-parametric monitoring of vat photopolymerization.*
> **Additive Manufacturing**, 105300.
> https://doi.org/10.1016/j.addma.2026.105300

A 10 MHz pulse-echo transducer embedded in a DLP 3D-printer's printhead listens
to every layer as it cures. These notebooks take you from raw ultrasonic
A-scans to a physics-informed, leak-free machine-learning "soft sensor" that
predicts part thickness, storage modulus, and degree of conversion — in real
time, from sound alone.

## The notebooks

| # | Notebook | You will learn | Highlights |
|---|---|---|---|
| 1 | `01_ultrasound_meets_3dprinting` | reading A-scans; B1/B2/B3 echoes; ToF | layer-by-layer **waveform animation**; I1-vs-I7 "race"; interactive 3D waterfall; waveform browser widget |
| 2 | `02_physics_informed_features` | the two-scale framework; 11 layer-wise descriptors | **within-layer curing animation**; rebuilding features from raw data; design-grid heatmaps |
| 3 | `03_learning_and_leakage` | 344-d representation; **data leakage**; LOIO CV; attention-fusion network | live leakage demo (random split vs LOIO) — a true peer-review story; full LOIO training run |
| 4 | `04_interpretation_and_deployment` | attention gates; feature importance; robustness | gate heatmaps ("does sensing beat the recipe?"); **virtual-sensor dashboard** |

Open in Colab (live once the repo is pushed to GitHub):

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wangyiquan1010-coder/IUM_teaching_colab/blob/main/notebooks/01_ultrasound_meets_3dprinting.ipynb) Notebook 1
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wangyiquan1010-coder/IUM_teaching_colab/blob/main/notebooks/02_physics_informed_features.ipynb) Notebook 2
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wangyiquan1010-coder/IUM_teaching_colab/blob/main/notebooks/03_learning_and_leakage.ipynb) Notebook 3
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wangyiquan1010-coder/IUM_teaching_colab/blob/main/notebooks/04_interpretation_and_deployment.ipynb) Notebook 4

Everything (code + 12 MB of data) ships in this repository — no extra downloads.
Notebook 3's training runs in ~4 min on a Colab GPU (Runtime → Change runtime
type → T4 GPU) or ~20–30 min on CPU with the default classroom settings.

## Repository layout

```
notebooks/   the four teaching notebooks (executed outputs included)
src/         ium_utils.py — loading, echo tracking, features, model, LOIO
data/        feature11.csv, condition_labels.csv, 3 sample-waveform .npz files
figs/        figures from the paper used as illustrations
tools/       extract_teaching_data.py, make_notebooks.py (maintainers only)
```

## Dataset (teaching subset)

- 50 printed cylinders (5 layer counts × 10 exposure intensities) — full
  **feature table** (11 physics-informed descriptors × 30 layer slots) and
  **labels** (Keyence thickness, rheometer storage modulus, Raman DoC).
- Raw waveforms for **three representative samples** (15 L @ I1, 15 L @ I7,
  30 L @ I7): every layer, 6 frames/layer, 62,509 points @ 2.5 GHz.
- See `DATA_CARD.md` for full provenance, units, and known caveats.

## For instructors

Notebook 1 ends with exercises. The series is designed for two 75-minute
sessions (NB1–2, NB3–4).

## License & citation

Data and figures © the authors, provided for teaching use with attribution.
If you use this material, cite the paper above.
