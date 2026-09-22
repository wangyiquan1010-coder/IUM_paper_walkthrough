# Listening to a Part Being Born — Ultrasonic Monitoring of 3D Printing, Hands-On

A four-notebook reproduction and walkthrough (graduate level) of the dataset and
code behind:

> Wang, Y., & Zhao, X. (2026). *Machine learning–aided in-situ ultrasonic
> characterization for multi-parametric monitoring of vat photopolymerization.*
> **Additive Manufacturing**, 105300.
> https://doi.org/10.1016/j.addma.2026.105300

A 10 MHz pulse-echo transducer embedded in a DLP 3D-printer's printhead listens
to every layer as it cures. These notebooks take you from raw ultrasonic
A-scans to a physics-informed, leak-free machine-learning "soft sensor" that
predicts part thickness, storage modulus, and degree of conversion — in real
time, from sound alone.

Every code cell is preceded by a short **Input / What this cell does / Output**
note saying which file or table it reads, what it computes, and which later cell
uses the result, and the code itself is commented for readers new to ultrasound
or to machine learning.

## The notebooks

| # | Notebook | You will learn | Highlights |
|---|---|---|---|
| 1 | `01_ultrasound_meets_3dprinting` | reading A-scans; the B1 / B2 / B3 echoes; time of flight | layer-by-layer **waveform animation**; weak-cure vs strong-cure "race"; interactive 3D waterfall with a fixed starting view; waveform browser widget |
| 2 | `02_physics_informed_features` | the two-scale framework; all 11 layer-wise descriptors, computed from raw samples with their formulas | **within-layer curing animation**, normalised so the two prints are comparable; design-grid heatmaps over all 50 prints |
| 3 | `03_model_and_training` | the 344-d representation; why the train/test split decides everything; the attention-fusion network | random split vs **leave-one-intensity-out**, measured side by side; full LOIO training run; the paper's architecture figure |
| 4 | `04_interpretation_and_deployment` | reading attention gates; feature importance; robustness | the fusion forward pass and **gate = sigmoid(W₂·ReLU(W₁·x))** written out by hand; permutation importance; sensor-noise degradation |

Open in Colab:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wangyiquan1010-coder/IUM_paper_walkthrough/blob/main/notebooks/01_ultrasound_meets_3dprinting.ipynb) Notebook 1
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wangyiquan1010-coder/IUM_paper_walkthrough/blob/main/notebooks/02_physics_informed_features.ipynb) Notebook 2
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wangyiquan1010-coder/IUM_paper_walkthrough/blob/main/notebooks/03_model_and_training.ipynb) Notebook 3
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wangyiquan1010-coder/IUM_paper_walkthrough/blob/main/notebooks/04_interpretation_and_deployment.ipynb) Notebook 4

Everything (code + 12 MB of data) ships in this repository — no extra downloads.
All four run end to end in under a minute on a laptop. Timed twice with
`nbconvert --execute`, they took 24–27 s, 18–21 s, 43–50 s and 37–41 s
respectively. Notebook 3 trains the
network on all ten folds, and the model is small enough that a GPU buys little:
its training loop takes 20 s on CPU against 14 s on CUDA on the same machine, so
the default Colab runtime is fine. Colab's free CPU has fewer cores than a
laptop, so expect somewhat longer there; we have not timed it on Colab itself.

## Repository layout

```
notebooks/   the four teaching notebooks (executed outputs included)
src/         ium_utils.py — loading, echo tracking, features, model, LOIO
data/        feature11.csv, condition_labels.csv, 3 sample-waveform .npz files
figs/        figures from the paper used as illustrations
tools/       extract_teaching_data.py, make_notebooks.py (maintainers only)
```

`tools/make_notebooks.py` is the source of truth for the notebooks: edit it,
re-run it, then re-execute the notebooks rather than editing the `.ipynb` files
by hand.

## Dataset (teaching subset)

- 50 printed cylinders (5 layer counts × 10 exposure intensities) — the full
  **feature table** (11 physics-informed descriptors × 30 layer slots) and
  **labels** (Keyence thickness, rheometer storage modulus, Raman DoC).
- Raw waveforms for **three representative samples** (15 L @ I1, 15 L @ I7,
  30 L @ I7): every layer, 6 frames/layer, 62,509 points @ 2.5 GHz.
- See `DATA_CARD.md` for full provenance, units, and known caveats.

## For instructors

Notebook 3 is the one to try first: it trains on all ten folds and is the only
one whose runtime depends on the machine you give it.

Results are reproduced, not quoted: the classroom budget (60 epochs, one seed)
gives LOIO R² ≈ 0.98 / 0.80 / 0.72 against the paper's 0.985 / 0.832 / 0.757.
Numbers shift slightly with hardware and seed, and the notebooks say so where it
matters.

## License & citation

Data and figures © the authors, provided for teaching use with attribution.
If you use this material, cite the paper above.
