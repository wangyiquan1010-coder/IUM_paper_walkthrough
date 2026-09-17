# -*- coding: utf-8 -*-
"""Extract the teaching subset from the original IUM dataset (READ-ONLY on originals).

Outputs into ../data/:
  - feature11.csv, condition_labels.csv   (verbatim copies)
  - waveforms_15L_I1.npz / waveforms_15L_I7.npz / waveforms_30L_I7.npz
      each: dict layer -> (n_frames, 62509) float32, keys 'layer_01'..,
      plus 'fs' (2.5e9), 'meta' (json string)
"""
import io, sys, os, json, shutil, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd

SRC = r"C:\Users\wangy\OneDrive - University of Pittsburgh\Desktop\IUM\IUM_50samples\IUM_50samples"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)

shutil.copyfile(os.path.join(SRC, "features", "feature11.csv"),
                os.path.join(OUT, "feature11.csv"))
shutil.copyfile(os.path.join(SRC, "condition_labels.csv"),
                os.path.join(OUT, "condition_labels.csv"))
print("copied feature11.csv, condition_labels.csv")

SAMPLES = {
    "waveforms_15L_I1":  (r"waveforms\1.21_15\1.21_10%",  15, 10, 10.70),
    "waveforms_15L_I7":  (r"waveforms\1.21_15\1.21_70%",  15, 70, 20.16),
    "waveforms_30L_I7":  (r"waveforms\1.21_30\1.21_70%",  30, 70, 20.16),
}

for name, (rel, n_layers, inten_pct, inten_mw) in SAMPLES.items():
    base = os.path.join(SRC, rel)
    layer_dirs = sorted([d for d in os.listdir(base)
                         if os.path.isdir(os.path.join(base, d))],
                        key=lambda d: int(d.rsplit("_", 1)[-1]))
    arrays = {}
    for d in layer_dirs:
        li = int(d.rsplit("_", 1)[-1])
        frames = sorted(glob.glob(os.path.join(base, d, "*.csv")),
                        key=lambda p: int(os.path.basename(p).rsplit("_", 1)[-1].split(".")[0]))
        wf = np.stack([pd.read_csv(f, header=None).values.ravel().astype(np.float32)
                       for f in frames])
        arrays[f"layer_{li:02d}"] = wf
    meta = dict(n_layers=n_layers, intensity_pct=inten_pct,
                intensity_mw_cm2=inten_mw, fs_hz=2.5e9,
                frames_per_layer=int(wf.shape[0]),
                source="IUM_50samples (Wang & Zhao, Add. Manuf. 2026, 105300)")
    np.savez_compressed(os.path.join(OUT, name + ".npz"),
                        meta=json.dumps(meta), **arrays)
    sz = os.path.getsize(os.path.join(OUT, name + ".npz")) / 1e6
    print(f"{name}: {len(arrays)} layers x {wf.shape} -> {sz:.1f} MB")
