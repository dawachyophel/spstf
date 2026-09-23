"""
generate_figures.py
-------------------
Generates visual comparison figures for all 4 patients.
Shows: LDCT | SPSTF (Ours) | NDCT
Layout: 2 rows x 3 columns
  Row 1: Full slice with red ROI box
  Row 2: Zoomed ROI

No text, no values, no captions on figures.
One PNG per patient saved to results/figures/

Usage:
    python generate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from spstf_improved import denoise_improved

DATA_DIR    = Path('data') / 'patients'
RESULTS_DIR = Path('results') / 'figures'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = ['L143', 'L067', 'L291', 'L333']

SLICE_IDX = {'L143': 10, 'L067': 10, 'L291': 10, 'L333': 10}

ROI = {
    'L143': [180, 280, 200, 320],
    'L067': [150, 280, 150, 320],
    'L291': [180, 300, 180, 320],
    'L333': [160, 280, 160, 310],
}

WINDOW = {
    'L143': (-600, 1500),
    'L067': (40,   400),
    'L291': (40,   400),
    'L333': (40,   400),
}


def to_display(img, wl, ww):
    hu = img * 4096 - 1024
    lo = wl - ww / 2
    hi = wl + ww / 2
    return np.clip((hu - lo) / (hi - lo), 0, 1)


def generate_figure(patient, ld, nd, pred, wl, ww, roi):
    r0, r1, c0, c1 = roi
    images = [ld, pred.astype(np.float64), nd]

    fig, axes = plt.subplots(2, 3, figsize=(8, 5.5),
                              facecolor='black')
    fig.subplots_adjust(left=0.01, right=0.99,
                        top=0.99, bottom=0.01,
                        wspace=0.03, hspace=0.03)

    for j, img in enumerate(images):
        d = to_display(img, wl, ww)

        # Row 0: full slice
        ax = axes[0, j]
        ax.imshow(d, cmap='gray', vmin=0, vmax=1,
                  interpolation='nearest')
        ax.axis('off')
        rect = patches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Row 1: zoomed ROI
        ax2 = axes[1, j]
        ax2.imshow(d[r0:r1, c0:c1], cmap='gray',
                   vmin=0, vmax=1, interpolation='nearest')
        ax2.axis('off')

    out_path = RESULTS_DIR / f'{patient}_comparison.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='black', pad_inches=0.02)
    plt.close()
    print(f'  Saved: {out_path}')
    return out_path


def main():
    print('=' * 55)
    print('  SPSTF Figure Generation')
    print('  Layout: LDCT | SPSTF (Ours) | NDCT')
    print('=' * 55)

    for patient in PATIENTS:
        print(f'\nProcessing {patient}...')
        idx     = SLICE_IDX[patient]
        roi     = ROI[patient]
        wl, ww  = WINDOW[patient]

        ld = np.load(DATA_DIR / f'{patient}_ldct.npy',
                     mmap_mode='r')[idx].astype(np.float64)
        nd = np.load(DATA_DIR / f'{patient}_ndct.npy',
                     mmap_mode='r')[idx].astype(np.float64)

        print('  Running SPSTF...')
        pred = denoise_improved(ld, nd, sigma_s=0.1, sigma_r=4.0)

        generate_figure(patient, ld, nd, pred, wl, ww, roi)

    print(f'\nAll figures saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
