# SPSTF: Adaptive Weighted Sparse Coding with Edge-Aware Transform Filtering for LDCT Denoising

Official Python implementation of the paper:

**"Adaptive weighted sparse coding with edge-aware transform filtering for clinically reliable low-dose computed tomography denoising"**

---

## Overview

SPSTF is a fully unsupervised LDCT image denoising framework that requires:
- No paired training data
- No pre-trained network weights
- No GPU acceleration

The method constructs a data-driven dictionary from grouped similar patches using SVD and incorporates three adaptive weight matrices into the sparse coding formulation, solved via closed-form weighted SVD shrinkage. An edge-aware recursive transform filter is applied as a post-processing step to recover fine anatomical boundaries.

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10 or later recommended.

---

## Dataset

Experiments use real quarter-dose CT image pairs from the **AAPM Mayo 2016 Low-Dose CT Grand Challenge**:

https://www.aapm.org/grandchallenge/lowdosect/

Four patients are used for evaluation:
- L143 (Chest)
- L067 (Abdomen)
- L291 (Abdomen)
- L333 (Abdomen)

---

## Usage

### Step 1: Prepare patient data from DICOM files

```bash
python prepare_patients.py
```

This loads DICOM files from `data/old_dataset/` and saves normalised npy arrays to `data/patients/`.

### Step 2: Run SPSTF evaluation

```bash
python run_spstf_patients.py --n_slices 20
```

Outputs PSNR, SSIM, and RMSE per patient. Results saved to `results/spstf_results.csv`.

For a quick test on 3 slices:

```bash
python run_spstf_patients.py --n_slices 3
```

### Step 3: Generate convergence plot (Fig. 2)

```bash
python generate_convergence_plot.py
```

Saves to `results/figures/convergence_plot.png`.

### Step 4: Generate visual comparison figures (Figs. 3-6)

```bash
python generate_figures.py
```

Generates LDCT / SPSTF / NDCT comparison figures for all 4 patients.
Saves to `results/figures/`.

---

## Core Method

The main denoising function is in `spstf_improved.py`:

```python
from spstf_improved import denoise_improved
import numpy as np

# ldct and ndct are normalised float64 arrays in [0, 1]
denoised = denoise_improved(ldct, ndct, sigma_s=0.1, sigma_r=4.0)
```

Blind noise estimation is used automatically when ndct is not available:

```python
denoised = denoise_improved(ldct, sigma_s=0.1, sigma_r=4.0)
```

---

## Parameters

All parameters are automatically configured from the estimated noise level nSig.
No manual tuning is required. See Table 1 in the paper for the full parameter schedule.

---

## Results

Evaluated on real quarter-dose CT pairs from the AAPM Mayo 2016 dataset:

| Patient | Region  | PSNR (dB)       | SSIM            | RMSE            |
|---------|---------|-----------------|-----------------|-----------------|
| L143    | Chest   | 47.82 +/- 0.23  | 0.9865 +/- 0.001| 1.04 +/- 0.03   |
| L067    | Abdomen | 48.01 +/- 0.26  | 0.9886 +/- 0.001| 1.01 +/- 0.03   |
| L291    | Abdomen | 42.99 +/- 0.21  | 0.9733 +/- 0.001| 1.81 +/- 0.04   |
| L333    | Abdomen | 48.16 +/- 0.30  | 0.9888 +/- 0.001| 1.00 +/- 0.03   |

---

## Citation

If you use this code, please cite:

```
@article{spstf2024,
  title={Adaptive weighted sparse coding with edge-aware transform filtering
         for clinically reliable low-dose computed tomography denoising},
  author={Lepcha, Dawachyophel and Goyal, Bhawna and Dogra, Ayush},
  journal={Scientific Reports},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License.
