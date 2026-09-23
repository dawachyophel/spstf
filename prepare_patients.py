"""
prepare_patients.py
-------------------
Loads 4 selected patients from raw DICOMs and saves per-patient npy files.

Patients:
  L143 — Chest       (matches original C140)
  L067 — Abdomen     (matches original L133)
  L291 — Abdomen     (matches original L218, best noise level)
  L333 — Abdomen     (matches original N300)

Output (all in D:\spstf_python\data\patients\):
  L143_ldct.npy, L143_ndct.npy   (234, 512, 512) float32 [0,1]
  L067_ldct.npy, L067_ndct.npy   (224, 512, 512) float32 [0,1]
  L291_ldct.npy, L291_ndct.npy   (343, 512, 512) float32 [0,1]
  L333_ldct.npy, L333_ndct.npy   (244, 512, 512) float32 [0,1]

Usage:
    python prepare_patients.py
"""

import os
import glob
import numpy as np
import pydicom
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

BASE_Q  = r'D:\spstf_python\data\old_dataset\quarter_3mm'
BASE_F  = r'D:\spstf_python\data\old_dataset\full_3mm'
OUT_DIR = Path('data') / 'patients'
OUT_DIR.mkdir(parents=True, exist_ok=True)

HU_MIN = -1024.0
HU_MAX =  3072.0

PATIENTS = ['L143', 'L067', 'L291', 'L333']


def normalise(img_hu):
    img = np.clip(img_hu, HU_MIN, HU_MAX)
    return ((img - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def load_patient(folder):
    files = sorted(glob.glob(
        os.path.join(folder, '**', '*.IMA'), recursive=True))
    if not files:
        files = sorted(glob.glob(
            os.path.join(folder, '**', '*.dcm'), recursive=True))
    slices = []
    for f in tqdm(files, desc=os.path.basename(folder), leave=False):
        ds  = pydicom.dcmread(f)
        img = ds.pixel_array.astype(np.float32)
        img = (img * float(ds.RescaleSlope)
               + float(ds.RescaleIntercept))
        slices.append(img)
    return np.array(slices, dtype=np.float32)


print('=' * 60)
print('  Preprocessing 4 patients from AAPM Mayo2016')
print('=' * 60)

for patient in PATIENTS:
    out_ld = OUT_DIR / f'{patient}_ldct.npy'
    out_nd = OUT_DIR / f'{patient}_ndct.npy'

    if out_ld.exists() and out_nd.exists():
        ld = np.load(out_ld, mmap_mode='r')
        nd = np.load(out_nd, mmap_mode='r')
        p  = np.mean([psnr(nd[i], ld[i], data_range=1.0)
                      for i in range(min(5, len(ld)))])
        nSig = np.mean([np.std(ld[i]-nd[i])*255
                        for i in range(min(5, len(ld)))])
        print(f'  {patient}: already exists  '
              f'{ld.shape}  PSNR={p:.2f}dB  nSig={nSig:.2f}')
        continue

    print(f'\n  Loading {patient}...')
    ldct = load_patient(os.path.join(BASE_Q, patient))
    ndct = load_patient(os.path.join(BASE_F, patient))

    assert ldct.shape == ndct.shape, \
        f'Shape mismatch: {ldct.shape} vs {ndct.shape}'

    ldct_n = normalise(ldct)
    ndct_n = normalise(ndct)

    np.save(out_ld, ldct_n)
    np.save(out_nd, ndct_n)

    p    = np.mean([psnr(ndct_n[i], ldct_n[i], data_range=1.0)
                    for i in range(min(5, len(ldct_n)))])
    nSig = np.mean([np.std(ldct_n[i]-ndct_n[i])*255
                    for i in range(min(5, len(ldct_n)))])
    print(f'  {patient}: {ldct_n.shape}  '
          f'PSNR={p:.2f}dB  nSig={nSig:.2f}  ✓')

print()
print('=' * 60)
print('  Summary')
print('=' * 60)
for f in sorted(OUT_DIR.glob('*.npy')):
    a    = np.load(f, mmap_mode='r')
    size = f.stat().st_size / 1e9
    print(f'  {f.name:<25} {str(a.shape):<22} {size:.2f} GB')

print()
print('  Next: python run_patients.py')
