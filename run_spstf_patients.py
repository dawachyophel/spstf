"""
run_spstf_patients.py
---------------------
Evaluates the proposed SPSTF method on real quarter-dose CT pairs.
Outputs PSNR, SSIM, and RMSE per patient.

No comparative methods. No GPU required. No training data required.

Dataset: AAPM Mayo 2016 Low-Dose CT Grand Challenge
Patients: L143 (Chest), L067 (Abdomen), L291 (Abdomen), L333 (Abdomen)

Usage:
    python run_spstf_patients.py --n_slices 20
    python run_spstf_patients.py --n_slices 5   (quick test)

Results saved to: results/spstf_results.csv
"""

import argparse
import csv
import time
import numpy as np
from pathlib import Path
from skimage.metrics import (peak_signal_noise_ratio as psnr_fn,
                              structural_similarity as ssim_fn)
from spstf_improved import denoise_improved

DATA_DIR    = Path('data') / 'patients'
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = {
    'L143': 'Chest',
    'L067': 'Abdomen',
    'L291': 'Abdomen',
    'L333': 'Abdomen',
}


def metrics(pred, ref):
    p = float(psnr_fn(ref, pred, data_range=1.0))
    s = float(ssim_fn(ref, pred, data_range=1.0))
    r = float(np.sqrt(np.mean((pred * 255 - ref * 255) ** 2)))
    return p, s, r


def evaluate_patient(patient: str, n_slices: int) -> dict:
    ld_path = DATA_DIR / f'{patient}_ldct.npy'
    nd_path = DATA_DIR / f'{patient}_ndct.npy'

    if not ld_path.exists():
        print(f'  {patient}: data not found. Run prepare_patients.py first.')
        return None

    ld_all = np.load(ld_path, mmap_mode='r')
    nd_all = np.load(nd_path, mmap_mode='r')
    n      = min(n_slices, len(ld_all))

    psnrs, ssims, rmses, times = [], [], [], []

    print(f'\n  Patient {patient} ({PATIENTS[patient]}) -- {n} slices')
    print(f'  {"Slice":>6}  {"nSig":>6}  {"PSNR_in":>8}  '
          f'{"PSNR_out":>9}  {"SSIM_out":>9}  {"RMSE_out":>9}  {"Time":>8}')
    print(f'  {"-"*70}')

    for i in range(n):
        ld = ld_all[i].astype(np.float64)
        nd = nd_all[i].astype(np.float64)

        from spstf_improved import estimate_noise
        nSig = float(np.clip(estimate_noise(ld, nd), 0.5, 79.0))

        t0   = time.perf_counter()
        pred = denoise_improved(ld, nd, sigma_s=0.1, sigma_r=4.0)
        t    = time.perf_counter() - t0

        p_in, _, _  = metrics(ld, nd)
        p, s, r     = metrics(pred.astype(np.float64), nd)

        psnrs.append(p); ssims.append(s)
        rmses.append(r); times.append(t)

        print(f'  {i+1:>6}  {nSig:>6.2f}  {p_in:>8.4f}  '
              f'{p:>9.4f}  {s:>9.4f}  {r:>9.4f}  {t:>7.1f}s')

    result = {
        'patient':    patient,
        'region':     PATIENTS[patient],
        'n_slices':   n,
        'PSNR_mean':  round(float(np.mean(psnrs)), 4),
        'PSNR_std':   round(float(np.std(psnrs)),  4),
        'SSIM_mean':  round(float(np.mean(ssims)), 4),
        'SSIM_std':   round(float(np.std(ssims)),  4),
        'RMSE_mean':  round(float(np.mean(rmses)), 4),
        'RMSE_std':   round(float(np.std(rmses)),  4),
        'time_mean':  round(float(np.mean(times)), 1),
    }

    print(f'\n  Summary:')
    print(f'    PSNR : {result["PSNR_mean"]:.4f} +/- {result["PSNR_std"]:.4f} dB')
    print(f'    SSIM : {result["SSIM_mean"]:.4f} +/- {result["SSIM_std"]:.4f}')
    print(f'    RMSE : {result["RMSE_mean"]:.4f} +/- {result["RMSE_std"]:.4f}')
    print(f'    Time : {result["time_mean"]:.1f} s/slice')

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SPSTF on real quarter-dose CT pairs')
    parser.add_argument('--n_slices', default=20, type=int,
                        help='Number of slices per patient (default: 20)')
    parser.add_argument('--patients', default='all',
                        help='Comma-separated patient IDs or "all"')
    args = parser.parse_args()

    patients = (list(PATIENTS.keys()) if args.patients == 'all'
                else args.patients.split(','))

    print('=' * 70)
    print('  SPSTF Evaluation -- Real Quarter-Dose CT Pairs')
    print('  AAPM Mayo 2016 Low-Dose CT Grand Challenge')
    print('=' * 70)

    all_results = []
    for patient in patients:
        result = evaluate_patient(patient, args.n_slices)
        if result:
            all_results.append(result)

    if not all_results:
        print('\nNo results. Check data directory.')
        return

    # Summary table
    print(f'\n{"=" * 70}')
    print('  Final Summary')
    print(f'{"=" * 70}')
    print(f'  {"Patient":<8} {"Region":<10} {"PSNR":>12} {"SSIM":>12} {"RMSE":>12}')
    print(f'  {"-" * 58}')
    for r in all_results:
        print(f'  {r["patient"]:<8} {r["region"]:<10} '
              f'{r["PSNR_mean"]:>6.4f}+/-{r["PSNR_std"]:<5.4f} '
              f'{r["SSIM_mean"]:>6.4f}+/-{r["SSIM_std"]:<5.4f} '
              f'{r["RMSE_mean"]:>6.4f}+/-{r["RMSE_std"]:<5.4f}')

    # Average
    avg_psnr = np.mean([r['PSNR_mean'] for r in all_results])
    avg_ssim = np.mean([r['SSIM_mean'] for r in all_results])
    avg_rmse = np.mean([r['RMSE_mean'] for r in all_results])
    print(f'  {"-" * 58}')
    print(f'  {"Average":<18} {avg_psnr:>6.4f}{"":>11} '
          f'{avg_ssim:>6.4f}{"":>11} {avg_rmse:>6.4f}')

    # Save CSV
    csv_path = RESULTS_DIR / 'spstf_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f'\n  Results saved: {csv_path}')


if __name__ == '__main__':
    main()
