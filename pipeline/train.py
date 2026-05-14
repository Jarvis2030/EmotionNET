from config import CONFIG, parse_args, apply_args_to_config
from utils import flag, set_seed
from load import load_raw_datasets
from segment import segment_all
from relabel import run_gmm_relabeling
from datasets import build_tensor_dataset
from trainer import run_loso_fold
from reporting import save_results

import pandas as pd

# SNN model — imported only when --snn flag is used
try:
    from model import EmotionNET, spike_rate_loss

    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False

import torch
from pathlib import Path

def main():
    args = parse_args()
    cfg = apply_args_to_config(args, CONFIG)
    set_seed(cfg['seed'])

    out_dir = Path(cfg['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    df_d, df_s = load_raw_datasets(cfg)  # load raw dataset

    df_seg_d, df_seg_s = segment_all(df_d, df_s, cfg) # segment

    df_seg_d = run_gmm_relabeling(df_seg_d, df_seg_s, CONFIG) # relabel
    
    seed_ds_full = build_tensor_dataset(df_seg_s, CONFIG) # Pre-build SEED-IV dataset (shared across folds)

    # LOSO
    subjects = sorted(df_seg_d['subject'].unique().tolist())
    if CONFIG['subjects']:
        subjects = [s for s in subjects if s in CONFIG['subjects']]
    flag(f'LOSO  —  {len(subjects)} subjects: {subjects}')

    all_results = []
    for i, subj in enumerate(subjects):
        print(f'\n{"─"*60}')
        print(f'  Fold {i+1}/{len(subjects)}  —  Subject {subj:02d}')
        print('─'*60, flush=True)

        result = run_loso_fold(df_seg_d, seed_ds_full, subj, CONFIG)
        
        all_results.append(result)
        print(f'  ✓ S{subj:02d}  BestVal={result["best_val_trial_acc"]:.3f}  '
              f'TestTrial={result["final_test_trial_acc"]:.3f}', flush=True)

        # Checkpoint after every subject
        rows_so_far = [{k: v for k, v in r.items() if k != 'history'} for r in all_results]
        ckpt_name = f"{CONFIG.get('_run_name', 'routeC')}_checkpoint.csv"
        pd.DataFrame(rows_so_far).to_csv(out_dir / ckpt_name, index=False)
        print(f'  Checkpoint → {out_dir}/{ckpt_name}', flush=True)

    save_results(all_results, CONFIG) # Save all result and plot and checkpoint

if __name__ == '__main__':
    main()