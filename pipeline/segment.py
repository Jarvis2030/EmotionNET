import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from utils import _parse_eeg

def artifact_reject_and_segment(df, config, dataset_tag):
    WIN    = config['window_size']
    STRIDE = config['stride']
    # Rejection threshold
    PTP_TH  = config['artifact_ptp_uv']   # 6.0 σ
    FLAT_TH = config['artifact_flat_uv']  # 0.01 σ
    kept, rejected = 0, 0
    rows = []

    # subject-video based grouping (avoid signal cutoff)
    groups = list(df.groupby(['subject', 'video']))
    n = len(groups) 

    for gi, ((subj, vid), g) in enumerate(groups):

        if gi % max(1, n // 10) == 0: # progress tracking
            print(f'  [{gi+1}/{n}] subj={subj} vid={vid}', flush=True)

        label = int(g['label'].iloc[0]) # get label from the first group
        
        # Retrieve data from each channel 
        g = g.sort_values('channel').head(config['n_channels'])
        ch_sigs = []
        # for each channels
        for _, row in g.iterrows():
            sig = row['EEG_clean']
            if not isinstance(sig, np.ndarray): # avoid format mismatch
                sig = _parse_eeg(sig)
            ch_sigs.append(sig)
        ch_sigs = [c for c in ch_sigs if len(c) > 0] # remove empty signal if any
       
        if not ch_sigs: # if no effective signal in this subject-video group, skip
            continue

        if len(ch_sigs) != config['n_channels']:
            raise ValueError(f'Channel amounts not match: Expected {config["n_channels"]}, got {len(ch_sigs[0])} instead for subject = {subj}, video = {vid}')

        # Do segmentation through the EEG data
        sig_len = min(len(c) for c in ch_sigs)
        for start in range(0, sig_len - WIN + 1, STRIDE):
            # Segmentation
            segs = [c[start:start+WIN] for c in ch_sigs]
            ptp = max(float(s.max() - s.min()) for s in segs) # peak-to-peak
            # Signal rejection
            if ptp > PTP_TH:
                rejected += 1; continue # remove artifact
            if max(float(s.std()) for s in segs) < FLAT_TH:
                rejected += 1; continue # remmove flat signal
            kept += 1 # nothing remove
            rows.append({
                'subject':  int(subj),
                'video':    int(vid),
                'label':    label,
                'dataset':  dataset_tag,
                'trial_id': f'{dataset_tag}__{subj}__{vid}',
                'EEG_array': np.stack(segs, axis=0),
            })

    # ---- checkpoint message -----
    total = kept + rejected
    print(f'  Kept={kept}, Rejected={rejected} ({rejected/max(total,1)*100:.1f}%)')
    df_seg = pd.DataFrame(rows)
    print(f'  Total segments: {len(df_seg)}')
    if len(df_seg):
        print(df_seg.groupby('subject')['label'].count().rename('n_seg').to_string())
    return df_seg

def segment_all(df_d, df_s, CONFIG):
    df_seg_d = artifact_reject_and_segment(df_d, CONFIG, 'dreamer')
    df_seg_s = artifact_reject_and_segment(df_s, CONFIG, 'seediv')
    del df_d, df_s

    # Reduce data for smoke test
    if CONFIG['_smoke']:
        # Cap to 80 segs per dataset to make build_tensor_dataset fast
        df_seg_d = df_seg_d.groupby('subject').head(10).reset_index(drop=True)
        # Ensure all 4 classes present in SEED-IV subset
        df_seg_s = df_seg_s.groupby('label').head(20).reset_index(drop=True)
    return df_seg_d, df_seg_s





