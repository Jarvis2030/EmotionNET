import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from domain_alignment import coral_align


def de_psd_1ch(sig, fs, bands):
    n = len(sig)
    if n == 0:
        return np.zeros(10, dtype=np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    psd_full = (np.abs(np.fft.rfft(sig))**2) / max(n, 1)
    psd_vals, de_vals = [], []
    for lo, hi in bands.values():
        mask = (freqs >= lo) & (freqs < hi)
        bp = float(psd_full[mask].mean()) if mask.any() else 0.0
        psd_vals.append(bp)
        de_vals.append(0.5 * np.log(2 * np.pi * np.e * (bp + 1e-10)))
    return np.array(psd_vals + de_vals, dtype=np.float32)


def seg_to_feature(df_seg, config):
    fs    = config['fs']
    bands = config['bands']
    feats = []
    for i, (_, row) in enumerate(df_seg.iterrows()):
        if i % 5000 == 0:
            print(f'  features {i}/{len(df_seg)}...', flush=True)
        eeg_2d = np.asarray(row['EEG_array'], dtype=np.float32)
        feat = np.concatenate([de_psd_1ch(eeg_2d[ch], fs, bands) for ch in range(eeg_2d.shape[0])])
        feats.append(feat)
    return np.stack(feats)


def run_gmm_relabeling(df_seg_d, df_seg_s, config):

    print('  Extracting DREAMER DE+PSD features...')
    Xd = seg_to_feature(df_seg_d, config)
    yd_orig = df_seg_d['label'].values.astype(int)

    print('  Extracting SEED-IV DE+PSD features...')
    Xs = seg_to_feature(df_seg_s, config)
    ys = df_seg_s['label'].values.astype(int)

    print('  StandardScaler...')
    scaler = StandardScaler()
    Xall_sc = scaler.fit_transform(np.vstack([Xd, Xs]))
    Xd_sc = Xall_sc[:len(Xd)]
    Xs_sc = Xall_sc[len(Xd):]

    Xs_aln, Xd_aln = coral_align(Xs_sc, Xd_sc)
    print(f'  DREAMER aligned: {Xd_aln.shape}  SEED-IV aligned: {Xs_aln.shape}')

    # Compute per-class means; fall back to global mean if a class is missing
    global_mean = Xs_aln.mean(axis=0)
    seediv_means_aln = np.stack([
        Xs_aln[ys == ci].mean(axis=0) if (ys == ci).any() else global_mean
        for ci in range(4)
    ])

    print('  Fitting GMM (k=4, diag, n_init=1)...')
    gmm = GaussianMixture(
        n_components=4, covariance_type='diag',
        means_init=seediv_means_aln, n_init=1, max_iter=100, random_state=config['seed'],
    )
    gmm.fit(Xd_aln)
    yd_gmm = gmm.predict(Xd_aln)

    print(f'\n  GMM converged: {gmm.converged_}')
    for ci in range(4):
        n = (yd_gmm == ci).sum()
        print(f'  Cluster {ci}: {n} ({n/len(yd_gmm)*100:.1f}%)')
    agree = (yd_orig == yd_gmm).mean()
    print(f'\n  Agreement with original label: {agree*100:.1f}%')

    assert len(yd_gmm) == len(df_seg_d)
    df_seg_d = df_seg_d.copy()
    df_seg_d['label_orig'] = df_seg_d['label'].values
    df_seg_d['label']      = yd_gmm

    return df_seg_d

