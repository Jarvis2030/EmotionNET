"""
EEG Cross-Subject Emotion Recognition Pipeline
DREAMER (GMM Re-labeling) + SEED-IV (Route C) → LOSO Training

Run:
    python run_pipeline.py
    python run_pipeline.py --subjects 1 2 3
    python run_pipeline.py --lambda_seed 0.5
"""

import sys
import copy
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from scipy.optimize import minimize
from scipy.signal import resample

from pipeline.domain_alignment import coral_align
from pipeline.SNN_data import mat_dataset_load, EEG_band_analysis, label_balancing_soft

# SNN model — imported only when --snn flag is used
try:
    from pipeline.model import EmotionNET, spike_rate_loss

    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ←  直接從 notebook-6 同步
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    'dreamer_csv':          './data/EEG_clean_table.csv',
    'seediv_csv':           './data/EEG_all_sessions_combined.csv',
    'out_dir':              './output',

    'fs':                   128,
    'n_channels':           14,
    'window_size':          384,
    'stride':               384,

    'artifact_ptp_uv':      8.0,    # z-score units (DREAMER is MATLAB z-scored)
    'artifact_flat_uv':     0.02,   # z-score std (flat signal threshold)

    'bands': {'delta':(1,4),'theta':(4,8),'alpha':(8,13),'beta':(13,30),'gamma':(30,45)},

    'in_channels':          9,
    'out_channels':         30,
    'lstm_hidden':          64,
    'lstm_layers':          1,
    'dropout':              0.25,
    'n_classes':            4,

    'lr':                   2e-4,
    'weight_decay':         1e-4,
    'lambda_mmd':           0.05,
    'lambda_seed':          1.0,
    'num_epochs':           60,
    'early_stop_patience':  15,
    'warmup_epochs':        10,   # train DREAMER only before CORAL alignment kicks in
    'coral_update_interval': 5,  # recompute CORAL matrix every N epochs
    'batch':                32,
    'val_ratio':            0.15,
    'seed':                 42,

    'subjects':             None,   # None = all

    'snn_out_channels'  : 50,
    'snn_hidden'        : 50,
    'snn_spike_weight'  : 1e-3,
    'snn_target_rate'   : 0.1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def flag(msg):
    ts = time.strftime('%H:%M:%S')
    print(f'\n{"="*60}', flush=True)
    print(f'[{ts}]  {msg}', flush=True)
    print('='*60, flush=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ds(x):
    x = np.asarray(x)
    n_new = int(x.shape[0] * 128 / 200)
    return resample(x, n_new, axis=0)


def _parse_eeg(x):
    if isinstance(x, str):
        return np.fromstring(x.strip('[]'), sep=',', dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load
# ─────────────────────────────────────────────────────────────────────────────

def load_dreamer(csv_path):
    flag(f'Loading DREAMER  {csv_path}')
    df = mat_dataset_load(csv_path, on_bad_lines='skip')
    df['dataset'] = 'dreamer'
    df['session_idx'] = -1
    key_cols = ['subject', 'label', 'video', 'channel', 'EEG_clean']
    before = len(df)
    df = df.dropna(subset=key_cols).reset_index(drop=True)
    if before - len(df):
        print(f'  Dropped {before-len(df)} NaN rows')
    df['subject']   = df['subject'].astype(int)
    df['label']     = df['label'].astype(int)
    df['video']     = df['video'].astype(int)
    df['channel']   = df['channel'].astype(int)
    df['EEG_clean'] = df['EEG_clean'].apply(_parse_eeg).apply(ds) # downsample
    print(f'  Shape: {df.shape}  |  Subjects: {sorted(df["subject"].unique().tolist())}')
    print(f'  Label counts:\n{df["label"].value_counts().sort_index().to_string()}')
    return df


def load_seediv(csv_path):
    flag(f'Loading SEED-IV  {csv_path}')
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df['dataset'] = 'seediv'
    df['session_idx'] = -1
    key_cols = ['subject', 'label', 'video', 'channel', 'EEG_clean']
    before = len(df)
    df = df.dropna(subset=key_cols).reset_index(drop=True)
    if before - len(df):
        print(f'  Dropped {before-len(df)} bad rows')
    df['subject']   = df['subject'].astype(int)
    df['label']     = df['label'].astype(int)
    df['video']     = df['video'].astype(int)
    df['channel']   = df['channel'].astype(int)
    df['EEG_clean'] = df['EEG_clean'].apply(_parse_eeg)
    print(f'  Shape: {df.shape}  |  Subjects: {sorted(df["subject"].unique().tolist())}')
    print(f'  Label counts:\n{df["label"].value_counts().sort_index().to_string()}')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Artifact Rejection + Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def artifact_reject_and_segment(df, config, dataset_tag):
    flag(f'Artifact rejection + segmentation  [{dataset_tag}]')
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
            raise ValueError(
                f'Channel amounts not match: Expected {config["n_channels"]},
                got {len(ch_sigs[0])} instead for subject = {subj}, video = {vid}'
            )

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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CORAL + GMM Re-labeling
# ─────────────────────────────────────────────────────────────────────────────

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
    flag('CORAL alignment + GMM re-labeling')

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

    print('  CORAL (DREAMER → SEED-IV space)...')
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

    # Save comparison plot
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    pca2 = PCA(n_components=2, random_state=42)
    Xd_2d = pca2.fit_transform(Xd_aln)
    ev = pca2.explained_variance_ratio_
    DARK_BG = '#0f0f1a'; PANEL_BG = '#1a1a2e'; GRID_COL = '#2e2e4a'; TEXT_COL = '#e0e0f0'
    COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52']
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(f'DREAMER: Original vs GMM  (agreement={agree*100:.1f}%)',
                 color='white', fontsize=13, fontweight='bold')
    for ai, ax in enumerate(axes):
        ax.set_facecolor(PANEL_BG); ax.spines[:].set_color(GRID_COL)
        ax.tick_params(colors=TEXT_COL); ax.grid(color=GRID_COL, alpha=0.5, linewidth=0.4)
        if ai == 0:
            for ci in range(4):
                m = yd_orig == ci
                ax.scatter(Xd_2d[m,0], Xd_2d[m,1], c=COLORS[ci], s=6, alpha=0.35, linewidths=0, label=f'Orig-{ci}')
            ax.set_title('Original Label', color=TEXT_COL)
        elif ai == 1:
            for ci in range(4):
                m = yd_gmm == ci
                ax.scatter(Xd_2d[m,0], Xd_2d[m,1], c=COLORS[ci], s=6, alpha=0.35, linewidths=0, label=f'GMM-{ci}')
            ax.set_title('GMM Label', color=TEXT_COL)
        else:
            agr = yd_orig == yd_gmm
            ax.scatter(Xd_2d[agr,0],  Xd_2d[agr,1],  c='#55A868', s=6, alpha=0.35, linewidths=0, label='Agree')
            ax.scatter(Xd_2d[~agr,0], Xd_2d[~agr,1], c='#C44E52', s=6, alpha=0.35, linewidths=0, label='Disagree')
            ax.set_title(f'Agreement ({agree*100:.1f}%)', color=TEXT_COL)
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=7, markerscale=2)
        ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', color=TEXT_COL)
        ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', color=TEXT_COL)
    plt.tight_layout()
    plt.savefig(out_dir / 'gmm_label_comparison.png', dpi=120, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f'  Plot → {out_dir}/gmm_label_comparison.png')

    return yd_gmm


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class EEGUnlabeledDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, _, tid = self.base[idx]
        return x, tid


def build_tensor_dataset(split_df, config):
    if len(split_df) == 0:
        raise ValueError('Empty DataFrame')
    x_list, y_list, tid_list = [], [], []
    for _, row in split_df.iterrows():
        eeg_2d = np.asarray(row['EEG_array'], dtype=np.float32)
        featured = EEG_band_analysis(fs=config['fs'], seg=eeg_2d, out_T=config['window_size'])
        x_list.append(featured)
        y_list.append(int(row['label']))
        tid_list.append(str(row['trial_id']))
    x = np.stack(x_list)
    y = np.array(y_list, dtype=np.int64)
    _, tid_ints = np.unique(tid_list, return_inverse=True)
    return TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).long(),
        torch.from_numpy(tid_ints.astype(np.int64)),
    )


class EEG2DCNNLSTM(nn.Module):
    def __init__(self, in_channels=9, eeg_channels=14,
                 out_channels=50, n_classes=4,
                 fs=128, decision_window=3,
                 dropout=0.3, lstm_hidden=64, lstm_layers=1):
        super().__init__()

        self.T       = fs * decision_window        # 384
        self.in_flat = in_channels * eeg_channels  # 126

        kernel_time = fs // 4   # 32
        stride_time = fs // 8   # 16

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(self.in_flat, out_channels,
                      kernel_size=kernel_time, stride=stride_time),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
        )

        # Activation function
        self.act = nn.GELU()

        # LSTM 
        self.lstm = nn.LSTM(
            input_size    = out_channels,  # 50
            hidden_size   = lstm_hidden,   # 64
            num_layers    = lstm_layers,
            batch_first   = True,
        )

        self.fusion_dim = lstm_hidden
        self.head = nn.Linear(lstm_hidden, n_classes)

    def extract_features(self, x):
        B = x.shape[0]
        z = x.permute(0, 1, 3, 2).reshape(B, self.in_flat, self.T)  # (B, 126, 384)
        z = self.cnn(z)          # (B, 50, T')
        z = self.act(z)          # (B, 50, T')  ← 取代 LIF
        z = z.permute(0, 2, 1)  # (B, T', 50)
        _, (h_n, _) = self.lstm(z)
        return h_n[-1]           # (B, 64)

    def forward(self, x, return_feat=False):
        feat   = self.extract_features(x)
        logits = self.head(feat)
        if return_feat:
            return logits, feat
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Latent-space CORAL
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_coral_matrix(model, drm_loader, seed_loader, device):
    """Compute CORAL whitening/coloring matrices in latent space (79-dim)."""
    model.eval()

    def collect_feats(loader, max_batches=50):
        feats = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch[0].to(device, dtype=torch.float32).permute(0, 1, 3, 2)
            feats.append(model.extract_features(x).cpu().numpy())
        return np.concatenate(feats, axis=0)

    Xd = collect_feats(drm_loader)   # (N_d, 79)
    Xs = collect_feats(seed_loader)  # (N_s, 79)

    # CORAL: whiten source (DREAMER), color with target (SEED-IV)
    def cov(X):
        X = X - X.mean(axis=0, keepdims=True)
        return (X.T @ X) / max(len(X) - 1, 1) + 1e-5 * np.eye(X.shape[1])

    Cd = cov(Xd)   # DREAMER cov
    Cs = cov(Xs)   # SEED-IV cov

    # Whiten DREAMER: Cd^{-1/2}
    Ud, Sd, _ = np.linalg.svd(Cd)
    W = Ud @ np.diag(1.0 / np.sqrt(Sd + 1e-10)) @ Ud.T

    # Color with SEED-IV: Cs^{1/2}
    Us, Ss, _ = np.linalg.svd(Cs)
    C = Us @ np.diag(np.sqrt(Ss + 1e-10)) @ Us.T

    # A = Cs^{1/2} @ Cd^{-1/2}
    # aligned = (drm_feat - mu_d) @ A.T + mu_s
    A    = torch.tensor(C @ W,           dtype=torch.float32, device=device)
    mu_d = torch.tensor(Xd.mean(axis=0), dtype=torch.float32, device=device)
    mu_s = torch.tensor(Xs.mean(axis=0), dtype=torch.float32, device=device)
    return A, mu_d, mu_s


def apply_coral(feat, A, mu_d, mu_s):
    """Transform DREAMER latent → SEED-IV latent space."""
    return (feat - mu_d) @ A.T + mu_s


def gaussian_kernel(x, y, sigmas=(1, 2, 4, 8, 16)):
    beta = 1.0 / (2.0 * torch.tensor(sigmas, device=x.device, dtype=x.dtype).view(-1, 1, 1))
    dist = torch.cdist(x, y, p=2).pow(2).unsqueeze(0)
    return torch.exp(-beta * dist).sum(dim=0)


def mmd_loss(source, target, sigmas=(1, 2, 4, 8, 16)):
    return (gaussian_kernel(source, source, sigmas).mean()
            + gaussian_kernel(target, target, sigmas).mean()
            - 2 * gaussian_kernel(source, target, sigmas).mean())

# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling
# ─────────────────────────────────────────────────────────────────────────────

def _softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)

def fit_temperature_from_logits(logits, labels, init_temp=1.5):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(logits) == 0:
        return 1.0

    def objective(log_t):
        T = np.exp(log_t[0])
        probs = _softmax_np(logits / T)
        nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
        return float(nll)

    res = minimize(objective, x0=np.array([np.log(init_temp)], dtype=np.float64), method='L-BFGS-B')
    T = float(np.exp(res.x[0])) if res.success else float(init_temp)
    return max(T, 1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_routeC(model, dataloader, criterion, device, n_classes=4, return_preds=False, temperature=1.0, return_logits=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    trial_preds  = defaultdict(list)
    trial_labels = {}
    all_logits = []   # ★ 新增
    all_labels = []   # 如果下面有用到一起加
    
    for data, labels, trial_ids in dataloader:
        data   = data.to(device,   dtype=torch.float32).permute(0, 1, 3, 2)
        labels = labels.to(device, dtype=torch.long)
        logits = model(data)
        logits_eval = logits / max(float(temperature), 1e-6)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * data.size(0)
        pred = logits_eval.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        for p, y, tid in zip(pred.cpu().numpy(), labels.cpu().numpy(), trial_ids.cpu().numpy()):
            trial_preds[int(tid)].append(int(p))
            trial_labels[int(tid)] = int(y)
    seg_acc  = correct / max(total, 1)
    seg_loss = total_loss / max(total, 1)
    y_true, y_pred = [], []
    for tid in sorted(trial_preds.keys()):
        counts = np.bincount(np.array(trial_preds[tid], dtype=np.int64), minlength=n_classes)
        y_pred.append(int(counts.argmax()))
        y_true.append(int(trial_labels[tid]))
    trial_acc = (np.array(y_true) == np.array(y_pred)).mean() if y_true else 0.0
    if return_preds and return_logits:
        return seg_loss, seg_acc, trial_acc, np.array(y_true), np.array(y_pred), np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)
    if return_preds:
        return seg_loss, seg_acc, trial_acc, np.array(y_true), np.array(y_pred)
    if return_logits:
        return seg_loss, seg_acc, trial_acc, np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)
    return seg_loss, seg_acc, trial_acc


def train_one_epoch_routeC(model, drm_loader, seed_loader, tgt_loader,
                            criterion, optimizer, device,
                            lambda_mmd=0.05, lambda_seed=1.0,
                            coral_params=None, warmup=False, config=None):
    """
    coral_params: (A, mu_d, mu_s) from compute_coral_matrix, or None
    warmup: if True, skip CE_S and MMD (DREAMER only)
    """
    model.train()
    total_loss = total_ce_d = total_ce_s = total_mmd_v = 0.0
    correct, total = 0, 0
    seed_iter = iter(seed_loader)
    tgt_iter  = iter(tgt_loader)

    for drm_data, drm_labels, _ in drm_loader:
        drm_data   = drm_data.to(device,   dtype=torch.float32).permute(0, 1, 3, 2)
        drm_labels = drm_labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        drm_logits, drm_feat = model(drm_data, return_feat=True)
        ce_d = criterion(drm_logits, drm_labels)

        if warmup or coral_params is None:
            # Warmup: only DREAMER CE, no SEED / MMD
            spike_reg = spike_rate_loss(model, config.get('snn_target_rate', 0.1),
                                        config.get('snn_spike_weight', 0.0)) \
                        if config.get('_use_snn', False) else 0.0
            loss = ce_d + spike_reg
            ce_s_val = mmd_val = 0.0
        else:
            try:
                seed_data, seed_labels, _ = next(seed_iter)
            except StopIteration:
                seed_iter = iter(seed_loader)
                seed_data, seed_labels, _ = next(seed_iter)
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter  = iter(tgt_loader)
                tgt_batch = next(tgt_iter)

            seed_data   = seed_data.to(device,   dtype=torch.float32).permute(0, 1, 3, 2)
            seed_labels = seed_labels.to(device, dtype=torch.long)
            tgt_data    = tgt_batch[0].to(device, dtype=torch.float32).permute(0, 1, 3, 2)

            # CORAL: align DREAMER latent → SEED-IV space, then classify
            A, mu_d, mu_s = coral_params
            drm_feat_aligned = apply_coral(drm_feat, A, mu_d, mu_s)
            drm_logits_aligned = model.head(drm_feat_aligned)

            seed_feat   = model.extract_features(seed_data)
            seed_logits = model.head(seed_feat)

            _, tgt_feat = model(tgt_data, return_feat=True)

            # CE_D on CORAL-aligned DREAMER, CE_S on raw SEED-IV
            ce_d = criterion(drm_logits_aligned, drm_labels)
            ce_s = criterion(seed_logits, seed_labels)
            mmd  = mmd_loss(drm_feat, tgt_feat)
            spike_reg = spike_rate_loss(model, config.get('snn_target_rate', 0.1),
                                        config.get('snn_spike_weight', 0.0)) \
                        if config.get('_use_snn', False) else 0.0
            loss = ce_d + lambda_seed * ce_s + lambda_mmd * mmd + spike_reg
            ce_s_val = ce_s.item()
            mmd_val  = mmd.item()

        loss.backward()
        optimizer.step()

        bs = drm_data.size(0)
        total_loss  += loss.item() * bs
        total_ce_d  += ce_d.item() * bs
        total_ce_s  += ce_s_val    * bs
        total_mmd_v += mmd_val     * bs
        pred = drm_logits.argmax(dim=1)
        correct += (pred == drm_labels).sum().item()
        total   += bs

    n = max(total, 1)
    return {'loss': total_loss/n, 'ce_drm': total_ce_d/n,
            'ce_seed': total_ce_s/n, 'mmd_loss': total_mmd_v/n, 'acc': correct/n}


# ─────────────────────────────────────────────────────────────────────────────
# LOSO Fold
# ─────────────────────────────────────────────────────────────────────────────

def loso_splits(df_seg_d, target_subject, config, seed_ds_prebuilt):
    source_df = df_seg_d[df_seg_d['subject'] != target_subject].copy()
    target_df = df_seg_d[df_seg_d['subject'] == target_subject].copy()
    if len(target_df) == 0:
        raise ValueError(f'No data for subject {target_subject}')
    source_trials = source_df[['trial_id', 'label']].drop_duplicates()
    train_trials, val_trials = train_test_split(
        source_trials, test_size=config['val_ratio'],
        random_state=config['seed'], stratify=source_trials['label'], shuffle=True,
    )
    train_df = source_df[source_df['trial_id'].isin(train_trials['trial_id'])]
    val_df   = source_df[source_df['trial_id'].isin(val_trials['trial_id'])]
    drm_train_ds = build_tensor_dataset(train_df,  config)
    drm_val_ds   = build_tensor_dataset(val_df,    config)
    tgt_test_ds  = build_tensor_dataset(target_df, config)
    tgt_unlab_ds = EEGUnlabeledDataset(tgt_test_ds)
    return drm_train_ds, drm_val_ds, tgt_test_ds, tgt_unlab_ds, seed_ds_prebuilt


def run_loso_fold(df_seg_d, seed_ds_prebuilt, target_subject, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    drm_train_ds, drm_val_ds, tgt_test_ds, tgt_unlab_ds, seed_ds = \
        loso_splits(df_seg_d, target_subject, config, seed_ds_prebuilt)

    drm_train_loader = DataLoader(drm_train_ds,  batch_size=config['batch'], shuffle=True,  drop_last=True)
    drm_val_loader   = DataLoader(drm_val_ds,    batch_size=config['batch'], shuffle=False, drop_last=False)
    seed_loader      = DataLoader(seed_ds,        batch_size=config['batch'], shuffle=True,  drop_last=True)
    tgt_unlab_loader = DataLoader(tgt_unlab_ds,  batch_size=config['batch'], shuffle=True,  drop_last=False)
    tgt_test_loader  = DataLoader(tgt_test_ds,   batch_size=config['batch'], shuffle=False, drop_last=False)

    if config.get('_use_snn', False):
        model = EEGSNNRouteC(
            in_channels=config['in_channels'],
            eeg_channels=config['n_channels'],
            out_channels=config['snn_out_channels'],
            n_classes=config['n_classes'],
            fs=config['fs'],
            decision_window=config['window_size'] // config['fs'],
            dropout=config['dropout'],
            lstm_hidden=config.get('snn_lstm_hidden', 64),
            lstm_layers=config.get('snn_lstm_layers', 1),
            beta1=config.get('snn_beta1', 0.9),
            beta2=config.get('snn_beta2', 0.9),
            threshold=config.get('snn_threshold', 0.3),
        ).to(device)
        print(f'  Model: EEGSNN (snntorch)  fusion_dim={model.fusion_dim}')
    else:
        model = EEG2DCNNLSTM(
            fs=config['fs'], input_time=config['window_size'] // config['fs'],
            in_channels=config['in_channels'], out_channels=config['out_channels'],
            n_classes=config['n_classes'], eeg_channels=config['n_channels'],
            lstm_hidden=config['lstm_hidden'], lstm_layers=config['lstm_layers'],
            dropout=config['dropout'],
        ).to(device)
        print(f'  Model: EEG2DCNNLSTM  fusion_dim={model.fusion_dim}')

    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.05))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    best_state = None; best_trial = -1.0; patience = 0; history = []
    best_temperature = 1.0
    best_val_trial_seen = -1.0   # ← purely for reporting; never affects training
    coral_params = None
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    warmup_epochs        = config['warmup_epochs']
    coral_update_interval = config['coral_update_interval']

    for epoch in range(config['num_epochs']):
        is_warmup = epoch < warmup_epochs

        # Update CORAL matrix: first time after warmup, then every K epochs
        if not config.get('_no_coral', False) and not is_warmup and (
            coral_params is None or
            (epoch - warmup_epochs) % coral_update_interval == 0
        ):
            print(f'  [Ep {epoch:03d}] Recomputing CORAL matrix...', flush=True)
            coral_params = compute_coral_matrix(model, drm_train_loader, seed_loader, device)
            print(f'  [Ep {epoch:03d}] CORAL matrix updated.', flush=True)

        train_stats = train_one_epoch_routeC(
            model, drm_train_loader, seed_loader, tgt_unlab_loader,
            criterion, optimizer, device,
            lambda_mmd=config['lambda_mmd'], lambda_seed=config['lambda_seed'],
            coral_params=coral_params, warmup=is_warmup, config=config,
        )
        val_loss,  val_acc,  val_trial  = evaluate_routeC(model, drm_val_loader,  criterion, device, config['n_classes'])
        test_loss, test_acc, test_trial = evaluate_routeC(model, tgt_test_loader, criterion, device, config['n_classes'])

        history.append({'epoch': epoch, **train_stats,
                        'val_loss': val_loss, 'val_acc': val_acc, 'val_trial_acc': val_trial,
                        'test_loss': test_loss, 'test_acc': test_acc, 'test_trial_acc': test_trial,
                        'temperature': best_temperature})

        print(f'[RouteC S{target_subject:02d}] Ep {epoch:03d} | '
              f'Loss {train_stats["loss"]:.4f} '
              f'(CE_D {train_stats["ce_drm"]:.4f} CE_S {train_stats["ce_seed"]:.4f} '
              f'MMD {train_stats["mmd_loss"]:.4f}) '
              f'Acc {train_stats["acc"]:.4f} | '
              f'Val {val_acc:.4f} Tr {val_trial:.4f} | '
              f'Test {test_acc:.4f} Tr {test_trial:.4f}', flush=True)

        # ── track best val trial for reporting (independent accumulator) ──────
        if val_trial > best_val_trial_seen:
            best_val_trial_seen = val_trial

        if val_trial > best_trial:
            best_trial = val_trial
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            # Save best checkpoint for this subject
            ckpt_path = out_dir / f'best_s{target_subject:02d}.pt'
            torch.save({'subject': target_subject, 'epoch': epoch,
                        'val_trial_acc': val_trial, 'temperature': best_temperature, 'state_dict': best_state}, ckpt_path)
        else:
            patience += 1
            if patience >= config['early_stop_patience']:
                print(f'  Early stopping at epoch {epoch}.', flush=True); break

    model.load_state_dict(best_state)
    test_loss, test_acc, test_trial, y_true, y_pred = evaluate_routeC(
        model, tgt_test_loader, criterion, device, config['n_classes'], return_preds=True, temperature=best_temperature)

    # ── Collect all test-set windows for demo / analysis ─────────────────────
    all_feats, all_labels_np, all_preds_np, all_probs_np = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tgt_test_loader:
            feats_b, labels_b = batch[0], batch[1]
            feats_b = feats_b.to(device, dtype=torch.float32).permute(0, 1, 3, 2)
            logits_b = model(feats_b)
            probs_b  = torch.softmax(logits_b, dim=-1)
            preds_b  = probs_b.argmax(dim=-1)
            all_feats.append(batch[0].cpu().numpy())      # (B, 9, 384, 14) — raw band feats
            all_labels_np.append(labels_b.cpu().numpy())
            all_preds_np.append(preds_b.cpu().numpy())
            all_probs_np.append(probs_b.cpu().numpy())
    all_feats_np  = np.concatenate(all_feats,     axis=0)  # (N, 9, 384, 14)
    all_labels_np = np.concatenate(all_labels_np, axis=0)  # (N,)
    all_preds_np  = np.concatenate(all_preds_np,  axis=0)  # (N,)
    all_probs_np  = np.concatenate(all_probs_np,  axis=0)  # (N, 4)
    test_npz_path = out_dir / f'all_test_data_s{target_subject:02d}.npz'
    np.savez(test_npz_path,
             eeg_feat=all_feats_np, labels=all_labels_np,
             preds=all_preds_np,    probs=all_probs_np,
             subject=np.array(target_subject))
    print(f'  Test data → {test_npz_path}  (N={len(all_labels_np)} windows)')

    return {'target_subject': target_subject,
            'best_val_trial_acc': best_val_trial_seen,   # ← fixed: real max seen
            'final_test_loss': test_loss, 'final_test_acc': test_acc,
            'final_test_trial_acc': test_trial, 'history': history,
            'y_true': y_true, 'y_pred': y_pred}


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_results, config):
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in r.items() if k not in ('history', 'y_true', 'y_pred')}
            for r in all_results]
    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_dir / 'LTSM_SNN_loso_results.csv', index=False)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    EMOTION_LABELS = ['Neutral', 'Sad', 'Fear', 'Happy']
    DARK_BG2 = '#0f0f1a'; PANEL_BG2 = '#1a1a2e'; TEXT_COL2 = '#e0e0f0'

    # collect all predictions across subjects
    all_true = np.concatenate([r['y_true'] for r in all_results if len(r.get('y_true', [])) > 0])
    all_pred = np.concatenate([r['y_pred'] for r in all_results if len(r.get('y_pred', [])) > 0])
    n_classes = config.get('n_classes', 4)
    labels    = list(range(n_classes))
    tick_lbl  = EMOTION_LABELS[:n_classes]

    # 1) aggregated CM across all subjects
    cm_agg = confusion_matrix(all_true, all_pred, labels=labels)
    cm_norm = cm_agg.astype(float) / cm_agg.sum(axis=1, keepdims=True).clip(min=1)
    pd.DataFrame(cm_agg,  index=tick_lbl, columns=tick_lbl).to_csv(out_dir / 'confusion_matrix.csv')
    pd.DataFrame(cm_norm, index=tick_lbl, columns=tick_lbl).to_csv(out_dir / 'confusion_matrix_norm.csv')
    print(f'Confusion matrix → {out_dir}/confusion_matrix.csv')

    # classification report
    report = classification_report(all_true, all_pred,
                                   labels=labels, target_names=tick_lbl, digits=4)
    print('\n' + report)
    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

    # 2) per-subject CMs
    subj_cms = {}
    for r in all_results:
        if len(r.get('y_true', [])) == 0: continue
        subj_cms[r['target_subject']] = confusion_matrix(
            r['y_true'], r['y_pred'], labels=labels)

    # 3) figure: aggregated CM + per-subject CMs
    n_subj   = len(subj_cms)
    ncols_s  = min(4, n_subj)
    nrows_s  = (n_subj + ncols_s - 1) // ncols_s
    total_rows = 1 + nrows_s          # row 0 = aggregated, rows 1+ = per-subject
    fig_cm, axes_cm = plt.subplots(
        total_rows, max(ncols_s, 1),
        figsize=(4.5 * max(ncols_s, 1), 4.5 * total_rows),
        squeeze=False)
    fig_cm.patch.set_facecolor(DARK_BG2)

    def _draw_cm(ax, cm, title):
        ax.set_facecolor(PANEL_BG2)
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_n, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set_xticks(range(n_classes)); ax.set_xticklabels(tick_lbl, rotation=30,
                                                              ha='right', color=TEXT_COL2, fontsize=8)
        ax.set_yticks(range(n_classes)); ax.set_yticklabels(tick_lbl, color=TEXT_COL2, fontsize=8)
        ax.set_xlabel('Predicted', color=TEXT_COL2, fontsize=8)
        ax.set_ylabel('True',      color=TEXT_COL2, fontsize=8)
        ax.set_title(title, color=TEXT_COL2, fontsize=9, fontweight='bold')
        ax.tick_params(colors=TEXT_COL2)
        for spine in ax.spines.values(): spine.set_edgecolor('#2e2e4a')
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm[i, j]
                pct = cm_n[i, j]
                color = 'white' if pct > 0.5 else TEXT_COL2
                ax.text(j, i, f'{val}\n({pct:.0%})',
                        ha='center', va='center', fontsize=7, color=color)
        return im

    # row 0: aggregated (centred in first ncols_s cols, span all)
    ax_agg = fig_cm.add_subplot(total_rows, 1, 1)   # full-width row
    # hide the individual cells in row 0
    for c in range(max(ncols_s, 1)):
        axes_cm[0][c].set_visible(False)
    # redraw as full-width
    fig_cm.add_subplot(total_rows, 1, 1)
    ax_agg = plt.subplot2grid((total_rows, max(ncols_s, 1)), (0, 0),
                               colspan=max(ncols_s, 1), fig=fig_cm)
    _draw_cm(ax_agg, cm_agg, 'Aggregated CM — All Subjects')

    # rows 1+: per-subject
    for idx, (subj, cm_s) in enumerate(sorted(subj_cms.items())):
        row = 1 + idx // ncols_s
        col = idx % ncols_s
        _draw_cm(axes_cm[row][col], cm_s, f'S{subj:02d}')
    # hide unused
    for idx in range(n_subj, nrows_s * ncols_s):
        axes_cm[1 + idx // ncols_s][idx % ncols_s].set_visible(False)

    plt.tight_layout()
    cm_png = out_dir / 'confusion_matrices.png'
    plt.savefig(cm_png, dpi=150, bbox_inches='tight', facecolor=DARK_BG2)
    plt.close(fig_cm)
    print(f'Confusion matrix figure → {cm_png}')

    flag('LOSO Results')
    print(df_res[['target_subject', 'best_val_trial_acc', 'final_test_acc', 'final_test_trial_acc']].to_string(index=False))
    print(f'\nMean Val Trial Acc  : {df_res["best_val_trial_acc"].mean():.4f}')
    print(f'Mean Test Trial Acc : {df_res["final_test_trial_acc"].mean():.4f}')

    DARK_BG = '#0f0f1a'; PANEL_BG = '#1a1a2e'; GRID_COL = '#2e2e4a'; TEXT_COL = '#e0e0f0'

    # ── 1. Per-subject epoch history CSV ─────────────────────────────────
    hist_rows = []
    for r in all_results:
        subj = r['target_subject']
        for h in r['history']:
            hist_rows.append({'subject': subj, **h})
    df_hist = pd.DataFrame(hist_rows)
    hist_csv = out_dir / 'epoch_history.csv'
    df_hist.to_csv(hist_csv, index=False)
    print(f'Epoch history → {hist_csv}')

    # ── 2. Mean loss/acc across subjects per epoch ───────────────────────
    loss_cols = [c for c in df_hist.columns
                 if c in ('loss', 'val_loss', 'test_loss',
                          'acc',  'val_acc',  'test_acc',
                          'ce_drm', 'ce_seed', 'mmd_loss')]
    df_mean = df_hist.groupby('epoch')[loss_cols].mean().reset_index()
    mean_csv = out_dir / 'mean_epoch_stats.csv'
    df_mean.to_csv(mean_csv, index=False)
    print(f'Mean epoch stats → {mean_csv}')

    # ── 3. Loss curves figure (mean across subjects) ─────────────────────
    has_loss = {'loss', 'val_loss', 'test_loss'}.issubset(df_mean.columns)
    has_acc  = {'acc',  'val_acc',  'test_acc' }.issubset(df_mean.columns)
    n_panels = int(has_loss) + int(has_acc)

    if n_panels > 0:
        fig2, axes = plt.subplots(1, n_panels,
                                  figsize=(7 * n_panels, 4.5), squeeze=False)
        fig2.patch.set_facecolor(DARK_BG)
        panel = 0
        epochs_x = df_mean['epoch'].values

        if has_loss:
            ax = axes[0][panel]; panel += 1
            ax.set_facecolor(PANEL_BG)
            ax.spines[:].set_color(GRID_COL); ax.tick_params(colors=TEXT_COL)
            ax.grid(color=GRID_COL, alpha=0.4, linewidth=0.6)
            ax.plot(epochs_x, df_mean['loss'],      color='#4C72B0', lw=1.8, label='Train Loss')
            ax.plot(epochs_x, df_mean['val_loss'],  color='#55A868', lw=1.8, label='Val Loss')
            ax.plot(epochs_x, df_mean['test_loss'], color='#DD8452', lw=1.8, label='Test Loss', linestyle='--')
            # sub-loss breakdown if available
            for col, lbl, clr in [('ce_drm',   'CE DREAMER', '#8172B2'),
                                   ('ce_seed',  'CE SEED-IV', '#937860'),
                                   ('mmd_loss', 'MMD',        '#da8bc3')]:
                if col in df_mean.columns:
                    ax.plot(epochs_x, df_mean[col], color=clr, lw=1.0,
                            linestyle=':', alpha=0.75, label=lbl)
            ax.set_xlabel('Epoch', color=TEXT_COL)
            ax.set_ylabel('Loss',  color=TEXT_COL)
            ax.set_title('Mean Loss (across subjects)', color=TEXT_COL, fontsize=11, fontweight='bold')
            ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)

        if has_acc:
            ax = axes[0][panel]
            ax.set_facecolor(PANEL_BG)
            ax.spines[:].set_color(GRID_COL); ax.tick_params(colors=TEXT_COL)
            ax.grid(color=GRID_COL, alpha=0.4, linewidth=0.6)
            ax.plot(epochs_x, df_mean['acc'],      color='#4C72B0', lw=1.8, label='Train Acc')
            ax.plot(epochs_x, df_mean['val_acc'],  color='#55A868', lw=1.8, label='Val Acc')
            ax.plot(epochs_x, df_mean['test_acc'], color='#DD8452', lw=1.8, label='Test Acc', linestyle='--')
            ax.axhline(0.25, color='#ff4444', lw=1.0, linestyle='--', label='Chance 25%')
            ax.set_xlabel('Epoch', color=TEXT_COL)
            ax.set_ylabel('Accuracy', color=TEXT_COL)
            ax.set_ylim(0, 1.05)
            ax.set_title('Mean Accuracy (across subjects)', color=TEXT_COL, fontsize=11, fontweight='bold')
            ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)

        plt.tight_layout()
        curve_png = out_dir / 'loss_acc_curves.png'
        plt.savefig(curve_png, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        plt.close(fig2)
        print(f'Loss/Acc curves → {curve_png}')

    # ── 4. Per-subject loss curves (one subplot per subject) ─────────────
    subjects_done = df_hist['subject'].unique()
    n_s = len(subjects_done)
    ncols = min(4, n_s); nrows = (n_s + ncols - 1) // ncols
    fig3, axes3 = plt.subplots(nrows, ncols,
                               figsize=(5.5 * ncols, 3.8 * nrows),
                               squeeze=False)
    fig3.patch.set_facecolor(DARK_BG)
    for idx, subj in enumerate(sorted(subjects_done)):
        ax = axes3[idx // ncols][idx % ncols]
        ax.set_facecolor(PANEL_BG)
        ax.spines[:].set_color(GRID_COL); ax.tick_params(colors=TEXT_COL, labelsize=7)
        ax.grid(color=GRID_COL, alpha=0.4, linewidth=0.5)
        df_s = df_hist[df_hist['subject'] == subj]
        ep   = df_s['epoch'].values
        if 'loss'      in df_s.columns: ax.plot(ep, df_s['loss'],      color='#4C72B0', lw=1.5, label='Train')
        if 'val_loss'  in df_s.columns: ax.plot(ep, df_s['val_loss'],  color='#55A868', lw=1.5, label='Val')
        if 'test_loss' in df_s.columns: ax.plot(ep, df_s['test_loss'], color='#DD8452', lw=1.5, label='Test', linestyle='--')
        ax.set_title(f'S{subj:02d}', color=TEXT_COL, fontsize=9, fontweight='bold')
        ax.set_xlabel('Epoch', color=TEXT_COL, fontsize=7)
        ax.set_ylabel('Loss',  color=TEXT_COL, fontsize=7)
    # hide unused subplots
    for idx in range(n_s, nrows * ncols):
        axes3[idx // ncols][idx % ncols].set_visible(False)
    fig3.suptitle('Per-Subject Loss Curves', color=TEXT_COL, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    persubj_png = out_dir / 'per_subject_loss_curves.png'
    plt.savefig(persubj_png, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig3)
    print(f'Per-subject loss curves → {persubj_png}')

    # ── 5. LOSO bar chart (original) ─────────────────────────────────────
    subjs = df_res['target_subject'].tolist()
    x = np.arange(len(subjs)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(subjs)*0.7), 5))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(PANEL_BG)
    ax.spines[:].set_color(GRID_COL); ax.tick_params(colors=TEXT_COL)
    ax.grid(axis='y', color=GRID_COL, alpha=0.5, linewidth=0.7)
    ax.bar(x - w/2, df_res['best_val_trial_acc'],   w, label='Best Val Trial',  color='#4C72B0', alpha=0.85)
    ax.bar(x + w/2, df_res['final_test_trial_acc'], w, label='Final Test Trial', color='#55A868', alpha=0.85)
    mv = df_res['best_val_trial_acc'].mean(); mt = df_res['final_test_trial_acc'].mean()
    ax.axhline(0.25, color='#ff4444', linestyle='--', linewidth=1, label='Chance 25%')
    ax.axhline(mv,   color='#4C72B0', linestyle=':', linewidth=1.5, label=f'Mean Val {mv:.3f}')
    ax.axhline(mt,   color='#55A868', linestyle=':', linewidth=1.5, label=f'Mean Test {mt:.3f}')
    ax.set_xticks(x); ax.set_xticklabels([f'S{s:02d}' for s in subjs], color=TEXT_COL, fontsize=9)
    ax.set_ylabel('Accuracy', color=TEXT_COL); ax.set_ylim(0, 1.05)
    ax.set_title('LTSM SNN LOSO — Val & Test Trial Accuracy', color=TEXT_COL, fontsize=13, fontweight='bold')
    ax.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / 'LTSM_SNN_loso_chart.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f'Chart → {out_dir}/LTSM_SNN_loso_chart.png')



# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # Step 1: Load
    df_d = load_dreamer(CONFIG['dreamer_csv'])
    df_s = load_seediv(CONFIG['seediv_csv'])

    # Step 2: Segment
    df_seg_d = artifact_reject_and_segment(df_d, CONFIG, 'dreamer')
    df_seg_s = artifact_reject_and_segment(df_s, CONFIG, 'seediv')
    del df_d, df_s

    if CONFIG['_smoke']:
        # Cap to 80 segs per dataset to make build_tensor_dataset fast
        df_seg_d = df_seg_d.groupby('subject').head(10).reset_index(drop=True)
        # Ensure all 4 classes present in SEED-IV subset
        df_seg_s = df_seg_s.groupby('label').head(20).reset_index(drop=True)
        print(f'  [SMOKE] df_seg_d: {len(df_seg_d)} rows  df_seg_s: {len(df_seg_s)} rows')

    # Step 3: GMM re-labeling
    yd_gmm = run_gmm_relabeling(df_seg_d, df_seg_s, CONFIG)
    assert len(yd_gmm) == len(df_seg_d)
    df_seg_d = df_seg_d.copy()
    df_seg_d['label_orig'] = df_seg_d['label'].values
    df_seg_d['label']      = yd_gmm
    flag('GMM labels merged  →  df_seg_d["label"] = GMM label')
    
    # 在 GMM label 上做 per-subject trial-level balancing
    df_seg_d = label_balancing_soft(df_seg_d, k_min=10, random_state=42)

    df_seg_d_bal = df_seg_d_bal.copy()
    df_seg_d_bal['trial_id'] = (
    df_seg_d_bal['dataset'].astype(str) + "__"
    + df_seg_d_bal['subject'].astype(str) + "__"
    + df_seg_d_bal['video'].astype(str)
    )

    print("re-added the trial id...")
    
    
    # 之後的 LOSO / training 都改用 df_seg_d_bal
    df_seg_d = df_seg_d_bal
    flag('Applied label_balancing on GMM labels → df_seg_d now balanced per subject')

    # Step 4: Pre-build SEED-IV dataset (shared across folds)
    flag('Pre-building SEED-IV TensorDataset')
    seed_ds_full = build_tensor_dataset(df_seg_s, CONFIG)
    print(f'  SEED-IV: {len(seed_ds_full)} segments')

    # Step 5: Route C LOSO
    subjects = sorted(df_seg_d['subject'].unique().tolist())
    if CONFIG['subjects']:
        subjects = [s for s in subjects if s in CONFIG['subjects']]
    flag(f'Route C LOSO  —  {len(subjects)} subjects: {subjects}')

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

    save_results(all_results, CONFIG)

    # ── best_overall.pt + demo_best_overall.npz ──────────────────────────────
    # Pick the subject with the highest best_val_trial_acc as the "overall best"
    best_subj_result = max(all_results, key=lambda r: r['best_val_trial_acc'])
    best_subj = best_subj_result['target_subject']
    src_pt    = out_dir / f'best_s{best_subj:02d}.pt'
    dst_pt    = out_dir / 'best_overall.pt'
    if src_pt.exists():
        import shutil
        shutil.copy2(src_pt, dst_pt)
        print(f'  best_overall.pt ← S{best_subj:02d}  '
              f'(val_trial={best_subj_result["best_val_trial_acc"]:.3f})')
    else:
        print(f'  [Warning] {src_pt} not found, skipping best_overall.pt')

    # demo_best_overall.npz — most confident correct window from best subject
    src_npz = out_dir / f'all_test_data_s{best_subj:02d}.npz'
    if src_npz.exists():
        d = np.load(src_npz, allow_pickle=True)
        eeg_feat = d['eeg_feat']   # (N, 9, 384, 14)
        labels   = d['labels']     # (N,)
        preds    = d['preds']      # (N,)
        probs    = d['probs']      # (N, 4)
        correct_mask = (labels == preds)
        if correct_mask.any():
            confs = probs[np.arange(len(probs)), preds]  # confidence of predicted class
            confs_correct = np.where(correct_mask, confs, -1.0)
            best_idx = int(np.argmax(confs_correct))
            demo_path = out_dir / 'demo_best_overall.npz'
            np.savez(demo_path,
                     eeg_feat=eeg_feat[best_idx],   # (9, 384, 14)
                     label=np.array(labels[best_idx]),
                     pred=np.array(preds[best_idx]),
                     probs=probs[best_idx],          # (4,)
                     conf=np.array(confs_correct[best_idx]),
                     subject=np.array(best_subj))
            print(f'  demo_best_overall.npz → label={labels[best_idx]} '
                  f'pred={preds[best_idx]} conf={confs_correct[best_idx]:.3f}')
        else:
            print(f'  [Warning] No correct predictions for S{best_subj:02d}, skipping demo.')
    else:
        print(f'  [Warning] {src_npz} not found, skipping demo_best_overall.npz')

    # all_test_data.npz — alias pointing to best subject's data
    src_all = out_dir / f'all_test_data_s{best_subj:02d}.npz'
    dst_all = out_dir / 'all_test_data.npz'
    if src_all.exists():
        shutil.copy2(src_all, dst_all)
        print(f'  all_test_data.npz ← S{best_subj:02d}')

    flag('ALL DONE')


if __name__ == '__main__':
    main()