import random
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
from scipy.signal import resample, butter, filtfilt, stft
from scipy.interpolate import interp1d

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

def gaussian_kernel(x, y, sigmas=(1, 2, 4, 8, 16)):
    beta = 1.0 / (2.0 * torch.tensor(sigmas, device=x.device, dtype=x.dtype).view(-1, 1, 1))
    dist = torch.cdist(x, y, p=2).pow(2).unsqueeze(0)
    return torch.exp(-beta * dist).sum(dim=0)


def mmd_loss(source, target, sigmas=(1, 2, 4, 8, 16)):
    return (gaussian_kernel(source, source, sigmas).mean()
            + gaussian_kernel(target, target, sigmas).mean()
            - 2 * gaussian_kernel(source, target, sigmas).mean())


def EEG_band_analysis(fs, seg, freq_bend = [(1,4), (4,8), (8,13), (13,30)], out_T = 1):
    C, T = seg.shape
    band_list = []

    for (low, high) in freq_bend:
        band_sig = np.zeros_like(seg)
        psd_all_ch = np.zeros_like(seg)

        # for each channel
        for ch in range(C):
            x = seg[ch,:]
            
            # STFT: Zxx shape = (n_freq, n_time)
            # 例如目標 1 秒視窗 -> nperseg ≈ fs * 1.0，但不能超過 T_raw
            nperseg = max(16, min(out_T, T))  # 至少 16 samples，最多 T_raw

            noverlap = int(nperseg * 0.5)
            noverlap = min(noverlap, nperseg - 1)  
            f, t, Zxx = stft(x, fs=fs, nperseg= nperseg, noverlap=noverlap)

            # PSD
            Pxx = np.abs(Zxx) ** 2  # (n_freq, n_time)

            mask = (f >= low) & (f <= high)
            if np.any(mask):
                power_t = Pxx[mask, :].mean(axis=0).astype(np.float32)
            else:
                power_t = np.zeros(Pxx.shape[1], dtype=np.float32)
            # power_t 現在是長度 n_time 的序列，要插值到 out_T
            n_time = power_t.shape[0]
            if n_time == 1:
                # 只有一個時間點，就直接複製
                seq = np.full((out_T,), power_t[0], dtype=np.float32)
            else:
                # 線性插值到 out_T 個時間點
                x_src = np.linspace(0, 1, n_time)
                x_tgt = np.linspace(0, 1, out_T)
                f_interp = interp1d(x_src, power_t, kind='linear')
                seq = f_interp(x_tgt).astype(np.float32)  # (out_T,)

            psd_all_ch[ch,:] = seq

            # decompose the original EEG
            b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
            band_sig[ch,:] = filtfilt(b, a, x)
        
        # append back to data
        # 4 band data, 4 PSD corresponding to each band, 1 original eeg
        band_list.append(band_sig)   # [C,T]
        band_list.append(psd_all_ch)

    band_list.append(seg)

    # Stack back to the 3D shape
    featured_x = np.stack(band_list, axis=0)
    return featured_x  # [9, C, T]