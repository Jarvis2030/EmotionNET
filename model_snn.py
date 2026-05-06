"""
SNN Model — Delta Encoder + 2 FC layers
使用 lava-dl (lava.lib.dl.slayer)

Input:  (B, 9, 14, 384)   ← EEG_band_analysis 輸出，T=384 timesteps
Output: (B, 4)             ← 4-class logits

Architecture:
    Delta encoder  : slayer.block.cuba.Dense  (delta coding, sparse activation)
    FC1            : slayer.block.cuba.Dense + LIF
    FC2            : slayer.block.cuba.Dense + LIF
    Readout head   : nn.Linear (rate coding → logits)

Run with --snn flag in run_pipeline.py
"""

import torch
import torch.nn as nn
import lava.lib.dl.slayer as slayer

# ─────────────────────────────────────────────────────────────────────────────
# Main SNN Model
# ─────────────────────────────────────────────────────────────────────────────

class EEGSNNRouteC(nn.Module):
    def __init__(self, in_channels=9, eeg_channels=14,
                 out_channels=50, hidden=50,
                 n_classes=4, fs=128, decision_window=3,
                 dropout=0.3):
        super().__init__()
        self.T           = fs * decision_window   # 384
        self.in_flat     = in_channels * eeg_channels  # 9*14 = 126
        self.out_channels = out_channels
        self.fusion_dim  = hidden   # for CORAL compatibility

        # kernel/stride same as SNNEMotionNet
        kernel_time = fs // 4   # 32
        stride_time = fs // 8   # 16
        time_steps  = (self.T - kernel_time) // stride_time + 1  # e.g. 23

        # ── CNN ──
        self.cnn = nn.Sequential(
            nn.Conv1d(self.in_flat, out_channels,
                      kernel_size=kernel_time, stride=stride_time),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
        )

        # ── Neuron params ──
        encoder_params = {
            'threshold'       : 0.3,
            'current_decay'   : 0.9,
            'voltage_decay'   : 0.9,
            'tau_grad'        : 1.0,
            'scale_grad'      : 1.0,
            'scale'           : 1 << 6,
            'norm'            : None,
            'dropout'         : None,
            'shared_param'    : True,
            'persistent_state': False,
            'requires_grad'   : True,
            'graded_spike'    : False,
        }
        dense_params = {
            'threshold'     : 0.1,
            'current_decay' : 1.0,
            'voltage_decay' : 0.1,
            'requires_grad' : True,
        }

        # ── SNN blocks ──
        self.encoder = nn.ModuleList([
            slayer.block.cuba.Input(encoder_params),
        ])
        self.snn = nn.ModuleList([
            slayer.block.cuba.Dense(dense_params, out_channels, hidden),
            slayer.block.cuba.Dense(dense_params, hidden,       hidden),
        ])

        # ── Readout ──
        self.head = nn.Linear(hidden, n_classes)

    def extract_features(self, x):
        """
        x: (B, 9, 384, 14) from training loop (after permute)
        """
        B = x.shape[0]
        # (B, 9, 384, 14) → (B, 126, 384)
        z = x.permute(0, 1, 3, 2).reshape(B, self.in_flat, self.T)

        # CNN: (B, 126, 384) → (B, out_channels, T')
        z = self.cnn(z)

        # lava Input block expects (B, C, T) — Conv1d output is already 3D
        for block in self.encoder:
            z = block(z)

        for block in self.snn:
            z = block(z)

        # rate coding: mean over T → (B, hidden)
        return z.mean(dim=-1)

    def forward(self, x, return_feat=False):
        feat   = self.extract_features(x)
        logits = self.head(feat)
        if return_feat:
            return logits, feat
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Loss helper (spike regularization)
# ─────────────────────────────────────────────────────────────────────────────

def spike_rate_loss(model, target_rate=0.1, weight=1e-3):
    total = 0.0
    count = 0
    for m in model.modules():
        if hasattr(m, 'neuron') and hasattr(m.neuron, 'spike'):
            s = m.neuron.spike
            if s is not None:
                total += (s.mean() - target_rate).pow(2)
                count += 1
    return weight * total if count > 0 else torch.tensor(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG additions for SNN
# ─────────────────────────────────────────────────────────────────────────────

SNN_CONFIG = {
    'snn_out_channels'  : 50,
    'snn_hidden'        : 50,
    'snn_spike_weight'  : 1e-3,
    'snn_target_rate'   : 0.1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Neuron params (CUBA LIF)
# ─────────────────────────────────────────────────────────────────────────────

def _cuba_params(threshold=1.0, current_decay=0.25, voltage_decay=0.03,
                 requires_grad=True):
    return {
        'threshold'     : threshold,
        'current_decay' : current_decay,
        'voltage_decay' : voltage_decay,
        'requires_grad' : requires_grad,
        'shared_param'  : True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Delta Encoder block
# ─────────────────────────────────────────────────────────────────────────────

class DeltaEncoderBlock(nn.Module):
    """
    Encode input as delta (frame difference) then pass through a Dense CUBA block.

    Input  : (B, C_in, H, T)   e.g. (B, 9, 14, 384)
    Output : (B, hidden, T)    spike rate tensor
    """
    def __init__(self, in_features, hidden, threshold=1.0,
                 current_decay=0.25, voltage_decay=0.03):
        super().__init__()
        self.in_features = in_features   # C_in * H = 9 * 14 = 126

        # slayer Dense expects (B, C, T)
        self.dense = slayer.block.cuba.Dense(
            neuron_params=_cuba_params(threshold, current_decay, voltage_decay),
            in_neurons=in_features,
            out_neurons=hidden,
            weight_norm=True,
            delay=False,
        )

    def forward(self, x):
        # x: (B, 9, 14, 384)
        B, C, H, T = x.shape
        x_flat = x.reshape(B, C * H, T)          # (B, 126, 384)

        # Delta coding: difference between consecutive timesteps
        delta = torch.zeros_like(x_flat)
        delta[:, :, 1:] = x_flat[:, :, 1:] - x_flat[:, :, :-1]
        delta[:, :, 0]  = x_flat[:, :, 0]        # first frame = raw

        # lava Dense expects (B, C, 1, 1, T)
        delta_5d = delta.unsqueeze(2).unsqueeze(3)  # (B, 126, 1, 1, 384)
        out = self.dense(delta_5d)                  # (B, hidden, 1, 1, T)
        return out.squeeze(2).squeeze(2)            # (B, hidden, T)


# ─────────────────────────────────────────────────────────────────────────────
# Main SNN Model
# ─────────────────────────────────────────────────────────────────────────────

class EEGSNNRouteC(nn.Module):
    """
    Delta Encoder + 2 FC (CUBA LIF) + linear readout head.

    Parameters
    ----------
    in_channels  : int  — number of band feature maps (default 9)
    eeg_channels : int  — number of EEG channels (default 14)
    hidden1      : int  — Delta encoder output neurons
    hidden2      : int  — FC1 output neurons
    hidden3      : int  — FC2 output neurons
    n_classes    : int  — classification head output
    T            : int  — number of timesteps (= window_size = 384)
    """

    def __init__(self, in_channels=9, eeg_channels=14,
                 hidden1=256, hidden2=128, hidden3=64,
                 n_classes=4, T=384,
                 threshold=1.0, current_decay=0.25, voltage_decay=0.03):
        super().__init__()
        self.T = T
        in_features = in_channels * eeg_channels   # 9 * 14 = 126

        np_ = _cuba_params(threshold, current_decay, voltage_decay)

        # Delta encoder
        self.delta_enc = DeltaEncoderBlock(
            in_features=in_features,
            hidden=hidden1,
            threshold=threshold,
            current_decay=current_decay,
            voltage_decay=voltage_decay,
        )

        # FC1
        self.fc1 = slayer.block.cuba.Dense(
            neuron_params=np_,
            in_neurons=hidden1,
            out_neurons=hidden2,
            weight_norm=True,
            delay=False,
        )

        # FC2
        self.fc2 = slayer.block.cuba.Dense(
            neuron_params=np_,
            in_neurons=hidden2,
            out_neurons=hidden3,
            weight_norm=True,
            delay=False,
        )

        # Readout: rate coding → logits
        # sum spikes over T, then linear projection
        self.head = nn.Linear(hidden3, n_classes)

        # For CORAL compatibility: expose fusion_dim
        self.fusion_dim = hidden3

    def extract_features(self, x):
        """
        Accept either:
        (B, 9, 14, 384) or
        (B, 9, 384, 14)
        Convert everything to (B, 9, 14, 384).
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got shape={x.shape}")

        # already (B, 9, 14, T)
        if x.shape[2] == 14 and x.shape[3] == self.T:
            pass

        # currently (B, 9, T, 14) -> permute to (B, 9, 14, T)
        elif x.shape[2] == self.T and x.shape[3] == 14:
            x = x.permute(0, 1, 3, 2)

        else:
            raise ValueError(
                f"Unexpected input shape {x.shape}. "
                f"Expected (B, 9, 14, {self.T}) or (B, 9, {self.T}, 14)."
            )

        s1 = self.delta_enc(x)              # (B, hidden1, T)
        # FC1: lava Dense expects (B, C, 1, 1, T)
        s1_5d = s1.unsqueeze(2).unsqueeze(3)
        s2_5d = self.fc1(s1_5d)             # (B, hidden2, 1, 1, T)
        s2    = s2_5d.squeeze(2).squeeze(2) # (B, hidden2, T)
        s2_5d = s2.unsqueeze(2).unsqueeze(3)
        s3_5d = self.fc2(s2_5d)             # (B, hidden3, 1, 1, T)
        s3    = s3_5d.squeeze(2).squeeze(2) # (B, hidden3, T)
        return s3.mean(dim=-1)              # (B, hidden3)  rate coding

    def forward(self, x, return_feat=False):
        feat   = self.extract_features(x)   # (B, hidden3)
        logits = self.head(feat)             # (B, 4)
        if return_feat:
            return logits, feat
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Loss helper (spike regularization)
# ─────────────────────────────────────────────────────────────────────────────

def spike_rate_loss(model, target_rate=0.1, weight=1e-3):
    """
    Safe version: only use spike tensors if they really exist.
    """
    total = 0.0
    count = 0
    device = next(model.parameters()).device

    for m in model.modules():
        if hasattr(m, 'neuron'):
            spike_attr = getattr(m.neuron, 'spike', None)

            # only use it if it is a tensor, not a function
            if isinstance(spike_attr, torch.Tensor):
                total = total + (spike_attr.mean() - target_rate).pow(2)
                count += 1

    if count > 0:
        return weight * total
    else:
        return torch.tensor(0.0, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG additions for SNN (merge into run_pipeline CONFIG)
# ─────────────────────────────────────────────────────────────────────────────

SNN_CONFIG = {
    'snn_hidden1'       : 256,
    'snn_hidden2'       : 128,
    'snn_hidden3'       : 64,
    'snn_threshold'     : 1.0,
    'snn_current_decay' : 0.25,
    'snn_voltage_decay' : 0.03,
    'snn_spike_weight'  : 1e-3,   # spike rate regularization weight
    'snn_target_rate'   : 0.1,    # target mean spike rate
}

model = EEGSNNRouteC()
x = torch.randn(4, 9, 384, 14)
print('Input:', x.shape)
with torch.no_grad():
    logits, feat = model(x, return_feat=True)
print('feat:', feat.shape)    # should be (4, 50)
print('logits:', logits.shape) # should be (4, 4)
print('OK')