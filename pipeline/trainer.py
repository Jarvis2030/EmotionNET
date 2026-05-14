
from collections import defaultdict
from utils import mmd_loss
import numpy as np
import torch
import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import build_tensor_dataset, EEGUnlabeledDataset
from model import EmotionNET, EmotionNET_ANN

# SNN model
try:
    from model import spike_rate_loss

    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, n_classes=4, return_preds=False, temperature=1.0, return_logits=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    trial_preds = defaultdict(list)
    trial_labels = {}
    all_logits = []
    all_labels = []
    
    for data, labels, trial_ids in dataloader:
        data = data.to(device, dtype=torch.float32).permute(0, 1, 3, 2)
        labels = labels.to(device, dtype=torch.long)

        logits = model(data) # soft prediction result
        logits_eval = logits / max(float(temperature), 1e-6)

        loss = criterion(logits, labels)
        total_loss += loss.item() * data.size(0) # total loss from each trial

        pred = logits_eval.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        for p, y, tid in zip(
            pred.cpu().numpy(),
            labels.cpu().numpy(),
            trial_ids.cpu().numpy(),
        ):
            trial_preds[int(tid)].append(int(p))
            trial_labels[int(tid)] = int(y)

    seg_acc = correct / max(total, 1)
    seg_loss = total_loss / max(total, 1)

    y_true, y_pred = [], []
    for tid in sorted(trial_preds.keys()):
        counts = np.bincount(
            np.array(trial_preds[tid], dtype=np.int64),
            minlength=n_classes,
        )
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


def train_one_epoch(model, drm_loader, seed_loader, tgt_loader,
                            criterion, optimizer, device,
                            lambda_mmd=0.05, lambda_seed=1.0,
                            warmup=False, config=None):

    model.train()
    
    total_loss = 0.0
    total_ce_d = 0.0
    total_ce_s = 0.0
    total_mmd_v = 0.0
    correct, total = 0, 0

    seed_iter = iter(seed_loader)
    tgt_iter = iter(tgt_loader)

    for drm_data, drm_labels, _ in drm_loader:
        drm_data = drm_data.to(device, dtype=torch.float32).permute(0, 1, 3, 2)
        drm_labels = drm_labels.to(device, dtype=torch.long)

        optimizer.zero_grad()

        drm_logits, drm_feat = model(drm_data, return_feat=True)
        ce_d = criterion(drm_logits, drm_labels)

        if warmup:
            ce_s_val = 0.0
            mmd_val = 0.0
        else:
            try:
                seed_data, seed_labels, _ = next(seed_iter)
            except StopIteration:
                seed_iter = iter(seed_loader)
                seed_data, seed_labels, _ = next(seed_iter)

            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_batch = next(tgt_iter)

            seed_data = seed_data.to(device, dtype=torch.float32).permute(0, 1, 3, 2)
            seed_labels = seed_labels.to(device, dtype=torch.long)
            tgt_data = tgt_batch[0].to(device, dtype=torch.float32).permute(0, 1, 3, 2)

            seed_feat = model.extract_features(seed_data)
            seed_logits = model.head(seed_feat)

            _, tgt_feat = model(tgt_data, return_feat=True)

            ce_s = criterion(seed_logits, seed_labels)
            mmd = mmd_loss(drm_feat, tgt_feat)

            ce_s_val = ce_s.item()
            mmd_val = mmd.item()

        spike_reg = (
            spike_rate_loss(
                model,
                config.get("snn_target_rate", 0.1),
                config.get("snn_spike_weight", 0.0),
            )
            if config is not None and config.get("_use_snn", False)
            else 0.0
        )

        loss = ce_d + lambda_seed * ce_s_val * 0 if warmup else ce_d
        if not warmup:
            loss = ce_d + lambda_seed * ce_s + lambda_mmd * mmd + spike_reg
        else:
            loss = ce_d + spike_reg

        loss.backward()
        optimizer.step()

        bs = drm_data.size(0)
        total_loss += loss.item() * bs
        total_ce_d += ce_d.item() * bs
        total_ce_s += ce_s_val * bs
        total_mmd_v += mmd_val * bs

        pred = drm_logits.argmax(dim=1)
        correct += (pred == drm_labels).sum().item()
        total += bs

    n = max(total, 1)
    
    return {
        "loss": total_loss / n,
        "ce_drm": total_ce_d / n,
        "ce_seed": total_ce_s / n,
        "mmd_loss": total_mmd_v / n,
        "acc": correct / n,
    }

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drm_train_ds, drm_val_ds, tgt_test_ds, tgt_unlab_ds, seed_ds = loso_splits(
        df_seg_d, target_subject, config, seed_ds_prebuilt
    )

    drm_train_loader = DataLoader(
        drm_train_ds,
        batch_size=config["batch"],
        shuffle=True,
        drop_last=True,
    )
    drm_val_loader = DataLoader(
        drm_val_ds,
        batch_size=config["batch"],
        shuffle=False,
        drop_last=False,
    )
    seed_loader = DataLoader(
        seed_ds,
        batch_size=config["batch"],
        shuffle=True,
        drop_last=True,
    )
    tgt_unlab_loader = DataLoader(
        tgt_unlab_ds,
        batch_size=config["batch"],
        shuffle=True,
        drop_last=False,
    )
    tgt_test_loader = DataLoader(
        tgt_test_ds,
        batch_size=config["batch"],
        shuffle=False,
        drop_last=False,
    )

    # 建立模型
    if config.get("_use_snn", False):
        model = EmotionNET(
            in_channels=config["in_channels"],
            eeg_channels=config["n_channels"],
            out_channels=config["snn_out_channels"],
            n_classes=config["n_classes"],
            fs=config["fs"],
            decision_window=config["window_size"] // config["fs"],
            dropout=config["dropout"],
            lstm_hidden=config.get("snn_lstm_hidden", 64),
            lstm_layers=config.get("snn_lstm_layers", 1),
            beta1=config.get("snn_beta1", 0.9),
            beta2=config.get("snn_beta2", 0.9),
            threshold=config.get("snn_threshold", 0.3),
        ).to(device)
        print(f"  Model: EEGSNN (snntorch)  fusion_dim={model.fusion_dim}")
    else:
        model = EmotionNET_ANN(
            fs=config["fs"],
            input_time=config["window_size"] // config["fs"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            n_classes=config["n_classes"],
            eeg_channels=config["n_channels"],
            lstm_hidden=config["lstm_hidden"],
            lstm_layers=config["lstm_layers"],
            dropout=config["dropout"],
        ).to(device)
        print(f"  Model: EEG2DCNNLSTM  fusion_dim={model.fusion_dim}")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.get("label_smoothing", 0.05)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    best_state = None
    best_trial = -1.0
    patience = 0
    history = []
    best_temperature = 1.0
    best_val_trial_seen = -1.0  # reporting only

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup_epochs = config.get("warmup_epochs", 0)

    # Training loop
    for epoch in range(config["num_epochs"]):
        is_warmup = epoch < warmup_epochs

        train_stats = train_one_epoch(
            model,
            drm_train_loader,
            seed_loader,
            tgt_unlab_loader,
            criterion,
            optimizer,
            device,
            lambda_mmd=config["lambda_mmd"],
            lambda_seed=config["lambda_seed"],
            warmup=is_warmup,
            config=config,
        )

        val_loss, val_acc, val_trial = evaluate(
            model,
            drm_val_loader,
            criterion,
            device,
            config["n_classes"],
        )
        test_loss, test_acc, test_trial = evaluate(
            model,
            tgt_test_loader,
            criterion,
            device,
            config["n_classes"],
        )

        history.append(
            {
                "epoch": epoch,
                **train_stats,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_trial_acc": val_trial,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_trial_acc": test_trial,
                "temperature": best_temperature,
            }
        )

        print(
            f"[S{target_subject:02d}] Ep {epoch:03d} | "
            f"Loss {train_stats['loss']:.4f} "
            f"(CE_D {train_stats['ce_drm']:.4f} "
            f"CE_S {train_stats['ce_seed']:.4f} "
            f"MMD {train_stats['mmd_loss']:.4f}) "
            f"Acc {train_stats['acc']:.4f} | "
            f"Val {val_acc:.4f} Tr {val_trial:.4f} | "
            f"Test {test_acc:.4f} Tr {test_trial:.4f}",
            flush=True,
        )

        # 追蹤最佳 val_trial（for reporting）
        if val_trial > best_val_trial_seen:
            best_val_trial_seen = val_trial

        # early stopping 依據 val_trial
        if val_trial > best_trial:
            best_trial = val_trial
            best_state = copy.deepcopy(model.state_dict())
            patience = 0

            ckpt_path = out_dir / f"best_s{target_subject:02d}.pt"
            torch.save(
                {
                    "subject": target_subject,
                    "epoch": epoch,
                    "val_trial_acc": val_trial,
                    "temperature": best_temperature,
                    "state_dict": best_state,
                },
                ckpt_path,
            )
        else:
            patience += 1
            if patience >= config["early_stop_patience"]:
                print(f"  Early stopping at epoch {epoch}.", flush=True)
                break

    # ── 用 best_state 做最後 test + 收集試驗資料 ──────────────────────────
    model.load_state_dict(best_state)

    test_loss, test_acc, test_trial, y_true, y_pred = evaluate(
        model,
        tgt_test_loader,
        criterion,
        device,
        config["n_classes"],
        return_preds=True,
        temperature=best_temperature,
    )

    all_feats, all_labels_np, all_preds_np, all_probs_np = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tgt_test_loader:
            feats_b, labels_b = batch[0], batch[1]
            feats_b = feats_b.to(device, dtype=torch.float32).permute(0, 1, 3, 2)

            logits_b = model(feats_b)
            probs_b = torch.softmax(logits_b, dim=-1)
            preds_b = probs_b.argmax(dim=-1)

            all_feats.append(batch[0].cpu().numpy())
            all_labels_np.append(labels_b.cpu().numpy())
            all_preds_np.append(preds_b.cpu().numpy())
            all_probs_np.append(probs_b.cpu().numpy())

    all_feats_np = np.concatenate(all_feats, axis=0)
    all_labels_np = np.concatenate(all_labels_np, axis=0)
    all_preds_np = np.concatenate(all_preds_np, axis=0)
    all_probs_np = np.concatenate(all_probs_np, axis=0)

    test_npz_path = out_dir / f"all_test_data_s{target_subject:02d}.npz"
    np.savez(
        test_npz_path,
        eeg_feat=all_feats_np,
        labels=all_labels_np,
        preds=all_preds_np,
        probs=all_probs_np,
        subject=np.array(target_subject),
    )
    print(f"  Test data → {test_npz_path}  (N={len(all_labels_np)} windows)")

    return {
        "target_subject": target_subject,
        "best_val_trial_acc": best_val_trial_seen,
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        "final_test_trial_acc": test_trial,
        "history": history,
        "y_true": y_true,
        "y_pred": y_pred,
    }
