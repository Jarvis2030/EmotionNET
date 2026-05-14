import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from utils import EEG_band_analysis

def mat_dataset_load(path = '/Users/linyuchun/Desktop/Project/SNN/data/EEG_clean_table.csv', eeg_prefix = 'EEG_clean'):
    seg_eeg = pd.read_csv(path)
    eeg_cols = [c for c in seg_eeg.columns if c.startswith(eeg_prefix)]

    seg_eeg[eeg_prefix] = (
        seg_eeg[eeg_cols]
        .apply(lambda row: row.dropna().astype(float).values, axis=1)
    )

    seg_eeg = seg_eeg.drop(columns=eeg_cols)
    return seg_eeg

def label_balancing(seg_eeg, random_state=42):
    """
    seg_eeg 至少需要欄位:
      - dataset   (e.g. 'dreamer', 'seed')
      - subject
      - video
      - label
    若有多個 session, 建議多一欄 session_idx 一起納入 trial_id.
    """

    rng = np.random.default_rng(random_state)
    df  = seg_eeg.copy()

    # ── 1. 建立 trial_id ────────────────────────────────────────────────────
    if "session_idx" in df.columns:
        df["trial_id"] = (
            df["dataset"].astype(str) + "__"
            + df["subject"].astype(str) + "__"
            + df["session_idx"].astype(str) + "__"
            + df["video"].astype(str)
        )
    else:
        df["trial_id"] = (
            df["dataset"].astype(str) + "__"
            + df["subject"].astype(str) + "__"
            + df["video"].astype(str)
        )

    # ── 2. Trial-level df（一個 trial 一列）────────────────────────────────
    agg_dict = {"dataset": "first", "subject": "first",
                "video": "first", "label": "first"}
    if "session_idx" in df.columns:
        agg_dict["session_idx"] = "first"

    df_trials = (
        df.groupby("trial_id")
          .agg(agg_dict)
          .reset_index()
    )

    labels_trial = df_trials["label"].values
    n0 = (labels_trial == 0).sum()
    n1 = (labels_trial == 1).sum()
    n2 = (labels_trial == 2).sum()
    n3 = (labels_trial == 3).sum()
    print(f"Trial-level counts before balance: [0:{n0}, 1:{n1}, 2:{n2}, 3:{n3}]")

    # ── 3. Per-subject stratified random balancing ──────────────────────────
    #
    #   做法：
    #   a) 對每個 subject，找出其 4 個 class 中最少的那個 class 的 trial 數 n_min_subj
    #   b) 每個 class 在該 subject 內 random sample n_min_subj 個 trial
    #   c) 合併所有 subject 的結果
    #
    #   這樣做的好處：
    #   - 不會因為某些 subject HAHV 特別多而讓其他 subject 消失
    #   - 每個 subject 在訓練集的 class 都是平衡的
    # ────────────────────────────────────────────────────────────────────────

    kept_trial_ids = []
    subjects = sorted(df_trials["subject"].unique())

    per_subj_stats = []
    for subj in subjects:
        sub_df = df_trials[df_trials["subject"] == subj]
        sub_labels = sub_df["label"].values

        counts = [int((sub_labels == c).sum()) for c in range(4)]
        n_min = min(c for c in counts if c > 0)   # 忽略完全沒有的 class

        sel_ids = []
        for c in range(4):
            c_ids = sub_df.loc[sub_labels == c, "trial_id"].values
            if len(c_ids) == 0:
                continue
            k = min(n_min, len(c_ids))
            chosen = rng.choice(c_ids, size=k, replace=False)
            sel_ids.extend(chosen.tolist())

        kept_trial_ids.extend(sel_ids)
        per_subj_stats.append({
            "subject": subj,
            "before": counts,
            "n_min": n_min,
            "kept_per_class": n_min,
        })

    # 印出 per-subject 統計
    print("\nPer-subject balancing summary:")
    print(f"{'Subj':>5}  {'Before [0,1,2,3]':>22}  {'n_min':>6}  {'kept_total':>10}")
    for s in per_subj_stats:
        print(f"{s['subject']:>5}  {str(s['before']):>22}  {s['n_min']:>6}  {s['n_min']*4:>10}")

    df_trials_bal = df_trials[df_trials["trial_id"].isin(kept_trial_ids)].reset_index(drop=True)

    print(f"\nGlobal trial-level counts after per-subject balance:")
    print(df_trials_bal["label"].value_counts().sort_index())
    print(f"Total balanced trials: {len(df_trials_bal)}")

    # ── 4. 回到 segment/channel level ──────────────────────────────────────
    keep_set  = set(kept_trial_ids)
    mask_keep = df["trial_id"].isin(keep_set)
    df_bal    = df[mask_keep].reset_index(drop=True)

    # 用 balanced trial 的 label 覆蓋（保險起見）
    label_map = df_trials_bal.set_index("trial_id")["label"].to_dict()
    df_bal["label"] = df_bal["trial_id"].map(label_map)

    print(f"\nSegment/channel-level label counts after balancing:")
    print(df_bal["label"].value_counts().sort_index())
    print(f"df_bal shape: {df_bal.shape}")

    # 清掉 helper 欄位
    df_bal = df_bal.drop(columns=["trial_id"])

    return df_bal



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

