import pandas as pd
from datasets import mat_dataset_load
from utils import ds, _parse_eeg

def load_dreamer(csv_path):
    df = mat_dataset_load(csv_path)
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

def load_raw_datasets(CONFIG):
    df_d = load_dreamer(CONFIG['dreamer_csv'])
    df_s = load_seediv(CONFIG['seediv_csv'])
    return df_d, df_s