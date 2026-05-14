# reporting.py
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def save_results(all_results, config):
    """
    Save LOSO results:
      - summary CSV
      - aggregated & per-subject confusion matrices (+ figure)
      - classification report
      - epoch history / mean stats CSV
      - loss/accuracy curves
      - LOSO bar chart
      - best_overall.pt, demo_best_overall.npz, all_test_data.npz
    """
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. summary CSV
    df_res = _save_summary_csv(all_results, out_dir)

    # 2. Confusion matrix + classification report + figure
    _save_confusion_matrices(all_results, config, out_dir)

    # 3. Epoch history / mean stats / curves
    df_hist, df_mean = _save_epoch_stats(all_results, out_dir)

    # 4. Loss/Acc 曲線 & per-subject loss 曲線 & LOSO bar chart
    _plot_loss_acc_curves(df_hist, df_mean, df_res, out_dir)

    # 5. best_overall.pt + demo_best_overall.npz + all_test_data.npz
    export_best_overall_artifacts(all_results, out_dir)

    # Summary
    print(df_res[["target_subject", "best_val_trial_acc",
                  "final_test_acc", "final_test_trial_acc"]].to_string(index=False))
    print(f'\nMean Val Trial Acc  : {df_res["best_val_trial_acc"].mean():.4f}')
    print(f'Mean Test Trial Acc : {df_res["final_test_trial_acc"].mean():.4f}')


def _save_summary_csv(all_results, out_dir: Path) -> pd.DataFrame:
    rows = [
        {k: v for k, v in r.items() if k not in ("history", "y_true", "y_pred")}
        for r in all_results
    ]
    df_res = pd.DataFrame(rows)
    csv_path = out_dir / "LTSM_SNN_loso_results.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"Results summary → {csv_path}")
    return df_res


def _save_confusion_matrices(all_results, config, out_dir: Path):
    EMOTION_LABELS = ["Neutral", "Sad", "Fear", "Happy"]
    DARK_BG2 = "#0f0f1a"
    PANEL_BG2 = "#1a1a2e"
    TEXT_COL2 = "#e0e0f0"

    # collect all predictions across subjects
    all_true = np.concatenate(
        [r["y_true"] for r in all_results if len(r.get("y_true", [])) > 0]
    )
    all_pred = np.concatenate(
        [r["y_pred"] for r in all_results if len(r.get("y_pred", [])) > 0]
    )
    n_classes = config.get("n_classes", 4)
    labels = list(range(n_classes))
    tick_lbl = EMOTION_LABELS[:n_classes]

    # 1) aggregated CM across all subjects
    cm_agg = confusion_matrix(all_true, all_pred, labels=labels)
    cm_norm = cm_agg.astype(float) / cm_agg.sum(axis=1, keepdims=True).clip(min=1)

    pd.DataFrame(cm_agg, index=tick_lbl, columns=tick_lbl).to_csv(
        out_dir / "confusion_matrix.csv"
    )
    pd.DataFrame(cm_norm, index=tick_lbl, columns=tick_lbl).to_csv(
        out_dir / "confusion_matrix_norm.csv"
    )
    print(f"Confusion matrix → {out_dir}/confusion_matrix.csv")

    # classification report
    report = classification_report(
        all_true, all_pred, labels=labels, target_names=tick_lbl, digits=4
    )
    print("\n" + report)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # 2) per-subject CMs
    subj_cms = {}
    for r in all_results:
        if len(r.get("y_true", [])) == 0:
            continue
        subj_cms[r["target_subject"]] = confusion_matrix(
            r["y_true"], r["y_pred"], labels=labels
        )

    # 3) figure: aggregated CM + per-subject CMs
    n_subj = len(subj_cms)
    ncols_s = min(4, n_subj)
    nrows_s = (n_subj + ncols_s - 1) // ncols_s
    total_rows = 1 + nrows_s  # row 0 = aggregated, rows 1+ = per-subject

    fig_cm, axes_cm = plt.subplots(
        total_rows,
        max(ncols_s, 1),
        figsize=(4.5 * max(ncols_s, 1), 4.5 * total_rows),
        squeeze=False,
    )
    fig_cm.patch.set_facecolor(DARK_BG2)

    def _draw_cm(ax, cm, title):
        ax.set_facecolor(PANEL_BG2)
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_n, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(
            tick_lbl, rotation=30, ha="right", color=TEXT_COL2, fontsize=8
        )
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(tick_lbl, color=TEXT_COL2, fontsize=8)
        ax.set_xlabel("Predicted", color=TEXT_COL2, fontsize=8)
        ax.set_ylabel("True", color=TEXT_COL2, fontsize=8)
        ax.set_title(title, color=TEXT_COL2, fontsize=9, fontweight="bold")
        ax.tick_params(colors=TEXT_COL2)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e2e4a")
        for i in range(n_classes):
            for j in range(n_classes):
                val = cm[i, j]
                pct = cm_n[i, j]
                color = "white" if pct > 0.5 else TEXT_COL2
                ax.text(
                    j,
                    i,
                    f"{val}\n({pct:.0%})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
        return im

    # row 0: aggregated (full-width)
    for c in range(max(ncols_s, 1)):
        axes_cm[0][c].set_visible(False)
    ax_agg = plt.subplot2grid(
        (total_rows, max(ncols_s, 1)), (0, 0), colspan=max(ncols_s, 1), fig=fig_cm
    )
    _draw_cm(ax_agg, cm_agg, "Aggregated CM — All Subjects")

    # rows 1+: per-subject
    for idx, (subj, cm_s) in enumerate(sorted(subj_cms.items())):
        row = 1 + idx // ncols_s
        col = idx % ncols_s
        _draw_cm(axes_cm[row][col], cm_s, f"S{subj:02d}")

    # hide unused
    for idx in range(n_subj, nrows_s * ncols_s):
        axes_cm[1 + idx // ncols_s][idx % ncols_s].set_visible(False)

    plt.tight_layout()
    cm_png = out_dir / "confusion_matrices.png"
    plt.savefig(cm_png, dpi=150, bbox_inches="tight", facecolor=DARK_BG2)
    plt.close(fig_cm)
    print(f"Confusion matrix figure → {cm_png}")


def _save_epoch_stats(all_results, out_dir: Path):
    # 1. Per-subject epoch history CSV
    hist_rows = []
    for r in all_results:
        subj = r["target_subject"]
        for h in r["history"]:
            hist_rows.append({"subject": subj, **h})
    df_hist = pd.DataFrame(hist_rows)
    hist_csv = out_dir / "epoch_history.csv"
    df_hist.to_csv(hist_csv, index=False)
    print(f"Epoch history → {hist_csv}")

    # 2. Mean loss/acc across subjects per epoch
    loss_cols = [
        c
        for c in df_hist.columns
        if c
        in (
            "loss",
            "val_loss",
            "test_loss",
            "acc",
            "val_acc",
            "test_acc",
            "ce_drm",
            "ce_seed",
            "mmd_loss",
        )
    ]
    df_mean = df_hist.groupby("epoch")[loss_cols].mean().reset_index()
    mean_csv = out_dir / "mean_epoch_stats.csv"
    df_mean.to_csv(mean_csv, index=False)
    print(f"Mean epoch stats → {mean_csv}")
    return df_hist, df_mean


def _plot_loss_acc_curves(df_hist, df_mean, df_res, out_dir: Path):
    DARK_BG = "#0f0f1a"
    PANEL_BG = "#1a1a2e"
    GRID_COL = "#2e2e4a"
    TEXT_COL = "#e0e0f0"

    # 3. Loss curves figure (mean across subjects)
    has_loss = {"loss", "val_loss", "test_loss"}.issubset(df_mean.columns)
    has_acc = {"acc", "val_acc", "test_acc"}.issubset(df_mean.columns)
    n_panels = int(has_loss) + int(has_acc)

    if n_panels > 0:
        fig2, axes = plt.subplots(
            1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False
        )
        fig2.patch.set_facecolor(DARK_BG)
        panel = 0
        epochs_x = df_mean["epoch"].values

        if has_loss:
            ax = axes[0][panel]
            panel += 1
            ax.set_facecolor(PANEL_BG)
            ax.spines[:].set_color(GRID_COL)
            ax.tick_params(colors=TEXT_COL)
            ax.grid(color=GRID_COL, alpha=0.4, linewidth=0.6)
            ax.plot(
                epochs_x,
                df_mean["loss"],
                color="#4C72B0",
                lw=1.8,
                label="Train Loss",
            )
            ax.plot(
                epochs_x,
                df_mean["val_loss"],
                color="#55A868",
                lw=1.8,
                label="Val Loss",
            )
            ax.plot(
                epochs_x,
                df_mean["test_loss"],
                color="#DD8452",
                lw=1.8,
                label="Test Loss",
                linestyle="--",
            )
            for col, lbl, clr in [
                ("ce_drm", "CE DREAMER", "#8172B2"),
                ("ce_seed", "CE SEED-IV", "#937860"),
                ("mmd_loss", "MMD", "#da8bc3"),
            ]:
                if col in df_mean.columns:
                    ax.plot(
                        epochs_x,
                        df_mean[col],
                        color=clr,
                        lw=1.0,
                        linestyle=":",
                        alpha=0.75,
                        label=lbl,
                    )
            ax.set_xlabel("Epoch", color=TEXT_COL)
            ax.set_ylabel("Loss", color=TEXT_COL)
            ax.set_title(
                "Mean Loss (across subjects)",
                color=TEXT_COL,
                fontsize=11,
                fontweight="bold",
            )
            ax.legend(
                facecolor=PANEL_BG,
                edgecolor=GRID_COL,
                labelcolor=TEXT_COL,
                fontsize=8,
            )

        if has_acc:
            ax = axes[0][panel]
            ax.set_facecolor(PANEL_BG)
            ax.spines[:].set_color(GRID_COL)
            ax.tick_params(colors=TEXT_COL)
            ax.grid(color=GRID_COL, alpha=0.4, linewidth=0.6)
            ax.plot(
                epochs_x,
                df_mean["acc"],
                color="#4C72B0",
                lw=1.8,
                label="Train Acc",
            )
            ax.plot(
                epochs_x,
                df_mean["val_acc"],
                color="#55A868",
                lw=1.8,
                label="Val Acc",
            )
            ax.plot(
                epochs_x,
                df_mean["test_acc"],
                color="#DD8452",
                lw=1.8,
                label="Test Acc",
                linestyle="--",
            )
            ax.axhline(
                0.25,
                color="#ff4444",
                lw=1.0,
                linestyle="--",
                label="Chance 25%",
            )
            ax.set_xlabel("Epoch", color=TEXT_COL)
            ax.set_ylabel("Accuracy", color=TEXT_COL)
            ax.set_ylim(0, 1.05)
            ax.set_title(
                "Mean Accuracy (across subjects)",
                color=TEXT_COL,
                fontsize=11,
                fontweight="bold",
            )
            ax.legend(
                facecolor=PANEL_BG,
                edgecolor=GRID_COL,
                labelcolor=TEXT_COL,
                fontsize=8,
            )

        plt.tight_layout()
        curve_png = out_dir / "loss_acc_curves.png"
        plt.savefig(curve_png, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig2)
        print(f"Loss/Acc curves → {curve_png}")

    # 4. Per-subject loss curves
    subjects_done = df_hist["subject"].unique()
    n_s = len(subjects_done)
    ncols = min(4, n_s)
    nrows = (n_s + ncols - 1) // ncols

    fig3, axes3 = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    fig3.patch.set_facecolor(DARK_BG)

    for idx, subj in enumerate(sorted(subjects_done)):
        ax = axes3[idx // ncols][idx % ncols]
        ax.set_facecolor(PANEL_BG)
        ax.spines[:].set_color(GRID_COL)
        ax.tick_params(colors=TEXT_COL, labelsize=7)
        ax.grid(color=GRID_COL, alpha=0.4, linewidth=0.5)
        df_s = df_hist[df_hist["subject"] == subj]
        ep = df_s["epoch"].values
        if "loss" in df_s.columns:
            ax.plot(ep, df_s["loss"], color="#4C72B0", lw=1.5, label="Train")
        if "val_loss" in df_s.columns:
            ax.plot(ep, df_s["val_loss"], color="#55A868", lw=1.5, label="Val")
        if "test_loss" in df_s.columns:
            ax.plot(
                ep,
                df_s["test_loss"],
                color="#DD8452",
                lw=1.5,
                label="Test",
                linestyle="--",
            )
        ax.set_title(f"S{subj:02d}", color=TEXT_COL, fontsize=9, fontweight="bold")
        ax.set_xlabel("Epoch", color=TEXT_COL, fontsize=7)
        ax.set_ylabel("Loss", color=TEXT_COL, fontsize=7)

    for idx in range(n_s, nrows * ncols):
        axes3[idx // ncols][idx % ncols].set_visible(False)

    fig3.suptitle(
        "Per-Subject Loss Curves",
        color=TEXT_COL,
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    persubj_png = out_dir / "per_subject_loss_curves.png"
    plt.savefig(persubj_png, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig3)
    print(f"Per-subject loss curves → {persubj_png}")

    # 5. LOSO bar chart
    subjs = df_res["target_subject"].tolist()
    x = np.arange(len(subjs))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(subjs) * 0.7), 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.spines[:].set_color(GRID_COL)
    ax.tick_params(colors=TEXT_COL)
    ax.grid(axis="y", color=GRID_COL, alpha=0.5, linewidth=0.7)

    ax.bar(
        x - w / 2,
        df_res["best_val_trial_acc"],
        w,
        label="Best Val Trial",
        color="#4C72B0",
        alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        df_res["final_test_trial_acc"],
        w,
        label="Final Test Trial",
        color="#55A868",
        alpha=0.85,
    )
    mv = df_res["best_val_trial_acc"].mean()
    mt = df_res["final_test_trial_acc"].mean()
    ax.axhline(
        0.25,
        color="#ff4444",
        linestyle="--",
        linewidth=1,
        label="Chance 25%",
    )
    ax.axhline(
        mv,
        color="#4C72B0",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean Val {mv:.3f}",
    )
    ax.axhline(
        mt,
        color="#55A868",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean Test {mt:.3f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s:02d}" for s in subjs], color=TEXT_COL, fontsize=9)
    ax.set_ylabel("Accuracy", color=TEXT_COL)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "LTSM SNN LOSO — Val & Test Trial Accuracy",
        color=TEXT_COL,
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(
        facecolor=PANEL_BG,
        edgecolor=GRID_COL,
        labelcolor=TEXT_COL,
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(
        out_dir / "LTSM_SNN_loso_chart.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=DARK_BG,
    )
    plt.close()
    print(f"Chart → {out_dir}/LTSM_SNN_loso_chart.png")


def export_best_overall_artifacts(all_results, out_dir: Path):
    """
    Create:
      - best_overall.pt
      - demo_best_overall.npz
      - all_test_data.npz
    based on the subject with the highest best_val_trial_acc.
    """
    out_dir = Path(out_dir)
    best_subj_result = max(all_results, key=lambda r: r["best_val_trial_acc"])
    best_subj = best_subj_result["target_subject"]

    # 1) best_overall.pt
    src_pt = out_dir / f"best_s{best_subj:02d}.pt"
    dst_pt = out_dir / "best_overall.pt"
    if src_pt.exists():
        shutil.copy2(src_pt, dst_pt)
        print(
            f"  best_overall.pt ← S{best_subj:02d} "
            f"(val_trial={best_subj_result['best_val_trial_acc']:.3f})"
        )
    else:
        print(f"  [Warning] {src_pt} not found, skipping best_overall.pt")

    # 2) demo_best_overall.npz
    src_npz = out_dir / f"all_test_data_s{best_subj:02d}.npz"
    if src_npz.exists():
        d = np.load(src_npz, allow_pickle=True)
        eeg_feat = d["eeg_feat"]
        labels = d["labels"]
        preds = d["preds"]
        probs = d["probs"]

        correct_mask = labels == preds
        if correct_mask.any():
            confs = probs[np.arange(len(probs)), preds]
            confs_correct = np.where(correct_mask, confs, -1.0)
            best_idx = int(np.argmax(confs_correct))

            demo_path = out_dir / "demo_best_overall.npz"
            np.savez(
                demo_path,
                eeg_feat=eeg_feat[best_idx],
                label=np.array(labels[best_idx]),
                pred=np.array(preds[best_idx]),
                probs=probs[best_idx],
                conf=np.array(confs_correct[best_idx]),
                subject=np.array(best_subj),
            )
            print(
                f"  demo_best_overall.npz → "
                f"label={labels[best_idx]} "
                f"pred={preds[best_idx]} "
                f"conf={confs_correct[best_idx]:.3f}"
            )
        else:
            print(
                f"  [Warning] No correct predictions for S{best_subj:02d}, skipping demo."
            )
    else:
        print(
            f"  [Warning] {src_npz} not found, skipping demo_best_overall.npz"
        )

    # 3) all_test_data.npz alias
    src_all = out_dir / f"all_test_data_s{best_subj:02d}.npz"
    dst_all = out_dir / "all_test_data.npz"
    if src_all.exists():
        shutil.copy2(src_all, dst_all)
        print(f"  all_test_data.npz ← S{best_subj:02d}")
    else:
        print(f"  [Warning] {src_all} not found, skipping all_test_data.npz")


