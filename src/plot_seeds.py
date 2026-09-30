import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Palette (dataviz skill default - first 3 slots are CVD/lightness validated)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"      # OURS (mean)
ORANGE = "#eb6834"    # baseline: GCN
AQUA = "#1baf7a"      # baseline: GCN + Virtual Node
MUTED = "#898781"     # individual-seed traces / axis ticks
GRID = "#e1e0d9"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "axes.edgecolor": SECONDARY_INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "grid.color": GRID,
    "grid.linestyle": "--",
    "grid.alpha": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titleweight": "bold",
})

GCN_BASELINE = 0.1399
GCN_VN_BASELINE = 0.1461
GCN_BASELINE2 = 0.1507
GCN_VN_BASELINE2 = 0.1595

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*Val (?:F1|Acc):\s*([\d.]+)\s*\|\s*Best:\s*([\d.]+)"
)
TEST_RE = re.compile(r"Final Test (?:F1|Accuracy):\s*([\d.]+)")


def parse_log(text):
    rows = [
        (int(e), float(l), float(v), float(b))
        for e, l, v, b in EPOCH_RE.findall(text)
    ]
    epochs = np.array(rows, dtype=np.float64)
    m = TEST_RE.search(text)
    test_metric = float(m.group(1)) if m else None
    return epochs, test_metric


def strip_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_dot_comparison(tmp, ax, df, value_col, title, ylabel):
    """Per-seed scatter vs. the two baselines. Dots encode position only, so
    zooming the y-axis near the baselines (unlike a bar's length-from-zero)
    does not distort the comparison."""
    if df.empty:
        ax.set_title(title, fontsize=15)
        return

    x = np.arange(len(df))
    ax.scatter(x, df[value_col], color=BLUE, s=110, zorder=3,
               label=f'GCN+CD (n={len(df)} seeds)')

    mean_val = df[value_col].mean()
    ax.axhline(y=mean_val, color=INK, linestyle=':', linewidth=1.5,
               label=f'GCN+CD mean ({mean_val:.4f})')
    if(tmp==0):
        ax.axhline(y=GCN_BASELINE, color=ORANGE, linestyle='--', linewidth=2.2,
                label=f'GCN ({GCN_BASELINE:.4f})')
        ax.axhline(y=GCN_VN_BASELINE, color=AQUA, linestyle='-.', linewidth=2.2,
                label=f'GCN + Virtual Node ({GCN_VN_BASELINE:.4f})')
    else:
        ax.axhline(y=GCN_BASELINE2, color=ORANGE, linestyle='--', linewidth=2.2,
                label=f'GCN ({GCN_BASELINE2:.4f})')
        ax.axhline(y=GCN_VN_BASELINE2, color=AQUA, linestyle='-.', linewidth=2.2,
                label=f'GCN + Virtual Node ({GCN_VN_BASELINE2:.4f})')

    for xi, yval in zip(x, df[value_col]):
        above = yval >= mean_val
        ax.annotate(f'{yval:.4f}', (xi, yval), textcoords='offset points',
                    xytext=(0, 11 if above else -11), ha='center',
                    va='bottom' if above else 'top',
                    fontsize=14, color=INK, zorder=4,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    if(tmp==0):
        y_min = min(df[value_col].min(), GCN_BASELINE, GCN_VN_BASELINE)
        y_max = max(df[value_col].max(), GCN_BASELINE, GCN_VN_BASELINE)
    else:
        y_min = min(df[value_col].min(), GCN_BASELINE2, GCN_VN_BASELINE2)
        y_max = max(df[value_col].max(), GCN_BASELINE2, GCN_VN_BASELINE2)
    pad = (y_max - y_min) * 0.25
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'])

    ax.set_title(title, fontsize=17)
    ax.set_xlabel('Seed', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    strip_axes(ax)
    ax.grid(True, axis='y')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=14)


# ---------------------------------------------------------------------------
# 1. Read + parse logs (seed 1 through 10)
# ---------------------------------------------------------------------------
epoch_rows = []
test_rows = []

for i in range(1, 11):
    filename = f"test{i}.log"
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found, skipping.")
        continue

    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    epochs_data, test_metric = parse_log(text)
    label_name = f"seed{i}"

    for row in epochs_data:
        epoch_rows.append({
            'Model': label_name,
            'Epoch': int(row[0]),
            'Loss': row[1],
            'Val_F1': row[2],
        })

    if test_metric is not None:
        test_rows.append({'Model': label_name, 'Test_F1': test_metric})

df_epochs = pd.DataFrame(epoch_rows)
df_test = pd.DataFrame(test_rows)

# ==========================================
# [1] Loss / Val F1 over epochs, side by side
#     * Val F1: a broken axis (0-0.02 / 0.135-max) would drop early-epoch
#       values (~0.09-0.13) into the empty gap between the two panels, so a
#       single continuous, zoomed axis is used instead - no data is lost, and
#       F1's range here is narrow enough that showing 0 buys nothing anyway.
#     * Val F1: the region where OURS's mean line has overtaken both
#       baselines is shaded, with an arrow annotation calling out the epoch.
# ==========================================
fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(15, 6))

# ---- Loss (left) ----
if not df_epochs.empty:
    for model_name in df_epochs['Model'].unique():
        subset = df_epochs[df_epochs['Model'] == model_name]
        ax_loss.plot(subset['Epoch'], subset['Loss'], color=MUTED, alpha=0.35, linewidth=1.1)

    mean_loss = df_epochs.groupby('Epoch')['Loss'].mean().reset_index()
    ax_loss.plot(mean_loss['Epoch'], mean_loss['Loss'], color=BLUE, linewidth=2.5,
                 label='GCN+CD (mean, n=10 seeds)')
    ax_loss.plot([], [], color=MUTED, alpha=0.6, linewidth=1.5, label='GCN+CD (individual seeds)')

ax_loss.set_title('Training Loss over Epochs')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
strip_axes(ax_loss)
ax_loss.grid(True, axis='y')
ax_loss.legend(frameon=False, loc='upper right')

# ---- Val F1 (right) ----
if not df_epochs.empty:
    for model_name in df_epochs['Model'].unique():
        subset = df_epochs[df_epochs['Model'] == model_name]
        ax_f1.plot(subset['Epoch'], subset['Val_F1'], color=MUTED, alpha=0.25, linewidth=1.0)

    stats = df_epochs.groupby('Epoch')['Val_F1'].agg(['mean', 'std']).reset_index()
    stats['std'] = stats['std'].fillna(0.0)
    ax_f1.fill_between(stats['Epoch'], stats['mean'] - stats['std'], stats['mean'] + stats['std'],
                        color=BLUE, alpha=0.15, linewidth=0, label='GCN+CD (mean ± std)')
    ax_f1.plot(stats['Epoch'], stats['mean'], color=BLUE, linewidth=2.5, label='GCN+CD (mean)')
    ax_f1.plot([], [], color=MUTED, alpha=0.5, linewidth=1.5, label='GCN+CD (individual seeds)')

    ax_f1.axhline(y=GCN_BASELINE, color=ORANGE, linestyle='--', linewidth=1.8,
                  label=f'GCN ({GCN_BASELINE:.4f})')
    ax_f1.axhline(y=GCN_VN_BASELINE, color=AQUA, linestyle='-.', linewidth=1.8,
                  label=f'GCN + Virtual Node ({GCN_VN_BASELINE:.4f})')

    val_min = min(df_epochs['Val_F1'].min(), GCN_BASELINE, GCN_VN_BASELINE)
    val_max = max(df_epochs['Val_F1'].max(), GCN_BASELINE, GCN_VN_BASELINE)
    pad = (val_max - val_min) * 0.1
    ax_f1.set_ylim(val_min - pad, val_max + pad)

    # First epoch where OURS's mean line has overtaken both baselines
    higher_baseline = max(GCN_BASELINE, GCN_VN_BASELINE)
    surpassed = stats[stats['mean'] >= higher_baseline]
    if not surpassed.empty:
        cross_epoch = surpassed['Epoch'].iloc[0]
        cross_val = surpassed['mean'].iloc[0]
        x_end = stats['Epoch'].max()

        ax_f1.axvspan(cross_epoch, x_end, color=BLUE, alpha=0.06, zorder=0)
        ax_f1.axvline(cross_epoch, color=BLUE, linestyle=':', linewidth=1.3, alpha=0.8)

        y_top = val_max + pad
        ax_f1.annotate(
            f'Surpasses both baselines\nfrom epoch {int(cross_epoch)}',
            xy=(cross_epoch, cross_val),
            xytext=(cross_epoch, y_top - (y_top - val_max) * 0.3),
            textcoords='data',
            ha='left' if cross_epoch < x_end * 0.7 else 'right',
            va='top',
            fontsize=9,
            fontweight='bold',
            color=BLUE,
            arrowprops=dict(arrowstyle='->', color=BLUE, linewidth=1.4, shrinkA=0, shrinkB=4),
        )

ax_f1.set_title('Validation F1 over Epochs')
ax_f1.set_xlabel('Epoch')
ax_f1.set_ylabel('Val F1')
strip_axes(ax_f1)
ax_f1.grid(True, axis='y')
handles, labels = ax_f1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax_f1.legend(by_label.values(), by_label.keys(), frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))

fig.tight_layout()
plt.show()

# ==========================================
# [2] Best Val F1 / Final Test F1 dot plots, side by side
#     * A bar's length must start at 0, which buries the real ~0.01-0.02 gap
#       between OURS and the baselines. A dot only encodes position, so the
#       y-axis can be zoomed to the baselines' neighborhood without distorting
#       the comparison.
# ==========================================
df_val_best = (
    df_epochs.groupby('Model', sort=False)['Val_F1'].max().reset_index()
    if not df_epochs.empty else pd.DataFrame(columns=['Model', 'Val_F1'])
)

fig, (ax_val_best, ax_test) = plt.subplots(1, 2, figsize=(15, 6))
plot_dot_comparison(0,ax_val_best, df_val_best, 'Val_F1', 'Best Val F1 per Seed', 'Val F1')
plot_dot_comparison(1,ax_test, df_test, 'Test_F1', 'Final Test F1 per Seed', 'Test F1')
fig.tight_layout()
plt.show()
