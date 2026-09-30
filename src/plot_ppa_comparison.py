"""OGBG-ppa: Best Val Accuracy / Final Test Accuracy dot comparison.

Same dot-plot format as plot_seeds.py's section [2] (Best Val F1 / Final Test
F1 per seed), adapted for ogbg-ppa:
  - Val/Test values are entered directly below (PROPOSAL*_VAL / PROPOSAL*_TEST)
    instead of being parsed from log files.
  - Two proposed models (제안 1 / 제안 2) get their own color each.
  - Leaderboard baselines (rank 11-17, ogbg-ppa) are drawn as thin shared-color
    reference lines with a value label at the right edge, rather than one
    legend entry + hline color each - with 7 of them, distinct hues would
    outrun what color can carry, so identity comes from the text label, not
    from hue (dataviz palette rule: extra series fold into a shared class).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------------------------------------------------------------------
# Korean-safe font: Segoe UI/Arial/DejaVu Sans have no Hangul glyphs, which is
# what was rendering 제안 1/제안 2 as tofu boxes. Pick the first Korean-capable
# font actually installed, and put it ahead of the Latin fallbacks.
# ---------------------------------------------------------------------------
_KOREAN_FONT_CANDIDATES = ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic"]
_installed = {f.name for f in fm.fontManager.ttflist}
_korean_font = next((name for name in _KOREAN_FONT_CANDIDATES if name in _installed), None)
if _korean_font is None:
    print("Warning: no Korean-capable font found (tried: "
          f"{', '.join(_KOREAN_FONT_CANDIDATES)}). Korean labels will render as "
          "boxes - install e.g. NanumGothic and clear the matplotlib font cache "
          "(delete ~/.cache/matplotlib or %USERPROFILE%\\.matplotlib and rerun).")

# ---------------------------------------------------------------------------
# Palette (dataviz skill default)
# ---------------------------------------------------------------------------
PROPOSAL1_COLOR = "#2a78d6"   # slot 1 (blue)  - 제안 1
PROPOSAL2_COLOR = "#eb6834"   # slot 2 (orange) - 제안 2
MUTED = "#898781"             # literature baselines / axis ticks
GRID = "#e1e0d9"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ([_korean_font] if _korean_font else []) + ["Segoe UI", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,  # Korean fonts commonly lack the unicode minus glyph
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

# ---------------------------------------------------------------------------
# Leaderboard baselines - ogbg-ppa, rank 11-17 (Test Accuracy, Validation Accuracy)
# https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-ppa
# ---------------------------------------------------------------------------
BASELINES = [
    {"name": "GIN+virtual node+FLAG", "test": 0.7245, "test_std": 0.0114, "val": 0.6789, "val_std": 0.0079},
    {"name": "GIN+virtual node",      "test": 0.7037, "test_std": 0.0107, "val": 0.6678, "val_std": 0.0105},
    {"name": "GCN+virtual node+FLAG", "test": 0.6944, "test_std": 0.0052, "val": 0.6638, "val_std": 0.0055},
    {"name": "GIN+FLAG",              "test": 0.6905, "test_std": 0.0092, "val": 0.6465, "val_std": 0.0070},
    {"name": "GIN",                   "test": 0.6892, "test_std": 0.0100, "val": 0.6562, "val_std": 0.0107},
    {"name": "GCN+virtual node",      "test": 0.6857, "test_std": 0.0061, "val": 0.6511, "val_std": 0.0048},
    # NOTE: test_std for GCN was cut off in the pasted leaderboard row - verify
    # against https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-ppa and fill in.
    {"name": "GCN",                   "test": 0.6839, "test_std": None,   "val": 0.6497, "val_std": 0.0034},
]

# ---------------------------------------------------------------------------
# Our results - fill in per-seed Best Val Acc / Final Test Acc for each proposal.
# The two lists don't need the same length; each becomes its own seed group on
# the x-axis.
# ---------------------------------------------------------------------------
PROPOSAL1_NAME = "제안 1"
PROPOSAL1_VAL = []   # TODO: fill in, e.g. [0.671, 0.665, 0.668, ...]
PROPOSAL1_TEST = []  # TODO: fill in

PROPOSAL2_NAME = "제안 2"
PROPOSAL2_VAL = []   # TODO: fill in
PROPOSAL2_TEST = []  # TODO: fill in


def strip_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_dot_comparison(ax, series, baselines, value_key, title, ylabel):
    """series: list of (name, color, values) for our proposals, plotted as
    per-seed dots + a mean hline. baselines: BASELINES-style dicts, drawn as
    thin muted reference lines with a value label at the right edge."""
    series = [(name, color, np.asarray(values, dtype=float))
              for name, color, values in series if len(values) > 0]

    if not series:
        ax.set_title(title, fontsize=11)
        ax.text(0.5, 0.5, "데이터 없음 (PROPOSAL*_VAL / PROPOSAL*_TEST를 채워주세요)",
                ha='center', va='center', fontsize=9, color=MUTED, transform=ax.transAxes)
        strip_axes(ax)
        return

    xticks, xtick_labels = [], []
    x_cursor = 0
    all_values = []

    for name, color, values in series:
        n = len(values)
        x = np.arange(x_cursor, x_cursor + n)
        ax.scatter(x, values, color=color, s=70, zorder=3, label=f'{name} (n={n} seeds)')

        mean_val = values.mean()
        ax.axhline(y=mean_val, color=color, linestyle=':', linewidth=1.5, alpha=0.9,
                   label=f'{name} mean ({mean_val:.4f})')

        for xi, yval in zip(x, values):
            ax.annotate(f'{yval:.4f}', (xi, yval), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8, color=SECONDARY_INK)

        xticks.extend(x)
        xtick_labels.extend([f'seed{i + 1}' for i in range(n)])
        all_values.append(values)
        x_cursor += n + 1  # gap between proposal groups

    x_max = x_cursor - 1
    all_values = np.concatenate(all_values)
    baseline_vals = [b[value_key] for b in baselines]

    y_min = min(all_values.min(), *baseline_vals)
    y_max = max(all_values.max(), *baseline_vals)
    pad = (y_max - y_min) * 0.25 if y_max > y_min else 0.01

    # Baseline values often sit close together (e.g. GCN/GCN+VN/GIN within a
    # few thousandths); a plain per-value label would overlap. Declutter the
    # label y-positions with a minimum gap and connect each to its true value
    # with a short leader line when the two diverge.
    sorted_baselines = sorted(baselines, key=lambda b: b[value_key])
    min_gap = (y_max - y_min + 2 * pad) * 0.08
    label_ys = []
    for b in sorted_baselines:
        y = b[value_key]
        if label_ys and y - label_ys[-1] < min_gap:
            y = label_ys[-1] + min_gap
        label_ys.append(y)

    for b, label_y in zip(sorted_baselines, label_ys):
        true_y = b[value_key]
        std = b.get(f"{value_key}_std")
        label = f"{b['name']} ({true_y:.4f}" + (f"±{std:.4f})" if std is not None else ")")
        ax.axhline(y=true_y, color=MUTED, linestyle='--', linewidth=1.1, alpha=0.6, zorder=1)
        if abs(label_y - true_y) > 1e-9:
            ax.plot([x_max + 0.15, x_max + 0.4], [true_y, label_y],
                    color=MUTED, linewidth=0.6, alpha=0.5, zorder=1)
        ax.annotate(label, xy=(x_max + 0.4, label_y), xytext=(4, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=7.5, color=MUTED)

    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(-0.6, x_max + 4.6)  # room on the right for baseline labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Seed')
    ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    strip_axes(ax)
    ax.grid(True, axis='y')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8)


# ==========================================
# Best Val Acc / Final Test Acc dot plots, side by side
#   * A bar's length must start at 0, which buries the real gap between
#     OURS and the baselines. A dot only encodes position, so the y-axis can
#     be zoomed to the baselines' neighborhood without distorting the
#     comparison.
# ==========================================
val_series = [
    (PROPOSAL1_NAME, PROPOSAL1_COLOR, PROPOSAL1_VAL),
    (PROPOSAL2_NAME, PROPOSAL2_COLOR, PROPOSAL2_VAL),
]
test_series = [
    (PROPOSAL1_NAME, PROPOSAL1_COLOR, PROPOSAL1_TEST),
    (PROPOSAL2_NAME, PROPOSAL2_COLOR, PROPOSAL2_TEST),
]

fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(17, 6))
plot_dot_comparison(ax_val, val_series, BASELINES, 'val', 'Best Val Accuracy per Seed', 'Val Accuracy')
plot_dot_comparison(ax_test, test_series, BASELINES, 'test', 'Final Test Accuracy per Seed', 'Test Accuracy')
fig.suptitle('OGBG-ppa: 제안 모델 vs. 리더보드 baseline', fontsize=13, y=1.02)
fig.tight_layout()
plt.savefig('ogbg_ppa_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
