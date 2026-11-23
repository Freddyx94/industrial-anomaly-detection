"""One chart style for every notebook.

Why: the same colours and quiet axes in every figure make plots from different stages easy to
compare, and a change is made in one place. The colours are the ones used in Stage 0.
"""

import matplotlib.pyplot as plt

# Series colours in a fixed order: the first series is always blue, the second orange, the third
# green-blue. The first three stay distinguishable for colour-blind readers; for more series, use
# separate small charts instead of more colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

INK = "#0b0b0b"  # titles and important text
INK_SECONDARY = "#52514e"  # axis labels, annotations
INK_MUTED = "#898781"  # tick marks, reference lines, background points
GRID = "#e1e0d9"
EDGE = "#c3c2b7"
SURFACE = "#fcfcfb"
MASK = "#e34948"  # outline of a ground-truth defect mask


def apply_style() -> None:
    """Quiet axes and grid, readable font sizes; call once at the top of a notebook."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
        "axes.edgecolor": EDGE, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
        "font.size": 10, "axes.titlesize": 11,
    })
