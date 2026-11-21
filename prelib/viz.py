"""Visual layer shared by every notebook: one figure style, diagrams, heat maps, token strips, animations (GIFs
that also play on GitHub) and slider-driven explorers that fall back to a static figure when no kernel is live."""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import FIG_DIR

# Okabe-Ito colours (distinguishable under colour-vision deficiency). One colour per model family, reused everywhere.
BLUE, ORANGE, GREEN, PINK, SKY, VERMILION, YELLOW, GREY, BLACK = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#8C8C8C", "#222222")
CYCLE = [BLUE, ORANGE, GREEN, PINK, SKY, VERMILION, GREY]
FAMILY = {"floor": BLACK, "n-gram": GREY, "window MLP": ORANGE, "RNN": SKY, "LSTM": PINK, "GRU": GREEN,
          "transformer": BLUE, "VAE": ORANGE, "GAN": PINK, "autoregressive": SKY, "masked": GREEN,
          "diffusion": BLUE, "flow": VERMILION, "real": BLACK}


def style():
    """Apply the series' figure style (call once at the top of a notebook)."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (6.4, 3.6), "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.prop_cycle": mpl.cycler(color=CYCLE), "lines.linewidth": 1.8,
        "image.cmap": "viridis", "figure.constrained_layout.use": True, "axes.xmargin": 0.03, "axes.ymargin": 0.05})


def live_kernel():
    try:
        ip = get_ipython()                      # noqa: F821  (defined by IPython)
    except NameError:
        return False
    return getattr(ip, "kernel", None) is not None


def explore(fn, **controls):
    """Interactive explorer. controls: name=(min, max, step, default) -> slider; name=[options] -> dropdown.

    In a running notebook (Jupyter, VS Code, Colab) this shows widgets and redraws when they move. Where no kernel
    is attached (GitHub's viewer, or a notebook executed in batch) it draws once at the default settings."""
    defaults = {k: (v[3] if isinstance(v, tuple) else v[0]) for k, v in controls.items()}
    if not live_kernel():
        print("Static preview at the default settings " + str(defaults) + ". Run the notebook to get the controls.")
        return fn(**defaults)
    import ipywidgets as W
    ws = {}
    for k, v in controls.items():
        if isinstance(v, tuple):
            cls = W.IntSlider if all(isinstance(x, (int, np.integer)) for x in v) else W.FloatSlider
            ws[k] = cls(min=v[0], max=v[1], step=v[2], value=v[3], description=k, continuous_update=False)
        else:
            ws[k] = W.Dropdown(options=list(v), value=v[0], description=k)
    return W.interact(fn, **ws)


def save_gif(fig, update, frames, name, fps=5, dpi=80):
    """Render update(frame) for each frame into figures/<name> and close the figure. Returns the relative path."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    frames = list(frames)
    FuncAnimation(fig, update, frames=frames, blit=False).save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"saved figures/{name}  ({len(frames)} frames, {os.path.getsize(path) / 1024:.0f} kB)")
    return "figures/" + name


def heat(ax, M, xlabels=None, ylabels=None, annotate=False, fmt="{:.2f}", cmap="viridis", vmin=None, vmax=None, cbar=None):
    """Heat map with optional per-cell values (use for matrices small enough to read)."""
    M = np.asarray(M)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, rotation=90 if max(map(len, map(str, xlabels))) > 3 else 0)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    if annotate:
        lo, hi = im.get_clim()
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                frac = (M[i, j] - lo) / (hi - lo + 1e-12)
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if (frac < 0.5) == (mpl.colormaps[im.get_cmap().name](0.0)[0] < 0.5) else "black")
    for sp in ax.spines.values():
        sp.set_visible(False)
    if cbar:
        ax.figure.colorbar(im, ax=ax, label=cbar, shrink=0.8)
    return im


def token_strip(ax, words, values=None, colors=None, cmap="Blues", vmax=None, wrap=16, fontsize=8, edge=None):
    """Draw a token sequence as a row (or wrapped rows) of boxes, shaded by `values` or filled with `colors`.
    `edge` is an optional list of colours (or None) used to outline chosen tokens."""
    n = len(words)
    rows = int(np.ceil(n / wrap))
    cm = mpl.colormaps[cmap]
    if values is not None:
        values = np.asarray(values, dtype=float)
        vmax = vmax or max(values.max(), 1e-9)
    for i, w in enumerate(words):
        r, c = divmod(i, wrap)
        if colors is not None:
            fc = colors[i]
        elif values is not None:
            fc = cm(0.05 + 0.85 * min(values[i] / vmax, 1.0))
        else:
            fc = "#F2F2F2"
        ec = edge[i] if edge is not None and edge[i] is not None else "white"
        ax.add_patch(plt.Rectangle((c, rows - 1 - r), 0.96, 0.9, fc=fc, ec=ec, lw=1.6 if ec != "white" else 0.5))
        lum = mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(fc))[2] if not isinstance(fc, tuple) else 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
        ax.text(c + 0.48, rows - 1 - r + 0.45, w, ha="center", va="center", fontsize=fontsize, color="white" if lum < 0.5 else "black")
    ax.set_xlim(0, wrap); ax.set_ylim(0, rows); ax.axis("off")


def diagram(ax, boxes, arrows=(), fontsize=9):
    """Schematic from boxes [(x, y, w, h, text, colour)] and arrows [(x0, y0, x1, y1) or (x0, y0, x1, y1, label)]."""
    for x, y, w, h, text, col in boxes:
        ax.add_patch(mpl.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                                fc=mpl.colors.to_rgba(col, 0.18), ec=col, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    for a in arrows:
        ax.annotate("", (a[2], a[3]), (a[0], a[1]), arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.0, shrinkA=0, shrinkB=0))
        if len(a) > 4:
            ax.text((a[0] + a[2]) / 2, (a[1] + a[3]) / 2 + 0.08, a[4], ha="center", fontsize=fontsize - 1, color=BLACK)
    ax.axis("off"); ax.set_aspect("equal"); ax.autoscale_view()


def pca2(X):
    """Project rows of X to their first two principal components."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def embedding_axes(ax, labels=("PC 1", "PC 2")):
    """Embedding plots carry no meaningful ticks: drop them and name the axes with a small corner cue."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.annotate(labels[0] + " \u2192", (0.02, 0.02), xycoords="axes fraction", fontsize=7, color=GREY)
    ax.annotate(labels[1] + " \u2192", (0.02, 0.07), xycoords="axes fraction", fontsize=7, color=GREY, rotation=90, va="bottom")
