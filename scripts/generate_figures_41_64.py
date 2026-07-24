"""Generate the 72 deterministic SVG figures for lessons 41–64.

The drawings follow the article notation rather than acting as decorative
cards.  Palatino, paper, ink and the five semantic accents match the textbook.
Run from any directory:

    python3 site/scripts/generate_figures_41_64.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from scipy.special import beta as beta_fn, gamma
from scipy.stats import beta as beta_dist
from scipy.stats import binom, cauchy, expon, norm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons"
RNG = np.random.default_rng(4164)

PAPER = "#fffef9"
INK = "#171915"
MUTED = "#696d65"
FAINT = "#969990"
GRID = "#deddd4"
BLUE = "#315f8c"
RED = "#b94a3b"
GREEN = "#38735d"
GOLD = "#a57920"
VIOLET = "#6f5a8f"
WASH = "#f5f3ea"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Palatino", "Times New Roman", "DejaVu Serif"],
        "font.size": 12.2,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": FAINT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "svg.fonttype": "path",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


def canvas(width: float = 10.4, height: float = 5.8):
    return plt.figure(figsize=(width, height), layout="constrained")


def clean(ax, *, xgrid: bool = False, ygrid: bool = False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(FAINT)
    ax.tick_params(length=3, width=0.7)
    if xgrid:
        ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, linewidth=0.7)


def diagram_axes(fig):
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def label(ax, x, y, text, color=INK, size=11.2, ha="left", va="center", weight="normal", **kwargs):
    return ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va, weight=weight, **kwargs)


def direct(ax, x, y, text, color, *, ha="left", va="center"):
    return ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=10.4,
        ha=ha,
        va=va,
        bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1.2, "alpha": 0.92},
    )


def box(ax, xy, width, height, text, *, color=BLUE, fill=PAPER, size=11.2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.016,rounding_size=0.018",
        linewidth=1.35,
        edgecolor=color,
        facecolor=fill,
    )
    ax.add_patch(patch)
    label(ax, xy[0] + width / 2, xy[1] + height / 2, text, color, size, "center")
    return patch


def arrow(ax, start, end, *, color=MUTED, curve=0.0, width=1.25, style="-|>", shrink=8):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=width,
        color=color,
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=shrink,
        shrinkB=shrink,
    )
    ax.add_patch(patch)
    return patch


def gaussian_ellipse(ax, mean, cov, *, color, levels=(1.0, 1.8, 2.6), alpha=1.0):
    vals, vecs = np.linalg.eigh(np.asarray(cov))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    for i, level in enumerate(levels):
        ax.add_patch(
            Ellipse(
                mean,
                2 * level * math.sqrt(vals[0]),
                2 * level * math.sqrt(vals[1]),
                angle=angle,
                fill=False,
                edgecolor=color,
                linewidth=1.8 if i == 0 else 1.0,
                alpha=alpha * (0.9 - i * 0.2),
            )
        )


def lesson41(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(8.2, 6.8), layout="constrained")
        sums = np.add.outer(np.arange(1, 7), np.arange(1, 7))
        ax.imshow(sums, origin="lower", cmap=mpl.colors.LinearSegmentedColormap.from_list("sum", [WASH, BLUE]), alpha=0.85)
        for i in range(6):
            for j in range(6):
                label(ax, j, i, str(sums[i, j]), INK, 11, "center")
                if sums[i, j] >= 10:
                    ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor=RED, linewidth=2))
        ax.set_xticks(range(6), range(1, 7))
        ax.set_yticks(range(6), range(1, 7))
        ax.set_xlabel("первый кубик $i$")
        ax.set_ylabel("второй кубик $j$")
        label(ax, 5.55, 0.2, "$i+j\\geq10$", RED, 11, "right")
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        xs = np.arange(0, 21)
        distributions = [
            (np.array([9, 10, 11]), np.array([0.2, 0.6, 0.2]), BLUE, "$\\sigma=0{,}63$"),
            (np.array([6, 8, 10, 12, 14]), np.array([0.1, 0.2, 0.4, 0.2, 0.1]), GREEN, "$\\sigma=2{,}19$"),
            (np.array([0, 10, 20]), np.array([0.18, 0.64, 0.18]), RED, "$\\sigma=6$"),
        ]
        for values, probs, color, text in distributions:
            ax.vlines(values, 0, probs, color=color, linewidth=4, alpha=0.88)
            ax.scatter(values, probs, color=color, s=30)
            direct(ax, values[-1] + 0.15, probs[-1], text, color)
        ax.axvline(10, color=INK, linestyle=(0, (3, 3)), linewidth=1)
        label(ax, 10, 0.68, "общее $\\mathbb{E}X=10$", INK, 10.5, "center")
        ax.set_xlabel("значение $X$")
        ax.set_ylabel("вероятность")
        ax.set_xlim(-1, 22)
        ax.set_ylim(0, 0.75)
        clean(ax, ygrid=True)
        return fig
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.8), layout="constrained", sharex=True, sharey=True)
    exact = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36
    for ax, n in zip(axes, [20, 200, 20000]):
        rolls = RNG.integers(1, 7, size=(n, 2)).sum(axis=1)
        counts = np.bincount(rolls, minlength=13)[2:13] / n
        ax.bar(np.arange(2, 13), counts, color=BLUE, alpha=0.65, width=0.72)
        ax.scatter(np.arange(2, 13), exact, color=RED, s=18, zorder=3)
        ax.set_title(f"$n={n:,}$".replace(",", " "), fontsize=12.5, fontstyle="italic")
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        clean(ax, ygrid=True)
    axes[0].set_ylabel("частота")
    axes[1].set_xlabel("сумма двух кубиков")
    return fig


def lesson42(index: int):
    if index == 1:
        fig = canvas()
        axes = fig.subplots(1, 2)
        a0, a1 = axes
        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
        a0.add_patch(Rectangle((0.05, 0.08), 0.9, 0.82, facecolor=WASH, edgecolor=MUTED, linewidth=1.2))
        a0.add_patch(Circle((0.42, 0.52), 0.27, facecolor=BLUE, alpha=0.18, edgecolor=BLUE, linewidth=1.8))
        a0.add_patch(Circle((0.62, 0.46), 0.28, facecolor=GOLD, alpha=0.2, edgecolor=GOLD, linewidth=1.8))
        label(a0, 0.25, 0.68, "$A$", BLUE, 14, "center")
        label(a0, 0.77, 0.33, "$B$", GOLD, 14, "center")
        label(a0, 0.5, 0.95, "$\\Omega$", MUTED, 12, "center")
        a1.add_patch(Rectangle((0.08, 0.12), 0.84, 0.76, facecolor=GOLD, alpha=0.14, edgecolor=GOLD, linewidth=1.8))
        a1.add_patch(Rectangle((0.08, 0.12), 0.28, 0.76, facecolor=BLUE, alpha=0.4, edgecolor=BLUE, linewidth=1.5))
        label(a1, 0.5, 0.95, "$B$ становится новым миром", GOLD, 11.5, "center")
        label(a1, 0.22, 0.5, "$A\\cap B$", BLUE, 12, "center")
        label(a1, 0.5, 0.05, "$P(A\\mid B)=P(A\\cap B)/P(B)$", INK, 11, "center")
        return fig
    if index == 2:
        fig = canvas()
        ax = diagram_axes(fig)
        box(ax, (0.04, 0.43), 0.13, 0.14, "10 000", color=INK)
        box(ax, (0.30, 0.66), 0.16, 0.14, "болезнь\n100", color=RED)
        box(ax, (0.30, 0.22), 0.16, 0.14, "здоровы\n9900", color=GREEN)
        box(ax, (0.67, 0.73), 0.18, 0.12, "+ : 90", color=RED)
        box(ax, (0.67, 0.55), 0.18, 0.12, "− : 10", color=MUTED)
        box(ax, (0.67, 0.29), 0.18, 0.12, "+ : 495", color=GOLD)
        box(ax, (0.67, 0.11), 0.18, 0.12, "− : 9405", color=GREEN)
        arrow(ax, (0.17, 0.50), (0.30, 0.73), color=RED)
        arrow(ax, (0.17, 0.50), (0.30, 0.29), color=GREEN)
        for start, end, color in [
            ((0.46, 0.73), (0.67, 0.79), RED), ((0.46, 0.73), (0.67, 0.61), MUTED),
            ((0.46, 0.29), (0.67, 0.35), GOLD), ((0.46, 0.29), (0.67, 0.17), GREEN),
        ]:
            arrow(ax, start, end, color=color)
        label(ax, 0.91, 0.51, "$P(D\\mid+)=\\frac{90}{90+495}$\n$=15{,}4\\%$", INK, 12, "center")
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    p = np.linspace(0.0001, 0.2, 500)
    sens = 0.9
    for spec, color in [(0.90, GOLD), (0.95, BLUE), (0.99, GREEN)]:
        ppv = sens * p / (sens * p + (1 - spec) * (1 - p))
        ax.plot(p * 100, ppv * 100, color=color, linewidth=2.2)
        direct(ax, 17.2, ppv[np.searchsorted(p, 0.172)] * 100, f"spec {int(spec*100)}%", color)
    ax.plot(p * 100, p * 100, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    direct(ax, 14.2, 14.2, "тест ничего не изменил", MUTED, ha="center")
    ax.set_xlabel("распространённость болезни, %")
    ax.set_ylabel("$P(D\\mid+)$, %")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 100)
    clean(ax, xgrid=True, ygrid=True)
    return fig


def lesson43(index: int):
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        for row in range(2):
            for col in range(5):
                x, y = 0.06 + col * 0.18, 0.58 - row * 0.27
                box(ax, (x, y), 0.13, 0.12, "100 000", color=FAINT, size=9.8)
                ax.scatter([x + 0.113], [y + 0.098], s=36, facecolor=PAPER, edgecolor=BLUE, linewidth=1.7, zorder=4)
        ax.scatter([0.06 + 2 * 0.18 + 0.095], [0.58 - 1 * 0.27 + 0.036], s=70, facecolor=PAPER, edgecolor=RED, linewidth=2.4, zorder=5)
        label(ax, 0.5, 0.88, "10 ожидаемых случайных совпадений + 1 истинное", INK, 12.5, "center", weight="bold")
        label(ax, 0.5, 0.10, "синее — случайное совпадение; красное — профиль виновного", MUTED, 10.8, "center")
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 4.8), layout="constrained")
        ax.set_xscale("log")
        priors = np.array([1e-6, 1e-4, 1e-2])
        lr = 1e4
        for y, p0, color in zip([0.72, 0.5, 0.28], priors, [BLUE, GOLD, RED]):
            ax.scatter([p0], [y], color=color, s=42)
            ax.scatter([p0 * lr], [y], facecolor=PAPER, edgecolor=color, linewidth=2, s=58)
            ax.annotate("", xy=(p0 * lr, y), xytext=(p0, y), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.7})
            direct(ax, p0, y + 0.08, f"prior {p0:g}:1", color, ha="center")
        ax.set_xlim(3e-7, 300)
        ax.set_ylim(0.1, 0.9)
        ax.set_yticks([])
        ax.set_xlabel("шансы виновности, логарифмическая шкала")
        label(ax, 0.5, 0.95, "одна улика: умножение шансов на $LR=10^4$", INK, 12, "center", transform=ax.transAxes)
        clean(ax, xgrid=True)
        return fig
    fig = canvas()
    ax = diagram_axes(fig)
    box(ax, (0.09, 0.58), 0.18, 0.14, "гипотеза $H$", color=BLUE)
    box(ax, (0.41, 0.72), 0.18, 0.14, "общая причина $C$", color=GOLD)
    box(ax, (0.72, 0.64), 0.16, 0.12, "улика $E_1$", color=RED)
    box(ax, (0.72, 0.36), 0.16, 0.12, "улика $E_2$", color=VIOLET)
    arrow(ax, (0.27, 0.65), (0.72, 0.70), color=BLUE)
    arrow(ax, (0.27, 0.65), (0.72, 0.42), color=BLUE)
    arrow(ax, (0.59, 0.79), (0.72, 0.70), color=GOLD)
    arrow(ax, (0.59, 0.79), (0.72, 0.42), color=GOLD, curve=0.18)
    label(ax, 0.5, 0.20, "$C$ связывает улики: $P(E_1,E_2\\mid H)$ не раскладывается автоматически", MUTED, 11.5, "center")
    return fig


def lesson44(index: int):
    if index == 1:
        fig, axes = plt.subplots(1, 4, figsize=(11.2, 4.4), layout="constrained", sharex=True)
        xs = np.linspace(0, 3.5, 400)
        for ax, n in zip(axes, [1, 2, 10, 50]):
            shape, scale = n, 1 / n
            density = np.where(xs > 0, xs ** (shape - 1) * np.exp(-xs / scale) / (gamma(shape) * scale**shape), 0)
            ax.plot(xs, density, color=BLUE, linewidth=2)
            ax.axvline(1, color=RED, linestyle=(0, (3, 3)), linewidth=1)
            ax.set_title(f"$n={n}$", fontsize=12, fontstyle="italic")
            ax.set_xlim(0, 3.2)
            ax.set_yticks([])
            clean(ax)
        axes[0].set_ylabel("плотность среднего")
        axes[1].set_xlabel("значение $\\overline{X}_n$")
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        n, p = 100, 0.3
        k = np.arange(10, 52)
        pmf = binom.pmf(k, n, p)
        ax.bar(k, pmf, width=0.85, color=BLUE, alpha=0.55)
        x = np.linspace(9.5, 51.5, 500)
        ax.plot(x, norm.pdf(x, n * p, math.sqrt(n * p * (1 - p))), color=RED, linewidth=2.2)
        mask = k <= 24
        ax.bar(k[mask], pmf[mask], width=0.85, color=GOLD, alpha=0.8)
        ax.axvline(24.5, color=INK, linestyle=(0, (3, 3)), linewidth=1.2)
        direct(ax, 24.5, pmf.max() * 0.88, "$24{,}5$ — поправка", INK, ha="center")
        ax.set_xlabel("число успехов $S_{100}$")
        ax.set_ylabel("вероятность / плотность")
        clean(ax, ygrid=True)
        return fig
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.8), layout="constrained", sharex=True)
    reps, n = 8000, 100
    independent = RNG.exponential(1, (reps, n)).mean(axis=1)
    ar = np.zeros((reps, n))
    eps = RNG.normal(0, math.sqrt(1 - 0.95**2), (reps, n))
    for t in range(1, n):
        ar[:, t] = 0.95 * ar[:, t - 1] + eps[:, t]
    dependent = ar.mean(axis=1) + 1
    heavy = RNG.standard_cauchy((reps, n)).mean(axis=1) + 1
    heavy_window = heavy[(heavy > -3) & (heavy < 5)]
    for ax, data, title, color in [
        (axes[0], independent, "независимые", BLUE),
        (axes[1], dependent, "зависимый ряд", GOLD),
        (axes[2], heavy_window, "Коши", RED),
    ]:
        ax.hist(data, bins=60, density=True, color=color, alpha=0.58)
        ax.axvline(1, color=INK, linewidth=1)
        ax.set_title(title, fontsize=12, fontstyle="italic")
        clean(ax)
    axes[0].set_ylabel("плотность среднего")
    axes[1].set_xlabel("среднее 100 наблюдений")
    return fig


def lesson45(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        p = np.linspace(0.001, 0.999, 600)
        for h, n, color in [(7, 10, GOLD), (14, 20, BLUE), (70, 100, RED)]:
            ll = h * np.log(p) + (n - h) * np.log1p(-p)
            y = np.exp(ll - ll.max())
            ax.plot(p, y, color=color, linewidth=2.1)
            label_x = {(7, 10): 0.90, (14, 20): 0.84, (70, 100): 0.76}[(h, n)]
            direct(ax, label_x, np.interp(label_x, p, y), f"{h}/{n}", color, ha="center")
        ax.axvline(0.7, color=INK, linestyle=(0, (3, 3)), linewidth=1)
        ax.set_xlabel("вероятность орла $p$")
        ax.set_ylabel("$L(p)/L(\\widehat p)$")
        clean(ax, ygrid=True)
        return fig
    if index == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), layout="constrained")
        p = np.linspace(0.02, 0.98, 500)
        ll = 14 * np.log(p) + 6 * np.log1p(-p)
        l = np.exp(ll)
        axes[0].plot(p, l, color=BLUE, linewidth=2.2)
        axes[0].set_title("likelihood", fontsize=12, fontstyle="italic")
        axes[1].plot(p, ll - ll.max(), color=RED, linewidth=2.2)
        p0 = 0.7
        curvature = -14 / p0**2 - 6 / (1 - p0) ** 2
        quad = 0.5 * curvature * (p - p0) ** 2
        axes[1].plot(p, quad, color=GOLD, linestyle=(0, (4, 3)), linewidth=1.4)
        direct(axes[1], 0.35, np.interp(0.35, p, quad), "квадратичное\nприближение", GOLD)
        for ax in axes:
            ax.axvline(p0, color=INK, linewidth=1, linestyle=(0, (3, 3)))
            ax.set_xlabel("$p$")
            clean(ax, ygrid=True)
        axes[0].set_ylabel("$L(p)$")
        axes[1].set_ylabel("$\\ell(p)-\\ell(\\widehat p)$")
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.2), layout="constrained")
    p = np.linspace(0, 1, 500)
    l = p**10
    ax.plot(p, l, color=RED, linewidth=2.5)
    ax.scatter([1], [1], s=55, facecolor=PAPER, edgecolor=RED, linewidth=2, zorder=4)
    ax.axvline(1, color=INK, linewidth=1)
    direct(ax, 0.98, 0.88, "$\\widehat p=1$ — граница", RED, ha="right")
    ax.plot(p, np.maximum(0, 1 - 50 * (p - 1) ** 2), color=BLUE, linestyle=(0, (4, 3)), linewidth=1.4)
    direct(ax, 0.79, 0.25, "симметричная парабола\nтеряет форму хвоста", BLUE)
    ax.set_xlabel("$p$")
    ax.set_ylabel("нормированное правдоподобие")
    clean(ax, ygrid=True)
    return fig


def lesson46(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 8.0), layout="constrained")
        p, n = 0.6, 400
        est = RNG.binomial(n, p, 100) / n
        se = np.sqrt(est * (1 - est) / n)
        lo, hi = est - 1.96 * se, est + 1.96 * se
        cover = (lo <= p) & (p <= hi)
        y = np.arange(100)
        for i in range(100):
            color = BLUE if cover[i] else RED
            ax.plot([lo[i], hi[i]], [y[i], y[i]], color=color, linewidth=1.25)
            ax.scatter([est[i]], [y[i]], color=color, s=7)
        ax.axvline(p, color=INK, linewidth=1.2)
        ax.set_xlim(0.48, 0.72)
        ax.set_ylim(-2, 101)
        ax.set_xlabel("оценка доли и 95%-й интервал")
        ax.set_ylabel("повторный опрос")
        ax.set_yticks([0, 24, 49, 74, 99], ["1", "25", "50", "75", "100"])
        label(ax, p, 102, "$p=0{,}6$", INK, 10.5, "center")
        clean(ax, xgrid=True)
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 6.0), layout="constrained")
        n, z = 10, 1.96
        for h in range(11):
            ph = h / n
            wald = (ph - z * math.sqrt(ph * (1 - ph) / n), ph + z * math.sqrt(ph * (1 - ph) / n))
            center = (ph + z**2 / (2 * n)) / (1 + z**2 / n)
            half = z * math.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
            wilson = (center - half, center + half)
            ax.plot(wald, [h + 0.12] * 2, color=RED, linewidth=2)
            ax.plot(wilson, [h - 0.12] * 2, color=BLUE, linewidth=2)
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.7, 10.7)
        ax.set_xlabel("возможное значение $p$")
        ax.set_ylabel("успехов $h$ из 10")
        direct(ax, 0.05, 9.7, "Вальд", RED)
        direct(ax, 0.05, 9.1, "Уилсон", BLUE)
        clean(ax, xgrid=True)
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), layout="constrained")
    x = np.linspace(-3, 4, 500)
    ys = [-(x**2) / 2, -np.where(x < 0, x**2 / 1.5, x**2 / 6)]
    for ax, y, title in zip(axes, ys, ["почти симметрично", "скошено"]):
        ax.plot(x, y, color=BLUE, linewidth=2.2)
        level = -1.92
        ax.axhline(level, color=GOLD, linewidth=1.3)
        mask = y >= level
        roots = x[np.where(np.diff(mask.astype(int)) != 0)[0]]
        for r in roots:
            ax.axvline(r, color=RED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(title, fontsize=12, fontstyle="italic")
        ax.set_xlabel("$\\theta-\\widehat\\theta$")
        clean(ax, ygrid=True)
    axes[0].set_ylabel("$\\ell(\\theta)-\\ell(\\widehat\\theta)$")
    return fig


def lesson47(index: int):
    if index == 1:
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.6), layout="constrained", sharex=True)
        x = np.linspace(0.001, 0.999, 500)
        prior = beta_dist.pdf(x, 8, 8)
        like = x**3 * (1 - x)
        like = like / np.trapezoid(like, x)
        post = beta_dist.pdf(x, 11, 9)
        for ax, y, color, title in zip(axes, [prior, like, post], [BLUE, GOLD, RED], ["prior", "likelihood 3/4", "posterior"]):
            ax.fill_between(x, y, color=color, alpha=0.2)
            ax.plot(x, y, color=color, linewidth=2)
            ax.set_title(title, fontsize=12, fontstyle="italic")
            ax.set_yticks([])
            clean(ax)
        axes[1].set_xlabel("вероятность орла $\\theta$")
        return fig
    if index == 2:
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), layout="constrained", sharex=True)
        x = np.linspace(0.001, 0.999, 500)
        configs = [(2, 2, 5, 3, "слабый prior"), (50, 50, 53, 51, "сильный prior")]
        for row, (a, b, ap, bp, title) in enumerate(configs):
            axes[row, 0].plot(x, beta_dist.pdf(x, a, b), color=BLUE, linewidth=2)
            axes[row, 1].plot(x, beta_dist.pdf(x, ap, bp), color=RED, linewidth=2)
            axes[row, 0].set_ylabel(title)
            direct(axes[row, 0], 0.05, axes[row, 0].get_ylim()[1] * 0.8, f"mean={a/(a+b):.2f}", BLUE)
            direct(axes[row, 1], 0.72, axes[row, 1].get_ylim()[1] * 0.8, f"mean={ap/(ap+bp):.2f}", RED)
            for ax in axes[row]:
                ax.set_yticks([])
                clean(ax)
        axes[0, 0].set_title("до данных", fontsize=12, fontstyle="italic")
        axes[0, 1].set_title("после 3/4", fontsize=12, fontstyle="italic")
        axes[1, 0].set_xlabel("$\\theta$")
        axes[1, 1].set_xlabel("$\\theta$")
        return fig
    fig = canvas()
    ax = diagram_axes(fig)
    box(ax, (0.05, 0.62), 0.16, 0.13, "$p(\\theta)$", color=BLUE)
    box(ax, (0.30, 0.62), 0.18, 0.13, "$D_1$", color=GOLD)
    box(ax, (0.57, 0.62), 0.18, 0.13, "$D_2$", color=GREEN)
    box(ax, (0.79, 0.62), 0.16, 0.13, "posterior", color=RED)
    for x0, x1, color in [(0.21, 0.30, GOLD), (0.48, 0.57, GREEN), (0.75, 0.79, RED)]:
        arrow(ax, (x0, 0.685), (x1, 0.685), color=color)
    box(ax, (0.30, 0.25), 0.45, 0.13, "$D_1\\cup D_2$", color=VIOLET)
    arrow(ax, (0.13, 0.62), (0.30, 0.315), color=VIOLET, curve=0.1)
    arrow(ax, (0.75, 0.315), (0.87, 0.62), color=VIOLET, curve=-0.1)
    label(ax, 0.52, 0.10, "$p(\\theta)p(D_1\\mid\\theta)p(D_2\\mid\\theta)$ — один и тот же продукт", MUTED, 11, "center")
    return fig


def lesson48(index: int):
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        box(ax, (0.04, 0.62), 0.25, 0.15, "$\\theta^{\\alpha-1}(1-\\theta)^{\\beta-1}$", color=BLUE, size=12)
        box(ax, (0.38, 0.62), 0.22, 0.15, "$\\theta^h(1-\\theta)^t$", color=GOLD, size=12)
        box(ax, (0.70, 0.62), 0.26, 0.15, "$\\theta^{\\alpha+h-1}(1-\\theta)^{\\beta+t-1}$", color=RED, size=11)
        arrow(ax, (0.29, 0.695), (0.38, 0.695), color=MUTED)
        arrow(ax, (0.60, 0.695), (0.70, 0.695), color=MUTED)
        label(ax, 0.50, 0.45, "показатели степеней складываются", INK, 12, "center")
        arrow(ax, (0.19, 0.61), (0.76, 0.36), color=BLUE, curve=-0.18)
        arrow(ax, (0.49, 0.61), (0.89, 0.36), color=GOLD, curve=-0.1)
        label(ax, 0.76, 0.27, "$\\alpha+h$", RED, 13, "center")
        label(ax, 0.89, 0.27, "$\\beta+t$", RED, 13, "center")
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        ns = np.array([1, 2, 5, 10, 20, 50, 100])
        alpha, beta = 2, 8
        y0 = np.zeros_like(ns, dtype=float)
        y1 = np.ones_like(ns, dtype=float)
        p0 = alpha / (ns + alpha + beta)
        p1 = (ns + alpha) / (ns + alpha + beta)
        for i, n in enumerate(ns):
            ax.plot([n, n], [y0[i], p0[i]], color=BLUE, alpha=0.65)
            ax.plot([n, n], [y1[i], p1[i]], color=RED, alpha=0.65)
        ax.scatter(ns, y0, facecolor=PAPER, edgecolor=BLUE, s=42, label="сырая 0")
        ax.scatter(ns, p0, color=BLUE, s=32)
        ax.scatter(ns, y1, facecolor=PAPER, edgecolor=RED, s=42, label="сырая 1")
        ax.scatter(ns, p1, color=RED, s=32)
        ax.axhline(alpha / (alpha + beta), color=GOLD, linestyle=(0, (3, 3)), linewidth=1.2)
        direct(ax, 53, 0.23, "prior mean 0,2", GOLD)
        ax.set_xscale("log")
        ax.set_xlabel("объём группы $n$")
        ax.set_ylabel("доля / posterior mean")
        ax.set_ylim(-0.05, 1.05)
        clean(ax, xgrid=True, ygrid=True)
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    n, p = 50, 0.2
    k = np.arange(0, 31)
    bi = binom.pmf(k, n, p)
    a, b = 2, 8
    bb = np.array([math.comb(n, int(x)) * beta_fn(a + x, b + n - x) / beta_fn(a, b) for x in k])
    ax.plot(k, bi, color=BLUE, linewidth=2.2)
    ax.plot(k, bb, color=RED, linewidth=2.2)
    ax.fill_between(k, bi, color=BLUE, alpha=0.1)
    ax.fill_between(k, bb, color=RED, alpha=0.08)
    direct(ax, 14, bi[14], "Binomial", BLUE)
    direct(ax, 22, bb[22], "Beta–Binomial", RED)
    ax.axvline(10, color=INK, linestyle=(0, (3, 3)), linewidth=1)
    ax.set_xlabel("успехов $H$ из 50")
    ax.set_ylabel("вероятность")
    clean(ax, ygrid=True)
    return fig


def lesson49(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        x = np.linspace(28, 125, 34)
        y = 2.2 + 0.12 * x + RNG.normal(0, 1.55, x.size)
        coef = np.polyfit(x, y, 1)
        pred = np.polyval(coef, x)
        ax.scatter(x, y, s=28, facecolor=PAPER, edgecolor=BLUE, linewidth=1.2, zorder=3)
        ax.plot(x, pred, color=RED, linewidth=2.2)
        for i in [3, 10, 18, 26, 31]:
            ax.plot([x[i], x[i]], [pred[i], y[i]], color=GOLD, linewidth=1.5)
        direct(ax, 92, np.polyval(coef, 92) + 0.5, "$\\widehat y=w_0+w_1x$", RED)
        ax.set_xlabel("площадь, м²")
        ax.set_ylabel("цена, млн рублей")
        clean(ax, xgrid=True, ygrid=True)
        return fig
    if index == 2:
        fig = canvas()
        ax = fig.add_subplot(111, projection="3d")
        xx, yy = np.meshgrid(np.linspace(-1.4, 1.4, 2), np.linspace(-1.4, 1.4, 2))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, color=BLUE, alpha=0.12, edgecolor=BLUE, linewidth=0.5)
        yv = np.array([0.8, 0.55, 1.8])
        yh = np.array([0.8, 0.55, 0.0])
        ax.quiver(0, 0, 0, *yv, color=INK, arrow_length_ratio=0.08, linewidth=2)
        ax.quiver(0, 0, 0, *yh, color=BLUE, arrow_length_ratio=0.1, linewidth=2)
        ax.plot([yh[0], yv[0]], [yh[1], yv[1]], [0, yv[2]], color=RED, linewidth=2)
        ax.text(*yv, "$y$", color=INK, fontsize=12)
        ax.text(yh[0], yh[1], 0.05, "$\\widehat y=X\\widehat w$", color=BLUE, fontsize=11)
        ax.text(yv[0], yv[1], 0.9, "$r$", color=RED, fontsize=12)
        ax.set_axis_off()
        ax.view_init(elev=23, azim=-57)
        return fig
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), layout="constrained", sharex=True)
    fitted = np.linspace(0, 10, 80)
    cases = [
        (RNG.normal(0, 0.7, 80), "без структуры", BLUE),
        (0.09 * (fitted - 5) ** 2 - 0.75 + RNG.normal(0, 0.25, 80), "осталась кривизна", RED),
        (RNG.normal(0, 0.16 + 0.11 * fitted), "растущий веер", GOLD),
        (0.75 * np.sin(1.8 * fitted) + RNG.normal(0, 0.2, 80), "временная волна", VIOLET),
    ]
    for ax, (res, title, color) in zip(axes.flat, cases):
        ax.scatter(fitted, res, color=color, s=12, alpha=0.7)
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_title(title, fontsize=11.5, fontstyle="italic")
        clean(ax, ygrid=True)
    axes[1, 0].set_xlabel("прогноз")
    axes[1, 1].set_xlabel("прогноз / время")
    axes[0, 0].set_ylabel("остаток")
    axes[1, 0].set_ylabel("остаток")
    return fig


def lesson50(index: int):
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        box(ax, (0.03, 0.54), 0.16, 0.14, "час $x$", color=INK)
        for y, text, color in [(0.76, "$1$", MUTED), (0.54, "$x$", BLUE), (0.32, "$x^2$", GOLD)]:
            box(ax, (0.34, y), 0.14, 0.11, text, color=color)
            arrow(ax, (0.19, 0.61), (0.34, y + 0.055), color=color)
        box(ax, (0.72, 0.54), 0.22, 0.14, "$w_0+w_1x+w_2x^2$", color=RED, size=12)
        for y, color in [(0.815, MUTED), (0.595, BLUE), (0.375, GOLD)]:
            arrow(ax, (0.48, y), (0.72, 0.61), color=color)
        label(ax, 0.5, 0.14, "линейность по весам, кривизна по времени", MUTED, 11.5, "center")
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        x_train = np.linspace(-1, 1, 18)
        local_rng = np.random.default_rng(5002)
        y_train = np.sin(2 * x_train) + local_rng.normal(0, 0.1, x_train.size)
        x = np.linspace(-1.35, 1.35, 500)
        ax.axvspan(-1, 1, color=WASH)
        label_positions = {1: (1.08, 1.55), 3: (1.08, 0.20), 9: (1.14, 4.75)}
        for degree, color in [(1, BLUE), (3, GREEN), (9, RED)]:
            coef = np.polyfit(x_train, y_train, degree)
            ax.plot(x, np.polyval(coef, x), color=color, linewidth=2 if degree != 9 else 1.7)
            label_x, label_y = label_positions[degree]
            direct(ax, label_x, label_y, f"степень {degree}", color)
        ax.scatter(x_train, y_train, color=INK, s=18, zorder=4)
        ax.axvline(-1, color=FAINT, linewidth=1)
        ax.axvline(1, color=FAINT, linewidth=1)
        ax.set_ylim(-8, 8)
        label(ax, 0, ax.get_ylim()[0], "обучающий диапазон", MUTED, 10.5, "center", "bottom")
        ax.set_xlabel("$x$")
        ax.set_ylabel("прогноз")
        clean(ax, ygrid=True)
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2), layout="constrained", sharey=True)
    a = np.linspace(0, 10, 120)
    prices = [10, 20, 30]
    for ax, interaction, title in zip(axes, [False, True], ["без взаимодействия", "с $p\\times a$"]):
        for p, color in zip(prices, [GREEN, BLUE, RED]):
            y = 100 - 2 * p + 3 * a + (0.12 * p * a if interaction else 0)
            ax.plot(a, y, color=color, linewidth=2)
            direct(ax, 8.8, y[-14], f"$p={p}$", color)
        ax.set_title(title, fontsize=12, fontstyle="italic")
        ax.set_xlabel("реклама $a$")
        clean(ax, ygrid=True)
    axes[0].set_ylabel("продажи")
    return fig


def lesson51(index: int):
    if index == 1:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), layout="constrained")
        e = np.linspace(-4, 4, 500)
        delta = 1
        losses = [
            (0.5 * e**2, e, BLUE, "квадрат"),
            (np.abs(e), np.sign(e), RED, "модуль"),
            (np.where(np.abs(e) <= delta, 0.5 * e**2, delta * (np.abs(e) - 0.5 * delta)),
             np.clip(e, -delta, delta), GREEN, "Хьюбер"),
        ]
        for loss, grad, color, text in losses:
            axes[0].plot(e, loss, color=color, linewidth=2)
            axes[1].plot(e, grad, color=color, linewidth=2)
        axes[0].set_title("потеря", fontsize=12, fontstyle="italic")
        axes[1].set_title("сила влияния $\\rho'(e)$", fontsize=12, fontstyle="italic")
        for ax in axes:
            ax.set_xlabel("остаток $e$")
            clean(ax, ygrid=True)
        direct(axes[0], 2.7, 0.5 * 2.7**2, "квадрат", BLUE)
        direct(axes[1], 2.8, 1.05, "модуль / Хьюбер", RED)
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(7.4, 6.4), layout="constrained")
        x = np.linspace(-4, 4, 300)
        y = np.linspace(-4, 4, 300)
        X, Y = np.meshgrid(x, y)
        Z = (X + Y - 2) ** 2 + 0.025 * (X - Y) ** 2
        ax.contour(X, Y, Z, levels=[0.25, 0.8, 2, 5, 10], colors=[FAINT] * 5, linewidths=1)
        for radius in [1.2, 2.0, 2.8]:
            ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor=BLUE, alpha=0.45, linewidth=1))
        ols = np.array([3.0, -1.0])
        ridge = np.array([0.86, 0.86])
        ax.scatter(*ols, color=RED, s=50)
        ax.scatter(*ridge, color=BLUE, s=50)
        direct(ax, ols[0], ols[1], "без штрафа", RED)
        direct(ax, ridge[0], ridge[1] + 0.3, "ridge", BLUE, ha="center")
        ax.set_xlabel("$w_1$")
        ax.set_ylabel("$w_2$")
        ax.set_aspect("equal")
        clean(ax, xgrid=True, ygrid=True)
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.3), layout="constrained")
    theta = np.linspace(0, 2 * np.pi, 400)
    for ax, kind, title in zip(axes, ["ridge", "lasso"], ["$L_2$: окружность", "$L_1$: ромб"]):
        X, Y = np.meshgrid(np.linspace(-2, 2, 220), np.linspace(-2, 2, 220))
        Z = 0.35 * (X - 1.4) ** 2 + 1.1 * (Y - 0.8) ** 2 + 0.5 * (X - 1.4) * (Y - 0.8)
        ax.contour(X, Y, Z, levels=[0.15, 0.4, 0.8, 1.4, 2.2], colors=[FAINT] * 5)
        if kind == "ridge":
            ax.plot(np.cos(theta), np.sin(theta), color=BLUE, linewidth=2)
            point = (0.83, 0.55)
        else:
            ax.add_patch(Polygon([(1, 0), (0, 1), (-1, 0), (0, -1)], fill=False, edgecolor=RED, linewidth=2))
            point = (1, 0)
        ax.scatter(*point, color=BLUE if kind == "ridge" else RED, s=42, zorder=3)
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.axvline(0, color=GRID, linewidth=0.8)
        ax.set_title(title, fontsize=12, fontstyle="italic")
        ax.set_aspect("equal")
        ax.set_xlabel("$w_1$")
        ax.set_ylabel("$w_2$")
        clean(ax)
    return fig


def lesson52(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        x_obs = np.array([-0.7, 0.0, 0.8])
        y_obs = np.array([-0.4, 0.2, 0.75])
        X = np.c_[np.ones(3), x_obs]
        sigma, tau = 0.25, 1.2
        S = np.linalg.inv(np.eye(2) / tau**2 + X.T @ X / sigma**2)
        m = S @ (X.T @ y_obs / sigma**2)
        draws = RNG.multivariate_normal(m, S, 70)
        x = np.linspace(-2.5, 2.5, 300)
        for w in draws:
            ax.plot(x, w[0] + w[1] * x, color=BLUE, alpha=0.09, linewidth=0.9)
        mean = m[0] + m[1] * x
        sd = np.sqrt(np.einsum("ij,jk,ik->i", np.c_[np.ones_like(x), x], S, np.c_[np.ones_like(x), x]))
        ax.fill_between(x, mean - 1.96 * sd, mean + 1.96 * sd, color=BLUE, alpha=0.16)
        ax.plot(x, mean, color=RED, linewidth=2.2)
        ax.scatter(x_obs, y_obs, color=INK, s=35, zorder=4)
        ax.axvspan(x_obs.min(), x_obs.max(), color=WASH, zorder=-2)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        clean(ax, ygrid=True)
        return fig
    if index == 2:
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.4), layout="constrained")
        specs = [
            ((0, 0), [[1.1, 0], [0, 1.1]], BLUE, "prior"),
            ((1.2, 0.8), [[1.4, -1.05], [-1.05, 0.9]], GOLD, "likelihood"),
            ((0.65, 0.48), [[0.36, -0.21], [-0.21, 0.25]], RED, "posterior"),
        ]
        for ax, (mean, cov, color, title) in zip(axes, specs):
            gaussian_ellipse(ax, mean, cov, color=color)
            ax.scatter(*mean, color=color, s=35)
            ax.set_xlim(-2.5, 3.5)
            ax.set_ylim(-2.5, 3.0)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=12, fontstyle="italic")
            ax.set_xlabel("$w_1$")
            clean(ax, xgrid=True, ygrid=True)
        axes[0].set_ylabel("$w_2$")
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    x = np.linspace(-3, 3, 300)
    mean = 0.2 + 0.65 * x
    param_sd = 0.12 + 0.12 * np.abs(x)
    noise_sd = 0.65
    total_sd = np.sqrt(param_sd**2 + noise_sd**2)
    ax.fill_between(x, mean - 1.96 * total_sd, mean + 1.96 * total_sd, color=GOLD, alpha=0.16)
    ax.fill_between(x, mean - 1.96 * param_sd, mean + 1.96 * param_sd, color=BLUE, alpha=0.28)
    ax.plot(x, mean, color=RED, linewidth=2)
    direct(ax, -2.7, mean[15] + 1.4, "новое наблюдение:\n$\\sigma^2+x^\\top Sx$", GOLD)
    direct(ax, 1.55, mean[-60] + 0.48, "условное среднее:\n$x^\\top Sx$", BLUE)
    ax.axvspan(-1, 1, color=WASH, zorder=-2)
    ax.set_xlabel("$x$")
    ax.set_ylabel("прогноз")
    clean(ax, ygrid=True)
    return fig


def lesson53(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(8.0, 6.5), layout="constrained")
        c0 = RNG.multivariate_normal([0.25, 0.25], [[0.025, 0], [0, 0.018]], 35)
        c1 = RNG.multivariate_normal([0.70, 0.68], [[0.025, 0.005], [0.005, 0.02]], 35)
        ax.scatter(c0[:, 0], c0[:, 1], color=BLUE, s=23, alpha=0.7)
        ax.scatter(c1[:, 0], c1[:, 1], color=RED, s=23, alpha=0.7)
        x = np.linspace(0, 1, 100)
        y = 0.85 - x
        ax.plot(x, y, color=INK, linewidth=1.7)
        p = np.array([0.78, 0.38])
        foot = np.array([0.575, 0.275])
        ax.plot([p[0], foot[0]], [p[1], foot[1]], color=GOLD, linewidth=1.7)
        ax.scatter(*p, facecolor=PAPER, edgecolor=GOLD, s=55, linewidth=2, zorder=4)
        arrow(ax, (0.5, 0.35), (0.66, 0.51), color=GREEN, shrink=1)
        direct(ax, 0.67, 0.53, "$w$", GREEN)
        direct(ax, 0.68, 0.31, "$s(x)/\\|w\\|$", GOLD)
        ax.set_xlabel("доля ссылок $x_1$")
        ax.set_ylabel("частота слова $x_2$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        clean(ax, xgrid=True, ygrid=True)
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(8.0, 6.5), layout="constrained")
        pts0 = RNG.multivariate_normal([0.25, 0.3], [[0.02, 0], [0, 0.025]], 35)
        pts1 = RNG.multivariate_normal([0.72, 0.65], [[0.025, 0], [0, 0.02]], 35)
        ax.scatter(pts0[:, 0], pts0[:, 1], color=BLUE, s=20, alpha=0.65)
        ax.scatter(pts1[:, 0], pts1[:, 1], color=RED, s=20, alpha=0.65)
        x = np.linspace(0, 1, 100)
        for intercept, color, text in [(1.05, GREEN, "порог 0,2"), (0.85, INK, "0,5"), (0.65, GOLD, "0,8")]:
            y = intercept - x
            ax.plot(x, y, color=color, linewidth=1.8)
            direct(ax, 0.07, intercept - 0.07, text, color)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        clean(ax, xgrid=True, ygrid=True)
        return fig
    fig = canvas()
    axes = fig.subplots(1, 2, subplot_kw={"projection": None})
    th = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    r1 = RNG.normal(0.55, 0.04, th.size)
    r2 = RNG.normal(1.15, 0.05, th.size)
    axes[0].scatter(r1 * np.cos(th), r1 * np.sin(th), color=BLUE, s=15)
    axes[0].scatter(r2 * np.cos(th), r2 * np.sin(th), color=RED, s=15)
    axes[0].add_patch(Circle((0, 0), 0.84, fill=False, edgecolor=INK, linewidth=1.7))
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].set_title("исходная плоскость", fontsize=12, fontstyle="italic")
    clean(axes[0])
    pos = axes[1].get_position()
    axes[1].remove()
    ax3 = fig.add_axes(pos, projection="3d")
    for r, color in [(r1, BLUE), (r2, RED)]:
        xx, yy = r * np.cos(th), r * np.sin(th)
        ax3.scatter(xx, yy, xx**2 + yy**2, color=color, s=14)
    xx, yy = np.meshgrid(np.linspace(-1.3, 1.3, 2), np.linspace(-1.3, 1.3, 2))
    ax3.plot_surface(xx, yy, np.ones_like(xx) * 0.72, color=GOLD, alpha=0.25)
    ax3.set_xlabel("$x_1$")
    ax3.set_ylabel("$x_2$")
    ax3.set_zlabel("$x_1^2+x_2^2$")
    ax3.set_title("поднятие в 3D", fontsize=12, fontstyle="italic")
    return fig


def lesson54(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(8.2, 6.6), layout="constrained")
        m0, m1 = np.array([2.0, 1.3]), np.array([4.7, 2.9])
        cov = np.array([[0.7, 0.28], [0.28, 0.42]])
        p0 = RNG.multivariate_normal(m0, cov, 55)
        p1 = RNG.multivariate_normal(m1, cov, 55)
        ax.scatter(p0[:, 0], p0[:, 1], color=BLUE, s=18, alpha=0.6)
        ax.scatter(p1[:, 0], p1[:, 1], color=RED, s=18, alpha=0.6)
        gaussian_ellipse(ax, m0, cov, color=BLUE)
        gaussian_ellipse(ax, m1, cov, color=RED)
        inv = np.linalg.inv(cov)
        w = inv @ (m1 - m0)
        b = -0.5 * (m1 @ inv @ m1 - m0 @ inv @ m0) + math.log(0.35 / 0.65)
        x = np.linspace(0, 6.5, 200)
        y = -(w[0] * x + b) / w[1]
        ax.plot(x, y, color=INK, linewidth=1.8)
        ax.scatter([*m0[:1], *m1[:1]], [m0[1], m1[1]], marker="x", color=[BLUE, RED], s=70)
        ax.set_xlabel("длина лепестка")
        ax.set_ylabel("ширина лепестка")
        ax.set_xlim(0, 6.5)
        ax.set_ylim(0, 4.5)
        clean(ax, xgrid=True, ygrid=True)
        return fig
    if index == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.5), layout="constrained", sharex=True, sharey=True)
        m0, m1 = np.array([-1.0, 0.0]), np.array([1.1, 0.25])
        cov0 = np.array([[0.65, 0.3], [0.3, 0.4]])
        cov1 = np.array([[0.25, -0.1], [-0.1, 0.8]])
        pts0 = RNG.multivariate_normal(m0, cov0, 55)
        pts1 = RNG.multivariate_normal(m1, cov1, 55)
        xx, yy = np.meshgrid(np.linspace(-3, 3, 250), np.linspace(-2.5, 2.5, 250))
        grid = np.dstack([xx, yy])
        for ax, mode in zip(axes, ["LDA", "QDA"]):
            ax.scatter(pts0[:, 0], pts0[:, 1], color=BLUE, s=14, alpha=0.6)
            ax.scatter(pts1[:, 0], pts1[:, 1], color=RED, s=14, alpha=0.6)
            if mode == "LDA":
                pooled = (cov0 + cov1) / 2
                gaussian_ellipse(ax, m0, pooled, color=BLUE)
                gaussian_ellipse(ax, m1, pooled, color=RED)
                i = np.linalg.inv(pooled)
                score = (grid @ i @ (m1 - m0)) - 0.5 * (m1 @ i @ m1 - m0 @ i @ m0)
            else:
                gaussian_ellipse(ax, m0, cov0, color=BLUE)
                gaussian_ellipse(ax, m1, cov1, color=RED)
                def q(mean, cov):
                    d = grid - mean
                    return -0.5 * np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d) - 0.5 * np.log(np.linalg.det(cov))
                score = q(m1, cov1) - q(m0, cov0)
            ax.contour(xx, yy, score, levels=[0], colors=[INK], linewidths=1.8)
            ax.set_title(mode, fontsize=12, fontstyle="italic")
            ax.set_xlabel("$x_1$")
            clean(ax)
        axes[0].set_ylabel("$x_2$")
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.5), layout="constrained")
    m0, m1 = np.array([-0.9, 0.55]), np.array([1.0, -0.25])
    cov0 = np.array([[0.7, 0.35], [0.35, 0.65]])
    cov1 = np.array([[0.55, -0.2], [-0.2, 0.5]])
    for m, c, color in [(m0, cov0, BLUE), (m1, cov1, RED)]:
        gaussian_ellipse(axes[0], m, c, color=color)
        x = np.linspace(-3, 3, 400)
        axes[1].plot(x, norm.pdf(x, m[0], math.sqrt(c[0, 0])), color=color, linewidth=2)
    axes[0].axvline(0.35, color=GOLD, linewidth=2)
    direct(axes[0], 0.38, 1.75, "известен только $x_1$", GOLD)
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$ отсутствует")
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-2.4, 2.4)
    axes[0].set_title("двумерные классы", fontsize=12, fontstyle="italic")
    axes[1].axvline(0.35, color=GOLD, linewidth=2)
    axes[1].set_xlabel("наблюдаемая координата $x_1$")
    axes[1].set_ylabel("маргинальная плотность")
    axes[1].set_title("проекция на ось", fontsize=12, fontstyle="italic")
    for ax in axes:
        clean(ax, ygrid=True)
    return fig


def lesson55(index: int):
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        labels = [("мобильные", BLUE), ("активные", GOLD), ("утро", GREEN), ("новички", VIOLET)]
        for i, (text, color) in enumerate(labels):
            y = 0.78 - i * 0.14
            box(ax, (0.05, y), 0.15, 0.09, text, color=color, size=9.8)
            arrow(ax, (0.20, y + 0.045), (0.39, 0.52), color=color)
        box(ax, (0.39, 0.44), 0.18, 0.16, "случайный\nмеханизм", color=INK)
        box(ax, (0.74, 0.65), 0.16, 0.13, "вариант A", color=BLUE)
        box(ax, (0.74, 0.26), 0.16, 0.13, "вариант B", color=RED)
        arrow(ax, (0.57, 0.52), (0.74, 0.715), color=BLUE)
        arrow(ax, (0.57, 0.52), (0.74, 0.325), color=RED)
        label(ax, 0.82, 0.53, "похожие смеси\nскрытых факторов", MUTED, 10.5, "center")
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.4), layout="constrained")
        x = np.linspace(-4, 4, 600)
        y = norm.pdf(x)
        observed = 2.15
        ax.plot(x, y, color=BLUE, linewidth=2.2)
        ax.fill_between(x, 0, y, where=np.abs(x) >= observed, color=RED, alpha=0.35)
        ax.axvline(observed, color=INK, linewidth=1.2)
        ax.axvline(-observed, color=INK, linewidth=1.2)
        direct(ax, observed, norm.pdf(observed) + 0.035, "$T_{obs}$", INK, ha="center")
        label(ax, 0, y.max() + 0.045, "$H_0$: распределение статистики без эффекта", MUTED, 11, "center")
        ax.set_xlabel("стандартизированная разность")
        ax.set_ylabel("плотность")
        ax.set_yticks([])
        clean(ax)
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    steps = np.arange(1, 121)
    for i in range(100):
        inc = RNG.normal(0, 1, steps.size)
        z = np.cumsum(inc) / np.sqrt(steps)
        crossed = np.any(np.abs(z) > 1.96)
        ax.plot(steps, z, color=RED if crossed else BLUE, alpha=0.16 if crossed else 0.08, linewidth=0.8)
    ax.axhline(1.96, color=INK, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.axhline(-1.96, color=INK, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.set_xlabel("момент просмотра")
    ax.set_ylabel("накопленный $z$")
    ax.set_ylim(-4.5, 4.5)
    clean(ax, ygrid=True)
    return fig


def lesson56(index: int):
    if index == 1:
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.6), layout="constrained", sharex=True, sharey=True)
        x = np.linspace(-2, 2, 25)
        axes[0].scatter(x, 0.45 * x + RNG.normal(0, 0.65, x.size), color=BLUE, s=18)
        axes[0].scatter(x, 0.45 * x + RNG.normal(0, 0.65, x.size), color=RED, s=18, alpha=0.55)
        axes[0].set_title("aleatoric", fontsize=12, fontstyle="italic")
        sparse_x = np.array([-1.8, -1.0, 1.1, 1.8])
        sparse_y = 0.5 * sparse_x + RNG.normal(0, 0.1, sparse_x.size)
        axes[1].scatter(sparse_x, sparse_y, color=INK, s=26)
        grid = np.linspace(-2, 2, 200)
        for slope in np.linspace(0.2, 0.8, 9):
            axes[1].plot(grid, slope * grid, color=BLUE, alpha=0.18)
        axes[1].set_title("epistemic", fontsize=12, fontstyle="italic")
        axes[2].scatter(np.linspace(-2, 0.7, 22), RNG.normal(0, 0.35, 22), color=BLUE, s=18)
        axes[2].scatter([1.8], [0.8], facecolor=PAPER, edgecolor=RED, linewidth=2, s=65)
        axes[2].axvspan(1.2, 2.2, color=RED, alpha=0.08)
        axes[2].set_title("distributional", fontsize=12, fontstyle="italic")
        for ax in axes:
            ax.axhline(0, color=GRID, linewidth=0.8)
            ax.set_xlabel("область признака")
            clean(ax)
        axes[0].set_ylabel("ответ / прогноз")
        return fig
    if index == 2:
        fig, (ax, hist_ax) = plt.subplots(2, 1, figsize=(8.4, 7.0), layout="constrained", height_ratios=[3, 1], sharex=True)
        pred = np.array([0.08, 0.18, 0.31, 0.45, 0.59, 0.72, 0.86, 0.94])
        obs = np.array([0.05, 0.14, 0.26, 0.51, 0.53, 0.62, 0.74, 0.80])
        sizes = np.array([700, 620, 500, 360, 240, 140, 60, 20])
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle=(0, (4, 3)))
        ax.scatter(pred, obs, s=20 + sizes / 7, color=BLUE, alpha=0.7, edgecolor=PAPER)
        for i in [-2, -1]:
            ax.plot([pred[i], pred[i]], [pred[i], obs[i]], color=RED, linewidth=1.5)
        ax.set_ylabel("наблюдаемая доля")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        clean(ax, xgrid=True, ygrid=True)
        hist_ax.bar(pred, sizes, width=0.09, color=GOLD, alpha=0.65)
        hist_ax.set_xlabel("предсказанная вероятность")
        hist_ax.set_ylabel("$n$")
        clean(hist_ax, ygrid=True)
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    coverage = np.linspace(0.1, 1, 200)
    useful = 0.025 + 0.18 * coverage**2.2
    random = np.full_like(coverage, 0.145)
    ax.plot(coverage * 100, useful * 100, color=BLUE, linewidth=2.3)
    ax.plot(coverage * 100, random * 100, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.7)
    work = 0.72
    ax.scatter([work * 100], [np.interp(work, coverage, useful) * 100], color=RED, s=55, zorder=4)
    direct(ax, work * 100, np.interp(work, coverage, useful) * 100 + 1.4, "рабочая точка", RED, ha="center")
    direct(ax, 84, useful[-35] * 100, "информативный отказ", BLUE)
    direct(ax, 72, 14.5, "случайный отказ", MUTED)
    ax.set_xlabel("coverage, %")
    ax.set_ylabel("selective risk, %")
    ax.set_xlim(10, 100)
    ax.set_ylim(0, 23)
    clean(ax, xgrid=True, ygrid=True)
    return fig


def lesson57(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 5.6), layout="constrained")
        p = np.linspace(0, 0.01, 400)
        miss = 10000 * p
        alarm = 20 * (1 - p)
        threshold = 20 / 10020
        ax.plot(p * 100, miss, color=RED, linewidth=2.2)
        ax.plot(p * 100, alarm, color=BLUE, linewidth=2.2)
        ax.fill_between(p * 100, np.minimum(miss, alarm), color=GREEN, alpha=0.1)
        ax.axvline(threshold * 100, color=INK, linestyle=(0, (3, 3)), linewidth=1.2)
        direct(ax, 0.72, miss[np.searchsorted(p, 0.0072)], "не эвакуировать", RED)
        direct(ax, 0.62, alarm[np.searchsorted(p, 0.0062)], "эвакуировать", BLUE)
        direct(ax, threshold * 100, 31, "$p_*\\approx0{,}2\\%$", INK, ha="center")
        ax.set_xlabel("вероятность пожара $p$, %")
        ax.set_ylabel("условная ожидаемая потеря")
        clean(ax, xgrid=True, ygrid=True)
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.6), layout="constrained")
        m = np.linspace(-4, 4, 500)
        curves = [
            ((m <= 0).astype(float), INK, "0/1"),
            (np.log1p(np.exp(-m)), BLUE, "logistic"),
            (np.maximum(0, 1 - m), RED, "hinge"),
            (np.exp(-m), GOLD, "exponential"),
        ]
        for y, color, text in curves:
            ax.plot(m, y, color=color, linewidth=2)
        direct(ax, 2.45, 0.98, "0/1", INK)
        direct(ax, -2.0, np.log1p(np.exp(2)), "logistic", BLUE)
        direct(ax, -1.1, 2.1, "hinge", RED)
        direct(ax, -1.4, 4.2, "exponential", GOLD)
        ax.set_xlabel("правильный margin $ys$")
        ax.set_ylabel("loss")
        ax.set_ylim(-0.05, 5)
        clean(ax, ygrid=True)
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    x = np.linspace(0, 80, 600)
    density = 0.65 * norm.pdf(x, 18, 7) + 0.35 * norm.pdf(x, 40, 13)
    density /= np.trapezoid(density, x)
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    median = x[np.searchsorted(cdf, 0.5)]
    q90 = x[np.searchsorted(cdf, 0.9)]
    mean = np.trapezoid(x * density, x)
    ax.fill_between(x, density, color=BLUE, alpha=0.16)
    ax.plot(x, density, color=BLUE, linewidth=2)
    for value, color, text in [(median, GREEN, "медиана"), (mean, RED, "среднее"), (q90, GOLD, "90-й квантиль")]:
        ax.axvline(value, color=color, linewidth=1.8)
        direct(ax, value, density.max() * (0.92 if text != "среднее" else 0.72), text, color, ha="center")
    ax.set_xlabel("спрос")
    ax.set_ylabel("плотность")
    ax.set_yticks([])
    clean(ax)
    return fig


def lesson58(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 5.6), layout="constrained")
        r = 0.2
        x = np.linspace(0.08, 0.32, 600)
        for n, color in [(50, GOLD), (200, BLUE), (2000, RED)]:
            sd = math.sqrt(r * (1 - r) / n)
            y = norm.pdf(x, r, sd)
            y /= y.max()
            ax.plot(x, y, color=color, linewidth=2)
            direct(ax, r + 1.2 * sd, np.interp(r + 1.2 * sd, x, y), f"$n={n}$", color)
        ax.axvline(r, color=INK, linewidth=1.2, linestyle=(0, (3, 3)))
        ax.set_xlabel("оценка риска $\\widehat R_n$")
        ax.set_ylabel("относительная плотность")
        ax.set_yticks([])
        clean(ax, xgrid=True)
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        m = 35
        true = 0.18 + RNG.uniform(0, 0.055, m)
        observed = true + RNG.normal(0, 0.018, m)
        order = np.argsort(observed)
        for i in range(m):
            color = RED if i == order[0] else FAINT
            ax.plot([i, i], [true[i], observed[i]], color=color, linewidth=1.2)
            ax.scatter([i], [true[i]], facecolor=PAPER, edgecolor=color, s=26)
            ax.scatter([i], [observed[i]], color=color, s=18)
        win = order[0]
        direct(ax, win, observed[win] - 0.018, "выбранный минимум", RED, ha="center")
        ax.set_xlabel("кандидат")
        ax.set_ylabel("risk")
        ax.set_xticks([])
        clean(ax, ygrid=True)
        return fig
    fig = canvas()
    ax = diagram_axes(fig)
        # Three distinct splitting questions.
    rows = [
        (0.72, "случайные строки", ["train", "test", "train", "test"], [BLUE, RED, BLUE, RED], "знакомый режим"),
        (0.47, "время", ["янв", "фев", "мар", "апр"], [BLUE, BLUE, BLUE, RED], "будущий период"),
        (0.22, "водители", ["A", "B", "C", "D"], [BLUE, BLUE, BLUE, RED], "новый водитель"),
    ]
    for y, name, cells, colors, goal in rows:
        label(ax, 0.03, y + 0.045, name, INK, 10.5)
        for j, (text, color) in enumerate(zip(cells, colors)):
            box(ax, (0.24 + 0.105 * j, y), 0.085, 0.09, text, color=color, size=9.5)
        arrow(ax, (0.68, y + 0.045), (0.78, y + 0.045), color=MUTED)
        label(ax, 0.80, y + 0.045, goal, MUTED, 10.3)
    return fig


def lesson59(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        c = np.linspace(0.05, 1, 300)
        bias = 0.55 * np.exp(-4.5 * c) + 0.03
        var = 0.035 * np.exp(3.2 * c)
        noise = np.full_like(c, 0.08)
        total = bias + var + noise
        for y, color, text in [(bias, BLUE, "bias²"), (var, RED, "variance"), (noise, GOLD, "noise"), (total, INK, "test error")]:
            ax.plot(c, y, color=color, linewidth=2.2 if color == INK else 1.8)
        direct(ax, 0.72, np.interp(0.72, c, var), "variance", RED)
        direct(ax, 0.18, np.interp(0.18, c, bias), "bias²", BLUE)
        direct(ax, 0.61, np.interp(0.61, c, total) + 0.045, "test error", INK)
        ax.set_xlabel("сложность модели")
        ax.set_ylabel("ошибка")
        ax.set_xticks([])
        clean(ax, ygrid=True)
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        complexity = np.logspace(-1, 2, 500)
        logc = np.log10(complexity)
        train = 0.23 / (1 + np.exp(8 * (logc - 1))) + 0.004
        test = 0.10 + 0.10 * (logc + 0.4) ** 2 / (1 + (logc + 0.4) ** 2)
        test += 0.34 * np.exp(-((logc - 1) / 0.16) ** 2)
        test -= 0.09 / (1 + np.exp(-4 * (logc - 1.3)))
        ax.semilogx(complexity, train, color=BLUE, linewidth=2)
        ax.semilogx(complexity, test, color=RED, linewidth=2.3)
        ax.axvline(10, color=INK, linestyle=(0, (3, 3)), linewidth=1.2)
        direct(ax, 10, test.max() + 0.015, "порог интерполяции", INK, ha="center")
        direct(ax, 35, np.interp(35, complexity, test), "второй спуск", RED)
        ax.set_xlabel("сложность / число параметров")
        ax.set_ylabel("ошибка")
        ax.set_xticks([0.1, 1, 10, 100], ["0,1", "1", "$n$", "100"])
        clean(ax, ygrid=True)
        return fig
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.8), layout="constrained")
    spectra = [
        np.array([8, 5, 3, 1.5, 0.8]),
        np.array([8, 4, 2, 0.7, 0.03]),
        np.array([8, 5, 3.5, 2.4, 1.2]),
    ]
    for ax, s, title in zip(axes, spectra, ["до порога", "у порога", "после"]):
        j = np.arange(1, len(s) + 1)
        ax.bar(j, s, color=BLUE, alpha=0.55)
        ax2 = ax.twinx()
        ax2.plot(j, 1 / s, color=RED, marker="o", linewidth=1.6)
        ax2.plot(j, s / (s**2 + 0.2), color=GOLD, marker=".", linewidth=1.2)
        ax.set_title(title, fontsize=12, fontstyle="italic")
        ax.set_xlabel("направление $j$")
        ax.set_yticks([])
        ax2.set_yticks([])
        clean(ax)
    axes[0].set_ylabel("сингулярное число $\\sigma_j$")
    return fig


def lesson60(index: int):
    if index == 1:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), layout="constrained")
        for i in range(11):
            axes[0].axvline(i / 10, color=GRID, linewidth=0.8)
            axes[0].axhline(i / 10, color=GRID, linewidth=0.8)
        pts = RNG.random((45, 2))
        axes[0].scatter(pts[:, 0], pts[:, 1], color=BLUE, s=17)
        axes[0].set_title("$10^2$ клеток", fontsize=12, fontstyle="italic")
        axes[0].set_aspect("equal")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        d = np.arange(1, 9)
        axes[1].bar(d, 10.0**d, color=RED, alpha=0.65)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("размерность $d$")
        axes[1].set_ylabel("$10^d$ клеток")
        axes[1].set_title("одинаковое разрешение", fontsize=12, fontstyle="italic")
        clean(axes[1], ygrid=True)
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        d = np.arange(1, 51)
        volume = np.pi ** (d / 2) / np.array([gamma(x / 2 + 1) for x in d])
        fraction = volume / (2.0**d)
        ax.semilogy(d, fraction, color=BLUE, linewidth=2.3)
        for k in [2, 3, 10, 20]:
            ax.scatter([k], [fraction[k - 1]], color=RED, s=35)
            offset = {2: (-25, 18), 3: (22, 25), 10: (0, 15), 20: (0, 15)}[k]
            ax.annotate(
                f"$d={k}$",
                xy=(k, fraction[k - 1]),
                xytext=offset,
                textcoords="offset points",
                color=RED,
                fontsize=10.4,
                ha="center",
                va="center",
                arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.9},
                bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1.2, "alpha": 0.92},
            )
        ax.set_xlabel("размерность $d$")
        ax.set_ylabel("$V_d/2^d$")
        clean(ax, xgrid=True, ygrid=True)
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), layout="constrained")
    dims = [2, 20, 200]
    colors = [GOLD, BLUE, RED]
    ratios = []
    for d, color in zip(dims, colors):
        q = RNG.normal(0, 1, d)
        pts = RNG.normal(0, 1, (1200, d))
        dist = np.linalg.norm(pts - q, axis=1) / math.sqrt(d)
        axes[0].hist(dist, bins=55, density=True, histtype="step", color=color, linewidth=2)
        direct(axes[0], np.quantile(dist, 0.78), np.histogram(dist, 55, density=True)[0].max() * 0.75, f"$d={d}$", color)
        ratios.append(dist.min() / dist.max())
    ds = np.array([2, 5, 10, 20, 50, 100, 200])
    ratio_curve = []
    for d in ds:
        q = RNG.normal(size=d)
        dist = np.linalg.norm(RNG.normal(size=(1200, d)) - q, axis=1)
        ratio_curve.append(dist.min() / dist.max())
    axes[1].plot(ds, ratio_curve, color=RED, marker="o", linewidth=2)
    axes[0].set_xlabel("нормированное расстояние")
    axes[0].set_ylabel("плотность")
    axes[1].set_xlabel("размерность")
    axes[1].set_ylabel("$d_{min}/d_{max}$")
    axes[1].set_xscale("log")
    for ax in axes:
        clean(ax, ygrid=True)
    return fig


def lesson61(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(8.0, 6.5), layout="constrained")
        origin = np.array([0.2, 0.1])
        full = np.array([1.2, 0.75])
        for _ in range(28):
            g = full + RNG.normal(0, 0.35, 2)
            ax.arrow(*origin, *g, color=BLUE, alpha=0.22, width=0.008, head_width=0.06, length_includes_head=True)
        ax.arrow(*origin, *full, color=RED, width=0.016, head_width=0.09, length_includes_head=True)
        direct(ax, 1.45, 0.92, "полный градиент", RED)
        label(ax, 0.72, -0.18, "облако мини-батчей", BLUE, 11, "center")
        ax.set_xlim(-0.2, 2)
        ax.set_ylim(-0.4, 1.55)
        ax.set_aspect("equal")
        ax.set_xlabel("$\\theta_1$")
        ax.set_ylabel("$\\theta_2$")
        clean(ax, xgrid=True, ygrid=True)
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(8.0, 6.5), layout="constrained")
        X, Y = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-2, 2, 300))
        Z = X**2 + 12 * Y**2
        ax.contour(X, Y, Z, levels=[0.5, 2, 5, 10, 20, 35], colors=[GRID] * 6)
        p_full = np.array([2.5, 1.4])
        p_sgd = p_full.copy()
        tf, ts = [p_full.copy()], [p_sgd.copy()]
        for _ in range(28):
            p_full -= 0.075 * np.array([2 * p_full[0], 24 * p_full[1]])
            p_sgd -= 0.075 * (np.array([2 * p_sgd[0], 24 * p_sgd[1]]) + RNG.normal(0, 2.0, 2))
            tf.append(p_full.copy())
            ts.append(p_sgd.copy())
        tf, ts = np.array(tf), np.array(ts)
        ax.plot(tf[:, 0], tf[:, 1], color=RED, linewidth=1.8, marker=".", ms=3)
        ax.plot(ts[:, 0], ts[:, 1], color=BLUE, linewidth=1.2, marker=".", ms=3)
        direct(ax, tf[6, 0], tf[6, 1], "полный", RED)
        direct(ax, ts[11, 0], ts[11, 1], "SGD", BLUE)
        ax.set_xlabel("$\\theta_1$")
        ax.set_ylabel("$\\theta_2$")
        clean(ax)
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    epoch = np.linspace(0, 100, 300)
    train = 0.9 * np.exp(-epoch / 25) + 0.05
    val = 0.65 * np.exp(-epoch / 20) + 0.18 + 0.000065 * (epoch - 42) ** 2
    best = epoch[np.argmin(val)]
    ax.plot(epoch, train, color=BLUE, linewidth=2.2)
    ax.plot(epoch, val, color=RED, linewidth=2.2)
    ax.axvline(best, color=INK, linestyle=(0, (3, 3)), linewidth=1.2)
    ax.axvspan(best, 100, color=RED, alpha=0.06)
    direct(ax, 63, np.interp(63, epoch, train), "train", BLUE)
    direct(ax, 63, np.interp(63, epoch, val), "validation", RED)
    direct(ax, best, val.min() + 0.08, "checkpoint", INK, ha="center")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("loss")
    clean(ax, ygrid=True)
    return fig


def lesson62(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(8.0, 6.5), layout="constrained")
        X, Y = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-2, 2, 300))
        Z = X**2 + 10 * Y**2
        ax.contour(X, Y, Z, levels=[0.5, 2, 5, 10, 20, 35], colors=[GRID] * 6)
        p0 = np.array([2.5, 1.5])
        paths = {}
        for mode, color in [("SGD", RED), ("Momentum", BLUE)]:
            p = p0.copy()
            v = np.zeros(2)
            path = [p.copy()]
            for _ in range(30):
                g = np.array([2 * p[0], 20 * p[1]])
                if mode == "Momentum":
                    v = 0.75 * v + g
                    step = v
                else:
                    step = g
                p -= 0.075 * step
                path.append(p.copy())
            paths[mode] = np.array(path)
            ax.plot(paths[mode][:, 0], paths[mode][:, 1], color=color, linewidth=1.8, marker=".", ms=3)
        direct(ax, paths["SGD"][6, 0], paths["SGD"][6, 1], "SGD", RED)
        direct(ax, paths["Momentum"][10, 0], paths["Momentum"][10, 1], "Momentum", BLUE)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        clean(ax)
        return fig
    if index == 2:
        fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.8), layout="constrained", sharex=True)
        t = np.arange(0, 70)
        g2 = np.zeros_like(t, dtype=float)
        g2[20] = 25
        adagrad = np.cumsum(g2)
        rms = np.zeros_like(g2)
        for i in range(1, len(t)):
            rms[i] = 0.9 * rms[i - 1] + 0.1 * g2[i]
        axes[0].stem(t, g2, linefmt=GOLD, markerfmt=" ", basefmt=" ")
        axes[0].plot(t, adagrad, color=RED, linewidth=2)
        axes[0].plot(t, rms, color=BLUE, linewidth=2)
        direct(axes[0], 47, adagrad[47], "AdaGrad: помнит", RED)
        direct(axes[0], 31, rms[31], "RMSProp: забывает", BLUE)
        axes[1].plot(t, 1 / np.sqrt(adagrad + 1), color=RED, linewidth=2)
        axes[1].plot(t, 1 / np.sqrt(rms + 1), color=BLUE, linewidth=2)
        axes[0].set_ylabel("накопитель $v$")
        axes[1].set_ylabel("$1/\\sqrt{v+1}$")
        axes[1].set_xlabel("шаг $t$")
        for ax in axes:
            clean(ax, ygrid=True)
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    t = np.arange(1, 31)
    beta = 0.9
    raw = 1 - beta**t
    corrected = raw / (1 - beta**t)
    ax.plot(t, raw, color=BLUE, linewidth=2.2)
    ax.plot(t, corrected, color=RED, linewidth=2.2)
    direct(ax, 14, raw[13] - 0.05, "$m_t$", BLUE)
    direct(ax, 16, 1.035, "$\\widehat m_t$", RED)
    ax.axhline(1, color=INK, linestyle=(0, (3, 3)), linewidth=1)
    ax.set_xlabel("шаг $t$")
    ax.set_ylabel("оценка постоянного градиента $g=1$")
    ax.set_ylim(0, 1.12)
    clean(ax, ygrid=True)
    return fig


def lesson63(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        true = 0.1
        data = []
        for m in [1, 10, 100, 1000]:
            mins = []
            for _ in range(300):
                mins.append(np.min(true + RNG.normal(0, 0.014, m)))
            data.append(mins)
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, color in zip(bp["boxes"], [GREEN, BLUE, GOLD, RED]):
            patch.set_facecolor(color)
            patch.set_alpha(0.28)
            patch.set_edgecolor(color)
        for median in bp["medians"]:
            median.set_color(INK)
        ax.axhline(true, color=INK, linestyle=(0, (3, 3)), linewidth=1.2)
        ax.set_xticklabels(["1", "10", "100", "1000"])
        ax.set_xlabel("число кандидатов $M$")
        ax.set_ylabel("minimum validation risk")
        clean(ax, ygrid=True)
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        M = np.logspace(0, 6, 400)
        delta = 0.05
        for n, color in [(500, RED), (2000, BLUE), (10000, GREEN)]:
            penalty = np.sqrt(np.log(2 * M / delta) / (2 * n))
            ax.semilogx(M, penalty, color=color, linewidth=2)
            direct(ax, 2.5e5, np.interp(2.5e5, M, penalty), f"$n={n}$", color)
        ax.set_xlabel("кандидатов $M$")
        ax.set_ylabel("$\\sqrt{\\log(2M/\\delta)/(2n)}$")
        clean(ax, xgrid=True, ygrid=True)
        return fig
    fig = canvas()
    ax = diagram_axes(fig)
    colors = [BLUE, GREEN, GOLD, VIOLET, RED]
    for i, color in enumerate(colors):
        x = 0.05 + i * 0.18
        box(ax, (x, 0.68), 0.14, 0.11, f"fold {i+1}", color=color, size=10)
    box(ax, (0.08, 0.34), 0.58, 0.17, "внутренний CV:\nвыбор степени и $\\lambda$", color=BLUE)
    box(ax, (0.76, 0.34), 0.16, 0.17, "внешний\ntest", color=RED)
    for i in range(4):
        arrow(ax, (0.12 + i * 0.18, 0.68), (0.37, 0.51), color=colors[i], shrink=5)
    arrow(ax, (0.77, 0.735), (0.84, 0.51), color=RED, shrink=5)
    label(ax, 0.5, 0.17, "внешний fold оценивает всю процедуру выбора", MUTED, 11.5, "center")
    return fig


def lesson64(index: int):
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        box(ax, (0.40, 0.40), 0.20, 0.18, "сервер\n$w_t\\to w_{t+1}$", color=INK)
        clients = [(0.08, 0.70), (0.40, 0.76), (0.72, 0.70), (0.08, 0.15), (0.72, 0.15)]
        for i, (x, y) in enumerate(clients, 1):
            box(ax, (x, y), 0.16, 0.12, f"клиент {i}\n$D_{i}$", color=[BLUE, GREEN, GOLD, VIOLET, RED][i - 1], size=9.8)
            arrow(ax, (0.50, 0.49), (x + 0.08, y + 0.06), color=FAINT, curve=0.12 if x < 0.4 else -0.12)
            arrow(ax, (x + 0.08, y + 0.02), (0.50, 0.45), color=[BLUE, GREEN, GOLD, VIOLET, RED][i - 1], curve=-0.12 if x < 0.4 else 0.12)
        label(ax, 0.5, 0.93, "вниз: модель · вверх: локальные обновления", MUTED, 11, "center")
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(8.0, 6.4), layout="constrained")
        X, Y = np.meshgrid(np.linspace(-3, 5, 320), np.linspace(-3, 4, 300))
        F1 = (X + 1.2) ** 2 + 1.8 * (Y - 0.7) ** 2
        F2 = 1.2 * (X - 2.5) ** 2 + (Y + 0.8) ** 2
        ax.contour(X, Y, F1, levels=[0.8, 2, 5, 10], colors=[BLUE] * 4, alpha=0.45)
        ax.contour(X, Y, F2, levels=[0.8, 2, 5, 10], colors=[RED] * 4, alpha=0.45)
        start = np.array([0.4, 2.6])
        paths = []
        for center, color in [(np.array([-1.2, 0.7]), BLUE), (np.array([2.5, -0.8]), RED)]:
            p = start.copy()
            path = [p.copy()]
            for _ in range(4):
                p -= 0.28 * 2 * (p - center)
                path.append(p.copy())
            path = np.array(path)
            paths.append(path)
            ax.plot(path[:, 0], path[:, 1], color=color, marker="o", ms=3, linewidth=1.8)
        avg = (paths[0][-1] + paths[1][-1]) / 2
        ax.scatter(*avg, color=VIOLET, s=65, zorder=5)
        direct(ax, avg[0], avg[1] - 0.35, "FedAvg", VIOLET, ha="center")
        ax.scatter(*start, color=INK, s=38)
        ax.set_xlabel("$w_1$")
        ax.set_ylabel("$w_2$")
        clean(ax)
        return fig
    fig, ax = plt.subplots(figsize=(8.0, 6.6), layout="constrained")
    triangle = np.array([[0.12, 0.12], [0.88, 0.12], [0.50, 0.84]])
    ax.add_patch(Polygon(triangle, fill=False, edgecolor=INK, linewidth=1.8))
    label(ax, 0.08, 0.08, "мало\nбайтов", BLUE, 12, "center", "top")
    label(ax, 0.92, 0.08, "редкая\nсвязь", GOLD, 12, "center", "top")
    label(ax, 0.50, 0.89, "малый\ndrift", GREEN, 12, "center", "bottom")
    arrow(ax, (0.22, 0.20), (0.45, 0.70), color=GREEN)
    label(ax, 0.29, 0.48, "control\nvariates", GREEN, 10, "center")
    arrow(ax, (0.78, 0.20), (0.55, 0.70), color=RED)
    label(ax, 0.71, 0.48, "меньше local\nepochs", RED, 10, "center")
    arrow(ax, (0.30, 0.16), (0.70, 0.16), color=VIOLET)
    label(ax, 0.50, 0.22, "quantization · sparsification", VIOLET, 10.5, "center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig


BUILDERS = {
    "41": lesson41,
    "42": lesson42,
    "43": lesson43,
    "44": lesson44,
    "45": lesson45,
    "46": lesson46,
    "47": lesson47,
    "48": lesson48,
    "49": lesson49,
    "50": lesson50,
    "51": lesson51,
    "52": lesson52,
    "53": lesson53,
    "54": lesson54,
    "55": lesson55,
    "56": lesson56,
    "57": lesson57,
    "58": lesson58,
    "59": lesson59,
    "60": lesson60,
    "61": lesson61,
    "62": lesson62,
    "63": lesson63,
    "64": lesson64,
}


def main():
    for lesson, builder in BUILDERS.items():
        lesson_dir = OUT / lesson
        lesson_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, 4):
            fig = builder(index)
            fig.savefig(
                lesson_dir / f"figure-{index}.svg",
                format="svg",
                bbox_inches="tight",
                pad_inches=0.16,
                metadata={"Title": f"Учебник ИИ, урок {lesson}, рисунок {index}"},
            )
            plt.close(fig)
    print(f"Generated {len(BUILDERS) * 3} SVG figures in {OUT}")


if __name__ == "__main__":
    main()
