"""Deterministic figures for lesson 36: matrices as transformations of space.

The columns of a matrix are the images of the basis; the determinant is signed
area; a circle becomes an ellipse whose semi-axes are the singular values, so the
condition number is the eccentricity of that ellipse. Grounded on the unit square,
a letter F, and a real photograph warped by a matrix. Numbers asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch, Circle
import matplotlib.cbook as cbook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "36"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "36"

PAPER = "#fffef9"; INK = "#171915"; MUTED = "#6e726a"; FAINT = "#969990"
GRID = "#deddd4"; LINE = "#c9c8be"; BLUE = "#315f8c"; RED = "#b94a3b"
GREEN = "#38735d"; GOLD = "#a57920"; VIOLET = "#6f5a8f"; WASH = "#f5f3ea"

mpl.rcParams.update({
    "font.family": "PT Sans", "font.size": 12, "axes.titlesize": 15,
    "axes.labelsize": 12, "axes.edgecolor": LINE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.16, "mathtext.fontset": "dejavuserif",
})


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def draw_grid(ax, A, lim=3, color=LINE, lw=0.7):
    ks = np.arange(-lim, lim + 1)
    ts = np.linspace(-lim, lim, 50)
    for k in ks:
        v = np.array([[k, k], [-lim, lim]])
        w = A @ v
        ax.plot(w[0], w[1], color=color, lw=lw, alpha=0.9, zorder=1)
        h = np.array([[-lim, lim], [k, k]])
        w = A @ h
        ax.plot(w[0], w[1], color=color, lw=lw, alpha=0.9, zorder=1)


def arrow(ax, tip, color, label, base=(0, 0), off=(0.12, 0.12)):
    ax.add_patch(FancyArrowPatch(base, tip, arrowstyle="-|>", mutation_scale=15,
                                 color=color, lw=2.2, zorder=5))
    ax.annotate(label, xy=tip, xytext=(tip[0] + off[0], tip[1] + off[1]),
                color=color, fontsize=12, zorder=6)


# --------------------------------------------- fig 36.1: columns are images of basis
def fig_basis() -> None:
    A = np.array([[2.0, 1.0], [0.0, 1.0]])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.2, 5.1))
    for ax in (ax0, ax1):
        ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.8); ax.set_ylim(-3.2, 3.8)
        ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xticks([]); ax.set_yticks([])
    # left: original
    draw_grid(ax0, np.eye(2), color=GRID)
    ax0.add_patch(Polygon([[0, 0], [1, 0], [1, 1], [0, 1]], closed=True, fc=BLUE, ec=BLUE, alpha=0.14, zorder=2))
    arrow(ax0, (1, 0), RED, "$e_1$"); arrow(ax0, (0, 1), BLUE, "$e_2$")
    arrow(ax0, (1, 1), GREEN, "$x=e_1+e_2$", off=(0.1, 0.15))
    ax0.set_title("до преобразования")
    # right: transformed
    draw_grid(ax1, A, color=LINE)
    a1 = A @ np.array([1, 0]); a2 = A @ np.array([0, 1]); ax_ = A @ np.array([1, 1])
    ax1.add_patch(Polygon([[0, 0], a1, a1 + a2, a2], closed=True, fc=GREEN, ec=GREEN, alpha=0.14, zorder=2))
    arrow(ax1, a1, RED, "$Ae_1=(2,0)$", off=(-0.2, -0.42))
    arrow(ax1, a2, BLUE, "$Ae_2=(1,1)$", off=(0.12, 0.12))
    arrow(ax1, ax_, GREEN, "$Ax=(3,1)$", off=(0.1, 0.16))
    ax1.set_title("после A с колонками (2,0) и (1,1)")
    assert np.allclose(ax_, [3, 1])
    fig.suptitle("Столбцы матрицы — это образы базисных векторов", y=1.02, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "basis.png")
    print("basis drawn, Ax =", ax_)


# --------------------------------------------- fig 36.2: determinant = signed area
def fig_determinant() -> None:
    mats = [
        (np.array([[2.0, 0.0], [0.0, 1.0]]), "$\\det=2$", "площадь ×2"),
        (np.array([[0.0, 1.0], [1.0, 0.0]]), "$\\det=-1$", "зеркало, площадь ×1"),
        (np.array([[1.0, 1.0], [1.0, 1.02]]), "$\\det=0{,}02$", "почти в линию"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    for ax, (A, tit, note) in zip(axes, mats):
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
        img = (A @ sq.T).T
        det = np.linalg.det(A)
        col = GREEN if det > 0 else RED
        ax.add_patch(Polygon([[0, 0], [1, 0], [1, 1], [0, 1]], closed=True, fc=GRID, ec=LINE, alpha=0.4, zorder=1))
        ax.add_patch(Polygon(img, closed=True, fc=col, ec=col, alpha=0.22, zorder=2))
        # vertex order arrows (orientation)
        for i in range(4):
            p, q = img[i], img[(i + 1) % 4]
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=col, lw=1.4, zorder=4))
        lim = 2.6
        ax.set_xlim(-0.6, lim); ax.set_ylim(-0.6, lim)
        ax.set_title(tit, fontsize=13)
        ax.text(0.5, -0.45, note, ha="center", fontsize=10, color=MUTED)
        assert abs(abs(det) - abs(np.linalg.det(A))) < 1e-9
    fig.suptitle("Определитель — ориентированная площадь единичного квадрата", y=1.03, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "determinant.png")
    print("determinant drawn")


# --------------------------------------------- fig 36.3: circle -> ellipse, singular values
def fig_singular() -> None:
    A = np.array([[2.0, 1.0], [0.0, 1.0]])
    U, s, Vt = np.linalg.svd(A)
    kappa = s[0] / s[1]
    print(f"singular values {s.round(3)}, kappa {kappa:.3f}")
    assert abs(s[0] - 2.288) < 0.01 and abs(s[1] - 0.874) < 0.01
    th = np.linspace(0, 2 * np.pi, 200)
    circ = np.array([np.cos(th), np.sin(th)])
    ell = A @ circ
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.set_aspect("equal")
    ax.plot(circ[0], circ[1], color=LINE, lw=1.6, ls=(0, (4, 3)), label="единичная окружность")
    ax.plot(ell[0], ell[1], color=BLUE, lw=2.4, label="её образ — эллипс")
    # semi-axes = singular directions * singular values
    for i, col in enumerate([RED, GREEN]):
        d = U[:, i] * s[i]
        ax.add_patch(FancyArrowPatch((0, 0), d, arrowstyle="-|>", mutation_scale=16, color=col, lw=2.4, zorder=5))
    ax.annotate(f"$\\sigma_1={s[0]:.2f}$", xy=U[:, 0] * s[0], xytext=(U[0, 0] * s[0] + 0.1, U[1, 0] * s[0] + 0.2), color=RED, fontsize=12)
    ax.annotate(f"$\\sigma_2={s[1]:.2f}$", xy=U[:, 1] * s[1], xytext=(U[0, 1] * s[1] - 1.2, U[1, 1] * s[1] + 0.15), color=GREEN, fontsize=12)
    ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.4, 2.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.set_title(f"Окружность становится эллипсом: полуоси равны $\\sigma_1,\\sigma_2$, а $\\kappa={kappa:.2f}$")
    save(fig, OUT / "singular.png")


# --------------------------------------------- margin: AB != BA on letter F
def side_compose() -> None:
    F = np.array([[0, 0], [0, 3], [2, 3], [2, 2.3], [0.7, 2.3], [0.7, 1.7],
                  [1.6, 1.7], [1.6, 1.0], [0.7, 1.0], [0.7, 0], [0, 0]], float).T
    F = F - F.mean(1, keepdims=True)
    R = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
    S = np.array([[1.8, 0.0], [0.0, 0.7]])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(4.2, 2.4))
    for ax, M, tit in [(a0, R @ S, "$RS$: сжать, потом повернуть"), (a1, S @ R, "$SR$: повернуть, потом сжать")]:
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(Polygon(F.T, closed=True, fc=GRID, ec=LINE, alpha=0.5))
        G = M @ F
        ax.add_patch(Polygon(G.T, closed=True, fc=BLUE, ec=BLUE, alpha=0.3))
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_title(tit, fontsize=8.5)
    fig.suptitle("порядок важен: $AB\\neq BA$", y=1.04, fontsize=10)
    fig.tight_layout()
    save(fig, SIDE / "compose.png")


# --------------------------------------------- margin: eigenvectors keep direction
def side_eigen() -> None:
    A = np.array([[2.0, 1.0], [0.0, 1.0]])
    w, V = np.linalg.eig(A)
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    draw_grid(ax, A, lim=2, color=LINE, lw=0.6)
    ax.axhline(0, color=GRID, lw=0.8); ax.axvline(0, color=GRID, lw=0.8)
    for i, col in enumerate([RED, GREEN]):
        v = V[:, i].real; v = v / np.linalg.norm(v) * 2.4
        ax.plot([-v[0], v[0]], [-v[1], v[1]], color=col, lw=2.0)
        ax.annotate(f"$\\lambda={w[i].real:.0f}$", xy=v * 0.7, color=col, fontsize=10)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
    ax.set_title("собственные прямые не поворачиваются", fontsize=9.5)
    save(fig, SIDE / "eigen.png")
    print("eigen: values", w.real.round(2))


# --------------------------------------------- margin: real photo warp (inverse gather vs forward scatter)
def side_warp() -> None:
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        img = plt.imread(f)
    g = img[..., :3].mean(2)[::6, ::6] / 255.0  # downsample grayscale
    h, w = g.shape
    ang = 0.5
    A = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    cy, cx = (h - 1) / 2, (w - 1) / 2
    # inverse gather (clean)
    out = np.zeros_like(g)
    ys, xs = np.mgrid[0:h, 0:w]
    coords = np.stack([xs - cx, ys - cy]).reshape(2, -1)
    src = np.linalg.inv(A) @ coords
    sx = np.rint(src[0] + cx).astype(int); sy = np.rint(src[1] + cy).astype(int)
    m = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    flat = out.reshape(-1)
    flat[m] = g[sy[m], sx[m]]
    gather = flat.reshape(h, w)
    # forward scatter (holes)
    scat = np.full_like(g, -1.0)
    dst = A @ np.stack([xs - cx, ys - cy]).reshape(2, -1)
    dx = np.rint(dst[0] + cx).astype(int); dy = np.rint(dst[1] + cy).astype(int)
    m2 = (dx >= 0) & (dx < w) & (dy >= 0) & (dy < h)
    scat[dy[m2], dx[m2]] = g.reshape(-1)[m2]
    holes = float((scat < 0).mean())
    scat_show = np.where(scat < 0, 0.5, scat)
    print(f"warp: forward-scatter holes {holes:.2%}")
    fig, axes = plt.subplots(1, 3, figsize=(4.4, 1.9))
    for ax, im, t in [(axes[0], g, "оригинал"), (axes[1], gather, "обратный проход"), (axes[2], scat_show, "прямой проход")]:
        ax.imshow(im, cmap="gray", vmin=0, vmax=1); ax.set_title(t, fontsize=8); ax.axis("off")
    fig.suptitle("поворот фото: обратный проход без дыр", y=1.06, fontsize=9)
    fig.tight_layout()
    save(fig, SIDE / "warp.png")


fig_basis()
fig_determinant()
fig_singular()
side_compose()
side_eigen()
side_warp()
print("lesson 36 figures written")
