"""Deterministic figures for lesson 35: unsupervised learning and autoencoders.

A digit compressed through a narrow code and reconstructed (blurry at few
dimensions, sharp at many), the reconstruction error against code size, and
the 2D latent space where digits cluster without any labels. On real
handwritten digits. Numbers reproduced and asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "35"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "35"

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


def data():
    D = load_digits()
    return D.data / 16.0, D.target, D.images


# ---------------------------- fig 35.1: reconstruction at several code sizes
def fig_reconstruction() -> None:
    X, y, images = data()
    i = np.where(y == 3)[0][0]
    fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.7))
    axes[0].imshow(X[i].reshape(8, 8), cmap="gray_r"); axes[0].set_title("оригинал\n64 числа", fontsize=10.5); axes[0].axis("off")
    for ax, k in zip(axes[1:], [2, 8, 16, 32]):
        p = PCA(n_components=k).fit(X)
        rec = p.inverse_transform(p.transform(X[i:i + 1]))[0]
        ax.imshow(rec.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        labels = {2: "размыто", 8: "грубо", 16: "почти чётко", 32: "чётко"}
        ax.set_title(f"код {k}\n{labels[k]}", fontsize=10.5)
        ax.axis("off")
    fig.suptitle("Сжать в короткий код и восстановить: чем длиннее код, тем чётче", y=1.05, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "reconstruction.png")
    print("reconstruction drawn")


# ---------------------------- fig 35.2: reconstruction error vs code size
def fig_error() -> None:
    X, y, images = data()
    ks = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    err, var = [], []
    for k in ks:
        p = PCA(n_components=k).fit(X)
        rec = p.inverse_transform(p.transform(X))
        err.append(np.mean((X - rec) ** 2)); var.append(p.explained_variance_ratio_.sum())
    err, var = np.array(err), np.array(var)
    print(f"error: k8={err[3]:.4f}, k16={err[4]:.4f}, k64={err[-1]:.4f}")
    assert abs(err[3] - 0.0239) < 0.002 and err[-1] < 1e-6
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(ks, err, color=RED, lw=2.4, marker="o", markersize=5, label="ошибка реконструкции")
    ax.fill_between(ks, 0, err, color=RED, alpha=0.07)
    ax.set_xlabel("размер кода (число измерений)")
    ax.set_ylabel("ошибка реконструкции (MSE)", color=RED)
    ax2 = ax.twinx()
    ax2.plot(ks, var * 100, color=BLUE, lw=2.0, ls=(0, (4, 3)), marker="s", markersize=4, label="сохранённая дисперсия")
    ax2.set_ylabel("сохранённая дисперсия, %", color=BLUE)
    ax2.set_ylim(0, 105)
    ax.set_ylim(0, 0.07)
    ax.annotate("почти всё восстановлено\nуже при 16 из 64", xy=(16, err[4]), xytext=(26, 0.04),
                fontsize=10, color=MUTED, arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    ax.set_title("Короче код — грубее восстановление")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    save(fig, OUT / "error.png")


# ---------------------------- fig 35.3: 2D latent space, clusters without labels
def fig_latent() -> None:
    X, y, images = data()
    Z = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    cmap = plt.get_cmap("tab10")
    for d in range(10):
        m = y == d
        ax.scatter(Z[m, 0], Z[m, 1], s=12, color=cmap(d), alpha=0.7, label=str(d))
    ax.set_xlabel("первая ось кода"); ax.set_ylabel("вторая ось кода")
    ax.set_title("Двумерный код: цифры сами собираются в группы")
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2, title="цифра")
    ax.text(0.02, 0.02, "цвет — истинная цифра, но код построен\nбез единой метки: структуру нашли сами данные",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    print(f"latent: 2D variance {PCA(n_components=2).fit(X).explained_variance_ratio_.sum():.3f}")
    save(fig, OUT / "latent.png")


# ------------------------------------------------ margins
def side_bottleneck() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    widths = [3.0, 2.0, 0.7, 2.0, 3.0]
    cols = [BLUE, BLUE, INK, GREEN, GREEN]
    labs = ["вход", "", "код", "", "выход"]
    x = 0.6
    for w, col, lab in zip(widths, cols, labs):
        ax.add_patch(Rectangle((x, 3 - w / 2), 0.7, w, fc=WASH, ec=col, lw=1.6))
        if lab:
            ax.text(x + 0.35, 3 - w / 2 - 0.35, lab, ha="center", va="top", fontsize=9, color=col)
        x += 2.3
    ax.text(6, 5.4, "узкий слой заставляет сжимать", ha="center", fontsize=9, color=INK)
    ax.set_title("бутылочное горло", fontsize=10.5)
    save(fig, SIDE / "bottleneck.png")


def side_anomaly() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    rng = np.random.default_rng(5)
    normal = rng.normal(0.02, 0.008, 60)
    anom = rng.normal(0.09, 0.01, 6)
    ax.hist(normal, bins=15, color=GREEN, alpha=0.7, label="обычные")
    ax.hist(anom, bins=6, color=RED, alpha=0.8, label="аномалии")
    ax.axvline(0.055, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.057, ax.get_ylim()[1] * 0.7, "порог", fontsize=8.5, color=INK)
    ax.set_xlabel("ошибка реконструкции", fontsize=9); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_title("аномалия = плохо сжимается", fontsize=10)
    save(fig, SIDE / "anomaly.png")


def side_denoise() -> None:
    X, y, images = data()
    i = np.where(y == 7)[0][0]
    rng = np.random.default_rng(3)
    noisy = np.clip(X[i] + rng.normal(0, 0.25, 64), 0, 1)
    p = PCA(n_components=12).fit(X)
    clean = p.inverse_transform(p.transform(noisy.reshape(1, -1)))[0]
    fig, axes = plt.subplots(1, 3, figsize=(4.2, 1.9))
    for ax, im, t in [(axes[0], noisy, "шумный вход"), (axes[1], None, ""), (axes[2], clean, "чистый выход")]:
        if im is None:
            ax.axis("off"); ax.annotate("", xy=(0.8, 0.5), xytext=(0.2, 0.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6, mutation_scale=12)); continue
        ax.imshow(np.clip(im, 0, 1).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1); ax.set_title(t, fontsize=8.5); ax.axis("off")
    fig.suptitle("деноизинг: убрать шум", y=1.05, fontsize=10)
    fig.tight_layout()
    save(fig, SIDE / "denoise.png")


fig_reconstruction()
fig_error()
fig_latent()
side_bottleneck()
side_anomaly()
side_denoise()
print("lesson 35 figures written")
