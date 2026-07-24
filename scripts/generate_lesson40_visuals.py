"""Deterministic figures for lesson 40: sound, embeddings and recommendations.

Real signal processing: the magnitude spectrum from an actual FFT (peaks land on the
true tone frequencies), the time-frequency tradeoff from a real STFT at short vs long
windows, and a 2D map of synthesized sounds embedded from real spectral features.
Signals are synthesized to isolate each idea; the spectral maths is genuine (scipy).
Numbers reproduced and asserted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "40"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "40"

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

FS = 8000


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------- fig 40.1: waveform + real FFT spectrum
def fig_three_reps() -> None:
    N = 1024
    t = np.arange(N) / FS
    f1, f2 = 500.0, 1250.0
    x = 1.0 * np.sin(2 * np.pi * f1 * t) + 0.6 * np.sin(2 * np.pi * f2 * t)
    w = np.hanning(N)
    X = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(N, 1 / FS)
    peaks = freqs[np.argsort(X)[::-1][:6]]
    print(f"three_reps: top peaks {sorted(peaks)[:4]} Hz (want ~500, 1250)")
    assert min(abs(peaks - f1)) < 20 and min(abs(peaks - f2)) < 20
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8.8, 5.4))
    a0.plot(t[:400] * 1000, x[:400], color=BLUE, lw=1.4)
    a0.set_xlabel("время, мс"); a0.set_ylabel("амплитуда")
    a0.set_title("Волна: сумма двух тонов во времени")
    a0.grid(True, color=GRID, lw=0.4, alpha=0.5); a0.set_axisbelow(True)
    a1.plot(freqs, X / X.max(), color=RED, lw=1.6)
    for f, lab in [(f1, "500 Гц"), (f2, "1250 Гц")]:
        k = np.argmin(abs(freqs - f))
        a1.annotate(lab, xy=(f, X[k] / X.max()), xytext=(f + 120, 0.7),
                    fontsize=10, color=INK, arrowprops=dict(arrowstyle="->", color=LINE, lw=1.0))
    a1.set_xlabel("частота, Гц"); a1.set_ylabel("сила частоты")
    a1.set_xlim(0, 2500)
    a1.set_title("Спектр (модуль ДПФ): пики ровно на этих частотах")
    a1.grid(True, color=GRID, lw=0.4, alpha=0.5); a1.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "three_reps.png")


# ---------------------------------------- fig 40.2: time-frequency tradeoff (real STFT)
def fig_tradeoff() -> None:
    dur = 0.5
    t = np.arange(int(dur * FS)) / FS
    x = np.sin(2 * np.pi * 800 * t) + np.sin(2 * np.pi * 900 * t)   # two close tones
    click = np.zeros_like(t); click[int(0.25 * FS):int(0.25 * FS) + 20] = 3.0  # sharp event
    x = x + click
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    for ax, nper, tit in [(a0, 64, "короткое окно"), (a1, 512, "длинное окно")]:
        f, tt, Z = sig.stft(x, FS, window="hann", nperseg=nper, noverlap=nper * 3 // 4)
        ax.pcolormesh(tt * 1000, f, np.abs(Z), shading="gouraud", cmap="magma")
        ax.set_ylim(0, 1600); ax.set_xlabel("время, мс")
        ax.set_title(tit + ("\nсобытие резкое, тоны слиты" if nper == 64 else "\nтоны разделены, событие размыто"), fontsize=11)
    a0.set_ylabel("частота, Гц")
    fig.suptitle("Компромисс времени и частоты: одно окно не даёт и то, и другое", y=1.02, fontsize=13.5)
    fig.tight_layout()
    save(fig, OUT / "tradeoff.png")
    print("tradeoff drawn (real STFT)")


# ---------------------------------------- synth sound bank for the embedding map
def synth(kind, seed, dur=0.5):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * FS)) / FS
    if kind == "низкий тон":
        f = rng.uniform(180, 260); x = np.sin(2 * np.pi * f * t)
    elif kind == "высокий тон":
        f = rng.uniform(1400, 2000); x = np.sin(2 * np.pi * f * t)
    elif kind == "аккорд":
        base = rng.uniform(220, 300); x = sum(np.sin(2 * np.pi * base * r * t) for r in (1, 1.26, 1.5))
    elif kind == "щелчки":
        x = np.zeros_like(t)
        for c in rng.integers(0, len(t), 12):
            x[c:c + 15] += rng.uniform(1, 3)
    elif kind == "шум":
        x = rng.standard_normal(len(t))
    else:  # свип
        f0, f1 = 300, 1800; x = sig.chirp(t, f0, dur, f1)
    return x / (np.abs(x).max() + 1e-9)


def features(x):
    f, _, Z = sig.stft(x, FS, window="hann", nperseg=256, noverlap=192)
    S = np.abs(Z) + 1e-9
    centroid = np.sum(f[:, None] * S, 0) / np.sum(S, 0)
    p = S / np.sum(S, 0)
    bw = np.sqrt(np.sum(((f[:, None] - centroid) ** 2) * p, 0))
    csum = np.cumsum(S, 0); rolloff = f[np.argmax(csum >= 0.85 * csum[-1], 0)]
    zcr = np.mean(np.abs(np.diff(np.sign(x))) > 0)
    flat = np.exp(np.mean(np.log(S), 0)) / (np.mean(S, 0) + 1e-9)
    return np.array([centroid.mean(), centroid.std(), bw.mean(), rolloff.mean(), zcr, flat.mean()])


# ---------------------------------------- fig 40.3: embedding map of sounds (real features + PCA)
def fig_embedding() -> None:
    kinds = ["низкий тон", "высокий тон", "аккорд", "щелчки", "шум", "свип"]
    cols = {"низкий тон": BLUE, "высокий тон": VIOLET, "аккорд": GREEN, "щелчки": GOLD, "шум": RED, "свип": INK}
    X, labels = [], []
    for k in kinds:
        for s in range(8):
            X.append(features(synth(k, s + hash(k) % 100)))
            labels.append(k)
    X = np.array(X); labels = np.array(labels)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Z = PCA(n_components=2, random_state=0).fit_transform(Xs)
    # nearest neighbour purity: same-kind share among 3 nearest
    from scipy.spatial.distance import cdist
    D = cdist(Z, Z); np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :3]
    purity = np.mean([np.mean(labels[nn[i]] == labels[i]) for i in range(len(Z))])
    print(f"embedding: 3-NN purity {purity:.2f}")
    assert purity > 0.6
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for k in kinds:
        m = labels == k
        ax.scatter(Z[m, 0], Z[m, 1], s=52, color=cols[k], alpha=0.75, label=k, edgecolors=PAPER, linewidths=0.6)
    ax.set_xlabel("первая ось эмбеддинга"); ax.set_ylabel("вторая ось эмбеддинга")
    ax.set_title(f"Карта звуков: похожие сближаются (чистота 3 соседей {purity:.0%})")
    ax.legend(loc="best", frameon=False, fontsize=10, title="тип звука")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "embedding.png")


# ---------------------------------------- margins
def side_leakage() -> None:
    N = 512
    t = np.arange(N) / FS
    x = np.sin(2 * np.pi * 617 * t)   # frequency between bins -> leakage
    freqs = np.fft.rfftfreq(N, 1 / FS)
    Xr = np.abs(np.fft.rfft(x)); Xh = np.abs(np.fft.rfft(x * np.hanning(N)))
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.plot(freqs, Xr / Xr.max(), color=RED, lw=1.4, label="прямоугольное")
    ax.plot(freqs, Xh / Xh.max(), color=BLUE, lw=1.4, label="окно Ханна")
    ax.set_xlim(300, 950); ax.set_yscale("log"); ax.set_ylim(1e-3, 1.5)
    ax.set_xlabel("частота, Гц", fontsize=9); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title("утечка спектра и окно", fontsize=9.5)
    save(fig, SIDE / "leakage.png")
    print("leakage drawn")


def side_cosine() -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.set_aspect("equal"); ax.set_xlim(-0.3, 3.2); ax.set_ylim(-0.3, 3.2)
    from matplotlib.patches import FancyArrowPatch
    for v, col, lab in [((1, 1), BLUE, "тихий"), ((3, 3), GREEN, "громкий"), ((2.6, 0.6), RED, "иной тембр")]:
        ax.add_patch(FancyArrowPatch((0, 0), v, arrowstyle="-|>", mutation_scale=12, color=col, lw=2.0))
        ax.annotate(lab, v, fontsize=8.5, color=col, xytext=(4, 3), textcoords="offset points")
    ax.plot([0, 3], [0, 3], color=LINE, lw=0.6, ls=(0, (3, 3)))
    ax.text(1.5, 2.9, "cos = 1: одно направление", fontsize=8, color=MUTED)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("косинус игнорирует громкость", fontsize=9)
    save(fig, SIDE / "cosine.png")


def side_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(4.3, 2.0))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 3)
    from matplotlib.patches import FancyArrowPatch, Rectangle
    steps = ["волна", "СТФТ", "признаки", "PCA", "сосед"]
    cols = [BLUE, VIOLET, GREEN, GOLD, RED]
    for i, (s, c) in enumerate(zip(steps, cols)):
        x = 0.3 + i * 2.35
        ax.add_patch(Rectangle((x, 1.1), 1.7, 0.9, fc=WASH, ec=c, lw=1.5))
        ax.text(x + 0.85, 1.55, s, ha="center", va="center", fontsize=9, color=c)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + 1.7, 1.55), (x + 2.35, 1.55), arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1.2))
    ax.set_title("конвейер аудио-эмбеддинга", fontsize=9.5)
    save(fig, SIDE / "pipeline.png")


fig_three_reps()
fig_tradeoff()
fig_embedding()
side_leakage()
side_cosine()
side_pipeline()
print("lesson 40 figures written")
