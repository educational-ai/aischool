"""Deterministic figures for lesson 87: generative editing, masks and deepfake detection.

Everything is computed from the REAL photograph scripts/data/vision-photo.jpg.

1. A mask is a contract: an "inpainting" edit with feathering plus global harmonisation
   provably changes pixels OUTSIDE the declared mask; the difference map measures it.
2. A spectral detector (radial power spectrum + logistic regression) separates real
   patches from patches produced by three concrete resampling pipelines.
3. Leave-one-generator-out: the same detector collapses on a generator it never saw.
4. JPEG recompression destroys the fingerprint.
5. Base rate turns a high-accuracy detector into a flood of false alarms.
6. Cost-optimal threshold and how confidently the detector is wrong on an unseen generator.

Every number quoted in the lesson text is computed here and asserted.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "scripts" / "data" / "vision-photo.jpg"
OUT = ROOT / "public" / "figures" / "lessons" / "87"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "87"

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

FACTS: dict[str, float | int | str] = {}


def save(fig, path, *, dpi=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------- image helpers
def load_rgb() -> np.ndarray:
    return np.asarray(Image.open(PHOTO).convert("RGB"), dtype=float) / 255.0


def gray(img: np.ndarray) -> np.ndarray:
    return img @ np.array([0.299, 0.587, 0.114])


def gauss_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(3 * sigma))
    t = np.arange(-r, r + 1)
    k = np.exp(-(t ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    k = gauss_kernel(sigma)
    out = a.copy()
    pad = len(k) // 2
    for axis in (0, 1):
        p = np.pad(out, [(pad, pad) if ax == axis else (0, 0) for ax in range(out.ndim)], mode="reflect")
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + out.shape[axis])
            acc += w * p[tuple(sl)]
        out = acc
    return out


def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=float) / 255.0


# ---------------------------------------------------------------- generators
def gen_resample(img: np.ndarray) -> np.ndarray:
    """Downscale by 2 and bicubic back: the classic upsampling fingerprint."""
    h, w = img.shape[:2]
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    small = pil.resize((w // 2, h // 2), Image.BICUBIC)
    return np.asarray(small.resize((w, h), Image.BICUBIC), dtype=float) / 255.0


def gen_smooth_noise(img: np.ndarray) -> np.ndarray:
    """Decoder-like smoothing plus a thin layer of synthetic grain."""
    rng = np.random.default_rng(8701)
    out = np.stack([blur(img[..., c], 0.9) for c in range(3)], axis=-1)
    return np.clip(out + rng.normal(0, 0.012, out.shape), 0, 1)


def gen_sharpen(img: np.ndarray) -> np.ndarray:
    """Over-sharpening: many pipelines push high frequencies to look 'crisp'."""
    out = np.stack([img[..., c] + 0.9 * (img[..., c] - blur(img[..., c], 1.4)) for c in range(3)], axis=-1)
    return np.clip(out, 0, 1)


GENERATORS = [("апсемплинг ×2", gen_resample), ("сглаживание+зерно", gen_smooth_noise),
              ("перерезкость", gen_sharpen)]


# ---------------------------------------------------------------- patches & features
PATCH = 64
STRIDE = 32


def patch_coords(shape, half: str):
    h, w = shape[:2]
    y0, y1 = (0, h // 2) if half == "top" else (h // 2, h)
    ys = range(y0, y1 - PATCH + 1, STRIDE)
    xs = range(0, w - PATCH + 1, STRIDE)
    return [(y, x) for y in ys for x in xs]


HANN = np.outer(np.hanning(PATCH), np.hanning(PATCH))
_yy, _xx = np.mgrid[0:PATCH, 0:PATCH]
_fy = np.minimum(_yy, PATCH - _yy)
_fx = np.minimum(_xx, PATCH - _xx)
_R = np.sqrt(_fy ** 2 + _fx ** 2)
NBINS = 24
_BIN = np.clip((_R / (PATCH / 2) * NBINS).astype(int), 0, NBINS - 1)


def radial_spectrum(patch: np.ndarray) -> np.ndarray:
    f = np.fft.fft2((patch - patch.mean()) * HANN)
    p = np.abs(f) ** 2
    prof = np.array([p[_BIN == b].mean() for b in range(NBINS)])
    prof = np.log10(prof + 1e-12)
    return prof - prof.mean()


def features(imgray: np.ndarray, half: str) -> np.ndarray:
    return np.array([radial_spectrum(imgray[y:y + PATCH, x:x + PATCH])
                     for y, x in patch_coords(imgray.shape, half)])


# ---------------------------------------------------------------- fig 87.1 mask contract
def fig_mask() -> None:
    img = load_rgb()
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((yy - 250) / 90.0) ** 2 + ((xx - 300) / 70.0) ** 2 <= 1.0).astype(float)

    # "inpainting": fill the hole with shifted texture, feather the seam, then harmonise globally
    src = np.roll(np.roll(img, 150, axis=0), -120, axis=1)
    soft = blur(mask, 6.0)                       # feathering leaks outside the binary mask
    feather = soft[..., None] * src + (1 - soft[..., None]) * img
    edit = np.clip(feather * 1.008 + 0.004, 0, 1)   # global colour harmonisation

    inside = mask > 0.5
    outside = ~inside
    tol = 1.0 / 255.0
    mask_share = float(inside.mean())

    d_f = np.abs(feather - img).max(axis=2)      # feathering only
    d = np.abs(edit - img).max(axis=2)           # feathering + harmonisation
    share_out_f = float((d_f[outside] > tol).mean())
    share_out = float((d[outside] > tol).mean())
    max_out = float(d_f[outside].max())
    mean_in = float(d[inside].mean())
    ring = (blur(mask, 3.0) > 1e-3) & outside    # ~9-пиксельная полоса сразу за контуром
    share_ring = float((d_f[ring] > tol).mean())

    FACTS.update(mask_share_pct=round(mask_share * 100, 1),
                 feather_outside_pct=round(share_out_f * 100, 1),
                 outside_changed_pct=round(share_out * 100, 1),
                 outside_max_levels=round(max_out * 255, 1),
                 ring_changed_pct=round(share_ring * 100, 1),
                 inside_mean_levels=round(mean_in * 255, 1))
    assert 2.0 < mask_share * 100 < 6.0, mask_share
    assert share_out > 0.9 and share_ring > 0.9, (share_out, share_ring)
    assert share_out_f < 0.2, share_out_f
    assert max_out * 255 > 20, max_out

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.4))
    axes[0].imshow(img); axes[0].set_title("исходник $x$", fontsize=11.5)
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title(f"маска $m$: {mask_share*100:.1f}% пикселей", fontsize=11.5)
    axes[2].imshow(edit); axes[2].set_title(r"результат $\widehat{x}$", fontsize=11.5)
    im = axes[3].imshow(np.clip(d * 6, 0, 1), cmap="magma")
    axes[3].contour(mask, levels=[0.5], colors=[GREEN], linewidths=1.4)
    axes[3].set_title(f"$|\\widehat{{x}}-x|$: вне маски\nизменено {share_out*100:.1f}% пикселей", fontsize=11.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Маска — это контракт, и он нарушается тише, чем кажется", y=1.0, fontsize=14)
    fig.tight_layout()
    save(fig, OUT / "mask_contract.png")

    # sidenote: histogram of the difference inside / outside
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bins = np.linspace(0, 0.25, 60)
    ax.hist(d_f[inside], bins=bins, color=RED, alpha=0.75, label="внутри маски", density=True)
    ax.hist(d_f[outside], bins=bins, color=BLUE, alpha=0.6, label="вне маски", density=True)
    ax.set_yscale("log"); ax.set_xlabel(r"$|\widehat{x}-x|$"); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("хвост правки уходит за контур", fontsize=11)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "diff_hist.png")
    print("mask:", json.dumps({k: FACTS[k] for k in list(FACTS)}, ensure_ascii=False))


# ---------------------------------------------------------------- fig 87.2 spectra
def fig_spectra() -> None:
    img = load_rgb()
    g = gray(img)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    fr = np.arange(NBINS) / NBINS
    prof_real = features(g, "top").mean(axis=0)
    ax.plot(fr, prof_real, color=INK, lw=2.6, label="исходные фрагменты")
    gaps = {}
    for (name, fn), c in zip(GENERATORS, [BLUE, GREEN, GOLD]):
        prof = features(gray(fn(img)), "top").mean(axis=0)
        ax.plot(fr, prof, color=c, lw=2.0, label=name)
        gaps[name] = float(np.abs(prof - prof_real)[-6:].mean())
    FACTS["gap_resample"] = round(gaps["апсемплинг ×2"], 2)
    FACTS["gap_sharpen"] = round(gaps["перерезкость"], 2)
    assert gaps["апсемплинг ×2"] > 0.5, gaps
    ax.axvspan(0.7, 1.0, color=WASH, alpha=0.8, zorder=0)
    ax.text(0.85, ax.get_ylim()[1] * 0.92, "высокие частоты", ha="center", fontsize=10, color=MUTED)
    ax.set_xlabel("нормированная пространственная частота")
    ax.set_ylabel(r"$\log_{10}$ мощности (центрировано)")
    ax.set_title("След генератора живёт в верхних частотах спектра")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "spectra.png")

    # sidenote: only the top of the spectrum, zoomed
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(fr[-8:], prof_real[-8:], color=INK, lw=2.4, marker="o", ms=4)
    for (name, fn), c in zip(GENERATORS, [BLUE, GREEN, GOLD]):
        prof = features(gray(fn(img)), "top").mean(axis=0)
        ax.plot(fr[-8:], prof[-8:], color=c, lw=1.8, marker="o", ms=3)
    ax.set_xlabel("частота"); ax.set_yticks([])
    ax.set_title("тот же хвост крупно", fontsize=11)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "spectra_zoom.png")
    print("spectra gaps:", gaps)


# ---------------------------------------------------------------- detectors
def build_scores():
    """Train real-vs-Gi on the top half, evaluate on the bottom half. Returns dict."""
    img = load_rgb()
    g = gray(img)
    fake_full = {name: gray(fn(img)) for name, fn in GENERATORS}

    Xr_tr, Xr_te = features(g, "top"), features(g, "bottom")
    models, scores = {}, {}
    for name in fake_full:
        Xf_tr = features(fake_full[name], "top")
        X = np.vstack([Xr_tr, Xf_tr])
        y = np.r_[np.zeros(len(Xr_tr)), np.ones(len(Xf_tr))]
        m = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
        models[name] = m
    for train_name, m in models.items():
        for test_name in fake_full:
            Xf_te = features(fake_full[test_name], "bottom")
            s = np.r_[m.predict_proba(Xr_te)[:, 1], m.predict_proba(Xf_te)[:, 1]]
            y = np.r_[np.zeros(len(Xr_te)), np.ones(len(Xf_te))]
            scores[(train_name, test_name)] = (y, s)
    return models, scores, g, fake_full, Xr_te


# ---------------------------------------------------------------- fig 87.3 LOGO matrix
def fig_transfer(models, scores):
    names = [n for n, _ in GENERATORS]
    M = np.array([[roc_auc_score(*scores[(a, b)]) for b in names] for a in names])
    FACTS["auc_diag_min"] = round(float(np.min(np.diag(M))), 3)
    FACTS["auc_own"] = round(float(M[0, 0]), 3)
    FACTS["auc_cross_worst"] = round(float(M.min()), 3)
    off = M[~np.eye(3, dtype=bool)]
    FACTS["auc_cross_mean"] = round(float(off.mean()), 3)
    assert np.min(np.diag(M)) > 0.8, M
    assert off.min() < 0.35, M

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=14, color=INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3), names, fontsize=10)
    ax.set_yticks(range(3), names, fontsize=10)
    ax.set_xlabel("проверяем на генераторе"); ax.set_ylabel("учили на генераторе")
    ax.set_title("ROC-AUC: по диагонали блеск, вне её — провал")
    save(fig, OUT / "transfer_matrix.png")
    print("transfer matrix:\n", np.round(M, 3))
    return M


# ---------------------------------------------------------------- fig 87.4 JPEG stress
def fig_jpeg(models):
    img = load_rgb()
    qs = [95, 85, 75, 60, 45, 30]
    curves = {}
    for name, fn in GENERATORS:
        m = models[name]
        row = []
        for q in qs:
            gr = gray(jpeg_roundtrip(img, q))
            gf = gray(jpeg_roundtrip(fn(img), q))
            Xr, Xf = features(gr, "bottom"), features(gf, "bottom")
            s = np.r_[m.predict_proba(Xr)[:, 1], m.predict_proba(Xf)[:, 1]]
            y = np.r_[np.zeros(len(Xr)), np.ones(len(Xf))]
            row.append(roc_auc_score(y, s))
        curves[name] = row
    FACTS["jpeg_q95_resample"] = round(curves["апсемплинг ×2"][0], 3)
    FACTS["jpeg_q30_resample"] = round(curves["апсемплинг ×2"][-1], 3)
    FACTS["jpeg_q30_worst"] = round(min(c[-1] for c in curves.values()), 3)
    assert curves["апсемплинг ×2"][-1] < curves["апсемплинг ×2"][0], curves

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for (name, _), c in zip(GENERATORS, [BLUE, GREEN, GOLD]):
        ax.plot(qs, curves[name], color=c, lw=2.2, marker="o", ms=5, label=name)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.text(90, 0.52, "уровень монетки", color=MUTED, fontsize=10)
    ax.invert_xaxis()
    ax.set_xlabel("качество JPEG при пересылке"); ax.set_ylabel("ROC-AUC на своём же генераторе")
    ax.set_ylim(0.3, 1.03)
    ax.set_title("Мессенджер пережал картинку — и след генератора смылся")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "jpeg_stress.png")
    print("jpeg curves:", {k: [round(v, 3) for v in vs] for k, vs in curves.items()})
    return curves


# ---------------------------------------------------------------- fig 87.5 base rate
def fig_base_rate():
    sens, spec = 0.90, 0.95
    N = 10000
    for pi, key in [(0.01, "ppv_01"), (0.10, "ppv_10"), (0.30, "ppv_30")]:
        ppv = sens * pi / (sens * pi + (1 - spec) * (1 - pi))
        FACTS[key] = round(ppv, 3)
    assert abs(FACTS["ppv_01"] - 0.154) < 0.002, FACTS["ppv_01"]
    tp, fn_, fp, tn = 90, 10, 495, 9405
    assert tp + fn_ + fp + tn == N
    FACTS["flags_total"] = tp + fp
    FACTS["flags_false_pct"] = round(fp / (tp + fp) * 100, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    def box(x, y, w, h, c, t, sub):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor=INK, lw=1.0, alpha=0.85))
        ax.text(x + w / 2, y + h / 2 + 0.25, t, ha="center", fontsize=12, color=PAPER, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.5, sub, ha="center", fontsize=10, color=PAPER)
    box(0.2, 7.8, 9.6, 1.8, MUTED, "10 000 файлов", "подделок 1%")
    box(0.2, 4.6, 1.4, 2.4, RED, "100", "подделки")
    box(2.2, 4.6, 7.6, 2.4, GREEN, "9 900", "настоящие")
    box(0.2, 1.4, 0.9, 2.4, RED, "90", "TP")
    box(1.3, 1.4, 0.3, 2.4, GOLD, "", "")
    box(2.2, 1.4, 2.6, 2.4, GOLD, "495", "FP")
    box(5.0, 1.4, 4.8, 2.4, GREEN, "9 405", "TN")
    ax.text(0.35, 1.0, "10 FN", fontsize=9.5, color=MUTED)
    ax.text(5.0, 0.4, f"тревог всего {tp+fp}, из них ложных {fp/(tp+fp)*100:.1f}%",
            fontsize=12, color=INK)
    ax.set_title("Sensitivity 0,90 и specificity 0,95 при 1% подделок", fontsize=13)

    ax = axes[1]
    pis = np.linspace(0.001, 0.5, 400)
    for s, c, lab in [(0.95, BLUE, "spec 0,95"), (0.99, GREEN, "spec 0,99"), (0.999, VIOLET, "spec 0,999")]:
        ax.plot(pis * 100, sens * pis / (sens * pis + (1 - s) * (1 - pis)), color=c, lw=2.2, label=lab)
    ax.scatter([1], [FACTS["ppv_01"]], s=60, color=RED, zorder=6)
    ax.annotate(f"{FACTS['ppv_01']:.3f}", (1, FACTS["ppv_01"]), textcoords="offset points",
                xytext=(12, 6), color=RED, fontsize=11)
    ax.set_xlabel("доля подделок в потоке, %"); ax.set_ylabel("PPV: доля истинных среди тревог")
    ax.set_title("Тот же детектор, разная редкость", fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "base_rate.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(pis * 100, sens * pis / (sens * pis + 0.05 * (1 - pis)), color=BLUE, lw=2.2)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlabel("доля подделок, %"); ax.set_ylabel("PPV")
    ax.set_title("половина тревог верна\nтолько после 5%", fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "ppv_curve.png")
    # where does PPV cross 1/2?
    cross = pis[np.argmax(sens * pis / (sens * pis + 0.05 * (1 - pis)) >= 0.5)]
    FACTS["ppv_half_pct"] = round(float(cross) * 100, 1)
    assert 4.5 < FACTS["ppv_half_pct"] < 6.0, FACTS["ppv_half_pct"]
    FACTS["ppv_half_exact_pct"] = round((1 - spec) / (sens + 1 - spec) * 100, 2)
    assert abs(FACTS["ppv_half_exact_pct"] - 5.26) < 0.01, FACTS["ppv_half_exact_pct"]
    print("base rate:", FACTS["ppv_01"], FACTS["ppv_10"], FACTS["ppv_30"], FACTS["ppv_half_pct"])


# ---------------------------------------------------------------- fig 87.6 cost & confident error
def fig_cost(scores):
    y_in, s_in = scores[("апсемплинг ×2", "апсемплинг ×2")]
    y_out, s_out = scores[("апсемплинг ×2", "перерезкость")]
    ts = np.linspace(0.001, 0.999, 999)

    pi, cfp, cfn = 0.01, 1.0, 20.0
    cost = []
    for t in ts:
        tpr = float((s_in[y_in == 1] >= t).mean())
        fpr = float((s_in[y_in == 0] >= t).mean())
        cost.append(cfn * pi * (1 - tpr) + cfp * (1 - pi) * fpr)
    cost = np.array(cost) * 10000
    t_best = float(ts[int(np.argmin(cost))])
    FACTS["t_best"] = round(t_best, 2)
    FACTS["cost_best"] = round(float(cost.min()), 1)
    FACTS["cost_half"] = round(float(cost[np.argmin(np.abs(ts - 0.5))]), 1)
    assert FACTS["cost_best"] <= FACTS["cost_half"], FACTS

    real_s = s_in[y_in == 0]
    own_s = s_in[y_in == 1]
    new_s = s_out[y_out == 1]
    FACTS["mean_score_real"] = round(float(real_s.mean()), 3)
    FACTS["mean_score_own"] = round(float(own_s.mean()), 3)
    FACTS["mean_score_new"] = round(float(new_s.mean()), 3)
    FACTS["new_called_real_pct"] = round(float((new_s < 0.5).mean()) * 100, 1)
    FACTS["new_confident_real_pct"] = round(float((new_s < 0.1).mean()) * 100, 1)
    assert FACTS["mean_score_new"] < FACTS["mean_score_own"], FACTS
    assert FACTS["new_called_real_pct"] > 60, FACTS

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    ax.plot(ts, cost, color=BLUE, lw=2.4)
    ax.axvline(t_best, color=RED, lw=1.6, ls=(0, (4, 3)))
    ax.scatter([t_best], [cost.min()], color=RED, s=55, zorder=6)
    ax.annotate(f"$t^*={t_best:.2f}$, цена {cost.min():.0f}", (t_best, cost.min()),
                textcoords="offset points", xytext=(14, 26), color=RED, fontsize=11)
    ax.set_xlabel("порог $t$"); ax.set_ylabel("ожидаемая цена на 10 000 файлов")
    ax.set_title(r"Цена ошибок: $c_{FP}=1$, $c_{FN}=20$, 1% подделок", fontsize=13)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)

    ax = axes[1]
    bins = np.linspace(0, 1, 26)
    ax.hist(real_s, bins=bins, color=GREEN, alpha=0.7, label="настоящие фрагменты")
    ax.hist(own_s, bins=bins, color=BLUE, alpha=0.7, label="свой генератор")
    ax.hist(new_s, bins=bins, color=RED, alpha=0.65, label="незнакомый генератор")
    ax.axvline(0.5, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.set_xlabel("оценка детектора «это подделка»"); ax.set_ylabel("число фрагментов")
    ax.set_title(f"Незнакомую подделку зовут настоящей в {FACTS['new_called_real_pct']:.0f}% случаев",
                 fontsize=13)
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT / "cost_scores.png")
    print("cost:", FACTS["t_best"], FACTS["cost_best"], FACTS["cost_half"],
          "scores:", FACTS["mean_score_real"], FACTS["mean_score_own"], FACTS["mean_score_new"],
          FACTS["new_called_real_pct"], FACTS["new_confident_real_pct"])


# ---------------------------------------------------------------- sidenote: calibration
def fig_calibration(scores):
    y_in, s_in = scores[("апсемплинг ×2", "апсемплинг ×2")]
    y_out, s_out = scores[("апсемплинг ×2", "сглаживание+зерно")]

    def curve(y, s, k=8):
        edges = np.linspace(0, 1, k + 1)
        xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (s >= a) & (s < b if b < 1 else s <= b)
            if sel.sum() >= 5:
                xs.append(s[sel].mean()); ys.append(y[sel].mean())
        return np.array(xs), np.array(ys)

    def ece(y, s, k=8):
        edges = np.linspace(0, 1, k + 1)
        tot = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (s >= a) & (s < b if b < 1 else s <= b)
            if sel.sum():
                tot += sel.mean() * abs(y[sel].mean() - s[sel].mean())
        return tot

    FACTS["ece_in"] = round(float(ece(y_in, s_in)), 3)
    FACTS["ece_shift"] = round(float(ece(y_out, s_out)), 3)
    assert FACTS["ece_shift"] > FACTS["ece_in"], FACTS

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=(0, (4, 3)))
    x1, y1 = curve(y_in, s_in); x2, y2 = curve(y_out, s_out)
    ax.plot(x1, y1, color=BLUE, lw=2.0, marker="o", ms=4, label=f"свой ген., ECE {FACTS['ece_in']:.2f}")
    ax.plot(x2, y2, color=RED, lw=2.0, marker="o", ms=4, label=f"чужой, ECE {FACTS['ece_shift']:.2f}")
    ax.set_xlabel("заявленная вероятность"); ax.set_ylabel("доля подделок")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("калибровка ломается\nпри смене генератора", fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, SIDE / "calibration.png")
    print("ece:", FACTS["ece_in"], FACTS["ece_shift"])


# ---------------------------------------------------------------- pooled detector
def fig_pooled(scores):
    """A detector trained on all three generators: does pooling save transfer?"""
    img = load_rgb()
    g = gray(img)
    names = [n for n, _ in GENERATORS]
    fake_full = {name: gray(fn(img)) for name, fn in GENERATORS}
    Xr_tr, Xr_te = features(g, "top"), features(g, "bottom")
    aucs = {}
    for held in names:
        seen = [n for n in names if n != held]
        Xf = np.vstack([features(fake_full[n], "top") for n in seen])
        X = np.vstack([Xr_tr, Xf])
        y = np.r_[np.zeros(len(Xr_tr)), np.ones(len(Xf))]
        m = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
        Xf_te = features(fake_full[held], "bottom")
        s = np.r_[m.predict_proba(Xr_te)[:, 1], m.predict_proba(Xf_te)[:, 1]]
        yy = np.r_[np.zeros(len(Xr_te)), np.ones(len(Xf_te))]
        aucs[held] = float(roc_auc_score(yy, s))
    FACTS["logo_pool_min"] = round(min(aucs.values()), 3)
    FACTS["logo_pool_max"] = round(max(aucs.values()), 3)
    FACTS["n_patches_train"] = len(Xr_tr)
    FACTS["n_patches_test"] = len(Xr_te)
    print("pooled LOGO:", {k: round(v, 3) for k, v in aucs.items()})

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    xs = np.arange(3)
    ax.bar(xs - 0.2, [roc_auc_score(*scores[(n, n)]) for n in names], width=0.38,
           color=BLUE, label="учили на этом же генераторе")
    ax.bar(xs + 0.2, [aucs[n] for n in names], width=0.38, color=RED,
           label="учили на двух других (leave-one-generator-out)")
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.set_xticks(xs, names, fontsize=10.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("ROC-AUC на отложенной половине снимка")
    ax.set_title("Незнакомый генератор рушит детектор, даже если учили на двух")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(True, axis="y", color=GRID, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
    save(fig, OUT / "logo_bars.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SIDE.mkdir(parents=True, exist_ok=True)
    fig_mask()
    fig_spectra()
    models, scores, _, _, _ = build_scores()
    fig_transfer(models, scores)
    fig_jpeg(models)
    fig_base_rate()
    fig_cost(scores)
    fig_calibration(scores)
    fig_pooled(scores)
    (ROOT / "scripts" / "data" / "lesson87_facts.json").write_text(
        json.dumps(FACTS, ensure_ascii=False, indent=2), encoding="utf8")
    print("\nFACTS:", json.dumps(FACTS, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
