"""Deterministic figures for lesson 11: Kepler's third law and exoplanets.

Recomputes every number quoted in the lesson (solar-system slope 1.4997,
solar-mass exoplanet slope 1.4988, all-planet slope 1.46, residual slope
-0.5, the 76% share of |d|<0.01, the threefold spread collapse) from
scripts/data/exoplanets.csv and reference planet tables.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "11"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "11"
DATA = ROOT / "scripts" / "data" / "exoplanets.csv"

PAPER = "#fffef9"
INK = "#171915"
MUTED = "#6e726a"
FAINT = "#969990"
GRID = "#deddd4"
LINE = "#c9c8be"
BLUE = "#315f8c"
RED = "#b94a3b"
GREEN = "#38735d"
GOLD = "#a57920"
VIOLET = "#6f5a8f"
WASH = "#f5f3ea"

mpl.rcParams.update(
    {
        "font.family": "PT Sans",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.16,
        "mathtext.fontset": "dejavuserif",
    }
)


def save(fig: plt.Figure, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def arrow(ax, start, end, *, color=MUTED, lw=1.5, rad=0.0, ls="solid"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                 connectionstyle=f"arc3,rad={rad}",
                                 color=color, linewidth=lw, linestyle=ls,
                                 mutation_scale=13, shrinkA=0, shrinkB=0))


def ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    k = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return k, y.mean() - k * x.mean()


# ------------------------------------------------------------------- data
PLANETS = [  # name, a (AU), T (yr)
    ("Меркурий", 0.387, 0.2408),
    ("Венера", 0.723, 0.6152),
    ("Земля", 1.000, 1.0000),
    ("Марс", 1.524, 1.8808),
    ("Юпитер", 5.203, 11.862),
    ("Сатурн", 9.537, 29.457),
    ("Уран", 19.19, 84.01),
    ("Нептун", 30.07, 164.79),
]
CERES = (2.77, 4.60)
PLUTO = (39.5, 248.0)

rows = [r for r in csv.DictReader(DATA.open(encoding="utf-8"))
        if r["pl_orbper"] and r["pl_orbsmax"] and r["st_mass"]
        and float(r["st_mass"]) > 0]
lnT = np.array([math.log(float(r["pl_orbper"]) / 365.25) for r in rows])
lna = np.array([math.log(float(r["pl_orbsmax"])) for r in rows])
lnM = np.array([math.log(float(r["st_mass"])) for r in rows])
rows_all = [r for r in csv.DictReader(DATA.open(encoding="utf-8"))
            if r["pl_orbper"] and r["pl_orbsmax"]]
k_3562, _ = ols([math.log(float(r["pl_orbsmax"])) for r in rows_all],
                [math.log(float(r["pl_orbper"]) / 365.25) for r in rows_all])
solar_mask = np.array([0.95 <= float(r["st_mass"]) <= 1.05 for r in rows])

sx = np.log([a for _, a, _ in PLANETS])
sy = np.log([t for _, _, t in PLANETS])
k_solar, b_solar = ols(sx, sy)
k_sun, b_sun = ols(lna[solar_mask], lnT[solar_mask])
k_all, b_all = ols(lna, lnT)
resid = lnT - 1.5 * lna
k_res, b_res = ols(lnM, resid)
corrT = lnT + 0.5 * lnM
k_corr, b_corr = ols(lna, corrT)
d_full = lnT - 1.5 * lna + 0.5 * lnM
share = (np.abs(d_full) < 0.01).mean()
std_before = (lnT - (k_all * lna + b_all)).std()
std_after = (corrT - (k_corr * lna + b_corr)).std()

print(f"all 3562 slope {k_3562:.4f}")
assert abs(k_3562 - 1.4597) < 5e-4
print(f"solar system slope {k_solar:.4f}, intercept {b_solar:.4f}")
print(f"solar-mass exo n={solar_mask.sum()} slope {k_sun:.4f}, intercept {b_sun:.4f}")
print(f"all planets n={len(rows)} slope {k_all:.4f}")
print(f"residual-vs-mass slope {k_res:.3f}")
print(f"corrected slope {k_corr:.3f}; spread {std_before:.3f} -> {std_after:.3f} "
      f"(x{std_before / std_after:.1f})")
print(f"|d|<0.01 share {share:.1%}")
assert abs(k_solar - 1.4997) < 5e-4
assert abs(k_sun - 1.4988) < 5e-3
assert abs(k_all - 1.46) < 5e-3
assert abs(k_res + 0.5) < 0.02
assert std_before / std_after > 2.8
assert 0.74 < share < 0.79


# ------------------------------------------- fig 11.1: solar system log-log
def fig_loglog_solar() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    xs = np.linspace(math.log(0.28), math.log(45), 50)
    ax.plot(np.exp(xs), np.exp(k_solar * xs + b_solar), color=BLUE, lw=1.8,
            zorder=2, label=None)
    offsets = {
        "Меркурий": (10, -13), "Венера": (12, -10), "Земля": (12, -8),
        "Марс": (12, -8), "Юпитер": (12, -6), "Сатурн": (12, -5),
        "Уран": (12, -4), "Нептун": (-16, 10),
    }
    for name, a, t in PLANETS:
        ax.scatter([a], [t], s=52, color=RED, edgecolor=PAPER,
                   linewidth=1.0, zorder=4)
        ax.annotate(name, (a, t), textcoords="offset points",
                    xytext=offsets[name], fontsize=10.5, color=INK,
                    ha="left" if offsets[name][0] > 0 else "right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 60)
    ax.set_ylim(0.12, 700)
    # slope triangle
    x0, x1 = 2.4, 24.0
    y0 = math.exp(k_solar * math.log(x0) + b_solar) * 0.42
    y1 = y0 * (x1 / x0) ** 1.5
    ax.plot([x0, x1], [y0, y0], color=MUTED, lw=1.1)
    ax.plot([x1, x1], [y0, y1], color=MUTED, lw=1.1)
    ax.plot([x0, x1], [y0, y1], color=MUTED, lw=1.1, ls=(0, (4, 3)))
    ax.text(math.sqrt(x0 * x1), y0 * 0.62, "1 по горизонтали", fontsize=10,
            color=MUTED, ha="center")
    ax.text(x1 * 1.13, math.sqrt(y0 * y1), "полтора\nпо вертикали",
            fontsize=10, color=MUTED, ha="left", va="center")
    ax.text(0.29, 300,
            "наклон МНК по восьми планетам:\n"
            r"$\hat k=1{,}4997$,  $\ln\hat C=0{,}0001$",
            fontsize=11.5, color=BLUE, va="top")
    ax.set_xlabel("большая полуось $a$, а.е. (логарифмическая шкала)")
    ax.set_ylabel("период $T$, лет (лог. шкала)")
    ax.set_title("Восемь планет на логарифмической бумаге")
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, OUT / "loglog-solar.png")


# --------------------------------------------- fig 11.2: transit method
def fig_transit() -> None:
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(9.4, 4.4), gridspec_kw={"width_ratios": [1, 1.55]})
    # left: star disk with planet crossing
    axl.set_xlim(-1.6, 1.6)
    axl.set_ylim(-1.5, 1.5)
    axl.set_aspect("equal")
    axl.axis("off")
    axl.add_patch(Circle((0, 0), 1.0, facecolor="#f3e2b7", edgecolor=GOLD,
                         linewidth=1.6, zorder=1))
    axl.plot([-1.5, 1.5], [0.32, 0.32], color=MUTED, lw=1.0,
             ls=(0, (4, 3)), zorder=2)
    for x, ghost in ((-1.28, True), (0.0, False), (1.28, True)):
        axl.add_patch(Circle((x, 0.32), 0.16, facecolor=INK if not ghost else LINE,
                             edgecolor=PAPER, linewidth=0.8, zorder=3))
    arrow(axl, (0.42, 0.62), (1.02, 0.62), color=MUTED, lw=1.3)
    axl.text(0, -1.32, "планета проходит по диску звезды",
             ha="center", fontsize=10.5, color=MUTED)
    axl.set_title("что происходит", fontsize=12.5)
    # right: light curve with two box dips
    t = np.linspace(0, 10, 1200)
    flux = np.ones_like(t)
    depth = 0.011
    for c in (2.2, 7.2):
        flux -= depth * np.clip(1 - np.abs(t - c) / 0.42, 0, 1) ** 0.25 * (
            np.abs(t - c) < 0.42)
    axr.plot(t, flux, color=BLUE, lw=1.7)
    axr.set_ylim(0.9835, 1.0045)
    axr.set_xlim(0, 10)
    axr.set_xlabel("время")
    axr.set_ylabel("блеск звезды, доли")
    axr.set_title("что видит фотометр", fontsize=12.5)
    axr.grid(True, color=GRID, lw=0.6, alpha=0.7)
    axr.set_axisbelow(True)
    axr.annotate("", (7.2, 0.9995), (2.2, 0.9995),
                 arrowprops=dict(arrowstyle="<|-|>", color=GREEN, lw=1.4))
    axr.text(4.7, 1.0008, "период $T$: между провалами",
             ha="center", fontsize=10.5, color=GREEN)
    axr.annotate("", (8.35, 1.0 - depth), (8.35, 1.0),
                 arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=1.3))
    axr.text(8.55, 1.0 - depth / 2,
             "глубина\n$(R_p/R_\\star)^2\\approx 1\\%$",
             fontsize=10.5, color=RED, va="center")
    fig.suptitle("Транзит: планету не видно, но виден её график", y=1.02,
                 fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "transit-method.png")


# ------------------------------------------ fig 11.3: exoplanets, same line
def fig_exo_line() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ea = np.exp(lna[solar_mask])
    et = np.exp(lnT[solar_mask])
    ax.scatter(ea, et, s=14, color=FAINT, alpha=0.55, linewidth=0, zorder=2,
               label=f"экзопланеты у звёзд $0{{,}}95$–$1{{,}}05\\,M_\\odot$ "
                     f"({solar_mask.sum()})")
    xs = np.linspace(math.log(0.008), math.log(45), 60)
    ax.plot(np.exp(xs), np.exp(k_sun * xs + b_sun), color=BLUE, lw=1.8,
            zorder=3, label="одна прямая МНК: наклон $1{,}4988$")
    for name, a, t in PLANETS:
        ax.scatter([a], [t], s=54, color=RED, edgecolor=PAPER, linewidth=1.0,
                   zorder=4)
    ax.annotate("Меркурий", (0.387, 0.2408), textcoords="offset points",
                xytext=(8, -14), fontsize=10, color=RED)
    ax.annotate("Нептун", (30.07, 164.79), textcoords="offset points",
                xytext=(-14, 10), fontsize=10, color=RED, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.006, 60)
    ax.set_ylim(0.004, 700)
    ax.annotate("тесные миры: смещение отбора,\nа не устройство Вселенной",
                (0.05, 0.011), (0.16, 0.0055), fontsize=10.5, color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("большая полуось $a$, а.е. (лог. шкала)")
    ax.set_ylabel("период $T$, лет (лог. шкала)")
    ax.set_title("Чужие миры на прямой Кеплера")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, OUT / "exo-same-line.png")


# -------------------------------------------- fig 11.4: polynomial overfit
def fig_poly() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    aa = np.array([a for _, a, _ in PLANETS])
    tt = np.array([t for _, _, t in PLANETS])
    coeffs = np.polyfit(aa, tt, 7)
    p_ceres = float(np.polyval(coeffs, CERES[0]))
    p_pluto = float(np.polyval(coeffs, PLUTO[0]))
    p_close = float(np.polyval(coeffs, 0.05))
    print(f"poly(linear) at Ceres {p_ceres:.2f} (law 4.61), "
          f"Pluto {p_pluto:.0f} (law {PLUTO[0] ** 1.5:.0f}), "
          f"a=0.05: {p_close:.3f}")
    assert p_pluto < -13000
    xs = np.linspace(0.0, 43.0, 900)
    ax.plot(xs, xs ** 1.5, color=BLUE, lw=1.9, zorder=3,
            label="закон $T=a^{3/2}$: два параметра")
    ax.plot(xs, np.polyval(coeffs, xs), color=RED, lw=1.8, ls=(0, (5, 3)),
            zorder=3, label="полином 7-й степени: ошибка обучения $=0$")
    ax.scatter(aa, tt, s=56, color=INK, edgecolor=PAPER, linewidth=1.0,
               zorder=5, label="восемь планет (обучение)")
    ax.scatter([CERES[0]], [CERES[1]], s=62, marker="D", color=GREEN,
               edgecolor=PAPER, linewidth=1.0, zorder=5)
    ax.scatter([PLUTO[0]], [PLUTO[1]], s=62, marker="D", color=GREEN,
               edgecolor=PAPER, linewidth=1.0, zorder=5)
    ax.annotate("Церера: обе модели\nсправляются", (CERES[0], CERES[1]),
                (5.5, -150), fontsize=10.5, color=GREEN,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2,
                                connectionstyle="arc3,rad=0.2"))
    ax.annotate("Плутон: закон даёт 248,\nнаблюдается 248", (PLUTO[0], PLUTO[1]),
                (26.5, 348), fontsize=10.5, color=GREEN, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.annotate("полином в точке Плутона:\n$-13\\,480$ лет — за краем\nобучения кривую никто не держит",
                (33.6, np.polyval(coeffs, 33.6)),
                (12.5, -320), fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2,
                                connectionstyle="arc3,rad=0.25"))
    ax.axhline(0, color=LINE, lw=1.0)
    ax.axvspan(0.387, 30.07, color=WASH, zorder=1)
    ax.text(15.2, -395, "диапазон обучения", fontsize=10, color=MUTED,
            ha="center")
    ax.set_xlim(0, 43)
    ax.set_ylim(-420, 460)
    ax.set_xlabel("большая полуось $a$, а.е.")
    ax.set_ylabel("период $T$, лет")
    ax.set_title("Нуль ошибки на обучении, провал на переносе")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, OUT / "poly-overfit.png")


# --------------------------------------- fig 11.5: residuals vs star mass
def fig_residuals() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.scatter(lnM, resid, s=12, color=FAINT, alpha=0.45, linewidth=0,
               zorder=2)
    xs = np.linspace(lnM.min(), lnM.max(), 40)
    ax.plot(xs, k_res * xs + b_res, color=RED, lw=1.9, zorder=3,
            label=f"МНК по остаткам: наклон ${k_res:.2f}$".replace(".", "{,}"))
    ax.plot(xs, -0.5 * xs, color=BLUE, lw=1.5, ls=(0, (5, 3)), zorder=3,
            label="теория: $-\\frac{1}{2}\\ln M$")
    ax.axhline(0, color=LINE, lw=1.0)
    ax.annotate("тяжёлые звёзды:\nпериод короче, точки ниже",
                (0.55, -0.34), (0.62, 0.42), fontsize=10.5, color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2,
                                connectionstyle="arc3,rad=0.25"))
    ax.annotate("красные карлики:\nвыше прямой",
                (-1.6, 0.78), (-2.45, 1.25), fontsize=10.5, color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("логарифм массы звезды, $\\ln M$ (в массах Солнца)")
    ax.set_ylabel("остаток $\\ln T-\\frac{3}{2}\\ln a$")
    ax.set_title("Остатки не шумят, а маршируют: минус одна вторая")
    ax.set_xlim(-2.6, 1.35)
    ax.set_ylim(-1.1, 1.65)
    ax.legend(loc="lower left", frameon=False, fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, OUT / "mass-residuals.png")


# ------------------------------------- fig 11.6: correction before / after
def fig_correction() -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 4.9), sharex=True,
                                   sharey=True)
    xs = np.linspace(math.log(0.008), math.log(45), 50)
    for ax, y, k, b, s, ttl in (
        (axl, lnT, k_all, b_all, std_before,
         "до: все звёзды вперемешку"),
        (axr, corrT, k_corr, b_corr, std_after,
         "после: $T\\mapsto T\\sqrt{M}$"),
    ):
        ax.scatter(lna, y, s=9, color=FAINT, alpha=0.4, linewidth=0, zorder=2)
        ax.plot(xs, k * xs + b, color=BLUE if ax is axr else RED, lw=1.8,
                zorder=3)
        ax.set_title(ttl, fontsize=12.5)
        ax.text(0.04, 0.96,
                (f"наклон ${k:.2f}$\nразброс ${s:.2f}$").replace(".", "{,}"),
                transform=ax.transAxes, fontsize=11, va="top",
                color=BLUE if ax is axr else RED)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("$\\ln a$")
    axl.set_ylabel("$\\ln T$  /  $\\ln T+\\frac{1}{2}\\ln M$")
    axl.set_xlim(-5.2, 4.2)
    axl.set_ylim(-7.6, 6.2)
    fig.suptitle("Скрытая переменная, впущенная в модель: облако сжимается втрое",
                 y=1.02, fontsize=15)
    fig.tight_layout()
    save(fig, OUT / "mass-correction.png")


# ------------------------------------------------------- margin: log paper
def side_log_paper() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.04, 1000)
    xs = np.array([0.2, 0.9, 4.0, 18.0, 60.0])
    ys = xs ** 1.5 * np.array([1.12, 0.93, 1.08, 0.95, 1.05])
    ax.scatter(xs, ys, s=34, color=RED, edgecolor=PAPER, linewidth=0.8,
               zorder=4)
    grid = np.linspace(math.log(0.12), math.log(90), 30)
    ax.plot(np.exp(grid), np.exp(1.5 * grid), color=BLUE, lw=1.7, zorder=3)
    ax.text(0.14, 220, "наклон = показатель\nстепени: $k\\approx 1{,}5$",
            fontsize=10, color=BLUE)
    ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    save(fig, SIDE / "log-paper.png")


# -------------------------------------------------- margin: Tycho quadrant
def side_quadrant() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.set_xlim(-0.15, 1.35)
    ax.set_ylim(-0.15, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    # wall
    ax.plot([0, 0], [0, 1.22], color=INK, lw=3.0)
    ax.plot([0, 1.28], [0, 0], color=INK, lw=3.0)
    # graduated arc
    th = np.linspace(0, math.pi / 2, 200)
    ax.plot(np.cos(th), np.sin(th), color=GOLD, lw=2.2)
    for t in np.linspace(0, math.pi / 2, 19):
        r0, r1 = (0.955, 1.0) if round(t / (math.pi / 8), 3) % 1 else (0.93, 1.0)
        ax.plot([r0 * math.cos(t), r1 * math.cos(t)],
                [r0 * math.sin(t), r1 * math.sin(t)], color=GOLD, lw=1.1)
    # sight line to star
    ang = math.radians(52)
    ax.plot([0, 1.16 * math.cos(ang)], [0, 1.16 * math.sin(ang)],
            color=BLUE, lw=1.5, ls=(0, (5, 3)))
    ax.scatter([1.2 * math.cos(ang)], [1.2 * math.sin(ang)], s=90, marker="*",
               color=BLUE, zorder=5)
    ax.text(0.24, 0.28, "дуга с делениями:\nчем длиннее,\nтем точнее отсчёт",
            fontsize=9.5, color=MUTED)
    ax.text(1.16 * math.cos(ang) - 0.05, 1.16 * math.sin(ang) + 0.09,
            "визир на звезду", fontsize=9.5, color=BLUE, ha="right")
    save(fig, SIDE / "tycho-quadrant.png")


# ---------------------------------------------- margin: circle vs ellipse
def side_eight_minutes() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.set_xlim(-1.55, 1.45)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    th = np.linspace(0, 2 * math.pi, 300)
    e = 0.20  # exaggerated vs Mars's 0.09
    a_e = 1.0
    b_e = a_e * math.sqrt(1 - e * e)
    ax.plot(a_e * np.cos(th) - a_e * e, b_e * np.sin(th), color=RED, lw=1.9,
            zorder=3)
    ax.plot(0.985 * np.cos(th) - 0.10, 0.985 * np.sin(th), color=BLUE, lw=1.5,
            ls=(0, (5, 3)), zorder=2)
    ax.scatter([0], [0], s=120, marker="*", color=GOLD, zorder=5)
    ax.text(0.04, -0.16, "Солнце", fontsize=9.5, color=GOLD)
    arrow(ax, (0.60, 1.12), (0.30, 0.93), color=INK, lw=1.3, rad=-0.2)
    ax.text(0.63, 1.12, "здесь круг врёт\nна 8 угловых минут", fontsize=9.5,
            color=INK, va="center")
    ax.text(-1.5, -1.1, "— эллипс Кеплера", fontsize=9.5, color=RED)
    ax.text(-0.1, -1.1, "- - лучший круг", fontsize=9.5, color=BLUE)
    save(fig, SIDE / "eight-minutes.png")


fig_loglog_solar()
fig_transit()
fig_exo_line()
fig_poly()
fig_residuals()
fig_correction()
side_log_paper()
side_quadrant()
side_eight_minutes()
print("lesson 11 figures written")
