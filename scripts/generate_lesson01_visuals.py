from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "01"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "01"

PAPER = "#fffef9"
INK = "#171915"
MUTED = "#6e726a"
GRID = "#deddd4"
LINE = "#c9c8be"
BLUE = "#315f8c"
RED = "#b94a3b"
GREEN = "#38735d"
GOLD = "#a57920"
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


def clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def add_box(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    title: str,
    subtitle: str,
    *,
    edge: str = LINE,
    fill: str = PAPER,
    title_color: str = INK,
) -> FancyBboxPatch:
    x, y = center
    w, h = size
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.35,
        edgecolor=edge,
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y + 0.035,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=title_color,
        zorder=4,
    )
    ax.text(
        x,
        y - 0.055,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
        zorder=4,
    )
    return patch


def arrow_between(
    ax: plt.Axes,
    a: tuple[float, float],
    b: tuple[float, float],
    patch_a: FancyBboxPatch,
    patch_b: FancyBboxPatch,
    *,
    color: str = MUTED,
    rad: float = 0.0,
    width: float = 1.4,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
) -> None:
    arrow = FancyArrowPatch(
        a,
        b,
        patchA=patch_a,
        patchB=patch_b,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=13,
        linewidth=width,
        color=color,
        shrinkA=4,
        shrinkB=0,
        zorder=2,
    )
    ax.add_patch(arrow)
    if label and label_xy:
        ax.text(
            *label_xy,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "fc": PAPER, "ec": "none"},
            zorder=5,
        )


def system_loop() -> None:
    fig, ax = plt.subplots(figsize=(14.4, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    centers = {
        "observe": (0.13, 0.60),
        "model": (0.37, 0.60),
        "decision": (0.62, 0.60),
        "world": (0.86, 0.60),
    }
    size = (0.18, 0.19)
    boxes = {
        "observe": add_box(
            ax,
            centers["observe"],
            size,
            "Наблюдение",
            "что и когда измерено",
            edge=BLUE,
            fill="#edf3f7",
        ),
        "model": add_box(
            ax,
            centers["model"],
            size,
            "Модель",
            r"$\hat y=f_{\theta}(x)$",
            edge=RED,
            fill="#f7ece9",
            title_color=RED,
        ),
        "decision": add_box(
            ax,
            centers["decision"],
            size,
            "Решение",
            "порог, правило, человек",
            edge=GOLD,
            fill="#f7f1df",
        ),
        "world": add_box(
            ax,
            centers["world"],
            size,
            "Изменившийся мир",
            "люди отвечают на действие",
            edge=GREEN,
            fill="#edf4ef",
        ),
    }
    arrow_between(ax, centers["observe"], centers["model"], boxes["observe"], boxes["model"])
    arrow_between(ax, centers["model"], centers["decision"], boxes["model"], boxes["decision"])
    arrow_between(ax, centers["decision"], centers["world"], boxes["decision"], boxes["world"])
    arrow_between(
        ax,
        centers["world"],
        centers["observe"],
        boxes["world"],
        boxes["observe"],
        color=GREEN,
        rad=-0.38,
        width=1.7,
        label="обратная связь меняет будущие данные",
        label_xy=(0.50, 0.25),
    )

    ax.text(
        0.50,
        0.91,
        "Модель — один узел системы, а не вся система",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.855,
        "Даже безупречное вычисление не исправляет плохое измерение, цель или действие.",
        ha="center",
        va="center",
        fontsize=12.5,
        color=MUTED,
    )

    failures = [
        ("данные", "пропуск, смещение, утечка"),
        ("цель", "не тот показатель"),
        ("проверка", "старый или слишком лёгкий тест"),
        ("действие", "нет права отмены"),
    ]
    xs = np.linspace(0.14, 0.86, len(failures))
    for x, (title, detail) in zip(xs, failures):
        ax.plot([x, x], [0.085, 0.16], color=LINE, linewidth=1)
        ax.text(x, 0.065, title, ha="center", fontsize=11, fontweight="bold")
        ax.text(x, 0.025, detail, ha="center", fontsize=9.5, color=MUTED)
    ax.text(0.03, 0.13, "Где ломается", rotation=90, va="center", color=RED, fontsize=10)
    save(fig, OUT / "system-loop.png")


def temporal_split() -> None:
    rng = np.random.default_rng(101)
    days = np.arange(90)
    load = (
        62
        + 8 * np.sin(2 * np.pi * days / 7)
        + 0.12 * days
        + rng.normal(0, 2.2, size=days.size)
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14.4, 8),
        gridspec_kw={"height_ratios": [1.15, 0.85], "hspace": 0.38},
    )
    ax = axes[0]
    ax.axvspan(0, 59.5, color=BLUE, alpha=0.08)
    ax.axvspan(59.5, 74.5, color=GOLD, alpha=0.11)
    ax.axvspan(74.5, 89, color=GREEN, alpha=0.10)
    ax.plot(days, load, color=INK, linewidth=1.35)
    ax.scatter(days[::4], load[::4], s=14, color=INK, zorder=3)
    ax.text(30, 80.5, "обучение", ha="center", color=BLUE, fontweight="bold")
    ax.text(67, 80.5, "настройка", ha="center", color=GOLD, fontweight="bold")
    ax.text(82, 80.5, "одна финальная проверка", ha="center", color=GREEN, fontweight="bold")
    ax.set_ylabel("нагрузка, кВт")
    ax.set_xlim(0, 89)
    ax.set_ylim(load.min() - 4, load.max() + 9)
    ax.set_title("Временное разбиение сохраняет направление времени", loc="left", pad=12)
    clean(ax)

    ax2 = axes[1]
    train = np.arange(0, 90, 3)
    valid = np.arange(1, 90, 6)
    test = np.array([i for i in range(90) if i not in set(train) | set(valid)])
    ax2.scatter(train, np.zeros_like(train) + 2, s=22, color=BLUE, label="train")
    ax2.scatter(valid, np.zeros_like(valid) + 1, s=22, color=GOLD, label="validation")
    ax2.scatter(test, np.zeros_like(test), s=22, color=GREEN, label="test")
    for x in [37, 61]:
        arrow = FancyArrowPatch(
            (x + 8, 0.05),
            (x, 1.92),
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.22",
            mutation_scale=12,
            color=RED,
            linewidth=1.2,
        )
        ax2.add_patch(arrow)
    ax2.text(
        50,
        2.48,
        "Случайное перемешивание: соседние часы узнают друг друга через признаки",
        ha="center",
        color=RED,
        fontsize=11,
    )
    ax2.set_yticks([0, 1, 2], ["test", "validation", "train"])
    ax2.set_xlabel("день наблюдения")
    ax2.set_xlim(-1, 90)
    ax2.set_ylim(-0.45, 2.7)
    clean(ax2)
    save(fig, OUT / "temporal-split.png")


def distribution_shift() -> None:
    rng = np.random.default_rng(2026)
    t = np.arange(120)
    shift = 72
    baseline = 4.5 + 0.25 * np.sin(t / 7) + rng.normal(0, 0.15, t.size)
    learned = 3.6 + 0.20 * np.sin(t / 8) + rng.normal(0, 0.14, t.size)
    learned[t >= shift] += 0.075 * (t[t >= shift] - shift) + 1.2
    baseline[t >= shift] += 0.018 * (t[t >= shift] - shift)
    window = 9
    kernel = np.ones(window) / window
    learned_smooth = np.convolve(np.pad(learned, (4, 4), mode="edge"), kernel, mode="valid")
    spread = 0.38 + 0.012 * np.maximum(t - shift, 0)

    fig, ax = plt.subplots(figsize=(14.4, 8))
    ax.axvspan(shift, t[-1], color=RED, alpha=0.055)
    ax.fill_between(
        t,
        learned_smooth - spread,
        learned_smooth + spread,
        color=RED,
        alpha=0.13,
        linewidth=0,
        label="схематический коридор $m(t)\\pm s(t)$",
    )
    ax.plot(t, learned_smooth, color=RED, linewidth=2.3, label="обучаемая модель")
    ax.plot(t, baseline, color=BLUE, linewidth=1.7, label="недельный baseline")
    ax.axvline(shift, color=INK, linewidth=1)
    ax.annotate(
        "сменилось расписание",
        xy=(shift, 7.3),
        xytext=(shift + 11, 8.3),
        arrowprops={
            "arrowstyle": "-|>",
            "color": INK,
            "connectionstyle": "arc3,rad=-0.12",
            "shrinkA": 4,
            "shrinkB": 4,
        },
        fontsize=11,
    )
    ax.text(
        18,
        7.9,
        "вчерашний тест всё ещё показывает 3,6 кВт",
        color=MUTED,
        fontsize=11,
    )
    ax.text(
        91,
        5.2,
        "простое правило\nстареет медленнее",
        color=BLUE,
        ha="center",
        fontsize=11,
    )
    ax.set_xlim(0, 119)
    ax.set_ylim(2.3, 9.0)
    ax.set_xlabel("день после запуска")
    ax.set_ylabel("средняя абсолютная ошибка, кВт")
    ax.set_title("Качество — функция времени, а не навсегда выданный сертификат", loc="left", pad=14)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    clean(ax)
    save(fig, OUT / "distribution-shift.png")


def autonomy_map() -> None:
    fig, ax = plt.subplots(figsize=(14.4, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axvspan(0, 0.5, color=BLUE, alpha=0.035)
    ax.axvspan(0.5, 1, color=RED, alpha=0.035)
    ax.axhspan(0.5, 1, color=GREEN, alpha=0.035)
    ax.axvline(0.5, color=LINE, linewidth=1)
    ax.axhline(0.5, color=LINE, linewidth=1)

    points = [
        ("калькулятор", 0.10, 0.09, (-2, 13)),
        ("QR-сканер", 0.22, 0.26, (9, -17)),
        ("шлагбаум", 0.14, 0.79, (8, 11)),
        ("спам-фильтр", 0.69, 0.28, (8, -17)),
        ("подсказка врачу", 0.80, 0.42, (-88, 12)),
        ("автопилот", 0.73, 0.82, (9, 11)),
        ("агент закупок", 0.91, 0.67, (-82, -20)),
    ]
    for label, x, y, offset in points:
        color = RED if x >= 0.5 else BLUE
        marker = "s" if y >= 0.5 else "o"
        ax.scatter([x], [y], s=70, color=color, marker=marker, zorder=3)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=11,
            ha="left",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": LINE,
                "linewidth": 0.8,
                "shrinkA": 5,
                "shrinkB": 5,
            },
        )

    ax.text(0.25, 0.95, "правило записано человеком", ha="center", color=BLUE, fontsize=11)
    ax.text(0.75, 0.95, "часть правила найдена по данным", ha="center", color=RED, fontsize=11)
    ax.text(0.03, 0.73, "действует\nсамостоятельно", ha="left", color=GREEN, fontsize=10)
    ax.text(0.03, 0.21, "советует\nчеловеку", ha="left", color=MUTED, fontsize=10)
    ax.set_xlabel("доля правила, выбранная по данным")
    ax.set_ylabel("автономность действия")
    ax.set_xticks([0, 0.5, 1], ["задано", "смешано", "обучено"])
    ax.set_yticks([0, 0.5, 1], ["совет", "проверка", "самостоятельно"])
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=8)
    ax.set_title("Обучаемость и автономность — разные координаты риска", loc="left", pad=14)
    clean(ax)
    save(fig, OUT / "autonomy-map.png")


def loss_and_cost() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 8), gridspec_kw={"wspace": 0.33})

    ax = axes[0]
    p = np.linspace(0.01, 0.99, 200)
    loss_positive = -np.log(p)
    loss_negative = -np.log(1 - p)
    ax.plot(p, loss_positive, color=BLUE, linewidth=2.1, label="истинный пик")
    ax.plot(p, loss_negative, color=RED, linewidth=2.1, label="обычный час")
    ax.axvline(0.5, color=LINE, linewidth=1)
    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("прогноз вероятности пика")
    ax.set_ylabel("log-loss")
    ax.set_title("Потеря обучает число", loc="left", pad=12)
    ax.legend(frameon=False, loc="upper center")
    clean(ax)

    ax2 = axes[1]
    warning_cost = 2 * (1 - p)
    silence_cost = 12 * p
    crossing = 1 / 7
    crossing_cost = 12 / 7
    ax2.axvspan(0, crossing, color=BLUE, alpha=0.045)
    ax2.axvspan(crossing, 1, color=GREEN, alpha=0.045)
    ax2.plot(
        p,
        warning_cost,
        color=BLUE,
        linewidth=2.25,
        label=r"предупредить: $2(1-p)$",
    )
    ax2.plot(
        p,
        silence_cost,
        color=RED,
        linewidth=2.25,
        label=r"молчать: $12p$",
    )
    ax2.axvline(crossing, color=GOLD, linewidth=1.4)
    ax2.scatter(
        [crossing],
        [crossing_cost],
        s=58,
        color=GOLD,
        edgecolor=PAPER,
        linewidth=1.2,
        zorder=4,
    )
    ax2.annotate(
        r"равные цены: $p^\star=1/7$",
        xy=(crossing, crossing_cost),
        xytext=(0.33, 3.7),
        arrowprops={
            "arrowstyle": "-|>",
            "color": GOLD,
            "connectionstyle": "arc3,rad=0.08",
            "shrinkA": 4,
            "shrinkB": 5,
        },
        color=INK,
        fontsize=11,
    )
    ax2.text(
        0.060,
        8.9,
        "выбираем молчать",
        color=MUTED,
        fontsize=10.5,
        rotation=90,
        rotation_mode="anchor",
        ha="center",
    )
    ax2.text(
        0.57,
        0.40,
        "выбираем предупредить",
        color=MUTED,
        fontsize=10.5,
        ha="center",
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 12.4)
    ax2.set_xticks([0, crossing, 0.5, 1], ["0", r"$1/7$", "0,5", "1"])
    ax2.set_xlabel("прогноз вероятности пика $p$")
    ax2.set_ylabel("ожидаемая цена действия")
    ax2.set_title("Цена решения задаёт порог", loc="left", pad=12)
    ax2.legend(frameon=False, loc="upper right")
    clean(ax2)
    save(fig, OUT / "loss-and-cost.png")


def side_empirical_risk() -> None:
    x = np.array([0.08, 0.18, 0.29, 0.41, 0.55, 0.66, 0.78, 0.91])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    thresholds = np.linspace(0, 1, 201)
    risk = np.array([np.mean((x >= t).astype(int) != y) for t in thresholds])

    fig, ax = plt.subplots(figsize=(4.4, 3))
    ax.step(thresholds, risk, where="post", color=RED, linewidth=2)
    best = thresholds[np.argmin(risk)]
    ax.axvline(best, color=GOLD, linewidth=1.2)
    ax.scatter([best], [risk.min()], color=GOLD, s=34, zorder=3)
    ax.text(best + 0.035, risk.min() + 0.035, f"min = {risk.min():.3f}", color=GOLD, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 0.7)
    ax.set_xlabel("порог $t$")
    ax.set_ylabel("$R(t)$")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.25, 0.5])
    clean(ax)
    save(fig, OUT / "empirical-risk.png", dpi=200)


def side_baseline() -> None:
    labels = ["вчера", "неделю назад", "линейная модель"]
    values = [6.1, 4.9, 4.7]
    colors = [LINE, BLUE, RED]
    fig, ax = plt.subplots(figsize=(4.4, 3))
    bars = ax.barh(labels, values, color=colors, height=0.5)
    for bar, value in zip(bars, values):
        ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=10)
    ax.set_xlim(0, 7)
    ax.set_xlabel("MAE, кВт")
    ax.invert_yaxis()
    clean(ax)
    save(fig, SIDE / "baseline-bars.png", dpi=200)


def main() -> None:
    system_loop()
    temporal_split()
    distribution_shift()
    autonomy_map()
    loss_and_cost()
    side_empirical_risk()
    side_baseline()


if __name__ == "__main__":
    main()
