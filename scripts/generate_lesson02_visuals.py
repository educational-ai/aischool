from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons" / "02"
SIDE = ROOT / "public" / "figures" / "sidenotes" / "02"

PAPER = "#fffef9"
INK = "#171915"
MUTED = "#6e726a"
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


T = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
H = np.array([20.0, 18.78, 15.12, 9.12, 0.70])
H0 = 20.0
Z = T**2 / 2
G_HAT = float(np.sum(Z * (H0 - H)) / np.sum(Z**2))


def clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    detail: str,
    *,
    edge: str,
    fill: str,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        edgecolor=edge,
        facecolor=fill,
        linewidth=1.35,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center", fontsize=10.5, color=MUTED)
    return patch


def connect(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    rad: float = 0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            linewidth=1.4,
            mutation_scale=13,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def measurement_chain() -> None:
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.4, 8),
        gridspec_kw={"width_ratios": [1.18, 1], "wspace": 0.18},
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.2, 22.6)
    ax.axis("off")
    ax.plot([0.24, 0.24], [0, 20.7], color=INK, linewidth=1.25)
    for value in range(0, 21, 2):
        length = 0.045 if value % 5 else 0.075
        ax.plot([0.24 - length, 0.24], [value, value], color=LINE, linewidth=1)
        if value % 5 == 0:
            ax.text(0.145, value, f"{value} м", ha="right", va="center", fontsize=10.5, color=MUTED)
    ax.add_patch(Rectangle((0.16, -0.25), 0.30, 0.25, facecolor=WASH, edgecolor=LINE))

    x_positions = [0.39, 0.43, 0.49, 0.56, 0.64]
    for idx, (time, height, x) in enumerate(zip(T, H, x_positions)):
        ax.plot([0.24, x - 0.025], [height, height], color=GRID, linewidth=0.9, linestyle="--")
        ax.scatter(
            [x],
            [height],
            s=155,
            facecolor=RED if idx == len(T) - 1 else BLUE,
            edgecolor=PAPER,
            linewidth=1.3,
            zorder=4,
        )
        if idx == 0:
            label_x, label_y, horizontal, vertical = x + 0.06, height + 0.10, "left", "bottom"
            label = "$t=0.0$ c"
        elif idx == 1:
            label_x, label_y, horizontal, vertical = x + 0.085, height - 0.18, "left", "top"
            label = f"$t={str(round(time,1)).replace('.', '{,}')}$ c\n$h={f'{height:.2f}'.replace('.', '{,}')}$ м"
        else:
            label_x, label_y, horizontal, vertical = x + 0.045, height + (0.48 if idx < 4 else 0.60), "left", "bottom"
            label = f"$t={str(round(time,1)).replace('.', '{,}')}$ c\n$h={f'{height:.2f}'.replace('.', '{,}')}$ м"
        ax.text(
            label_x,
            label_y,
            label,
            ha=horizontal,
            va=vertical,
            fontsize=10.5,
            color=INK,
        )
    curve_t = np.linspace(0, 2, 120)
    curve_x = 0.39 + 0.25 * (curve_t / 2) ** 1.3
    curve_h = H0 - 0.5 * G_HAT * curve_t**2
    ax.plot(curve_x, curve_h, color=LINE, linewidth=1.1, linestyle=(0, (3, 3)))
    ax.text(0.03, 21.8, "Один опыт превращается в пять пар $(t_i,h_i)$", fontsize=17, fontweight="bold")
    ax.text(0.03, 20.8, "Положение измеряется линейкой, время — отдельным прибором.", color=MUTED, fontsize=12)
    ax.text(0.78, 3.5, "штриховая линия —\nещё не закон,\nа подсказка глазу", ha="center", color=MUTED, fontsize=10.5)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    boxes = [
        rounded_box(ax2, (0.05, 0.72), (0.36, 0.17), "Секундомер", "$t_i$ в секундах", edge=BLUE, fill="#edf3f7"),
        rounded_box(ax2, (0.58, 0.72), (0.36, 0.17), "Шкала высоты", "$h_i$ в метрах", edge=GREEN, fill="#edf4ef"),
        rounded_box(ax2, (0.315, 0.43), (0.36, 0.17), "Строка опыта", "$(t_i,h_i)$", edge=GOLD, fill="#f7f1df"),
        rounded_box(ax2, (0.315, 0.14), (0.36, 0.17), "Гипотеза", "$h=h_0-gt^2/2$", edge=RED, fill="#f7ece9"),
    ]
    connect(ax2, (0.23, 0.72), (0.43, 0.60), color=BLUE)
    connect(ax2, (0.76, 0.72), (0.57, 0.60), color=GREEN)
    connect(ax2, (0.495, 0.43), (0.495, 0.31), color=GOLD)
    ax2.text(0.5, 0.94, "Приборы не знают формулы", ha="center", fontsize=17, fontweight="bold")
    ax2.text(
        0.5,
        0.055,
        "Формула появляется после измерения —\nкак проверяемое предположение о механизме.",
        ha="center",
        fontsize=11.5,
        color=MUTED,
    )
    save(fig, OUT / "measurement-chain.png")


def least_squares() -> None:
    fig = plt.figure(figsize=(14.4, 8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.18, 1], height_ratios=[3.0, 1.0], hspace=0.16, wspace=0.30)
    ax = fig.add_subplot(grid[0, 0])
    axr = fig.add_subplot(grid[1, 0], sharex=ax)
    axs = fig.add_subplot(grid[:, 1])

    tt = np.linspace(0, 2.04, 240)
    fitted = H0 - 0.5 * G_HAT * tt**2
    reference = H0 - 0.5 * 9.81 * tt**2
    ax.plot(tt, reference, color=BLUE, linewidth=1.7, linestyle=(0, (5, 4)), label="$g=9{,}81$ м/с²")
    ax.plot(tt, fitted, color=RED, linewidth=2.2, label=fr"МНК: $\hat g={f'{G_HAT:.2f}'.replace('.', '{{,}}')}$ м/с²")
    ax.scatter(T, H, s=48, color=INK, zorder=4, label="измерения")
    for time, height in zip(T[1:], H[1:]):
        ax.annotate(
            f"{height:.2f}".replace(".", ","),
            (time, height),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color=MUTED,
        )
    ax.set_ylabel("высота, м")
    ax.set_xlim(-0.03, 2.08)
    ax.set_ylim(-0.3, 21)
    ax.set_title("Один параметр выбирается всей совокупностью точек", loc="left", pad=12)
    ax.legend(frameon=False, loc="lower left")
    clean(ax)
    ax.tick_params(labelbottom=False)

    predicted = H0 - 0.5 * G_HAT * T**2
    residuals = H - predicted
    bar_colors = [GREEN if value >= 0 else RED for value in residuals]
    axr.axhline(0, color=LINE, linewidth=1)
    axr.bar(T, residuals * 100, width=0.12, color=bar_colors)
    axr.set_xlabel("время, с")
    axr.set_ylabel("остаток,\nсм")
    axr.set_ylim(-7, 4)
    clean(axr)

    g_values = np.linspace(8.8, 10.5, 240)
    sse = np.array([np.sum((H - (H0 - 0.5 * value * T**2)) ** 2) for value in g_values])
    minimum = np.sum((H - predicted) ** 2)
    axs.plot(g_values, sse, color=RED, linewidth=2.4)
    axs.scatter([G_HAT], [minimum], s=70, color=GOLD, zorder=4)
    axs.axvline(G_HAT, color=GOLD, linewidth=1.1)
    axs.annotate(
        fr"$\hat g={f'{G_HAT:.2f}'.replace('.', '{{,}}')}$",
        xy=(G_HAT, minimum),
        xytext=(9.75, sse.max() * 0.25),
        arrowprops={"arrowstyle": "-|>", "color": GOLD, "connectionstyle": "arc3,rad=-0.18", "shrinkB": 6},
        color=GOLD,
        fontsize=12,
    )
    axs.set_xlabel("$g$, м/с²")
    axs.set_ylabel("SSE, м²")
    axs.set_title("Минимум суммы квадратов", loc="left", pad=12)
    clean(axs)
    fig.suptitle("Подгонка параметра не меняет физическую форму модели", x=0.08, ha="left", fontsize=18, fontweight="bold")
    save(fig, OUT / "least-squares.png")


def model_families() -> None:
    rng = np.random.default_rng(2026)
    train_t = np.linspace(0, 1.5, 8)

    def truth(t: np.ndarray) -> np.ndarray:
        return H0 - 4.905 * t**2 + 0.36 * t**3

    train_h = truth(train_t) + rng.normal(0, 0.055, size=train_t.size)
    z = train_t**2 / 2
    g_fit = float(np.sum(z * (H0 - train_h)) / np.sum(z**2))

    def physical(t: np.ndarray) -> np.ndarray:
        return H0 - 0.5 * g_fit * t**2

    beta = float(np.sum(train_t**3 * (train_h - physical(train_t))) / np.sum(train_t**6))

    def hybrid(t: np.ndarray) -> np.ndarray:
        return physical(t) + beta * t**3

    poly = np.polyfit(train_t, train_h, 7)
    tt = np.linspace(0, 2.75, 420)
    true_values = truth(tt)
    physical_values = physical(tt)
    hybrid_values = hybrid(tt)
    free_values = np.polyval(poly, tt)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14.4, 8), gridspec_kw={"height_ratios": [2.45, 1], "hspace": 0.12})
    for current in (ax, ax2):
        current.axvspan(1.5, 2.75, color=RED, alpha=0.05)
        current.axvline(1.5, color=INK, linewidth=1.1)
    ax.plot(tt, true_values, color=INK, linewidth=1.5, linestyle=(0, (5, 4)))
    ax.plot(tt, physical_values, color=BLUE, linewidth=2.1)
    ax.plot(tt, hybrid_values, color=RED, linewidth=2.3)
    ax.plot(tt, free_values, color=VIOLET, linewidth=1.8)
    ax.scatter(train_t, train_h, s=42, color=INK, zorder=4)
    ax.set_ylim(-12, 22)
    ax.set_ylabel("высота, м")
    ax.set_title("Одинаково точные на опыте модели продолжают кривую по-разному", loc="left", pad=12)
    ax.text(0.10, 20.5, "измерения", color=INK, fontsize=10.5)
    ax.text(1.58, 20.5, "новый режим", color=MUTED, fontsize=10.5)
    label_box = {"boxstyle": "round,pad=0.16", "fc": PAPER, "ec": "none", "alpha": 0.92}
    ax.text(2.06, physical(np.array([2.06]))[0] + 1.5, "механика", color=BLUE, fontsize=10.5, bbox=label_box)
    ax.text(2.18, hybrid(np.array([2.18]))[0] + 1.2, "гибрид", color=RED, fontsize=10.5, bbox=label_box)
    free_at = float(np.polyval(poly, 1.85))
    ax.text(1.83, np.clip(free_at, -8, 18), "полином 7-й степени", color=VIOLET, fontsize=10.5, bbox=label_box)
    ax.text(2.27, truth(np.array([2.27]))[0] - 2.0, "скрытая траектория", color=INK, fontsize=10.5, bbox=label_box)
    clean(ax)
    ax.tick_params(labelbottom=False)

    errors = {
        "механика": np.abs(physical_values - true_values),
        "гибрид": np.abs(hybrid_values - true_values),
        "полином 7": np.abs(free_values - true_values),
    }
    ax2.plot(tt, errors["механика"], color=BLUE, linewidth=1.9)
    ax2.plot(tt, errors["гибрид"], color=RED, linewidth=2.1)
    clipped_poly_error = np.clip(errors["полином 7"], 0, 12)
    ax2.plot(tt, clipped_poly_error, color=VIOLET, linewidth=1.7)
    overflow = errors["полином 7"] > 12
    if np.any(overflow):
        first_overflow = int(np.flatnonzero(overflow)[0])
        ax2.scatter(
            [tt[first_overflow]],
            [11.65],
            marker="^",
            s=64,
            color=VIOLET,
            zorder=5,
            clip_on=False,
        )
        ax2.text(
            tt[first_overflow] - 0.03,
            10.55,
            "ошибка выше 12 м,\nвне шкалы",
            ha="right",
            va="top",
            color=VIOLET,
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.18", "fc": PAPER, "ec": "none", "alpha": 0.94},
        )
    ax2.set_ylim(0, 12)
    ax2.set_xlabel("время, с")
    ax2.set_ylabel("абсолютная\nошибка, м")
    clean(ax2)
    save(fig, OUT / "model-families.png")


def identifiability_ridge() -> None:
    mass = 80.0
    rho = 1.2
    g = 9.81
    observed_v = 5.0
    product = 2 * mass * g / (rho * observed_v**2)
    cd = np.linspace(0.55, 1.85, 260)
    area = np.linspace(26, 86, 260)
    CD, AREA = np.meshgrid(cd, area)
    velocity = np.sqrt(2 * mass * g / (rho * CD * AREA))
    loss = ((velocity - observed_v) / 0.08) ** 2

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.4, 8), gridspec_kw={"wspace": 0.30})
    ax.contourf(CD, AREA, np.minimum(loss, 25), levels=[0, 0.5, 2, 6, 12, 25], colors=["#edf4ef", "#e4ece7", "#f3ecda", "#f1ddd7", "#ead0c8"])
    contour = ax.contour(CD, AREA, loss, levels=[1, 4, 9], colors=[GREEN, GOLD, RED], linewidths=[1.8, 1.3, 1.0])
    ax.clabel(contour, inline=True, fontsize=9, fmt={1: r"$1\sigma$", 4: r"$2\sigma$", 9: r"$3\sigma$"})
    ridge_cd = np.linspace(0.62, 1.82, 200)
    ridge_area = product / ridge_cd
    ax.plot(ridge_cd, ridge_area, color=INK, linewidth=1.6, linestyle=(0, (6, 4)))
    pairs = [(0.8, product / 0.8), (1.2, product / 1.2), (1.6, product / 1.6)]
    for index, (c_value, a_value) in enumerate(pairs, start=1):
        ax.scatter([c_value], [a_value], s=55, color=[BLUE, RED, VIOLET][index - 1], zorder=4)
        ax.annotate(
            f"{index}",
            (c_value, a_value),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xlim(cd.min(), cd.max())
    ax.set_ylim(area.min(), area.max())
    ax.set_xlabel("коэффициент сопротивления $C_d$")
    ax.set_ylabel("площадь купола $A$, м²")
    ax.set_title("Долина одинаковой скорости", loc="left", pad=12)
    clean(ax)

    labels = [f"{index}\n$C_d={str(c_value).replace('.', '{,}')}$\n$A={str(a_value).replace('.', '{,}')}$ м²" for index, (c_value, a_value) in enumerate(pairs, start=1)]
    velocities = [np.sqrt(2 * mass * g / (rho * c_value * a_value)) for c_value, a_value in pairs]
    bars = ax2.bar(labels, velocities, color=[BLUE, RED, VIOLET], width=0.55)
    ax2.axhline(observed_v, color=INK, linewidth=1.2, linestyle=(0, (5, 4)))
    for bar, value in zip(bars, velocities):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f} м/с".replace(".", ","), ha="center", fontsize=11)
    ax2.set_ylim(0, 5.9)
    ax2.set_ylabel("терминальная скорость")
    ax2.set_title("$C_dA$ одинаково — прогноз не различает параметры", loc="left", pad=12)
    clean(ax2)
    fig.suptitle("Одно наблюдение определяет произведение, а не два множителя", x=0.08, ha="left", fontsize=18, fontweight="bold")
    save(fig, OUT / "identifiability-ridge.png")


def heat_balance() -> None:
    hours = np.linspace(0, 24, 97)
    dt = hours[1] - hours[0]
    outside = 4 + 3 * np.sin(2 * np.pi * (hours - 5) / 24)
    heater = np.where((hours >= 6) & (hours <= 21), 5.5, 2.2)
    observed = np.zeros_like(hours)
    physical = np.zeros_like(hours)
    hybrid = np.zeros_like(hours)
    observed[0] = physical[0] = hybrid[0] = 19.5
    capacity = 7.5
    leakage = 0.65
    for index in range(1, len(hours)):
        door_loss = 4.0 if 14 <= hours[index - 1] <= 16.5 else 0.0
        observed[index] = observed[index - 1] + dt * (
            heater[index - 1] - leakage * (observed[index - 1] - outside[index - 1]) - door_loss
        ) / capacity
        physical[index] = physical[index - 1] + dt * (
            heater[index - 1] - leakage * (physical[index - 1] - outside[index - 1])
        ) / capacity
        learned_door = 3.7 if 14 <= hours[index - 1] <= 16.5 else 0.0
        hybrid[index] = hybrid[index - 1] + dt * (
            heater[index - 1] - leakage * (hybrid[index - 1] - outside[index - 1]) - learned_door
        ) / capacity

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.4, 8), gridspec_kw={"width_ratios": [0.88, 1.35], "wspace": 0.28})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    building = FancyBboxPatch((0.22, 0.28), 0.56, 0.42, boxstyle="round,pad=0.02,rounding_size=0.025", facecolor="#edf3f7", edgecolor=BLUE, linewidth=1.5)
    ax.add_patch(building)
    ax.add_patch(Rectangle((0.43, 0.28), 0.14, 0.25, facecolor=PAPER, edgecolor=LINE))
    ax.text(0.50, 0.57, "воздух внутри", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.50, 0.49, "$C\\,dT/dt$", ha="center", fontsize=13, color=BLUE)
    connect(ax, (0.04, 0.49), (0.20, 0.49), color=RED)
    ax.text(0.04, 0.55, "обогрев $Q$", ha="left", color=RED, fontsize=11)
    connect(ax, (0.80, 0.49), (0.96, 0.49), color=GREEN)
    ax.text(0.96, 0.55, "потери $k(T-T_{out})$", ha="right", color=GREEN, fontsize=10.5)
    ax.text(0.50, 0.83, "Тепловой баланс задаёт форму", ha="center", fontsize=17, fontweight="bold")
    ax.text(0.50, 0.16, "$C\\,\\dfrac{dT}{dt}=Q-k(T-T_{out})$", ha="center", fontsize=15)
    ax.text(0.50, 0.08, "открытая дверь появится как структурированный остаток", ha="center", color=MUTED, fontsize=10.5)

    ax2.axvspan(14, 16.5, color=GOLD, alpha=0.09)
    ax2.plot(hours, observed, color=INK, linewidth=1.6, linestyle=(0, (5, 4)), label="измерение")
    ax2.plot(hours, physical, color=BLUE, linewidth=2.0, label="физическая модель")
    ax2.plot(hours, hybrid, color=RED, linewidth=2.2, label="физика + остаток")
    ax2.plot(hours, outside, color=GREEN, linewidth=1.1, alpha=0.75, label="снаружи")
    ax2.text(15.25, 23.0, "дверь открыта", ha="center", color=GOLD, fontsize=10.5)
    ax2.set_xlim(0, 24)
    ax2.set_ylim(2, 24)
    ax2.set_xlabel("время суток, ч")
    ax2.set_ylabel("температура, °C")
    ax2.set_title("Остаток переносит повторяющийся эффект", loc="left", pad=12)
    ax2.legend(frameon=False, ncol=2, loc="lower left")
    clean(ax2)
    save(fig, OUT / "heat-balance.png")


def model_source_map() -> None:
    fig, ax = plt.subplots(figsize=(14.4, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    columns = [
        (0.055, "До данных", BLUE, "#edf3f7", [
            ("закон", "$h=h_0+v_0t-gt^2/2$"),
            ("единицы", "м, с, м/с²"),
            ("границы", "нет воздуха и вращения"),
        ]),
        (0.365, "Из опыта", RED, "#f7ece9", [
            ("параметры", "$\\hat g$, смещение датчика"),
            ("остаток", "$r_i=h_i-\\hat h_i$"),
            ("неопределённость", "разброс и корреляции"),
        ]),
        (0.675, "На новом режиме", GREEN, "#edf4ef", [
            ("прогноз", "ошибка вне train"),
            ("механизм", "ответ на вмешательство"),
            ("решение", "допустимые последствия"),
        ]),
    ]
    for x, heading, color, fill, rows in columns:
        ax.text(x + 0.13, 0.89, heading, ha="center", fontsize=16, fontweight="bold", color=color)
        for index, (title, detail) in enumerate(rows):
            y = 0.68 - index * 0.19
            rounded_box(ax, (x, y), (0.26, 0.13), title, detail, edge=color, fill=fill)
    ax.text(0.340, 0.51, "+", ha="center", va="center", fontsize=24, color=MUTED)
    ax.text(0.650, 0.51, "+", ha="center", va="center", fontsize=24, color=MUTED)
    ax.text(0.5, 0.96, "Три источника доказательств нельзя заменить одной метрикой", ha="center", fontsize=18, fontweight="bold")
    ax.text(
        0.5,
        0.10,
        "Структура задаёт возможные продолжения · данные выбирают числа · новый режим проверяет перенос",
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    save(fig, OUT / "model-source-map.png")


def side_forces() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Circle((0.5, 0.52), 0.09, facecolor="#edf3f7", edgecolor=BLUE, linewidth=1.3))
    ax.plot([0.42, 0.58], [0.65, 0.65], color=INK, linewidth=1.1)
    ax.plot([0.42, 0.5], [0.65, 0.61], color=INK, linewidth=1.0)
    ax.plot([0.58, 0.5], [0.65, 0.61], color=INK, linewidth=1.0)
    connect(ax, (0.5, 0.62), (0.5, 0.90), color=GREEN)
    connect(ax, (0.5, 0.42), (0.5, 0.12), color=RED)
    ax.text(0.55, 0.84, "$F_{drag}=\\rho C_dAv^2/2$", color=GREEN, fontsize=10)
    ax.text(0.55, 0.18, "$mg$", color=RED, fontsize=11)
    ax.text(0.5, 0.97, "при $v=v_\\infty$ силы равны", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.02, "равенство сил определяет скорость,\nно видит $C_d$ и $A$ только произведением", ha="center", color=MUTED, fontsize=9.2)
    save(fig, SIDE / "parachute-forces.png", dpi=200)


def side_units() -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows = [
        ("$h_0$", "$[\\mathrm{m}]$", BLUE),
        ("$v_0t$", "$[\\mathrm{m/s}]\\,[\\mathrm{s}]=[\\mathrm{m}]$", GREEN),
        ("$gt^2/2$", "$[\\mathrm{m/s^2}]\\,[\\mathrm{s^2}]=[\\mathrm{m}]$", RED),
    ]
    for index, (term, units, color) in enumerate(rows):
        y = 0.80 - index * 0.24
        ax.add_patch(FancyBboxPatch((0.06, y - 0.10), 0.88, 0.18, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=WASH, edgecolor=LINE))
        ax.text(0.16, y, term, ha="center", va="center", fontsize=13, color=color)
        ax.text(0.56, y, units, ha="center", va="center", fontsize=10.5)
    ax.text(0.5, 0.96, "складывать можно величины одной размерности", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(0.5, 0.045, "миллисекунда, поданная как секунда,\nувеличит квадратичный член в миллион раз", ha="center", color=MUTED, fontsize=9.2)
    save(fig, SIDE / "units-ledger.png", dpi=200)


def main() -> None:
    measurement_chain()
    least_squares()
    model_families()
    identifiability_ridge()
    heat_balance()
    model_source_map()
    side_forces()
    side_units()


if __name__ == "__main__":
    main()
