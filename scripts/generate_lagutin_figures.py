"""Deterministic editorial figures for the first textbook module.

The visual language follows the article typography: Palatino, restrained ink,
one semantic accent per claim, direct labels instead of detached legends.
Every SVG is meant to be read together with a numbered article caption.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "figures" / "lessons"

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
        "font.size": 12.5,
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
    fig = plt.figure(figsize=(width, height), layout="constrained")
    return fig


def clean(ax, *, xgrid: bool = False, ygrid: bool = False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(FAINT)
    ax.tick_params(length=3, width=0.7)
    if xgrid:
        ax.grid(True, axis="x", color=GRID, linewidth=0.7)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, linewidth=0.7)


def label(ax, x, y, text, color=INK, size=11.5, ha="left", va="center", weight="normal"):
    ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va, weight=weight)


def box(ax, xy, width, height, text, *, color=BLUE, fill=PAPER, size=12, radius=0.02):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=1.35,
        edgecolor=color,
        facecolor=fill,
    )
    ax.add_patch(patch)
    label(ax, xy[0] + width / 2, xy[1] + height / 2, text, color=color, size=size, ha="center")
    return patch


def arrow(ax, start, end, *, color=MUTED, style="-|>", width=1.25, curve=0.0, shrink=10):
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


def diagram_axes(fig):
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def direct_line_label(ax, x, y, text, color, *, dy=0.0, ha="left"):
    ax.text(
        x,
        y + dy,
        text,
        color=color,
        fontsize=10.8,
        ha=ha,
        va="center",
        bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1.5, "alpha": 0.93},
    )


def lesson01(index: int):
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        nodes = [
            (0.06, 0.59, 0.17, 0.15, "наблюдения", BLUE),
            (0.30, 0.59, 0.17, 0.15, "модель", VIOLET),
            (0.54, 0.59, 0.17, 0.15, "решение", RED),
            (0.78, 0.59, 0.17, 0.15, "мир", GREEN),
        ]
        for x, y, w, h, text, color in nodes:
            box(ax, (x, y), w, h, text, color=color, fill=PAPER)
        for left, right in zip(nodes[:-1], nodes[1:]):
            arrow(ax, (left[0] + left[2], 0.665), (right[0], 0.665))
        ax.plot([0.865, 0.865, 0.145], [0.59, 0.39, 0.39], color=GOLD, linewidth=1.25)
        arrow(ax, (0.145, 0.39), (0.145, 0.59), color=GOLD, shrink=6)
        label(ax, 0.505, 0.27, "обратная связь: что изменилось после действия?", GOLD, 12, ha="center")
        label(ax, 0.505, 0.87, "правило поведения живёт во всём цикле, а не в одном блоке", MUTED, 11, ha="center")
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axvspan(50, 100, color=WASH)
        ax.axhspan(50, 100, color=WASH, alpha=0.55)
        points = [
            (8, 12, "калькулятор", MUTED),
            (12, 84, "шлагбаум", GOLD),
            (78, 72, "фильтр спама", RED),
            (84, 28, "помощник врача", GREEN),
            (50, 45, "маршрутизатор", VIOLET),
        ]
        for x, y, text, color in points:
            ax.scatter([x], [y], s=52, facecolor=PAPER, edgecolor=color, linewidth=1.8, zorder=3)
            ax.annotate(text, (x, y), xytext=(7, 7), textcoords="offset points", color=color, fontsize=10.8)
        ax.set_xlabel("какая доля правила найдена по данным")
        ax.set_ylabel("насколько самостоятельно действует система")
        ax.set_xticks([0, 50, 100], ["записано", "смешано", "обучено"])
        ax.set_yticks([0, 50, 100], ["совет", "совместно", "сама"])
        clean(ax, xgrid=True, ygrid=True)
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    x = np.linspace(0, 1, 250)
    train = 0.08 + 0.05 * x
    future = 0.10 + 0.05 * x + 0.30 / (1 + np.exp(-18 * (x - 0.64)))
    ax.plot(x, train, color=BLUE, linewidth=2.2)
    ax.plot(x, future, color=RED, linewidth=2.2)
    ax.fill_between(x, train, future, where=future >= train, color=RED, alpha=0.12)
    ax.axvline(0.62, color=FAINT, linewidth=1, linestyle=(0, (4, 4)))
    direct_line_label(ax, 0.18, train[45], "ошибка в знакомой среде", BLUE, dy=-0.025)
    direct_line_label(ax, 0.72, future[180], "ошибка после смены условий", RED, dy=0.035)
    label(ax, 0.62, 0.5, "меняется мир", MUTED, 10.5, ha="center")
    ax.set_xlabel("время после запуска")
    ax.set_ylabel("доля ошибочных решений")
    ax.set_ylim(0, 0.58)
    ax.set_yticks([0, 0.2, 0.4], ["0%", "20%", "40%"])
    ax.set_xticks([])
    clean(ax, ygrid=True)
    return fig


def lesson02(index: int):
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        t = np.linspace(0, 1.2, 250)
        truth = 1 - 0.62 * t**2 + 0.14 * t**3
        physics = 1 - 0.62 * t**2
        hybrid = physics + 0.135 * t**3
        ax.axvspan(0.82, 1.2, color=WASH)
        ax.scatter(np.linspace(0.05, 0.78, 14), truth[np.linspace(10, 162, 14).astype(int)],
                   s=18, color=INK, zorder=4)
        ax.plot(t, physics, color=BLUE, linewidth=2)
        ax.plot(t, hybrid, color=RED, linewidth=2.4)
        ax.plot(t, truth, color=GREEN, linewidth=1.4, linestyle=(0, (4, 3)))
        direct_line_label(ax, 0.18, 0.90, "закон механики", BLUE)
        direct_line_label(ax, 0.55, 0.79, "закон + обученная поправка", RED)
        direct_line_label(ax, 0.91, 0.61, "скрытая траектория", GREEN)
        label(ax, 1.01, 0.17, "проверка\nпереноса", MUTED, 10.5, ha="center")
        ax.set_xlabel("время полёта")
        ax.set_ylabel("высота")
        ax.set_xticks([])
        ax.set_yticks([])
        clean(ax)
        return fig
    if index == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), layout="constrained", sharey=True)
        x = np.linspace(-2, 2, 250)
        truth = 0.2 + 0.27 * x + 0.28 * x**2
        for ax, span, title in zip(axes, [0.8, 1.8], ["узкий опыт", "широкий опыт"]):
            mask = np.abs(x) <= span
            coef = np.polyfit(x[mask], truth[mask], 1)
            pred = np.polyval(coef, x)
            ax.plot(x, truth, color=GREEN, linewidth=2.2)
            ax.plot(x, pred, color=RED, linewidth=2)
            ax.scatter(x[mask][::15], truth[mask][::15], color=INK, s=16, zorder=3)
            ax.axvspan(-span, span, color=BLUE, alpha=0.07)
            ax.set_title(title, fontsize=13, fontstyle="italic")
            ax.set_xlabel("режим системы")
            ax.set_xticks([])
            clean(ax, ygrid=True)
        axes[0].set_ylabel("наблюдаемый эффект")
        direct_line_label(axes[0], -1.8, truth[12], "реальный закон", GREEN)
        direct_line_label(axes[1], 0.7, np.polyval(np.polyfit(x[np.abs(x)<=1.8], truth[np.abs(x)<=1.8], 1), 0.7),
                          "линейная модель", RED, dy=-0.11)
        return fig
    fig = canvas()
    ax = diagram_axes(fig)
    box(ax, (0.06, 0.58), 0.19, 0.14, "физический мир", color=GREEN)
    box(ax, (0.34, 0.58), 0.19, 0.14, "датчик", color=GOLD)
    box(ax, (0.62, 0.58), 0.19, 0.14, "таблица", color=BLUE)
    box(ax, (0.62, 0.26), 0.19, 0.14, "модель", color=VIOLET)
    for a, b in [((0.25, 0.65), (0.34, 0.65)), ((0.53, 0.65), (0.62, 0.65)), ((0.715, 0.58), (0.715, 0.40))]:
        arrow(ax, a, b)
    arrow(ax, (0.62, 0.33), (0.25, 0.33), color=RED)
    label(ax, 0.435, 0.78, "калибровка", GOLD, 10.5, ha="center")
    label(ax, 0.435, 0.25, "прогноз возвращается в мир", RED, 10.5, ha="center")
    label(ax, 0.91, 0.65, "каждая стрелка\nсодержит допущение", MUTED, 11, ha="center")
    return fig


def lesson03(index: int):
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        stages = [
            (0.03, "явление", GREEN),
            (0.23, "измерение", GOLD),
            (0.43, "признаки", BLUE),
            (0.63, "строка", VIOLET),
            (0.83, "решение", RED),
        ]
        for x, text, color in stages:
            box(ax, (x, 0.56), 0.14, 0.13, text, color=color, size=11)
        for a, b in zip(stages[:-1], stages[1:]):
            arrow(ax, (a[0] + 0.14, 0.625), (b[0], 0.625), shrink=7)
        losses = ["контекст", "точность", "единицы", "редкие случаи"]
        for i, text in enumerate(losses):
            x = 0.30 + i * 0.18
            arrow(ax, (x, 0.54), (x, 0.38), color=FAINT, style="-", shrink=2)
            label(ax, x, 0.31, text, MUTED, 10.2, ha="center")
        label(ax, 0.5, 0.83, "таблица хранит след мира, а не сам мир", INK, 13, ha="center", weight="bold")
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        rng = np.random.default_rng(17)
        t = np.arange(0, 96)
        base = 42 + 13 * np.sin(2 * np.pi * (t - 5) / 24) + rng.normal(0, 2.2, len(t))
        observed = base.copy()
        missing = ((t % 24 >= 12) & (t % 24 <= 16) & (base > 44)) | ((t + 3) % 29 == 0)
        observed[missing] = np.nan
        ax.plot(t, base, color=FAINT, linewidth=1.2, label="_nolegend_")
        ax.plot(t, observed, color=BLUE, linewidth=2.1)
        ax.scatter(t[~missing], observed[~missing], s=13, color=BLUE)
        ax.scatter(t[missing], base[missing], s=30, facecolor=PAPER, edgecolor=RED, linewidth=1.5)
        direct_line_label(ax, 8, base[8], "видимые показания", BLUE, dy=9)
        direct_line_label(ax, 56, base[56], "пропуски зависят от величины", RED, dy=10)
        ax.set_xlabel("четыре суток, часы")
        ax.set_ylabel("показание сенсора")
        ax.set_yticks([])
        clean(ax, ygrid=True)
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), layout="constrained")
    matrix = np.array([[42, 8, 2], [6, 28, 5], [1, 7, 31]])
    ax = axes[0]
    ax.imshow(matrix, cmap=mpl.colors.LinearSegmentedColormap.from_list("paperblue", [PAPER, BLUE]))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color=PAPER if matrix[i, j] > 24 else INK, fontsize=14)
    ax.set_xticks(range(3), ["кошка", "лиса", "собака"])
    ax.set_yticks(range(3), ["эксперт A", "эксперт B", "эксперт C"])
    ax.set_title("разметка не всегда единогласна", fontsize=13, fontstyle="italic")
    axes[1].axis("off")
    box(axes[1], (0.12, 0.64), 0.22, 0.12, "объект 17", color=VIOLET)
    box(axes[1], (0.62, 0.64), 0.22, 0.12, "объект 17", color=VIOLET)
    box(axes[1], (0.12, 0.27), 0.22, 0.12, "обучение", color=BLUE)
    box(axes[1], (0.62, 0.27), 0.22, 0.12, "тест", color=RED)
    arrow(axes[1], (0.23, 0.64), (0.23, 0.39), color=BLUE)
    arrow(axes[1], (0.73, 0.64), (0.73, 0.39), color=RED)
    arrow(axes[1], (0.34, 0.70), (0.62, 0.70), color=RED, curve=-0.2)
    label(axes[1], 0.48, 0.86, "дубликат создаёт утечку", RED, 12, ha="center", weight="bold")
    return fig


def lesson04(index: int):
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        cells = 11
        x0, y0, w, h = 0.08, 0.55, 0.075, 0.12
        symbols = ["1", "0", "1", "1", "□", "0", "1", "□", "□", "□", "□"]
        for i in range(cells):
            ax.add_patch(Rectangle((x0 + i * w, y0), w, h, facecolor=PAPER, edgecolor=INK, linewidth=1))
            label(ax, x0 + (i + 0.5) * w, y0 + h / 2, symbols[i], INK, 13, ha="center")
        head_x = x0 + 4 * w
        box(ax, (head_x - 0.01, 0.29), w + 0.02, 0.12, "головка", color=RED, size=10.5)
        arrow(ax, (head_x + w / 2, 0.41), (head_x + w / 2, 0.55), color=RED, shrink=4)
        box(ax, (0.70, 0.25), 0.21, 0.16, "таблица правил\n(состояние, символ)", color=BLUE, size=10.5)
        arrow(ax, (0.69, 0.33), (head_x + w + 0.02, 0.33), color=BLUE)
        label(ax, 0.5, 0.82, "одна и та же машина читает разные программы как данные", MUTED, 12, ha="center")
        return fig
    if index == 2:
        fig = canvas()
        ax = diagram_axes(fig)
        criteria = [
            (0.18, 0.66, "похож на человека", RED),
            (0.50, 0.75, "фактически верен", GREEN),
            (0.72, 0.42, "объясняет ход", BLUE),
        ]
        for x, y, text, color in criteria:
            ax.add_patch(Circle((x, y), 0.145, facecolor=color, edgecolor="none", alpha=0.13))
            ax.add_patch(Circle((x, y), 0.145, facecolor="none", edgecolor=color, linewidth=1.6))
            label(ax, x, y, text, color, 11, ha="center", weight="bold")
        ax.scatter([0.47], [0.58], s=60, color=INK, zorder=4)
        label(ax, 0.47, 0.50, "полезный ответ", INK, 10.5, ha="center")
        label(ax, 0.49, 0.19, "одна поведенческая оценка не измеряет все три свойства", MUTED, 12, ha="center")
        return fig
    fig = canvas()
    ax = diagram_axes(fig)
    programs = ["останавливается", "печатает 0", "зацикливается", "?"]
    for i, text in enumerate(programs):
        y = 0.75 - i * 0.16
        box(ax, (0.08, y), 0.25, 0.10, f"P{i+1}: {text}", color=BLUE if i < 3 else RED, size=10.5)
        box(ax, (0.63, y), 0.22, 0.10, "да" if i < 2 else ("нет" if i == 2 else "противоречие"),
            color=GREEN if i < 2 else RED, size=10.5)
        arrow(ax, (0.33, y + 0.05), (0.63, y + 0.05))
    box(ax, (0.39, 0.38), 0.18, 0.18, "идеальный\nрешатель\nостановки", color=VIOLET, size=11)
    label(ax, 0.5, 0.13, "программа может получить собственное описание на вход", MUTED, 11.5, ha="center")
    return fig


def lesson05(index: int):
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        circles = [
            (0.31, 0.57, 0.23, BLUE, "статистика\nнеопределённость"),
            (0.56, 0.57, 0.23, VIOLET, "машинное обучение\nпрогноз"),
            (0.47, 0.36, 0.23, RED, "ИИ-система\nдействие"),
        ]
        for x, y, r, color, text in circles:
            ax.add_patch(Circle((x, y), r, edgecolor=color, facecolor=color, alpha=0.11, linewidth=1.5))
            label(ax, x, y, text, color, 12, ha="center", weight="bold")
        label(ax, 0.85, 0.62, "области пересекаются,\nно задают разные вопросы", MUTED, 11.5, ha="center")
        return fig
    if index == 2:
        fig = canvas()
        ax = diagram_axes(fig)
        stages = [
            (0.06, 0.56, 0.20, "оценить\nспрос ± ошибка", BLUE),
            (0.40, 0.56, 0.20, "предсказать\nследующий час", VIOLET),
            (0.74, 0.56, 0.20, "выпустить\nавтобусы", RED),
        ]
        for x, y, w, text, color in stages:
            box(ax, (x, y), w, 0.16, text, color=color, size=11.5)
        arrow(ax, (0.26, 0.64), (0.40, 0.64))
        arrow(ax, (0.60, 0.64), (0.74, 0.64))
        ax.plot([0.84, 0.84, 0.16], [0.56, 0.39, 0.39], color=GREEN, linewidth=1.25)
        arrow(ax, (0.16, 0.39), (0.16, 0.56), color=GREEN, shrink=6)
        label(ax, 0.50, 0.28, "пассажиры меняют поведение: действие создаёт новые данные", GREEN, 11.5, ha="center")
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    n = np.arange(10, 501)
    se = 0.5 / np.sqrt(n)
    decision = 2 * se + 0.018
    ax.plot(n, se, color=BLUE, linewidth=2.2)
    ax.plot(n, decision, color=RED, linewidth=2.2)
    ax.fill_between(n, se, decision, color=GOLD, alpha=0.12)
    direct_line_label(ax, 70, se[60], "неопределённость оценки", BLUE, dy=0.007)
    direct_line_label(ax, 250, decision[240], "риск решения с запасом", RED, dy=0.006)
    label(ax, 365, 0.055, "цена ошибки не исчезает\nпри большой выборке", GOLD, 11, ha="center")
    ax.set_xlabel("число наблюдений")
    ax.set_ylabel("масштаб риска")
    ax.set_yticks([])
    clean(ax, ygrid=True)
    return fig


def lesson06(index: int):
    modes = [
        ("классификация", "готовый класс", RED),
        ("регрессия", "готовое число", BLUE),
        ("кластеризация", "структура входов", VIOLET),
        ("semi-supervised", "редкие метки", GOLD),
        ("подкрепление", "отложенная награда", GREEN),
    ]
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        for i, (name, signal, color) in enumerate(modes):
            y = 0.82 - i * 0.16
            box(ax, (0.06, y), 0.28, 0.10, name, color=color, size=10.8)
            arrow(ax, (0.34, y + 0.05), (0.55, y + 0.05), color=color)
            box(ax, (0.55, y), 0.28, 0.10, signal, color=color, fill=WASH, size=10.8)
        label(ax, 0.46, 0.93, "какой сигнал говорит алгоритму, что получилось лучше?", MUTED, 12, ha="center")
        return fig
    if index == 2:
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 5.8), layout="constrained")
        rng = np.random.default_rng(8)
        x = np.r_[rng.normal(-1, 0.35, 24), rng.normal(1, 0.35, 24)]
        y = np.r_[rng.normal(0.7, 0.45, 24), rng.normal(-0.5, 0.45, 24)]
        colors = np.array([BLUE] * 24 + [RED] * 24)
        axes[0].scatter(x, y, c=colors, s=23)
        axes[0].axvline(0, color=INK, linewidth=1)
        axes[0].set_title("даны классы", fontsize=12.5, fontstyle="italic")
        axes[1].scatter(x, y, c=VIOLET, s=23)
        axes[1].scatter([-1, 1], [0.7, -0.5], s=120, facecolor=PAPER, edgecolor=GOLD, linewidth=2)
        axes[1].set_title("видна структура", fontsize=12.5, fontstyle="italic")
        axes[2].scatter(x, y, facecolor=PAPER, edgecolor=FAINT, s=23)
        axes[2].scatter(x[[4, 27, 39]], y[[4, 27, 39]], c=[BLUE, RED, RED], s=44)
        axes[2].set_title("меток мало", fontsize=12.5, fontstyle="italic")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            clean(ax)
        return fig
    fig = canvas()
    ax = diagram_axes(fig)
    for i in range(7):
        x = 0.07 + i * 0.13
        color = GREEN if i == 6 else BLUE
        ax.add_patch(Circle((x, 0.58), 0.035, facecolor=color, edgecolor="none"))
        if i < 6:
            arrow(ax, (x + 0.035, 0.58), (x + 0.095, 0.58), color=FAINT, shrink=1)
        label(ax, x, 0.47, f"a{i+1}", MUTED, 10, ha="center")
    arrow(ax, (0.85, 0.64), (0.85, 0.78), color=GREEN, shrink=3)
    box(ax, (0.74, 0.78), 0.22, 0.11, "награда +1", color=GREEN, size=11)
    arrow(ax, (0.76, 0.84), (0.20, 0.84), color=GOLD, curve=0.18, shrink=8)
    label(ax, 0.50, 0.26, "какому из шести действий принадлежит успех?", GOLD, 11.5, ha="center")
    label(ax, 0.50, 0.15, "ответ приходит после цепочки", MUTED, 11.5, ha="center")
    return fig


def lesson07(index: int):
    rng = np.random.default_rng(12)
    a = rng.normal((-0.8, 0.45), (0.55, 0.48), (48, 2))
    b = rng.normal((0.75, -0.35), (0.58, 0.52), (48, 2))
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        ax.scatter(a[:, 0], a[:, 1], facecolor=PAPER, edgecolor=BLUE, s=31)
        ax.scatter(b[:, 0], b[:, 1], facecolor=PAPER, edgecolor=RED, s=31)
        xx = np.linspace(-2.5, 2.5, 2)
        ax.plot(xx, 0.25 - 0.8 * xx, color=INK, linewidth=1.6)
        ax.fill_between(xx, 0.25 - 0.8 * xx, 2.5, color=BLUE, alpha=0.06)
        ax.fill_between(xx, -2.5, 0.25 - 0.8 * xx, color=RED, alpha=0.06)
        label(ax, -1.45, 1.65, "полезное", BLUE, 11)
        label(ax, 1.25, -1.45, "нежелательное", RED, 11)
        label(ax, 0.25, 0.15, "$w^T x+b=0$", INK, 11, ha="center")
        ax.set_xlabel("признак 1")
        ax.set_ylabel("признак 2")
        ax.set_xticks([])
        ax.set_yticks([])
        clean(ax)
        return fig
    if index == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), layout="constrained")
        thresholds = [0.34, 0.72]
        for ax, tau in zip(axes, thresholds):
            cm = np.array([[int(82 * tau), int(18 * (1 - tau) + 5)],
                           [int(26 * tau + 4), int(74 * (1 - 0.55 * tau))]])
            ax.imshow(cm, cmap=mpl.colors.LinearSegmentedColormap.from_list("pblue", [PAPER, BLUE]))
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                            color=PAPER if cm[i, j] > 45 else INK)
            ax.set_xticks([0, 1], ["пропустить", "заблокировать"])
            ax.set_yticks([0, 1], ["обычное", "спам"])
            ax.set_title(f"порог $\\tau={tau:.2f}$", fontsize=13, fontstyle="italic")
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    score = np.linspace(0.02, 0.98, 12)
    observed = np.clip(score + 0.12 * np.sin(score * 7) - 0.05, 0, 1)
    ax.plot([0, 1], [0, 1], color=FAINT, linewidth=1.3, linestyle=(0, (4, 3)))
    ax.plot(score, observed, color=RED, marker="o", markersize=5, linewidth=2)
    ax.fill_between(score, score, observed, color=GOLD, alpha=0.12)
    direct_line_label(ax, 0.1, 0.13, "идеальная калибровка", FAINT)
    direct_line_label(ax, 0.61, observed[7], "реальная модель", RED, dy=0.08)
    ax.set_xlabel("обещанная вероятность")
    ax.set_ylabel("наблюдаемая доля класса")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0,5", "1"])
    ax.set_yticks([0, 0.5, 1], ["0", "0,5", "1"])
    clean(ax, xgrid=True, ygrid=True)
    return fig


def lesson08(index: int):
    rng = np.random.default_rng(21)
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        t = np.arange(72)
        actual = 12 + 0.09 * t + 2.4 * np.sin(t / 5) + rng.normal(0, 0.7, len(t))
        pred = 12 + 0.09 * t + 2.1 * np.sin((t - 1) / 5)
        width = 1.3 + 0.025 * t
        ax.fill_between(t, pred - width, pred + width, color=BLUE, alpha=0.12)
        ax.plot(t, pred, color=BLUE, linewidth=2)
        ax.scatter(t, actual, s=13, color=INK)
        direct_line_label(ax, 8, pred[8], "прогноз", BLUE, dy=1.5)
        label(ax, 57, pred[57] + width[57] + 0.8, "интервал расширяется", MUTED, 10.8, ha="center")
        ax.set_xlabel("время поездки")
        ax.set_ylabel("задержка, мин")
        ax.set_xticks([])
        clean(ax, ygrid=True)
        return fig
    if index == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), layout="constrained", sharey=True)
        fitted = np.linspace(2, 20, 70)
        residual_good = rng.normal(0, 1.2, 70)
        residual_bad = 0.055 * (fitted - 11) ** 2 - 1.5 + rng.normal(0, 0.65, 70)
        for ax, res, title in zip(axes, [residual_good, residual_bad], ["остатки без узора", "пропущена нелинейность"]):
            ax.axhline(0, color=INK, linewidth=1)
            ax.scatter(fitted, res, s=19, facecolor=PAPER, edgecolor=BLUE if ax is axes[0] else RED)
            ax.set_title(title, fontsize=13, fontstyle="italic")
            ax.set_xlabel("прогноз $\\hat y$")
            clean(ax, ygrid=True)
        axes[0].set_ylabel("остаток $y-\\hat y$")
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    outliers = np.arange(0, 11)
    mae = 2 + 0.42 * outliers
    rmse = np.sqrt(4 + 3.3 * outliers**2)
    ax.plot(outliers, mae, color=BLUE, linewidth=2.2)
    ax.plot(outliers, rmse, color=RED, linewidth=2.2)
    direct_line_label(ax, 7.1, mae[7], "MAE", BLUE, dy=-0.7)
    direct_line_label(ax, 7.1, rmse[7], "RMSE", RED, dy=1.0)
    ax.set_xlabel("величина одного редкого промаха")
    ax.set_ylabel("значение метрики")
    ax.set_xticks([0, 5, 10])
    clean(ax, ygrid=True)
    return fig


def lesson09(index: int):
    rng = np.random.default_rng(4)
    points = np.vstack(
        [
            rng.normal((-1.2, 0.5), (0.32, 0.38), (38, 2)),
            rng.normal((0.1, -0.8), (0.38, 0.28), (38, 2)),
            rng.normal((1.2, 0.55), (0.32, 0.35), (38, 2)),
        ]
    )
    if index == 1:
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 5.8), layout="constrained")
        centers = np.array([[-0.3, 1.1], [0.1, 0.0], [0.7, -0.4]])
        colors = np.array([BLUE, RED, GREEN])
        for step, ax in enumerate(axes):
            dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels_ = dist.argmin(axis=1)
            ax.scatter(points[:, 0], points[:, 1], c=colors[labels_], s=17, alpha=0.72)
            ax.scatter(centers[:, 0], centers[:, 1], marker="x", c=colors, s=90, linewidths=2.4)
            ax.set_title(["назначить центры", "пересчитать группы", "сдвинуть центры"][step],
                         fontsize=12.5, fontstyle="italic")
            ax.set_xticks([])
            ax.set_yticks([])
            clean(ax)
            new_centers = np.array([points[labels_ == j].mean(axis=0) for j in range(3)])
            centers = centers * 0.25 + new_centers * 0.75
        return fig
    if index == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), layout="constrained")
        base = points.copy()
        for ax, scale_y, title in zip(axes, [1, 5], ["сопоставимые шкалы", "вторая шкала растянута"]):
            p = base.copy()
            p[:, 1] *= scale_y
            color = np.where(p[:, 0] > 0, RED, BLUE)
            ax.scatter(p[:, 0], p[:, 1], c=color, s=18, alpha=0.72)
            ax.set_title(title, fontsize=13, fontstyle="italic")
            ax.set_xlabel("признак $x_1$")
            ax.set_ylabel("признак $x_2$")
            ax.set_xticks([])
            ax.set_yticks([])
            clean(ax)
        return fig
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), layout="constrained")
    theta = np.linspace(0, np.pi, 90)
    moon1 = np.c_[np.cos(theta), np.sin(theta)] + rng.normal(0, 0.05, (90, 2))
    moon2 = np.c_[1 - np.cos(theta), 0.45 - np.sin(theta)] + rng.normal(0, 0.05, (90, 2))
    axes[0].scatter(moon1[:, 0], moon1[:, 1], color=BLUE, s=14)
    axes[0].scatter(moon2[:, 0], moon2[:, 1], color=RED, s=14)
    axes[0].set_title("структура глазами человека", fontsize=13, fontstyle="italic")
    allp = np.vstack([moon1, moon2])
    wrong = allp[:, 0] > 0.5
    axes[1].scatter(allp[~wrong, 0], allp[~wrong, 1], color=BLUE, s=14)
    axes[1].scatter(allp[wrong, 0], allp[wrong, 1], color=RED, s=14)
    axes[1].axvline(0.5, color=INK, linewidth=1)
    axes[1].set_title("$k$-средних режет иначе", fontsize=13, fontstyle="italic")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        clean(ax)
    return fig


def lesson10(index: int):
    rng = np.random.default_rng(41)
    p1 = rng.normal((-0.9, 0.2), (0.48, 0.5), (70, 2))
    p2 = rng.normal((0.9, -0.1), (0.48, 0.5), (70, 2))
    if index == 1:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        ax.scatter(p1[:, 0], p1[:, 1], facecolor=PAPER, edgecolor=FAINT, s=21)
        ax.scatter(p2[:, 0], p2[:, 1], facecolor=PAPER, edgecolor=FAINT, s=21)
        labelled1, labelled2 = p1[[4, 22, 55]], p2[[7, 27, 61]]
        ax.scatter(labelled1[:, 0], labelled1[:, 1], color=BLUE, s=48, label="метка A")
        ax.scatter(labelled2[:, 0], labelled2[:, 1], color=RED, s=48, label="метка B")
        ax.axvline(0, color=INK, linewidth=1.2)
        label(ax, -0.9, 1.45, "неразмеченные точки\nзадают форму облака", MUTED, 11, ha="center")
        ax.set_xticks([])
        ax.set_yticks([])
        clean(ax)
        return fig
    if index == 2:
        fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
        allp = np.vstack([p1, p2])
        score = 1 / (1 + np.exp(-2.3 * allp[:, 0]))
        uncertainty = np.abs(score - 0.5)
        selected = np.argsort(uncertainty)[:9]
        ax.scatter(allp[:, 0], allp[:, 1], c=score, cmap=mpl.colors.LinearSegmentedColormap.from_list("br", [BLUE, PAPER, RED]),
                   s=21)
        ax.scatter(allp[selected, 0], allp[selected, 1], s=85, facecolor="none", edgecolor=GOLD, linewidth=2)
        ax.axvline(0, color=INK, linewidth=1)
        label(ax, 0.03, 1.55, "эксперту показывают\nточки у границы", GOLD, 11, ha="center")
        ax.set_xticks([])
        ax.set_yticks([])
        clean(ax)
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    steps = np.arange(9)
    rewards = np.array([0, 0, 0, -0.1, 0, 0, 0, 0, 1.0])
    ax.axhline(0, color=FAINT, linewidth=1)
    ax.vlines(steps, 0, rewards, color=np.where(rewards > 0, GREEN, RED), linewidth=3)
    ax.scatter(steps, rewards, c=np.where(rewards > 0, GREEN, RED), s=42, zorder=3)
    for i in range(8):
        ax.annotate("", xy=(i + 0.88, 0.22), xytext=(i + 0.12, 0.22),
                    arrowprops={"arrowstyle": "-|>", "color": FAINT, "lw": 1})
    label(ax, 4, 0.32, "цепочка действий", MUTED, 11, ha="center")
    label(ax, 8, 1.15, "награда приходит поздно", GREEN, 11, ha="center")
    ax.set_xlabel("шаг агента")
    ax.set_ylabel("награда")
    ax.set_xticks(steps, [f"$a_{i+1}$" for i in steps])
    ax.set_ylim(-0.25, 1.35)
    clean(ax, ygrid=True)
    return fig


def lesson11(index: int):
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        stages = [
            (0.03, "вопрос\nо падении", BLUE),
            (0.22, "измерить\n$s,t$", GOLD),
            (0.41, "построить\n$s(t^2)$", VIOLET),
            (0.60, "оценить\n$g$", RED),
            (0.79, "проверить\nостатки", GREEN),
        ]
        for x, text, color in stages:
            box(ax, (x, 0.55), 0.15, 0.15, text, color=color, size=11)
        for a, b in zip(stages[:-1], stages[1:]):
            arrow(ax, (a[0] + 0.15, 0.625), (b[0], 0.625), shrink=5)
        ax.plot([0.865, 0.865, 0.105], [0.55, 0.39, 0.39], color=GREEN, linewidth=1.25)
        arrow(ax, (0.105, 0.39), (0.105, 0.55), color=GREEN, shrink=6)
        label(ax, 0.49, 0.27, "если остатки имеют узор, меняем модель или опыт", GREEN, 11.5, ha="center")
        return fig
    if index == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), layout="constrained")
        rng = np.random.default_rng(2)
        t = np.linspace(0.2, 1.25, 12)
        s = 4.905 * t**2 + rng.normal(0, 0.12, len(t))
        axes[0].scatter(t, s, color=RED, s=30)
        axes[0].plot(t, 4.905 * t**2, color=BLUE, linewidth=2)
        axes[0].set_xlabel("$t$, с")
        axes[0].set_ylabel("$s$, м")
        axes[0].set_title("кривая по времени", fontsize=13, fontstyle="italic")
        u = t**2
        coef = np.dot(u, s) / np.dot(u, u)
        axes[1].scatter(u, s, color=RED, s=30)
        axes[1].plot([0, u.max()], [0, coef * u.max()], color=BLUE, linewidth=2)
        axes[1].set_xlabel("$t^2$, с²")
        axes[1].set_ylabel("$s$, м")
        axes[1].set_title("прямая после линеаризации", fontsize=13, fontstyle="italic")
        for ax in axes:
            clean(ax, xgrid=True, ygrid=True)
        return fig
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    t = np.linspace(0.1, 1.4, 200)
    delays = [0.00, 0.06, 0.12]
    for delay, color in zip(delays, [BLUE, GOLD, RED]):
        error = (t + delay) ** 2 - t**2
        ax.plot(t, error, color=color, linewidth=2.1)
        direct_line_label(ax, 1.14, error[np.searchsorted(t, 1.14)], f"$\\delta={delay:.2f}$ с", color, dy=0.03)
    ax.set_xlabel("истинное время $t$, с")
    ax.set_ylabel("ошибка в $t^2$")
    clean(ax, ygrid=True)
    return fig


def lesson12(index: int):
    if index == 1:
        fig = canvas()
        ax = diagram_axes(fig)
        nodes = [
            (0.04, 0.62, "источник", GREEN),
            (0.25, 0.62, "сырой журнал", GOLD),
            (0.46, 0.62, "очистка", BLUE),
            (0.67, 0.62, "разметка", VIOLET),
            (0.83, 0.62, "релиз", RED),
        ]
        for x, y, text, color in nodes:
            box(ax, (x, y), 0.13, 0.12, text, color=color, size=10.5)
        for a, b in zip(nodes[:-1], nodes[1:]):
            arrow(ax, (a[0] + 0.13, 0.68), (b[0], 0.68), shrink=5)
        notes = [("кто и когда?", 0.145), ("что удалено?", 0.355), ("по какому правилу?", 0.565), ("кто согласовал?", 0.75)]
        for text, x in notes:
            arrow(ax, (x, 0.60), (x, 0.43), color=FAINT, style="-", shrink=2)
            label(ax, x, 0.36, text, MUTED, 10.2, ha="center")
        label(ax, 0.50, 0.86, "provenance превращает файл в воспроизводимый объект", INK, 12.5, ha="center", weight="bold")
        return fig
    if index == 2:
        fig = canvas()
        ax = diagram_axes(fig)
        people = [("ученик A", BLUE), ("ученик B", RED), ("ученик C", GREEN)]
        for row, (name, color) in enumerate(people):
            y = 0.75 - row * 0.22
            for col in range(5):
                ax.add_patch(Rectangle((0.08 + col * 0.075, y), 0.055, 0.085, facecolor=color, edgecolor="none", alpha=0.8))
            label(ax, 0.015, y + 0.04, name, color, 10.5)
        box(ax, (0.58, 0.66), 0.15, 0.12, "train", color=BLUE)
        box(ax, (0.79, 0.66), 0.15, 0.12, "test", color=RED)
        arrow(ax, (0.42, 0.78), (0.58, 0.72), color=BLUE)
        arrow(ax, (0.42, 0.56), (0.79, 0.72), color=RED)
        arrow(ax, (0.42, 0.34), (0.58, 0.72), color=BLUE)
        arrow(ax, (0.42, 0.76), (0.79, 0.72), color=RED, curve=-0.18)
        label(ax, 0.76, 0.49, "строки одного человека\nоказались в обеих частях", RED, 11, ha="center")
        return fig
    fig = canvas()
    ax = diagram_axes(fig)
    checks = [
        ("источник и лицензия", 0.87, GREEN),
        ("единица наблюдения", 0.72, GREEN),
        ("пропуски и повторы", 0.58, GOLD),
        ("группы и время", 0.45, GOLD),
        ("ограничения применения", 0.28, RED),
    ]
    for i, (text, value, color) in enumerate(checks):
        y = 0.80 - i * 0.14
        label(ax, 0.07, y, text, INK, 11)
        ax.add_patch(Rectangle((0.42, y - 0.025), 0.48, 0.05, facecolor=GRID, edgecolor="none"))
        ax.add_patch(Rectangle((0.42, y - 0.025), 0.48 * value, 0.05, facecolor=color, edgecolor="none"))
        label(ax, 0.92, y, f"{round(value * 100)}%", color, 10.5, ha="right")
    label(ax, 0.50, 0.94, "паспорт не ставит оценку данным — он делает риски видимыми", MUTED, 11.5, ha="center")
    return fig


BUILDERS = {
    "01": lesson01,
    "02": lesson02,
    "03": lesson03,
    "04": lesson04,
    "05": lesson05,
    "06": lesson06,
    "07": lesson07,
    "08": lesson08,
    "09": lesson09,
    "10": lesson10,
    "11": lesson11,
    "12": lesson12,
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
                metadata={"Title": f"Контур, урок {lesson}, рисунок {index}"},
            )
            plt.close(fig)
    print(f"Generated {len(BUILDERS) * 3} SVG figures in {OUT}")


if __name__ == "__main__":
    main()
