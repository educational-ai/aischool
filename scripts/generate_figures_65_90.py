#!/usr/bin/env python3
"""Generate the 78 article figures for lessons 65–90 and QA contact sheets.

The Markdown directives are the source of truth for paths, titles, captions and
accessible descriptions.  Every lesson has three deliberately different
argumentative figures.  SVG is the publication artifact; PNG thumbnails live
only under qa/ for visual inspection.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "content" / "lessons"
PUBLIC = ROOT / "public"
QA = ROOT / "qa" / "figures-65-90"
THUMBS = QA / "thumbs"

PAPER = "#fffdf7"
INK = "#20231f"
MUTED = "#68716b"
GRID = "#dedbd2"
BLUE = "#315f8c"
RED = "#b94a3b"
GREEN = "#38735d"
GOLD = "#a57920"
VIOLET = "#6f5a8f"
CYAN = "#4e8791"
PALE_BLUE = "#dce9f2"
PALE_RED = "#f2dfda"
PALE_GREEN = "#dcebe3"
PALE_GOLD = "#efe5c9"
PALE_VIOLET = "#e8e1ed"
COLORS = [BLUE, RED, GREEN, GOLD, VIOLET, CYAN]
PALES = [PALE_BLUE, PALE_RED, PALE_GREEN, PALE_GOLD, PALE_VIOLET]

mpl.rcParams.update(
    {
        "font.family": "Palatino",
        "font.size": 10.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titleweight": "regular",
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    }
)

FIGURE_RE = re.compile(
    r':::figure\{src="(?P<src>[^"]+)"\s+id="(?P<id>[^"]+)"\s+'
    r'title="(?P<title>[^"]+)"\s+alt="(?P<alt>[^"]+)"\}\n'
    r"(?P<caption>.*?)\n:::",
    re.S,
)


def parse_specs() -> list[dict[str, str | int]]:
    specs: list[dict[str, str | int]] = []
    for lesson in range(65, 91):
        text = (LESSONS / f"{lesson:02d}.md").read_text(encoding="utf-8")
        found = list(FIGURE_RE.finditer(text))
        if len(found) != 3:
            raise RuntimeError(f"lesson {lesson:02d}: expected 3 figures, got {len(found)}")
        for index, match in enumerate(found, 1):
            spec: dict[str, str | int] = match.groupdict()
            spec["lesson"] = lesson
            spec["index"] = index
            specs.append(spec)
    return specs


def clean_ax(ax, *, grid: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(True, color=GRID, lw=0.7, alpha=0.8, zorder=0)
    ax.tick_params(length=3, width=0.7)


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=11.2, pad=8, color=INK)


def note(ax, x: float, y: float, text: str, color: str = INK, size: float = 9) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        bbox=dict(boxstyle="round,pad=.22", fc=PAPER, ec="none", alpha=0.92),
        zorder=8,
    )


def node(ax, xy, label, fc=PALE_BLUE, ec=BLUE, radius=0.09, fontsize=9.5):
    x, y = xy
    patch = Circle((x, y), radius, fc=fc, ec=ec, lw=1.4, zorder=3)
    ax.add_patch(patch)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=4)
    return patch


def box(ax, xy, w, h, label, fc=PALE_BLUE, ec=BLUE, fontsize=9.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=.015,rounding_size=.025",
        fc=fc,
        ec=ec,
        lw=1.35,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, zorder=4)
    return patch


def arrow(ax, a, b, color=MUTED, lw=1.35, style="-|>", rad=0.0, alpha=1.0):
    patch = FancyArrowPatch(
        a,
        b,
        arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=10,
        color=color,
        lw=lw,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def frame(fig, title: str, lesson: int, index: int):
    fig.suptitle(title, x=0.055, y=0.965, ha="left", va="top", fontsize=15.5, color=INK)
    fig.text(
        0.055,
        0.025,
        f"Урок {lesson:02d} · рисунок {index}",
        ha="left",
        va="bottom",
        fontsize=8.8,
        color=MUTED,
    )


def axes_grid(fig, rows=1, cols=1, widths=None):
    gs = fig.add_gridspec(
        rows,
        cols,
        left=0.07,
        right=0.96,
        bottom=0.13,
        top=0.82,
        wspace=0.34,
        hspace=0.46,
        width_ratios=widths,
    )
    return [fig.add_subplot(gs[r, c]) for r in range(rows) for c in range(cols)]


def heat(ax, matrix, xlabels=None, ylabels=None, cmap="Blues", vmin=None, vmax=None):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right")
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def mini_image(ax, x, y, w, h, color, kind=0, alpha=1):
    ax.add_patch(Rectangle((x, y), w, h, fc="#f4f0e8", ec=GRID, lw=0.8, alpha=alpha))
    if kind % 4 == 0:
        ax.add_patch(Circle((x + .5 * w, y + .55 * h), .24 * min(w, h), fc=color, ec="none", alpha=.82))
    elif kind % 4 == 1:
        ax.add_patch(Polygon([(x+.16*w,y+.18*h),(x+.84*w,y+.18*h),(x+.55*w,y+.82*h)], fc=color, ec="none", alpha=.82))
    elif kind % 4 == 2:
        ax.plot([x+.12*w,x+.88*w],[y+.25*h,y+.75*h], color=color, lw=3, alpha=.82)
        ax.plot([x+.12*w,x+.88*w],[y+.72*h,y+.30*h], color=color, lw=1.4, alpha=.65)
    else:
        for q in range(3):
            ax.add_patch(Circle((x + (.28+.22*q)*w, y + (.35+.18*(q%2))*h), .12*min(w,h), fc=color, ec="none", alpha=.78))


def line_common(ax, xlabel="", ylabel=""):
    clean_ax(ax, grid=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def draw_65(fig, i):
    if i == 1:
        a, b, c = axes_grid(fig, 1, 3)
        panel_title(a, "ориентированный граф")
        pts = {"A":(.18,.66),"B":(.55,.82),"C":(.78,.45),"D":(.32,.22)}
        for k,p in pts.items(): node(a,p,k,radius=.065)
        for u,v,w in [("A","B",.5),("A","D",.5),("B","C",1),("C","A",.6),("C","D",.4),("D","C",1)]:
            arrow(a,pts[u],pts[v],lw=1+2*w,color=BLUE)
        a.set(xlim=(0,1),ylim=(0,1)); a.axis("off")
        panel_title(b, "матрица $P$")
        m=np.array([[0,.5,0,.5],[0,0,1,0],[.6,0,0,.4],[0,0,1,0]])
        heat(b,m,list("ABCD"),list("ABCD"),"Blues",0,1)
        for r in range(4):
            for q in range(4): b.text(q,r,f"{m[r,q]:.1f}",ha="center",va="center",fontsize=8)
        panel_title(c, "поток $p_0\\to p_1$")
        x=np.arange(4); p0=np.array([.55,.15,.15,.15]); p1=p0@m
        c.bar(x-.17,p0,.34,label="$p_0$",color=PALE_BLUE,edgecolor=BLUE)
        c.bar(x+.17,p1,.34,label="$p_1$",color=BLUE,alpha=.8)
        c.set_xticks(x,list("ABCD")); c.set_ylim(0,.65); c.legend(frameon=False); clean_ax(c,grid=True)
    elif i == 2:
        axs=axes_grid(fig,1,2)
        for j,ax in enumerate(axs):
            panel_title(ax,"без телепортации" if j==0 else "$\\alpha=0{,}85$")
            pts=[(.15,.55),(.42,.78),(.68,.72),(.72,.32),(.42,.25)]
            for k,p in enumerate(pts): node(ax,p,chr(65+k),radius=.055,fc=PALES[k],ec=COLORS[k])
            edges=[(0,1),(1,2),(2,3),(3,4),(4,2)]
            for u,v in edges: arrow(ax,pts[u],pts[v],color=BLUE)
            if j:
                for u in range(5):
                    arrow(ax,pts[u],pts[(u+2)%5],color=GOLD,lw=.8,style="->",alpha=.5,rad=.18)
                note(ax,.5,.04,"пунктир: случайный прыжок",GOLD,8.5)
            else:
                ax.add_patch(FancyBboxPatch((.34,.16),.48,.66,boxstyle="round,pad=.02",fc="none",ec=RED,lw=1.3,ls="--"))
                note(ax,.60,.90,"ловушка",RED)
            ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    else:
        ax=axes_grid(fig)[0]; rng=np.random.default_rng(65)
        x=rng.lognormal(2,1,220); y=.002*x**.72*rng.lognormal(0,.7,220)
        ax.scatter(np.log1p(x),np.log(y),s=16,c=BLUE,alpha=.42,edgecolors="none")
        ids=np.argsort(y/(x+.5))[-4:]
        ax.scatter(np.log1p(x[ids]),np.log(y[ids]),s=42,c=RED,zorder=4)
        for k,q in enumerate(ids): note(ax,np.log1p(x[q])+.12,np.log(y[q])+.12,f"v{q}",RED,8)
        line_common(ax,"$\\log(1+d^{in})$","$\\log \\pi$")
        panel_title(ax,"одинаковая степень — разный поток")


def draw_66(fig, i):
    rng=np.random.default_rng(66)
    if i==1:
        axs=axes_grid(fig,1,3)
        for ax,n in zip(axs,[100,1000,10000]):
            s=np.cumsum(rng.choice([-1,1],n)); t=np.arange(1,n+1)
            ax.plot(t,s,color=BLUE,lw=1)
            ax.fill_between(t,-2*np.sqrt(t),2*np.sqrt(t),color=PALE_BLUE,alpha=.55)
            ax.plot(t,np.sqrt(t),color=GOLD,lw=1);ax.plot(t,-np.sqrt(t),color=GOLD,lw=1)
            panel_title(ax,f"$n={n:,}$".replace(",", " "))
            line_common(ax,"шаг","$S_k$")
    elif i==2:
        a,b=axes_grid(fig,1,2); x=np.arange(101)
        a.plot(x,x/100,color=BLUE,lw=2.3);a.scatter([10,50,90],[.1,.5,.9],c=RED,zorder=3)
        line_common(a,"начальный капитал $i$","$P_i(\\tau_{100}<\\tau_0)$");panel_title(a,"вероятность успеха")
        b.plot(x,x*(100-x),color=RED,lw=2.3);b.scatter([10,50,90],[900,2500,900],c=BLUE,zorder=3)
        line_common(b,"начальный капитал $i$","$\\mathbb{E}\\tau$");panel_title(b,"время остановки")
    else:
        a,b=axes_grid(fig,1,2)
        steps=rng.normal([.08,.03],[1,.7],(1600,2)); path=np.cumsum(steps,axis=0)
        a.plot(path[:,0],path[:,1],color=BLUE,lw=.8);a.scatter(*path[::250].T,c=np.arange(len(path[::250])),cmap="viridis",s=20)
        a.set_aspect("equal");line_common(a,"восток, км","север, км");panel_title(a,"траектория буя")
        lags=np.unique(np.geomspace(1,200,30).astype(int)); msd=[]
        for lag in lags: msd.append(np.mean(np.sum((path[lag:]-path[:-lag])**2,axis=1)))
        b.loglog(lags,msd,"o-",color=BLUE,ms=3);b.loglog(lags,msd[2]/lags[2]*lags,"--",color=GOLD,label="наклон 1")
        line_common(b,"лаг, шаги","$\\langle|\\Delta r|^2\\rangle$");b.legend(frameon=False);panel_title(b,"средний квадрат смещения")


def draw_67(fig, i):
    rng=np.random.default_rng(67)
    if i==1:
        axs=axes_grid(fig,3,1); x=np.linspace(-7,7,500)
        d=.52*np.exp(-(x+3)**2/1.5)+.48*np.exp(-(x-3)**2/1.2)
        axs[0].plot(x,d,color=BLUE,lw=2);panel_title(axs[0],"целевая плотность");axs[0].set_yticks([]);clean_ax(axs[0])
        trace=np.empty(500);trace[0]=-3
        for t in range(1,500):
            y=trace[t-1]+rng.normal(0,1.3)
            pi=lambda z:.52*np.exp(-(z+3)**2/1.5)+.48*np.exp(-(z-3)**2/1.2)
            trace[t]=y if rng.random()<min(1,pi(y)/pi(trace[t-1])) else trace[t-1]
        axs[1].plot(trace,color=RED,lw=.8);panel_title(axs[1],"зависимая траектория");clean_ax(axs[1])
        axs[2].hist(trace[80:],35,density=True,color=PALE_BLUE,edgecolor=BLUE);axs[2].plot(x,d/np.trapezoid(d,x),color=INK,lw=1.3)
        panel_title(axs[2],"частоты после прогрева");clean_ax(axs[2])
    elif i==2:
        axs=axes_grid(fig,2,3)
        sigmas=[.05,.8,6]
        for col,sig in enumerate(sigmas):
            tr=np.empty(1200);tr[0]=5;acc=0
            for t in range(1,len(tr)):
                y=tr[t-1]+rng.normal(0,sig)
                if rng.random()<min(1,np.exp(-(y*y-tr[t-1]**2)/2)):tr[t]=y;acc+=1
                else:tr[t]=tr[t-1]
            ax=axs[col];ax.plot(tr[:350],color=COLORS[col],lw=.7);panel_title(ax,f"$\\sigma={sig}$ · принято {acc/(len(tr)-1):.0%}");clean_ax(ax)
            ac=[np.corrcoef(tr[:-k],tr[k:])[0,1] for k in range(1,50)]
            ax=axs[3+col];ax.plot(range(1,50),ac,color=COLORS[col]);ax.axhline(0,color=GRID,lw=.8);line_common(ax,"лаг","$\\rho_k$")
    else:
        ax=axes_grid(fig)[0]; x=np.linspace(-3,3,180);y=np.linspace(-3,3,180);X,Y=np.meshgrid(x,y)
        Z=np.exp(-.5*(3*X**2-5*X*Y+3*Y**2))
        ax.contour(X,Y,Z,levels=8,colors=GRID)
        for color,cov,label in [(RED,np.eye(2)*.15,"изотропный"),(BLUE,np.array([[.12,.1],[.1,.12]]),"ориентированный")]:
            tr=np.zeros((180,2));tr[0]=[-2,-2]
            inv=np.array([[3,-2.5],[-2.5,3]])
            for t in range(1,len(tr)):
                prop=tr[t-1]+rng.multivariate_normal([0,0],cov)
                logratio=-.5*(prop@inv@prop-tr[t-1]@inv@tr[t-1])
                tr[t]=prop if np.log(rng.random())<min(0,logratio) else tr[t-1]
            ax.plot(*tr.T,color=color,lw=1,label=label,alpha=.9)
        ax.legend(frameon=False);ax.set_aspect("equal");line_common(ax,"$\\beta_1$","$\\beta_2$");panel_title(ax,"коррелированный posterior")


def draw_68(fig, i):
    if i==1:
        ax=axes_grid(fig)[0]; ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        top=list("МАМА МЫЛА"); bottom=list("ХФХФ ЦЫПФ")
        xs=np.linspace(.1,.9,len(top))
        for x,a,b in zip(xs,top,bottom):
            ax.text(x,.72,a,ha="center",fontsize=16,color=BLUE)
            ax.text(x,.28,b,ha="center",fontsize=16,color=RED)
            if a!=" ":arrow(ax,(x,.65),(x,.37),color=GRID,lw=1)
        for inds,color in [([0,2],GOLD),([1,3,7],GREEN)]:
            for q in inds: ax.add_patch(Circle((xs[q],.72),.038,fc="none",ec=color,lw=1.4))
        note(ax,.5,.5,"повторы сохраняют позиции",INK,10)
    elif i==2:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        pts=[(.12,.2),(.28,.55),(.43,.31),(.57,.67),(.72,.42),(.88,.78)]
        heights=[.1,.45,.3,.75,.52,.92]
        for k,p in enumerate(pts):
            node(ax,p,f"$k_{k}$",radius=.045,fc=PALE_GOLD if k in [1,3] else PALE_BLUE,ec=GOLD if k in [1,3] else BLUE,fontsize=8)
        for k in range(len(pts)-1):arrow(ax,pts[k],pts[k+1],color=BLUE,rad=(-1)**k*.08)
        arrow(ax,pts[3],pts[2],color=RED,rad=.28)
        ax.plot([p[0] for p in pts],[.08+.75*h for h in heights],color=GRID,lw=1,ls="--")
        note(ax,.31,.73,"жадный пик",RED);note(ax,.78,.91,"лучший ключ",GREEN)
    else:
        ax=axes_grid(fig)[0]; n=np.array([50,100,200,500,1000,2000])
        curves=[1-np.exp(-n/k) for k in [1200,520,260]]
        for y,c,l in zip(curves,[GOLD,BLUE,GREEN],["униграммы","биграммы","триграммы"]):
            ax.plot(n,y,"o-",color=c,label=l);ax.fill_between(n,np.maximum(0,y-.08),np.minimum(1,y+.08),color=c,alpha=.10)
        ax.set_xscale("log");ax.set_ylim(0,1);line_common(ax,"длина шифротекста","доля восстановленных букв");ax.legend(frameon=False);panel_title(ax,"больше контекста окупается после данных")


def draw_69(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        root=(.12,.5);node(ax,root,"$s_0$",radius=.055)
        acts=[(.38,.72),(.38,.28)]
        for p,l,c in zip(acts,["коротко","надёжно"],[RED,GREEN]):
            box(ax,(p[0]-.08,p[1]-.055),.16,.11,l,fc=PALE_RED if c==RED else PALE_GREEN,ec=c)
            arrow(ax,root,(p[0]-.09,p[1]),color=c)
        leaves=[(.7,.87),(.7,.57),(.7,.34),(.7,.13)]
        labels=["вовремя\n0,7","перекрыто\n0,3","станция\n0,95","задержка\n0,05"]
        for p,l in zip(leaves,labels):node(ax,p,l,radius=.07,fontsize=8)
        for u,v in [(acts[0],leaves[0]),(acts[0],leaves[1]),(acts[1],leaves[2]),(acts[1],leaves[3])]:arrow(ax,u,v,color=MUTED)
        note(ax,.88,.72,"$R+\\gamma V(s')$",BLUE,10)
    elif i==2:
        axs=axes_grid(fig,1,4); goal=(4,4); danger=(2,3)
        for k,ax in enumerate(axs):
            V=np.zeros((5,5))
            for r in range(5):
                for c in range(5):
                    dist=abs(goal[0]-r)+abs(goal[1]-c)
                    V[r,c]=max(0,(k*2-dist))/max(1,k*2) if k else 0
            V[danger]=-0.8
            heat(ax,V,cmap="RdYlGn",vmin=-1,vmax=1);ax.set_xticks([]);ax.set_yticks([]);panel_title(ax,f"$k={k if k<3 else 8}$")
    else:
        axs=axes_grid(fig,3,1);t=np.arange(24);load=2+1.5*np.exp(-((t-19)/3)**2)+.3*np.sin(t)
        price=.12+.22*((t>=17)&(t<=21));charge=np.clip(3+np.cumsum(np.where(t<7,.35,np.where((t>=17)&(t<=21),-.55,0))),0,6)
        axs[0].plot(t,load,color=BLUE,label="нагрузка");axs[0].fill_between(t,0,price*10,color=PALE_GOLD,label="тариф ×10");axs[0].legend(frameon=False,ncol=2);panel_title(axs[0],"спрос и цена")
        axs[1].plot(t,charge,color=GREEN,lw=2);axs[1].set_ylim(0,6);panel_title(axs[1],"заряд, кВт·ч")
        act=np.diff(np.r_[charge[0],charge]);axs[2].bar(t,act,color=np.where(act>=0,GREEN,RED));panel_title(axs[2],"действие: + заряд, − разряд")
        for ax in axs:clean_ax(ax,grid=True);ax.set_xlim(0,23)


def draw_70(fig, i):
    rng=np.random.default_rng(70)
    if i==1:
        axs=axes_grid(fig,2,1);T=180;choices=np.r_[rng.integers(0,3,60),rng.choice(3,120,p=[.08,.12,.8])]
        rewards=rng.random(T)<np.array([.12,.16,.23])[choices]
        axs[0].scatter(np.arange(T),choices,c=np.array(COLORS)[choices],s=10);axs[0].set_yticks([0,1,2],["A","B","C"]);panel_title(axs[0],"исследование сменяется использованием");clean_ax(axs[0])
        regret=np.cumsum(.23-np.array([.12,.16,.23])[choices]);axs[1].plot(regret,color=RED,lw=2);line_common(axs[1],"раунд","накопленный regret")
    elif i==2:
        ax=axes_grid(fig)[0];means=np.array([.20,.24,.18]);n=np.array([20,200,8]);bonus=np.sqrt(np.log(1000)/n)
        ax.errorbar(range(3),means,yerr=bonus,fmt="o",color=BLUE,ecolor=GOLD,capsize=5,lw=2)
        ax.scatter(2,means[2]+bonus[2],s=60,c=RED,zorder=4,label="следующий выбор")
        ax.set_xticks(range(3),[f"{c}\\n$N={q}$" for c,q in zip("ABC",n)]);ax.set_ylim(0,1.2);line_common(ax,"рука","среднее + бонус");ax.legend(frameon=False);panel_title(ax,"оптимизм UCB")
    else:
        ax=axes_grid(fig)[0];groups=["новые","постоянные","все"];A=[.30,.12,.23];B=[.34,.16,.21];x=np.arange(3)
        ax.bar(x-.18,A,.36,color=BLUE,label="вариант A");ax.bar(x+.18,B,.36,color=GREEN,label="вариант B")
        ax.set_xticks(x,groups);ax.set_ylim(0,.4);line_common(ax,"контекст","CTR");ax.legend(frameon=False);panel_title(ax,"средняя смесь меняет победителя")


def draw_71(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        pts=[(.1+.18*k,.5) for k in range(5)]
        for k,p in enumerate(pts):node(ax,p,"цель" if k==4 else f"$s_{k}$",radius=.055,fc=PALE_GREEN if k==4 else PALE_BLUE,ec=GREEN if k==4 else BLUE)
        for k in range(4):arrow(ax,pts[k],pts[k+1],color=GREEN,lw=1+1.2*k)
        for k,p in enumerate(pts[:-1]):note(ax,p[0],.25,f"$\\delta={.5**(3-k):.2f}$",RED,8)
        note(ax,.5,.82,"новость о награде идёт назад",INK,10)
    elif i==2:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        box(ax,(.06,.58),.18,.16,"переход\n$(s,a,r,s')$",PALE_BLUE,BLUE)
        box(ax,(.36,.68),.18,.14,"$Q_\\theta(s,a)$",PALE_GREEN,GREEN)
        box(ax,(.36,.38),.18,.14,"$Q_{\\bar\\theta}(s',a')$",PALE_GOLD,GOLD)
        box(ax,(.70,.52),.2,.18,"TD-цель\nи loss",PALE_RED,RED)
        arrow(ax,(.24,.66),(.36,.75),BLUE);arrow(ax,(.24,.64),(.36,.45),GOLD);arrow(ax,(.54,.75),(.70,.64),GREEN);arrow(ax,(.54,.45),(.70,.58),GOLD)
        arrow(ax,(.79,.52),(.54,.45),RED,style="->",rad=-.28);note(ax,.62,.19,"медленное копирование $\\theta\\to\\bar\\theta$",MUTED,9)
    else:
        axs=axes_grid(fig,3,1);t=np.arange(48);tariff=.1+.25*((t>31)&(t<41));soc=5+2*np.sin((t-8)/48*2*np.pi)
        adv=np.sin(t/5)*.5;est=adv.copy();est[28:36]*=-.7
        axs[0].plot(t,tariff,color=GOLD);axs[0].plot(t,soc/20,color=GREEN);panel_title(axs[0],"тариф и заряд (масштабирован)")
        axs[1].plot(t,adv,color=INK,label="истинный");axs[1].plot(t,est,color=RED,label="critic");axs[1].axhline(0,color=GRID);axs[1].legend(frameon=False,ncol=2);panel_title(axs[1],"advantage")
        prob=1/(1+np.exp(-np.cumsum(est)*.15));axs[2].plot(t,prob,color=BLUE);axs[2].set_ylim(0,1);panel_title(axs[2],"вероятность разряда")
        for ax in axs:clean_ax(ax,grid=True);ax.axvspan(28,36,color=PALE_RED,alpha=.55)


def draw_72(fig, i):
    if i==1:
        a,b=axes_grid(fig,1,2);m=np.array([[.5,.2,.8],[.8,.5,.2],[.2,.8,.5]])
        heat(a,m,["камень","ножницы","бумага"],["камень","ножницы","бумага"],"RdYlGn",0,1);panel_title(a,"win rate")
        b.axis("off");b.set(xlim=(0,1),ylim=(0,1));pts=[(.5,.83),(.18,.25),(.82,.25)]
        for p,l,c in zip(pts,["камень","ножницы","бумага"],COLORS[:3]):node(b,p,l,fc=PALES[COLORS[:3].index(c)],ec=c,radius=.09,fontsize=8)
        arrow(b,pts[0],pts[1],RED,rad=.12);arrow(b,pts[1],pts[2],GREEN,rad=.12);arrow(b,pts[2],pts[0],BLUE,rad=.12);panel_title(b,"цикл доминирования")
    elif i==2:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        levels=[[(.5,.85)],[(.25,.58),(.5,.58),(.75,.58)],[(.15,.25),(.34,.25),(.48,.25),(.63,.25),(.82,.25)]]
        for d,lev in enumerate(levels):
            for q,p in enumerate(lev):node(ax,p,f"{['s','a','s′'][d]}{q}",radius=.045,fc=PALES[d],ec=COLORS[d],fontsize=8)
        for p in levels[1]:arrow(ax,levels[0][0],p,BLUE)
        for k,p in enumerate(levels[2]):arrow(ax,levels[1][min(2,k//2)],p,GOLD)
        note(ax,.5,.06,"selection  >  expansion  >  value  >  backup",INK,9.5)
        note(ax,.36,.69,"PUCT",BLUE);note(ax,.73,.40,"новый узел",RED)
    else:
        axs=axes_grid(fig,1,3);rng=np.random.default_rng(72);m=rng.uniform(.2,.8,(8,8));m=(m+1-m.T)/2;np.fill_diagonal(m,.5)
        heat(axs[0],m,cmap="RdYlGn",vmin=0,vmax=1);axs[0].set_title("матрица игр",loc="left")
        elo=np.linspace(1300,1700,8)+rng.normal(0,40,8);axs[1].barh(range(8),elo,color=BLUE);axs[1].set_yticks(range(8),[f"v{k}" for k in range(8)]);clean_ax(axs[1]);panel_title(axs[1],"Elo")
        pred=1/(1+10**((elo[None,:]-elo[:,None])/400));res=m-pred;heat(axs[2],res,cmap="RdBu_r",vmin=-.3,vmax=.3);panel_title(axs[2],"остатки: циклы")


def draw_73(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        for k in range(5):
            x=.1+.19*k;box(ax,(x-.055,.42),.11,.16,"RNN",PALE_BLUE,BLUE)
            ax.text(x,.74,f"$x_{k+1}$",ha="center");arrow(ax,(x,.69),(x,.58),GREEN)
            ax.text(x,.22,f"$h_{k+1}$",ha="center")
            if k<4:arrow(ax,(x+.055,.5),(x+.19-.055,.5),GOLD,lw=2-.25*k)
        note(ax,.5,.88,"одни веса на каждом шаге",INK,10)
    elif i==2:
        ax=axes_grid(fig)[0];lag=np.arange(50)
        for u,c in zip([.8,1,1.15],[BLUE,GREEN,RED]):ax.plot(lag,np.log10(np.maximum(1e-9,u**lag)),color=c,lw=2,label=f"$u={u}$")
        ax.axhspan(-7,-5,color=PALE_RED,alpha=.6,label="численно мало");line_common(ax,"временной лаг","$\\log_{10}\\|\\partial h_T/\\partial h_t\\|$");ax.legend(frameon=False);panel_title(ax,"произведение производных")
    else:
        a,b=axes_grid(fig,1,2);t=np.arange(120);s1=.4*np.sin(t/13)+t/130;s2=.4*np.sin(t/13)-t/180+.7
        a.plot(t,s1,color=BLUE);a.plot(t,s2,color=RED);a.scatter([80,80],[s1[80],s2[80]],c=[BLUE,RED]);line_common(a,"цикл","сенсор");panel_title(a,"одна точка — разные тренды")
        true=np.maximum(0,120-t);pred=true+6*np.sin(t/10);base=120-np.full_like(t,80)
        b.plot(t,true,color=INK,label="RUL");b.plot(t,pred,color=BLUE,label="RNN");b.plot(t,base,color=GOLD,label="последняя точка");line_common(b,"цикл","остаток");b.legend(frameon=False);panel_title(b,"прогноз ресурса")


def draw_74(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        arrow(ax,(.08,.68),(.92,.68),BLUE,lw=2.8);note(ax,.12,.76,"$c_{t-1}$",BLUE)
        for x,l,c in [(.30,"$\\times f_t$",GOLD),(.52,"$+i_t\\tilde c_t$",GREEN),(.76,"$\\tanh\\times o_t$",VIOLET)]:box(ax,(x-.08,.58),.16,.18,l,PALES[[GOLD,GREEN,VIOLET].index(c)+2 if c!=VIOLET else 4],c,8.5)
        arrow(ax,(.52,.25),(.52,.58),GREEN);box(ax,(.42,.18),.2,.10,"новая запись",PALE_GREEN,GREEN,8)
        arrow(ax,(.76,.58),(.76,.35),VIOLET);note(ax,.76,.27,"$h_t$",VIOLET)
    elif i==2:
        axs=axes_grid(fig,5,1);t=np.arange(100);key=15;answer=82
        inp=np.zeros(100);inp[key]=1
        f=np.full(100,.98);f[key]=.2;i0=np.zeros(100);i0[key]=.95;o=np.zeros(100);o[answer:]=.9
        cell=np.zeros(100)
        for k in range(1,100):cell[k]=f[k]*cell[k-1]+i0[k]*inp[k]
        for ax,y,l,c in zip(axs,[inp,f,i0,cell,o],["вход","forget","input","cell","output"],COLORS[:5]):
            ax.plot(t,y,color=c);ax.set_yticks([]);panel_title(ax,l);clean_ax(ax);ax.axvline(key,color=GRID);ax.axvline(answer,color=GRID)
    else:
        axs=axes_grid(fig,2,1);t=np.arange(14*24);load=2+.7*np.sin(2*np.pi*t/24-1)+.25*np.sin(2*np.pi*t/(24*7))
        pred=2+.62*np.sin(2*np.pi*t/24-1.1)+.18*np.sin(2*np.pi*t/(24*7))
        base=np.roll(load,24*7);base[:24*7]=np.nan
        axs[0].plot(t[-120:],load[-120:],color=INK,label="факт");axs[0].plot(t[-120:],pred[-120:],color=BLUE,label="LSTM");axs[0].plot(t[-120:],base[-120:],color=GOLD,label="неделю назад");axs[0].legend(frameon=False,ncol=3);panel_title(axs[0],"пять суток")
        err=np.abs(load-pred);axs[1].plot(t[-120:],err[-120:],color=RED);axs[1].fill_between(t[-120:],0,err[-120:],where=load[-120:]>np.quantile(load,.9),color=PALE_RED);panel_title(axs[1],"ошибка на пиках")
        for ax in axs:clean_ax(ax,grid=True)


def draw_75(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        rows=[("байты",list("D0 9D D0 B5 D0 B9...")),("символы",list("нейросеть учится")),("слова",["нейросеть","учится"]),("подслова",["нейро","сеть"," уч","ится"])]
        ys=[.78,.58,.38,.18]
        for (label,toks),y,c in zip(rows,ys,COLORS[:4]):
            ax.text(.02,y,label,ha="left",va="center",color=c)
            x=.17
            for tok in toks[:10]:
                w=max(.045,min(.17,.025*len(tok)+.025));box(ax,(x,y-.045),w,.09,tok,PALES[ys.index(y)],c,7.5);x+=w+.012
                if x>.94:break
            ax.text(.96,y,str(len(toks)),ha="right",va="center",color=MUTED)
    elif i==2:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        cols=["вид","code points","UTF-8","токены"];rows=[("й","U+0439","D0 B9","[742]"),("й","U+0438 U+0306","D0 B8 CC 86","[81, 992]"),("emoji ZWJ","3 code points","11 bytes","[4182, 9921]")]
        for c,x in zip(cols,[.10,.36,.65,.88]):ax.text(x,.86,c,ha="center",color=MUTED)
        for r,row in enumerate(rows):
            y=.66-.22*r
            for q,(text,x) in enumerate(zip(row,[.10,.36,.65,.88])):box(ax,(x-.09,y-.055),.18,.11,text,PALES[r],COLORS[r],8)
        arrow(ax,(.36,.40),(.36,.61),GOLD,style="<->");note(ax,.5,.51,"NFC",GOLD,9)
    else:
        ax=axes_grid(fig)[0];langs=["en","ru","de","tr","ar","hi","zh","sw"];med=[56,71,63,88,83,96,62,105];lo=np.array(med)-12;hi=np.array(med)+18;x=np.arange(len(langs))
        ax.errorbar(x,med,yerr=[np.array(med)-lo,hi-np.array(med)],fmt="o",color=BLUE,ecolor=GOLD,capsize=4)
        ax.set_xticks(x,langs);line_common(ax,"язык","токены на 100 символов");panel_title(ax,"один смысл расходует разный контекст")


def draw_76(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        words=["робот","стакан","стол","он"];xs=[.1,.32,.54,.82]
        for x,w,c in zip(xs,words,COLORS[:4]):box(ax,(x-.07,.72),.14,.11,w,PALES[xs.index(x)],c,8.5)
        note(ax,.82,.52,"query $q_i$",VIOLET);arrow(ax,(.82,.69),(.82,.57),VIOLET)
        weights=[.08,.17,.62,.13]
        for x,w in zip(xs,weights):arrow(ax,(.80,.48),(x,.32),BLUE,lw=.6+4*w,alpha=.8)
        for x,w in zip(xs,weights):note(ax,x,.24,f"$a={w:.2f}$",BLUE,8)
        note(ax,.5,.10,"выход = сумма value с этими весами",INK,9.5)
    elif i==2:
        a,b=axes_grid(fig,1,2);m=np.array([[.7,.1,.1,.1],[.2,.5,.2,.1],[.1,.3,.5,.1],[.1,.15,.25,.5]])
        heat(a,m,list("АБВГ"),list("АБВГ"),"Blues",0,.8);panel_title(a,"полная карта")
        causal=np.tril(m);causal=causal/causal.sum(1,keepdims=True);heat(b,causal,list("АБВГ"),list("АБВГ"),"Blues",0,.8);panel_title(b,"causal mask")
        for r in range(4):
            for q in range(r+1,4):b.add_patch(Rectangle((q-.5,r-.5),1,1,fc=PALE_RED,ec="none",alpha=.75))
    else:
        ax=axes_grid(fig)[0];n=np.array([128,256,512,1024,2048,4096])
        full=n**2/1e6;local=n*512/1e6
        ax.plot(n,full,"o-",color=RED,label="полная $n^2$");ax.plot(n,local,"o-",color=BLUE,label="окно 512")
        ax.set_xscale("log",base=2);ax.set_yscale("log");line_common(ax,"токены $n$","млн связей");ax.legend(frameon=False);panel_title(ax,"контекст удваивается — карта учетверяется")


def draw_77(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        ys=[.75,.48,.21];labels=["LN  >  attention","LN  >  MLP","следующий блок"];cols=[BLUE,GREEN,VIOLET]
        x0=.15
        box(ax,(.08,.80),.16,.09,"$X\\in\\mathbb{R}^{n\\times d}$",PALE_GOLD,GOLD,8)
        for y,l,c in zip(ys,labels,cols):
            box(ax,(.37,y-.06),.24,.12,l,PALES[cols.index(c)],c)
            arrow(ax,(.24,y+.05 if y==.75 else y+.27),(.37,y),c)
            arrow(ax,(.61,y),(.82,y),c)
            arrow(ax,(.27,y+.14),(.78,y+.14),MUTED,lw=1.2)
            note(ax,.78,y+.14,"residual",MUTED,8)
        note(ax,.86,.21,"$Y$",VIOLET)
    elif i==2:
        a,b=axes_grid(fig,1,2);layers=np.arange(25)
        a.plot(layers,1+.08*np.sin(layers/3),color=BLUE,label="residual");a.plot(layers,np.exp(layers/11),color=RED,label="без residual")
        a.set_yscale("log");line_common(a,"слой","норма активации");a.legend(frameon=False);panel_title(a,"сигнал")
        b.plot(layers[::-1],np.exp(-layers/6),color=RED,label="без residual");b.plot(layers[::-1],.5+.2*np.cos(layers/5),color=BLUE,label="residual")
        b.set_yscale("log");line_common(b,"слой","норма градиента");b.legend(frameon=False);panel_title(b,"обратный путь")
    else:
        a,b=axes_grid(fig,1,2);steps=np.arange(1,101)
        variants=[("полный",1.8-0.55*np.log1p(steps)/np.log(101),BLUE),("без residual",1.9-.28*np.log1p(steps)/np.log(101),RED),("без позиции",1.85-.42*np.log1p(steps)/np.log(101),GOLD),("без MLP",1.82-.37*np.log1p(steps)/np.log(101),VIOLET)]
        for l0,c0,col in variants:a.plot(steps,c0,color=col,label=l0)
        line_common(a,"шаг, тыс.","validation loss");a.legend(frameon=False,fontsize=8);panel_title(a,"ablation")
        vals=[.94,.58,.51,.72];b.bar(range(4),vals,color=[v[2] for v in variants]);b.set_xticks(range(4),["полный","−res","−pos","−MLP"],rotation=25);b.set_ylim(0,1);clean_ax(b,grid=True);panel_title(b,"порядок слов: accuracy")


def draw_78(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        toks=["Я","вижу","синий","дом","."]
        for k,t in enumerate(toks):box(ax,(.08+.17*k,.68),.13,.10,t,PALE_BLUE,BLUE)
        for k,t in enumerate(toks[1:]):box(ax,(.08+.17*k,.28),.13,.10,t,PALE_GREEN,GREEN)
        ax.text(.02,.73,"вход",ha="right");ax.text(.02,.33,"цель",ha="right")
        for k in range(4):arrow(ax,(.145+.17*k,.66),(.145+.17*k,.40),GOLD)
        note(ax,.5,.52,"сдвиг на один токен",GOLD)
    elif i==2:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        stages=[("сырой веб",.08,1.0),("язык",.25,.82),("качество",.42,.63),("dedup",.59,.48),("смесь",.78,.40)]
        for k,(l,x,h) in enumerate(stages):
            ax.add_patch(Rectangle((x,.18),.10,.62*h,fc=PALES[k],ec=COLORS[k],lw=1.3))
            ax.text(x+.05,.12,l,ha="center",fontsize=8)
            if k<len(stages)-1:arrow(ax,(x+.10,.18+.31*h),(stages[k+1][1],.18+.31*stages[k+1][2]),MUTED,lw=4*h,alpha=.5)
        note(ax,.50,.88,"каждый фильтр меняет состав",INK,10)
    else:
        ax=axes_grid(fig)[0];c=np.array([0,.01,.03,.1,.2,.4])
        orig=.52+.43*(1-np.exp(-12*c));new=.50+.16*(1-np.exp(-6*c))
        ax.plot(c,orig,"o-",color=BLUE,label="исходные вопросы");ax.plot(c,new,"o-",color=GREEN,label="новые варианты")
        ax.fill_between(c,new,orig,color=PALE_RED,alpha=.45,label="выигрыш памяти")
        line_common(ax,"доля загрязнения","accuracy");ax.set_ylim(.45,1);ax.legend(frameon=False);panel_title(ax,"утечка поднимает только знакомую форму")


def draw_79(fig, i):
    if i==1:
        a,b=axes_grid(fig,1,2);N=np.array([1,2,4,8,16,32,64]);L=1+.9*N**-.28
        a.plot(N,L,"o-",color=BLUE);line_common(a,"$N$","loss");panel_title(a,"линейные оси")
        b.loglog(N,L-1,"o-",color=BLUE);line_common(b,"$N$","$L-L_\\infty$");panel_title(b,"log–log: наклон $-\\alpha$")
    elif i==2:
        ax=axes_grid(fig)[0];n=np.logspace(7,10,120);d=np.logspace(8,12,120);N,D=np.meshgrid(n,d)
        Z=1+1.8*(N/1e8)**-.25+1.4*(D/1e9)**-.22
        cs=ax.contourf(np.log10(N),np.log10(D),Z,levels=18,cmap="YlGnBu_r")
        for C in [1e17,1e18,1e19]:
            nn=np.logspace(7,10,100);dd=C/(6*nn);mask=(dd>=1e8)&(dd<=1e12);ax.plot(np.log10(nn[mask]),np.log10(dd[mask]),color=PAPER,lw=1.2)
        optn=np.logspace(7.5,9.5,6);optd=20*optn;ax.scatter(np.log10(optn),np.log10(optd),c=RED,s=24,label="compute-optimal")
        line_common(ax,"$\\log_{10}N$","$\\log_{10}D$");ax.legend(frameon=False);panel_title(ax,"долина бюджета")
    else:
        ax=axes_grid(fig)[0];scale=np.arange(1,11);prob=.2+.065*scale;acc=(prob>.5).astype(float)*.55+.18
        ax.plot(scale,prob,"o-",color=BLUE,label="$P$(верный ответ)");ax.step(scale,acc,where="mid",color=RED,label="exact match")
        ax.axhline(.5,color=GRID);line_common(ax,"масштаб модели","метрика");ax.set_ylim(0,1);ax.legend(frameon=False);panel_title(ax,"порог превращает плавность в скачок")


def draw_80(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];x=np.linspace(-5,5,300);p=1/(1+np.exp(-x))
        ax.plot(x,p,color=BLUE,lw=2.5);ax.axhline(.5,color=GRID);ax.axvline(0,color=GRID)
        for pr in [.5,.75,.9]:
            xx=np.log(pr/(1-pr));ax.scatter(xx,pr,c=RED);note(ax,xx+.28,pr+.04,f"{pr:.2f}",RED,8)
        line_common(ax,"$r_i-r_j$","$P(i\\succ j)$");panel_title(ax,"логистическая шкала предпочтения")
    elif i==2:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        left=[(.16,.7),(.28,.5),(.12,.3)];right=[(.72,.7),(.86,.5),(.72,.3)]
        for k,p in enumerate(left+right):node(ax,p,chr(65+k),radius=.055,fc=PALE_BLUE if k<3 else PALE_GREEN,ec=BLUE if k<3 else GREEN)
        for group,col in [(left,BLUE),(right,GREEN)]:
            for u in range(3):
                for v in range(u+1,3):arrow(ax,group[u],group[v],col,rad=.12)
        arrow(ax,left[1],right[1],RED,lw=1);note(ax,.5,.58,"один мост",RED)
        ax.errorbar([.16,.28,.12,.72,.86,.72],[.13]*6,xerr=[.03,.04,.03,.11,.12,.10],fmt="o",color=GOLD)
        note(ax,.5,.10,"ширина интервала зависит от связности",GOLD,9)
    else:
        ax=axes_grid(fig)[0];step=np.arange(80);score=.4+.5*(1-np.exp(-step/20));human=.4+.32*(1-np.exp(-step/18))-.003*np.maximum(step-45,0)
        ax.plot(step,score,color=BLUE,label="reward model");ax.plot(step,human,color=GREEN,label="люди");ax.axvspan(45,79,color=PALE_RED,alpha=.55)
        line_common(ax,"сила оптимизации","оценка");ax.legend(frameon=False);panel_title(ax,"за пределами разметки proxy расходится")


def draw_81(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        labels=["pretrain","SFT","пары","оптимизация","оценка"];xs=np.linspace(.05,.82,5)
        for k,(x,l) in enumerate(zip(xs,labels)):box(ax,(x,.45),.13,.15,l,PALES[k],COLORS[k])
        for k in range(4):arrow(ax,(xs[k]+.13,.525),(xs[k+1],.525),COLORS[k])
        arrow(ax,(xs[-1]+.065,.43),(xs[2]+.065,.38),RED,rad=-.28);note(ax,.55,.20,"новый сбор, не утечка test",RED,8.5)
    elif i==2:
        a,b=axes_grid(fig,1,2)
        for ax,title in [(a,"PPO: online"),(b,"DPO: offline")]:
            ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1));panel_title(ax,title)
        for p,l,c in [((.08,.65),"policy",BLUE),((.40,.65),"samples",GREEN),((.70,.65),"reward",RED),((.40,.25),"critic + KL",GOLD)]:box(a,p,.2,.14,l,PALES[[BLUE,GREEN,RED,GOLD].index(c)],c,8)
        arrow(a,(.28,.72),(.40,.72),BLUE);arrow(a,(.60,.72),(.70,.72),GREEN);arrow(a,(.80,.65),(.60,.32),RED);arrow(a,(.40,.32),(.18,.65),GOLD,rad=-.18)
        for p,l,c in [((.08,.62),"пары",GREEN),((.40,.62),"log-ratio",BLUE),((.70,.62),"DPO loss",RED),((.40,.24),"reference",GOLD)]:box(b,p,.2,.14,l,PALES[[GREEN,BLUE,RED,GOLD].index(c)],c,8)
        arrow(b,(.28,.69),(.40,.69),GREEN);arrow(b,(.60,.69),(.70,.69),BLUE);arrow(b,(.50,.38),(.50,.62),GOLD)
    else:
        axs=axes_grid(fig,2,2);t=np.arange(70);ys=[.3+.5*(1-np.exp(-t/20)),.3+.34*(1-np.exp(-t/17))-.0025*np.maximum(t-40,0),80+1.5*t,0.01*np.exp(t/22)]
        labs=["learned reward","human score","длина, токены","KL к reference"];cols=[BLUE,GREEN,GOLD,RED]
        for ax,y,l,c in zip(axs,ys,labs,cols):
            ax.plot(t,y,color=c,lw=2);ax.axvline(40,color=GRID);panel_title(ax,l);clean_ax(ax,grid=True)


def draw_82(fig, i):
    rng=np.random.default_rng(82)
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        box(ax,(.05,.62),.15,.14,"шум $z$",PALE_GOLD,GOLD)
        box(ax,(.30,.62),.18,.14,"generator",PALE_GREEN,GREEN)
        box(ax,(.68,.62),.20,.14,"discriminator",PALE_RED,RED)
        box(ax,(.30,.25),.18,.14,"real data",PALE_BLUE,BLUE)
        arrow(ax,(.20,.69),(.30,.69),GOLD);arrow(ax,(.48,.69),(.68,.69),GREEN);arrow(ax,(.48,.32),(.68,.66),BLUE)
        arrow(ax,(.78,.60),(.40,.51),RED,rad=-.18);note(ax,.60,.42,"gradient к $G$",RED,8)
    elif i==2:
        axs=axes_grid(fig,1,3);angles=np.linspace(0,2*np.pi,8,endpoint=False);centers=np.c_[2*np.cos(angles),2*np.sin(angles)]
        for k,ax in enumerate(axs):
            real=np.vstack([rng.normal(c,.13,(80,2)) for c in centers]);ax.scatter(*real.T,s=3,c=GRID)
            keep=[0,1,2,3,4] if k==1 else ([1,5] if k==2 else range(8))
            gen=np.vstack([rng.normal(centers[q],.18,(40,2)) for q in keep]);ax.scatter(*gen.T,s=7,c=RED,alpha=.65)
            ax.set_aspect("equal");ax.axis("off");panel_title(ax,["разброс","пять мод","collapse"][k])
    else:
        a,b,c=axes_grid(fig,1,3)
        for k in range(9):mini_image(a,.05+(k%3)*.31,.07+(k//3)*.28,.25,.22,COLORS[k%5],k)
        a.set(xlim=(0,1),ylim=(0,1));a.axis("off");panel_title(a,"случайные samples")
        cls=np.array([.18,.11,.08,.16,.06,.09,.12,.05,.09,.06]);b.barh(range(10),cls,color=BLUE);b.axvline(.1,color=RED,ls="--");b.set_yticks(range(10),[str(k) for k in range(10)]);clean_ax(b);panel_title(b,"покрытие классов")
        d=np.sort(rng.gamma(2,.25,60));c.plot(d,np.linspace(0,1,len(d)),color=GREEN);c.axvline(.18,color=RED);line_common(c,"nearest-train distance","CDF");panel_title(c,"копирование")


def draw_83(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        box(ax,(.04,.68),.12,.12,"$z$",PALE_GOLD,GOLD);box(ax,(.23,.65),.20,.18,"mapping\n$z\\to w$",PALE_VIOLET,VIOLET)
        arrow(ax,(.16,.74),(.23,.74),GOLD)
        res=["4²","16²","64²","256²"];xs=[.50,.62,.74,.86]
        for k,(x,r) in enumerate(zip(xs,res)):box(ax,(x,.53-.035*k),.10,.16+.07*k,r,PALES[k],COLORS[k],8);arrow(ax,(.43,.74),(x,.70),VIOLET,lw=1,rad=.05*k)
        note(ax,.72,.25,"coarse  >  fine",INK,10)
    elif i==2:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        for r in range(4):
            for c in range(5):mini_image(ax,.05+c*.185,.08+r*.205,.15,.16,COLORS[r],kind=c+r)
        ax.text(.5,.94,"fine style  >",ha="center",color=MUTED);ax.text(.01,.5,"coarse\nstyle",ha="center",va="center",rotation=90,color=MUTED)
    else:
        a,b,c=axes_grid(fig,1,3);mini_image(a,.08,.18,.84,.62,BLUE,0);panel_title(a,"исходник");a.axis("off")
        x=np.linspace(0,1,80);X,Y=np.meshgrid(x,x);pert=.5*np.sin(45*X+21*Y)+.5*np.sin(30*X-38*Y);b.imshow(pert,cmap="RdBu",vmin=-1,vmax=1);b.axis("off");panel_title(b,"$10\\times\\delta$")
        steps=np.arange(15);c.plot(steps,3-.35*steps,color=BLUE,label="верный");c.plot(steps,-1+.45*steps,color=RED,label="цель");c.axvline(7,color=GRID);line_common(c,"PGD-шаг","logit");c.legend(frameon=False);panel_title(c,"граница решения")


def draw_84(fig, i):
    rng=np.random.default_rng(84)
    if i==1:
        ax=axes_grid(fig)[0];m=rng.normal(0,1,(6,6));m[np.arange(6),np.arange(6)]+=3;m[1,4]=2.8
        heat(ax,m,[f"T{k}" for k in range(6)],[f"I{k}" for k in range(6)],"YlGnBu");panel_title(ax,"диагональ positives; яркая вне диагонали — false negative")
    elif i==2:
        ax=axes_grid(fig)[0];centers=[(-1,-.5),(1,-.2),(0,1.1)]
        for k,c0 in enumerate(centers):
            im=rng.normal(c0,.22,(16,2));tx=rng.normal(c0,.24,(16,2))
            ax.scatter(*im.T,s=40,facecolors="none",edgecolors=COLORS[k],label=f"image {k}")
            ax.scatter(*tx.T,s=22,marker="s",c=COLORS[k],alpha=.65)
        line_common(ax,"embedding 1","embedding 2");ax.legend(frameon=False,ncol=3,fontsize=8);panel_title(ax,"круги — изображения, квадраты — тексты")
    else:
        axs=axes_grid(fig,1,2)
        for ax,title,good in [(axs[0],"web encoder",2),(axs[1],"после satellite fine-tuning",4)]:
            ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1));panel_title(ax,title)
            mini_image(ax,.02,.28,.24,.48,GREEN,3)
            for k in range(5):
                fc=PALE_GREEN if k<good else PALE_RED;ec=GREEN if k<good else RED
                box(ax,(.32,.76-.15*k),.62,.11,f"{k+1}. caption · sim {0.42-.03*k:.2f}",fc,ec,7.5)


def draw_85(fig, i):
    rng=np.random.default_rng(85)
    if i==1:
        a,b=axes_grid(fig,1,2)
        for c0,col in [((-2,-1),BLUE),((1.7,1),RED),((1.2,-1.5),GREEN)]:
            pts=rng.normal(c0,.2,(18,2));a.scatter(*pts.T,c=col,s=14)
        a.scatter(*rng.normal(0,1,(30,2)).T,facecolors="none",edgecolors=GRID);line_common(a,"$z_1$","$z_2$");panel_title(a,"autoencoder: острова")
        for c0,col in [((-.7,-.3),BLUE),((.6,.4),RED),((.4,-.6),GREEN)]:
            pts=rng.normal(c0,.42,(80,2));b.scatter(*pts.T,c=col,s=7,alpha=.25)
        line_common(b,"$z_1$","$z_2$");panel_title(b,"VAE: перекрывающиеся облака")
    elif i==2:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        box(ax,(.05,.62),.16,.14,"$x$",PALE_BLUE,BLUE);box(ax,(.30,.62),.20,.14,"encoder",PALE_GREEN,GREEN);box(ax,(.57,.72),.14,.11,"$\\mu$",PALE_GOLD,GOLD);box(ax,(.57,.48),.14,.11,"$\\log\\sigma^2$",PALE_RED,RED)
        box(ax,(.57,.18),.14,.11,"$\\varepsilon\\sim N(0,I)$",PALE_VIOLET,VIOLET,8);box(ax,(.79,.54),.15,.15,"$z=\\mu+\\sigma\\varepsilon$",PALE_BLUE,BLUE,8)
        arrow(ax,(.21,.69),(.30,.69),BLUE);arrow(ax,(.50,.69),(.57,.775),GREEN);arrow(ax,(.50,.65),(.57,.535),GREEN);arrow(ax,(.71,.235),(.79,.59),VIOLET);arrow(ax,(.71,.775),(.79,.63),GOLD);arrow(ax,(.71,.535),(.79,.58),RED)
    else:
        a,b=axes_grid(fig,1,2);rate=np.linspace(.05,6,80);dist=.22+1.8*np.exp(-rate/1.4)
        a.plot(rate,dist,color=BLUE,lw=2);pts=[.2,1.5,4.8]
        for x,c in zip(pts,[RED,GREEN,GOLD]):a.scatter(x,.22+1.8*np.exp(-x/1.4),c=c,s=38)
        line_common(a,"KL rate","reconstruction distortion");panel_title(a,"rate–distortion frontier")
        b.axis("off");b.set(xlim=(0,1),ylim=(0,1))
        for k,(x,c,l) in enumerate(zip([.05,.37,.69],[RED,GREEN,GOLD],["острова","компромисс","collapse"])):
            mini_image(b,x,.46,.25,.34,c,k);b.text(x+.125,.35,l,ha="center",fontsize=8)
        panel_title(b,"три режима")


def draw_86(fig, i):
    rng=np.random.default_rng(86)
    if i==1:
        axs=axes_grid(fig,2,5)
        alphas=[.98,.8,.5,.2,.02];base=np.zeros((28,28));yy,xx=np.ogrid[:28,:28];base[(xx-14)**2+(yy-14)**2<75]=1
        for k,a0 in enumerate(alphas):
            im=np.sqrt(a0)*base+np.sqrt(1-a0)*rng.normal(0,1,base.shape)
            axs[k].imshow(im,cmap="gray",vmin=-2,vmax=2);axs[k].axis("off");panel_title(axs[k],f"$\\bar\\alpha={a0}$")
        t=np.arange(1000);bar=np.exp(-5*(t/999)**2);snr=bar/(1-bar+1e-6)
        axs[5].plot(t,snr,color=BLUE);axs[5].set_yscale("log");line_common(axs[5],"$t$","SNR");axs[5].set_position([.12,.13,.76,.25])
        for ax in axs[6:]:ax.remove()
    elif i==2:
        axs=axes_grid(fig,2,4)
        for k,ax in enumerate(axs[:4]):
            n=32;base=np.zeros((n,n));yy,xx=np.ogrid[:n,:n];base[((xx-16)/(7+2*k))**2+((yy-16)/(5+1*k))**2<1]=1
            im=base+rng.normal(0,.9-.2*k,(n,n));ax.imshow(im,cmap="gray",vmin=-2,vmax=2);ax.axis("off");panel_title(ax,f"$t={750-200*k}$")
            spec=np.abs(np.fft.rfft2(im));rad=spec.mean(0);axs[4+k].plot(rad[:16],color=COLORS[k]);axs[4+k].set_yticks([]);clean_ax(axs[4+k]);panel_title(axs[4+k],"спектр")
    else:
        axs=axes_grid(fig,1,3);x=np.linspace(-3,3,20);y=np.linspace(-3,3,20);X,Y=np.meshgrid(x,y)
        for k,ax in enumerate(axs):
            s=.35+1.0*k;Z=np.exp(-((X-1.2)**2+Y**2)/(2*s**2))+np.exp(-((X+1.2)**2+Y**2)/(2*s**2))
            gy,gx=np.gradient(np.log(Z+1e-5),y,x);ax.contourf(X,Y,Z,15,cmap="Blues");ax.quiver(X[::2,::2],Y[::2,::2],gx[::2,::2],gy[::2,::2],color=INK,alpha=.6)
            ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([]);panel_title(ax,["данные","промежуточно","почти Gaussian"][k])


def draw_87(fig, i):
    if i==1:
        axs=axes_grid(fig,1,4)
        for k,ax in enumerate(axs):ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1));panel_title(ax,["исходник","маска","результат","$|\\Delta|$"][k])
        mini_image(axs[0],.08,.12,.84,.70,BLUE,3);axs[1].add_patch(Rectangle((.26,.27),.48,.42,fc=RED,alpha=.8))
        mini_image(axs[2],.08,.12,.84,.70,GREEN,0);axs[3].add_patch(Rectangle((.22,.24),.56,.48,fc=PALE_RED,ec=RED,lw=2));axs[3].add_patch(Rectangle((.12,.16),.76,.64,fc="none",ec=GOLD,ls="--"))
    elif i==2:
        a,b=axes_grid(fig,1,2)
        for ax,title,parts in [
            (a,"100 подделок",[("найдены · 90",.90,GREEN),("пропущены · 10",.10,RED)]),
            (b,"9 900 настоящих",[("ложная тревога · 495",.05,GOLD),("верно пропущены · 9 405",.95,BLUE)]),
        ]:
            ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1));panel_title(ax,title)
            x=.05
            for label,width,color in parts:
                ax.add_patch(Rectangle((x,.40),.90*width,.24,fc=color,ec=PAPER,lw=1.5,alpha=.82))
                if width>.18:
                    ax.text(x+.45*width,.52,label,ha="center",va="center",fontsize=8.2,color=PAPER)
                else:
                    ax.annotate(label,xy=(x+.45*width,.65),xytext=(x+.45*width,.79),
                                ha="center",va="bottom",fontsize=8,color=color,
                                arrowprops=dict(arrowstyle="-",color=color,lw=.9))
                x+=.90*width
        fig.text(.50,.105,"среди 585 тревог истинны 90 · PPV = 15,4%",ha="center",fontsize=10,color=INK)
    else:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        root=(.1,.5);node(ax,root,"файл",radius=.06)
        stages=[(.30,.72,"подпись"),(.30,.30,"watermark"),(.56,.72,"детектор"),(.56,.30,"источники"),(.80,.50,"эксперт")]
        for k,(x,y,l) in enumerate(stages):box(ax,(x-.08,y-.055),.16,.11,l,PALES[k],COLORS[k],8)
        arrow(ax,root,(.22,.72),BLUE);arrow(ax,root,(.22,.30),GREEN);arrow(ax,(.38,.72),(.48,.72),GOLD);arrow(ax,(.38,.30),(.48,.30),VIOLET);arrow(ax,(.64,.72),(.72,.52),RED);arrow(ax,(.64,.30),(.72,.48),RED)
        note(ax,.80,.25,"уровень уверенности\n+ действие",INK,8.5)


def draw_88(fig, i):
    if i==1:
        a,b=axes_grid(fig,1,2);t=np.linspace(0,1,14);y=1.1+2*t-2.4*t*t
        a.plot(t,y,"o-",color=GREEN);line_common(a,"время","$y$");panel_title(a,"согласованная парабола")
        y2=y.copy();y2[7:]+=np.linspace(.4,-.2,len(y2[7:]));b.plot(t,y2,"o-",color=RED);b.plot(t,y,"--",color=GRID);b.axvline(t[7],color=GOLD);line_common(b,"время","$y$");panel_title(b,"правдоподобные кадры, неверный путь")
    elif i==2:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        cams=[(.13,.25),(.82,.25)];point=(.53,.78)
        for k,c in enumerate(cams):
            tri=Polygon([(c[0]-.05,c[1]-.04),(c[0]+.05,c[1]-.04),(c[0],c[1]+.05)],fc=PALE_BLUE,ec=BLUE);ax.add_patch(tri);ax.text(c[0],.12,f"камера {k+1}",ha="center")
            arrow(ax,c,point,GOLD,style="-",lw=1.3)
            ax.plot([c[0]-.12,c[0]+.12],[c[1]+.17,c[1]+.17],color=INK,lw=2)
        node(ax,point,"$X$",radius=.04,fc=PALE_RED,ec=RED)
        ax.plot([.63,.95],[.42,.42],color=RED,lw=1.3);note(ax,.79,.47,"эпиполярная линия",RED,8)
    else:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        box(ax,(.07,.60),.18,.14,"общий\nпрефикс",PALE_BLUE,BLUE)
        branches=[(.52,.78,"влево",GREEN),(.52,.52,"прямо",GOLD),(.52,.26,"вправо",RED)]
        for x,y,l,c in branches:
            box(ax,(x-.08,y-.05),.16,.10,l,PALES[[GREEN,GOLD,RED].index(c)+2 if c!=RED else 1],c)
            arrow(ax,(.25,.67),(x-.08,y),c)
            xs=np.linspace(x+.11,.92,30);offset={GREEN:.16,GOLD:0,RED:-.16}[c];ys=y+.4*offset*np.linspace(0,1,30)**1.4
            ax.plot(xs,ys,color=c,lw=2);ax.fill_between(xs,ys-.015-.06*np.linspace(0,1,30),ys+.015+.06*np.linspace(0,1,30),color=c,alpha=.12)
        note(ax,.77,.09,"конус = неопределённость rollout",MUTED,8.5)


def draw_89(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        pts=[(.16,.68),(.40,.78),(.66,.68),(.66,.32),(.40,.22),(.16,.32)];labs=["цель","план","tool","наблюдение","проверка","ответ"];cols=COLORS[:6]
        for p,l,c in zip(pts,labs,cols):box(ax,(p[0]-.075,p[1]-.05),.15,.10,l,PALES[cols.index(c)%5],c,8)
        for k in range(len(pts)-1):arrow(ax,pts[k],pts[k+1],cols[k])
        arrow(ax,pts[4],pts[1],RED,rad=.24);arrow(ax,pts[4],pts[5],GREEN)
        note(ax,.51,.49,"ошибка  >  новый план",RED,8.5)
    elif i==2:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        box(ax,(.06,.58),.18,.14,"результат",PALE_BLUE,BLUE)
        checks=[(.38,.75,"unit test",GREEN),(.38,.50,"schema",GOLD),(.38,.25,"human approval",VIOLET)]
        for x,y,l,c in checks:box(ax,(x,y-.055),.22,.11,l,PALES[[GREEN,GOLD,VIOLET].index(c)+2 if c!=VIOLET else 4],c);arrow(ax,(.24,.65),(x,y),c)
        box(ax,(.74,.48),.18,.16,"evidence\nbundle",PALE_GREEN,GREEN)
        for x,y,l,c in checks:arrow(ax,(x+.22,y),(.74,.56),c)
    else:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        box(ax,(.05,.65),.18,.13,"user goal",PALE_BLUE,BLUE);box(ax,(.05,.28),.18,.13,"web page\nuntrusted",PALE_RED,RED)
        box(ax,(.38,.48),.20,.17,"agent\ncontext",PALE_GOLD,GOLD);box(ax,(.72,.68),.20,.13,"secret tool",PALE_VIOLET,VIOLET);box(ax,(.72,.28),.20,.13,"outbound tool",PALE_GREEN,GREEN)
        arrow(ax,(.23,.71),(.38,.60),BLUE);arrow(ax,(.23,.34),(.38,.52),RED);arrow(ax,(.58,.59),(.72,.73),VIOLET);arrow(ax,(.58,.53),(.72,.35),GREEN)
        ax.add_patch(Rectangle((.65,.18),.31,.70,fc="none",ec=RED,lw=1.4,ls="--"));note(ax,.81,.91,"capability gate",RED,8)


def draw_90(fig, i):
    if i==1:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        widths=[.9,.72,.55,.39,.25];labels=["тема","решение","данные + target","метрика + baseline","проверяемый вопрос"]
        for k,(w,l) in enumerate(zip(widths,labels)):
            x=.5-w/2;y=.82-.15*k
            ax.add_patch(Polygon([(x,y),(x+w,y),(x+w-.07,y-.11),(x+.07,y-.11)],fc=PALES[k],ec=COLORS[k],lw=1.2))
            ax.text(.5,y-.055,l,ha="center",va="center",fontsize=9)
    elif i==2:
        ax=axes_grid(fig)[0];ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1))
        labs=["snapshot","split","baseline","модель","ошибки","интерактив","защита"];xs=np.linspace(.03,.87,len(labs))
        for k,(x,l) in enumerate(zip(xs,labs)):
            box(ax,(x,.55),.10,.11,l,PALES[k%5],COLORS[k%6],7)
            if k<len(labs)-1:arrow(ax,(x+.10,.605),(xs[k+1],.605),COLORS[k%6])
            if k in [1,3,5]:
                ax.add_patch(Polygon([(x+.05,.40),(x+.09,.34),(x+.05,.28),(x+.01,.34)],fc=PALE_GOLD,ec=GOLD));ax.text(x+.05,.34,"OK",ha="center",va="center",fontsize=7,color=GREEN)
        note(ax,.5,.16,"артефакт  >  проверка  >  следующий этап",INK,9.5)
    else:
        axs=axes_grid(fig,1,3);groups=["зима","лето","пропуски","пики"];score=[.78,.72,.56,.61];err=[.03,.04,.08,.07]
        axs[0].errorbar(range(4),score,yerr=err,fmt="o",color=BLUE,ecolor=GOLD,capsize=4);axs[0].set_xticks(range(4),groups,rotation=28);axs[0].set_ylim(.4,.9);clean_ax(axs[0],grid=True);panel_title(axs[0],"метрика по режимам")
        for k,ax in enumerate(axs[1:]):
            ax.axis("off");ax.set(xlim=(0,1),ylim=(0,1));mini_image(ax,.12,.40,.76,.42,RED if k else GOLD,k+2);box(ax,(.12,.12),.76,.18,["уверенная ошибка","сдвиг периода"][k],PALE_RED,RED,8);panel_title(ax,f"контрпример {k+1}")


DRAWERS = {
    65: draw_65, 66: draw_66, 67: draw_67, 68: draw_68, 69: draw_69,
    70: draw_70, 71: draw_71, 72: draw_72, 73: draw_73, 74: draw_74,
    75: draw_75, 76: draw_76, 77: draw_77, 78: draw_78, 79: draw_79,
    80: draw_80, 81: draw_81, 82: draw_82, 83: draw_83, 84: draw_84,
    85: draw_85, 86: draw_86, 87: draw_87, 88: draw_88, 89: draw_89,
    90: draw_90,
}


def add_svg_accessibility(path: Path, title: str, description: str) -> None:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(path)
    root = tree.getroot()
    ns = "{http://www.w3.org/2000/svg}"
    title_el = ET.Element(f"{ns}title")
    title_el.text = title
    desc_el = ET.Element(f"{ns}desc")
    desc_el.text = description
    root.insert(0, desc_el)
    root.insert(0, title_el)
    root.set("role", "img")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def render(spec: dict[str, str | int]) -> Path:
    lesson = int(spec["lesson"])
    index = int(spec["index"])
    title = str(spec["title"])
    fig = plt.figure(figsize=(9.2, 5.25), dpi=150)
    frame(fig, title, lesson, index)
    DRAWERS[lesson](fig, index)

    out = PUBLIC / str(spec["src"]).lstrip("/")
    out.parent.mkdir(parents=True, exist_ok=True)
    thumb = THUMBS / f"{lesson:02d}-{index}.png"
    THUMBS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches=None)
    fig.savefig(thumb, format="png", dpi=110, bbox_inches=None)
    plt.close(fig)
    description = f'{spec["alt"]}. {str(spec["caption"]).strip()}'
    add_svg_accessibility(out, title, description)
    return thumb


def contact_sheets(thumbs: list[Path]) -> list[Path]:
    QA.mkdir(parents=True, exist_ok=True)
    sheets: list[Path] = []
    cols, rows = 3, 3
    cell_w, cell_h = 820, 475
    for page, start in enumerate(range(0, len(thumbs), cols * rows), 1):
        chunk = thumbs[start : start + cols * rows]
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), PAPER)
        for k, path in enumerate(chunk):
            im = Image.open(path).convert("RGB")
            im.thumbnail((cell_w - 20, cell_h - 20), Image.Resampling.LANCZOS)
            x = (k % cols) * cell_w + (cell_w - im.width) // 2
            y = (k // cols) * cell_h + (cell_h - im.height) // 2
            canvas.paste(im, (x, y))
        out = QA / f"contact-{page:02d}.png"
        canvas.save(out, quality=94)
        sheets.append(out)
    return sheets


def main() -> None:
    specs = parse_specs()
    thumbs = []
    for spec in specs:
        thumbs.append(render(spec))
        print(f'{int(spec["lesson"]):02d}-{int(spec["index"])}  {spec["title"]}')
    sheets = contact_sheets(thumbs)
    print(f"\nGenerated {len(specs)} SVGs and {len(sheets)} contact sheets.")


if __name__ == "__main__":
    main()
