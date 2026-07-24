"use client";

import { useEffect, useId, useMemo, useRef } from "react";

declare global {
  interface Window {
    KonturInt?: {
      __core: boolean;
      mountAll: (scope?: ParentNode) => void;
    };
    __konturScriptPromises?: Map<string, Promise<void>>;
  }
}

function loadScript(source: string) {
  window.__konturScriptPromises ??= new Map();
  const cached = window.__konturScriptPromises.get(source);
  if (cached) return cached;

  const promise = new Promise<void>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      `script[data-kontur-source="${source}"]`,
    );
    if (existing?.dataset.loaded === "true") {
      resolve();
      return;
    }
    const script = existing ?? document.createElement("script");
    script.src = source;
    script.defer = true;
    script.dataset.konturSource = source;
    script.addEventListener(
      "load",
      () => {
        script.dataset.loaded = "true";
        resolve();
      },
      { once: true },
    );
    script.addEventListener("error", () => reject(new Error(`Не загружен ${source}`)), {
      once: true,
    });
    if (!existing) document.head.appendChild(script);
  });
  window.__konturScriptPromises.set(source, promise);
  return promise;
}

const widgetBundles: Record<string, string> = {
  "model-residual-lab": "g10-data",
  "dataset-forensics": "g10-data",
  "turing-jury": "g10-data",
  "discipline-layers": "g10-data",
  "learning-signals": "g10-data",
  "classifier-studio": "g10-data",
  "regression-workbench": "g10-data",
  "clustering-lens": "g10-data",
  "label-budget-game": "g10-data",
  "galileo-lab": "g10-data",
  "dataset-passport-audit": "g10-data",
};

export function WidgetMount({
  name,
  title,
  options = {},
}: {
  name: string;
  title?: string;
  options?: Record<string, string | number | boolean>;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const cueRef = useRef<HTMLParagraphElement>(null);
  const cueId = useId();
  const serialized = useMemo(() => JSON.stringify(options), [options]);

  useEffect(() => {
    const node = ref.current as (HTMLDivElement & { __konturDestroy?: () => void }) | null;
    const cue = cueRef.current;
    let cancelled = false;
    let resizeObserver: ResizeObserver | null = null;
    let media: MediaQueryList | null = null;
    let scrollRegion: HTMLElement | null = null;
    const safeName = name.replace(/[^a-z0-9-]/gi, "");
    const safeBundle = (widgetBundles[safeName] ?? safeName).replace(/[^a-z0-9-]/gi, "");

    function updateMobileOverflow() {
      if (!cue || !scrollRegion || !media) return;
      const isScrollable =
        media.matches && scrollRegion.scrollWidth > scrollRegion.clientWidth + 4;
      cue.hidden = !isScrollable;

      if (isScrollable) {
        scrollRegion.dataset.mobileScrollable = "true";
        scrollRegion.setAttribute("role", "region");
        scrollRegion.setAttribute(
          "aria-label",
          `${title ?? "Интерактивный график"}. Продолжение доступно горизонтальной прокруткой.`,
        );
        scrollRegion.setAttribute("aria-describedby", cueId);
        return;
      }

      delete scrollRegion.dataset.mobileScrollable;
      scrollRegion.removeAttribute("role");
      scrollRegion.removeAttribute("aria-label");
      scrollRegion.removeAttribute("aria-describedby");
    }

    async function mount() {
      await loadScript("/interactive/core.js");
      await loadScript(`/interactive/widgets/${safeBundle}.js`);
      if (!cancelled && node) {
        window.KonturInt?.mountAll(node.parentNode ?? document);
        scrollRegion = node.querySelector<HTMLElement>(
          ".kontur-int > .kontur-int-row:not(.controls)",
        );
        if (scrollRegion) {
          const canvas = scrollRegion.querySelector<HTMLCanvasElement>(
            ".kontur-int-canvas",
          );
          const logicalWidth = Number.parseFloat(canvas?.style.maxWidth ?? "");
          const mobileWidth = Number.isFinite(logicalWidth)
            ? Math.min(logicalWidth, 920)
            : 920;
          scrollRegion.style.setProperty(
            "--kontur-mobile-canvas-width",
            `${mobileWidth}px`,
          );
          media = window.matchMedia("(max-width: 640px)");
          resizeObserver = new ResizeObserver(updateMobileOverflow);
          resizeObserver.observe(scrollRegion);
          media.addEventListener("change", updateMobileOverflow);
          updateMobileOverflow();
        }
      }
    }
    mount().catch((error) => {
      if (node) {
        node.textContent =
          error instanceof Error ? error.message : "Интерактив не загрузился";
        node.classList.add("widget-mount--error");
      }
    });
    return () => {
      cancelled = true;
      resizeObserver?.disconnect();
      media?.removeEventListener("change", updateMobileOverflow);
      if (scrollRegion) {
        delete scrollRegion.dataset.mobileScrollable;
        scrollRegion.style.removeProperty("--kontur-mobile-canvas-width");
        scrollRegion.removeAttribute("role");
        scrollRegion.removeAttribute("aria-label");
        scrollRegion.removeAttribute("aria-describedby");
      }
      node?.__konturDestroy?.();
    };
  }, [cueId, name, title]);

  return (
    <div className="widget-figure" aria-label={title ?? name}>
      {title ? <h3 className="widget-figure__title">{title}</h3> : null}
      <p ref={cueRef} id={cueId} className="widget-figure__mobile-pan" hidden>
        График шире экрана — листайте по горизонтали
        <span aria-hidden="true"> →</span>
      </p>
      <div ref={ref} data-ai-widget={name} data-opts={serialized}>
        <p className="widget-loading">Загружается живая иллюстрация…</p>
      </div>
    </div>
  );
}
