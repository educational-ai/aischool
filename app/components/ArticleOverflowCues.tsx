"use client";

import { useEffect, useRef } from "react";

const MOBILE_QUERY = "(max-width: 640px)";

export function ArticleOverflowCues() {
  const markerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const article = markerRef.current?.closest(".article-markdown");
    if (!article) return;

    const displays = Array.from(
      article.querySelectorAll<HTMLElement>(".katex-display"),
    );
    if (displays.length === 0) return;

    const media = window.matchMedia(MOBILE_QUERY);

    function updateEdge(display: HTMLElement) {
      const atEnd =
        display.scrollLeft + display.clientWidth >= display.scrollWidth - 2;
      display.dataset.mobileScrollEnd = atEnd ? "true" : "false";
    }

    function updateDisplay(display: HTMLElement) {
      const isScrollable =
        media.matches && display.scrollWidth > display.clientWidth + 4;

      if (isScrollable) {
        display.dataset.mobileScrollable = "true";
        display.setAttribute("role", "region");
        display.setAttribute(
          "aria-label",
          "Длинная формула. Прокрутите по горизонтали, чтобы прочитать её полностью.",
        );
        updateEdge(display);
        return;
      }

      delete display.dataset.mobileScrollable;
      delete display.dataset.mobileScrollEnd;
      display.removeAttribute("role");
      display.removeAttribute("aria-label");
    }

    function updateAll() {
      displays.forEach(updateDisplay);
    }

    const scrollHandlers = displays.map((display) => {
      const onScroll = () => updateEdge(display);
      display.addEventListener("scroll", onScroll, { passive: true });
      return { display, onScroll };
    });

    const resizeObserver = new ResizeObserver(updateAll);
    displays.forEach((display) => resizeObserver.observe(display));
    media.addEventListener("change", updateAll);
    const frame = window.requestAnimationFrame(updateAll);

    return () => {
      window.cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      media.removeEventListener("change", updateAll);
      scrollHandlers.forEach(({ display, onScroll }) => {
        display.removeEventListener("scroll", onScroll);
        delete display.dataset.mobileScrollable;
        delete display.dataset.mobileScrollEnd;
        display.removeAttribute("role");
        display.removeAttribute("aria-label");
      });
    };
  }, []);

  return <span ref={markerRef} className="article-overflow-cues" aria-hidden="true" />;
}
