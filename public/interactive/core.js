// Kontur Interactive
// A small dependency-free runtime for figures embedded in textbook articles.
// Article markup declares only:
//   <div data-ai-widget="name" data-opts='{"key":"value"}'></div>
// A widget registers a builder:
//   KonturInt.register("name", (root, opts, K) => { ...; return destroy; });
(function () {
  "use strict";

  if (window.KonturInt && window.KonturInt.__core) {
    window.dispatchEvent(new Event("kontur-int-ready"));
    return;
  }

  var registry = new Map();
  var controlSequence = 0;

  var COLORS = {
    blue: "#315f8c",
    red: "#b94a3b",
    green: "#38735d",
    gold: "#a57920",
    violet: "#6f5a8f",
    ink: "#171915",
    muted: "#6e726a",
    grid: "#deddd4",
    axis: "#9b9a91",
    paper: "#fffef9",
    wash: "#f6f5ef",
  };

  function element(tag, className, attributes) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (attributes) {
      Object.keys(attributes).forEach(function (key) {
        var value = attributes[key];
        if (key === "text") node.textContent = value;
        else if (key === "htmlFor") node.htmlFor = value;
        else node.setAttribute(key, value);
      });
    }
    return node;
  }

  function row(parent, className) {
    var node = element("div", "kontur-int-row" + (className ? " " + className : ""));
    parent.appendChild(node);
    return node;
  }

  function makeCanvas(parent, width, height, options) {
    options = options || {};
    var canvas = element("canvas", "kontur-int-canvas", {
      role: "img",
      "aria-label": options.label || "Интерактивная иллюстрация",
    });
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = "100%";
    canvas.style.maxWidth = (options.maxWidth || width) + "px";
    canvas.style.height = "auto";
    canvas.style.touchAction = options.drag === false ? "auto" : "none";
    parent.appendChild(canvas);

    var context = canvas.getContext("2d");
    var observer = null;
    var resizeFrame = 0;
    var lastCssWidth = 0;
    var lastCssHeight = 0;
    var state = { canvas: canvas, ctx: context, w: width, h: height };

    function resize() {
      var cssWidth = canvas.getBoundingClientRect().width || width;
      var cssHeight = cssWidth * height / width;
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var pixelWidth = Math.round(cssWidth * dpr);
      var pixelHeight = Math.round(cssHeight * dpr);
      if (
        canvas.width === pixelWidth
        && canvas.height === pixelHeight
        && Math.abs(lastCssWidth - cssWidth) < 0.25
        && Math.abs(lastCssHeight - cssHeight) < 0.25
      ) return false;
      lastCssWidth = cssWidth;
      lastCssHeight = cssHeight;
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
      context.setTransform(
        dpr * cssWidth / width,
        0,
        0,
        dpr * cssHeight / height,
        0,
        0,
      );
      return true;
    }

    resize();
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(function () {
        if (resizeFrame) return;
        resizeFrame = requestAnimationFrame(function () {
          resizeFrame = 0;
          if (resize() && options.onResize) options.onResize();
        });
      });
      observer.observe(parent);
      observer.observe(canvas);
    }

    // A canvas can mount before fonts, rails and responsive rules have settled.
    // Draw once more on the next frame even when the first measured width did
    // not change; otherwise a late intrinsic resize can leave a blank surface
    // until the reader touches a control.
    resizeFrame = requestAnimationFrame(function () {
      resizeFrame = 0;
      resize();
      if (options.onResize) options.onResize();
    });

    state.resize = resize;
    state.destroy = function () {
      if (observer) observer.disconnect();
      if (resizeFrame) cancelAnimationFrame(resizeFrame);
    };
    return state;
  }

  function scale(domainStart, domainEnd, rangeStart, rangeEnd) {
    var slope = (rangeEnd - rangeStart) / (domainEnd - domainStart || 1);
    var fn = function (value) {
      return rangeStart + (value - domainStart) * slope;
    };
    fn.inv = function (value) {
      return domainStart + (value - rangeStart) / slope;
    };
    return fn;
  }

  function slider(parent, config, onInput) {
    controlSequence += 1;
    var id = "kontur-range-" + controlSequence;
    var wrap = element("div", "kontur-int-control");
    var label = element("label", "kontur-int-label", { htmlFor: id });
    var name = element("span", "kontur-int-label-name", { text: config.label });
    var value = element("output", "kontur-int-label-value");
    label.appendChild(name);
    label.appendChild(value);
    var input = element("input", "kontur-int-range", {
      id: id,
      type: "range",
      min: config.min,
      max: config.max,
      step: config.step == null ? "any" : config.step,
      value: config.value,
      "aria-label": config.label,
    });
    var format = config.format || function (number) {
      return String(Math.round(number * 1000) / 1000);
    };
    function renderValue() {
      value.value = format(Number(input.value)) + (config.unit || "");
      value.textContent = value.value;
    }
    input.addEventListener("input", function () {
      renderValue();
      onInput(Number(input.value));
    });
    wrap.appendChild(label);
    wrap.appendChild(input);
    parent.appendChild(wrap);
    renderValue();
    return {
      input: input,
      get: function () { return Number(input.value); },
      set: function (next) {
        input.value = next;
        renderValue();
      },
    };
  }

  function segmented(parent, config, onChange) {
    var wrap = element("fieldset", "kontur-int-control kontur-int-control--segmented");
    if (config.label) {
      wrap.appendChild(element("legend", "kontur-int-label-name", { text: config.label }));
    }
    var group = element("div", "kontur-int-segments");
    var current = config.value;
    var buttons = [];
    config.options.forEach(function (option) {
      var button = element(
        "button",
        "kontur-int-segment" + (option.value === current ? " is-active" : ""),
        { type: "button", text: option.label, "aria-pressed": option.value === current ? "true" : "false" },
      );
      button.addEventListener("click", function () {
        current = option.value;
        buttons.forEach(function (item) {
          var active = item.button === button;
          item.button.classList.toggle("is-active", active);
          item.button.setAttribute("aria-pressed", active ? "true" : "false");
        });
        onChange(option.value);
      });
      buttons.push({ value: option.value, button: button });
      group.appendChild(button);
    });
    wrap.appendChild(group);
    parent.appendChild(wrap);
    return {
      set: function (next) {
        current = next;
        buttons.forEach(function (item) {
          var active = item.value === next;
          item.button.classList.toggle("is-active", active);
          item.button.setAttribute("aria-pressed", active ? "true" : "false");
        });
      },
    };
  }

  function readout(parent) {
    var node = element("dl", "kontur-int-readout");
    parent.appendChild(node);
    return {
      set: function (items) {
        node.replaceChildren();
        items.forEach(function (item) {
          var group = element("div", "kontur-int-metric");
          group.appendChild(element("dt", null, { text: item.label }));
          var value = element("dd", null, { text: item.value });
          if (item.color) value.style.color = item.color;
          group.appendChild(value);
          node.appendChild(group);
        });
      },
    };
  }

  function caption(parent, text) {
    var node = element("p", "kontur-int-caption", { text: text });
    parent.appendChild(node);
    return node;
  }

  function hint(parent, text) {
    var node = element("p", "kontur-int-hint", { text: text });
    parent.appendChild(node);
    return node;
  }

  function drag(target, logicalSize, handlers) {
    var activePointer = null;
    function point(event) {
      var rect = target.getBoundingClientRect();
      var px = (event.clientX - rect.left) / rect.width;
      var py = (event.clientY - rect.top) / rect.height;
      return {
        x: px * logicalSize.w,
        y: py * logicalSize.h,
        px: px,
        py: py,
      };
    }
    function down(event) {
      activePointer = event.pointerId;
      target.setPointerCapture(event.pointerId);
      if (handlers.down) handlers.down(point(event), event);
      event.preventDefault();
    }
    function move(event) {
      var current = point(event);
      if (activePointer === event.pointerId && handlers.move) {
        handlers.move(current, event);
        event.preventDefault();
      } else if (handlers.hover) {
        handlers.hover(current, event);
      }
    }
    function up(event) {
      if (activePointer !== event.pointerId) return;
      if (handlers.up) handlers.up(point(event), event);
      activePointer = null;
      if (target.hasPointerCapture(event.pointerId)) target.releasePointerCapture(event.pointerId);
    }
    target.addEventListener("pointerdown", down);
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", up);
    target.addEventListener("pointercancel", up);
    return function () {
      target.removeEventListener("pointerdown", down);
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", up);
      target.removeEventListener("pointercancel", up);
    };
  }

  function rafThrottle(fn) {
    var frame = 0;
    var args = null;
    return function () {
      args = arguments;
      if (frame) return;
      frame = requestAnimationFrame(function () {
        frame = 0;
        fn.apply(null, args);
      });
    };
  }

  function visibleLoop(root, step) {
    var running = true;
    var visible = true;
    var frame = 0;
    var start = 0;
    var last = 0;
    var observer = null;
    function tick(time) {
      if (!running) return;
      if (!start) start = time;
      if (visible) step((time - start) / 1000, time, last ? (time - last) / 1000 : 0);
      last = time;
      frame = requestAnimationFrame(tick);
    }
    if (typeof IntersectionObserver !== "undefined") {
      observer = new IntersectionObserver(function (entries) {
        visible = Boolean(entries[0] && entries[0].isIntersecting);
      }, { rootMargin: "160px" });
      observer.observe(root);
    }
    frame = requestAnimationFrame(tick);
    return {
      stop: function () {
        running = false;
        cancelAnimationFrame(frame);
        if (observer) observer.disconnect();
      },
    };
  }

  function mountNode(node) {
    if (node.__konturMounted) return;
    var name = node.getAttribute("data-ai-widget");
    var builder = registry.get(name);
    if (!builder) return;
    node.__konturMounted = true;
    var options = {};
    var raw = node.getAttribute("data-opts");
    if (raw) {
      try { options = JSON.parse(raw); } catch { options = {}; }
    }
    var root = element("figure", "kontur-int");
    node.replaceChildren(root);
    try {
      var destroy = builder(root, options, API);
      node.__konturDestroy = function () {
        if (typeof destroy === "function") destroy();
        node.__konturMounted = false;
      };
    } catch (error) {
      root.replaceChildren(element("p", "kontur-int-error", {
        text: "Интерактив не загрузился: " + (error && error.message ? error.message : String(error)),
      }));
      console.error("[KonturInt]", name, error);
    }
  }

  function mountAll(scope) {
    var root = scope && scope.querySelectorAll ? scope : document;
    root.querySelectorAll("[data-ai-widget]").forEach(mountNode);
  }

  function register(name, builder) {
    registry.set(name, builder);
    document.querySelectorAll('[data-ai-widget="' + name + '"]').forEach(mountNode);
  }

  var API = {
    __core: true,
    COLORS: COLORS,
    element: element,
    row: row,
    makeCanvas: makeCanvas,
    scale: scale,
    slider: slider,
    segmented: segmented,
    readout: readout,
    caption: caption,
    hint: hint,
    drag: drag,
    rafThrottle: rafThrottle,
    visibleLoop: visibleLoop,
    register: register,
    mountAll: mountAll,
  };

  window.KonturInt = API;
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { mountAll(document); }, { once: true });
  } else {
    queueMicrotask(function () { mountAll(document); });
  }
  window.dispatchEvent(new Event("kontur-int-ready"));
})();
