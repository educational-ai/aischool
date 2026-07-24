import type { ComponentPropsWithoutRef } from "react";
import Image from "next/image";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkDirective from "remark-directive";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { visit } from "unist-util-visit";
import { ArticleOverflowCues } from "./ArticleOverflowCues";
import { SideCalculation } from "./SideCalculation";
import { WidgetMount } from "./WidgetMount";

type DirectiveNode = {
  type: string;
  name?: string;
  attributes?: Record<string, string>;
  data?: {
    hName?: string;
    hProperties?: Record<string, string>;
  };
};

function articleDirectives() {
  return (tree: unknown) => {
    visit(tree as Parameters<typeof visit>[0], (node: DirectiveNode) => {
      if (!["containerDirective", "leafDirective", "textDirective"].includes(node.type)) {
        return;
      }
      const name = node.name ?? "";
      if (
        ![
          "definition",
          "proposition",
          "proof",
          "exercise",
          "answer",
          "sidenote",
          "problem",
          "figure",
          "widget",
          "sidecalc",
        ].includes(name)
      ) {
        return;
      }
      node.data ??= {};
      node.data.hName =
        name === "widget" || name === "sidecalc"
          ? "div"
          : name === "figure"
            ? "figure"
            : "aside";
      node.data.hProperties = {
        ...(node.attributes ?? {}),
        "data-directive": name,
      };
    });
  };
}

type DirectiveProps = ComponentPropsWithoutRef<"aside"> & {
  "data-directive"?: string;
  title?: string;
  label?: string;
  points?: string;
  src?: string;
  alt?: string;
  caption?: string;
  credit?: string;
  mode?: string;
};

const components: Components = {
  aside({ children, node, ...rawProps }) {
    void node;
    const props = rawProps as DirectiveProps;
    const kind = props["data-directive"];
    if (!kind) return <aside {...rawProps}>{children}</aside>;
    const { title, label, points, src, alt, caption, credit, mode, ...asideProps } = props;
    if (kind === "sidenote") {
      return (
        <aside
          className={`sidenote${mode ? ` sidenote--${mode}` : ""}`}
          aria-label={title}
        >
          <span>{label ?? "С полей"}</span>
          {title ? <strong>{title}</strong> : null}
          {src ? (
            <figure className="sidenote__media">
              <Image
                src={src}
                alt={alt ?? caption ?? title ?? ""}
                width={440}
                height={300}
                sizes="220px"
                unoptimized
              />
              {caption || credit ? (
                <figcaption>
                  {caption}
                  {credit ? <small>{credit}</small> : null}
                </figcaption>
              ) : null}
            </figure>
          ) : null}
          <div>{children}</div>
        </aside>
      );
    }
    if (kind === "problem") {
      return (
        <aside className="article-problem" {...asideProps}>
          <div className="article-problem__points">
            <span>{points ?? "—"}</span>
            <small>{points === "1" ? "балл" : "баллов"}</small>
          </div>
          <div className="article-problem__body">{children}</div>
        </aside>
      );
    }
    if (kind === "answer") {
      return (
        <details className="article-answer">
          <summary>{title ?? "Показать ответ"}</summary>
          <div>{children}</div>
        </details>
      );
    }
    const names: Record<string, string> = {
      definition: "Определение",
      proposition: "Утверждение",
      proof: "Разбор",
      exercise: "Упражнение",
    };
    return (
      <aside className={`article-block article-block--${kind}`}>
        <header>
          <span>{names[kind] ?? kind}</span>
          {title ? <strong>{title}</strong> : null}
        </header>
        <div>{children}</div>
      </aside>
    );
  },
  figure({ children, node, ...rawProps }) {
    void node;
    const props = rawProps as ComponentPropsWithoutRef<"figure"> & {
      "data-directive"?: string;
      src?: string;
      title?: string;
      alt?: string;
      id?: string;
      wide?: string;
    };
    if (props["data-directive"] !== "figure" || !props.src) {
      return <figure {...rawProps}>{children}</figure>;
    }
    const { src, title, alt, id, wide } = props;
    const number = id?.match(/^fig-(\d{2})-(\d+)$/);
    const visibleTitle =
      title && /^Рис\./u.test(title)
        ? title
        : `${number ? `Рис. ${Number(number[1])}.${number[2]}. ` : ""}${title ?? "Рисунок"}`;
    return (
      <figure
        className={`article-figure${wide === "true" ? " article-figure--wide" : ""}`}
        id={id}
      >
        <div className="article-figure__mobile-tools">
          <span>Рисунок шире экрана — проведите по нему</span>
          <a
            href={src}
            target="_blank"
            rel="noreferrer"
            aria-label={`Открыть в полном размере: ${visibleTitle}`}
          >
            Открыть целиком ↗
          </a>
        </div>
        <div
          className="article-figure__frame"
          role="region"
          aria-label={`${visibleTitle}. На узком экране рисунок прокручивается по горизонтали.`}
        >
          <Image
            src={src}
            alt={alt ?? title ?? ""}
            width={1440}
            height={800}
            sizes="(max-width: 860px) 100vw, 1040px"
            unoptimized
          />
        </div>
        <figcaption>
          <span>{visibleTitle}</span>
          <div>{children}</div>
        </figcaption>
      </figure>
    );
  },
  div({ children, node, ...rawProps }) {
    void node;
    const props = rawProps as ComponentPropsWithoutRef<"div"> & {
      "data-directive"?: string;
      name?: string;
      title?: string;
      kind?: string;
    };
    if (props["data-directive"] === "sidecalc" && props.kind) {
      const options = Object.fromEntries(
        Object.entries(props).filter(
          ([key]) => !["data-directive", "kind", "node", "children"].includes(key),
        ),
      ) as Record<string, string>;
      return <SideCalculation kind={props.kind} options={options} />;
    }
    if (props["data-directive"] !== "widget" || !props.name) {
      return <div {...rawProps}>{children}</div>;
    }
    const options = Object.fromEntries(
      Object.entries(props).filter(
        ([key]) => !["data-directive", "name", "title", "node", "children"].includes(key),
      ),
    ) as Record<string, string>;
    return <WidgetMount name={props.name} title={props.title} options={options} />;
  },
  a({ children, ...props }) {
    const external = typeof props.href === "string" && /^https?:/.test(props.href);
    return (
      <a {...props} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}>
        {children}
      </a>
    );
  },
  h2({ children }) {
    const text = String(children);
    const id = text
      .toLocaleLowerCase("ru-RU")
      .replace(/[^\p{L}\p{N}\s-]/gu, "")
      .trim()
      .replace(/\s+/g, "-");
    return <h2 id={id}>{children}</h2>;
  },
};

export function ArticleMarkdown({ markdown }: { markdown: string }) {
  // remark-directive treats the second half of times and ratios such as
  // `12:30` or `3:2` as an empty text directive. Escaping that colon keeps the
  // visible text unchanged and prevents an invalid <div> from appearing inside
  // the surrounding paragraph during hydration.
  const safeMarkdown = markdown.replace(
    /([\p{L}\p{N}]):(?=[\p{L}\p{N}])/gu,
    "$1\\:",
  );

  return (
    <div className="article-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, remarkDirective, articleDirectives]}
        rehypePlugins={[rehypeKatex]}
        components={components}
      >
        {safeMarkdown}
      </ReactMarkdown>
      <ArticleOverflowCues />
    </div>
  );
}
