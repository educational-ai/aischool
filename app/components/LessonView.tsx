"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import {
  lessonById,
  lessonHref,
  lessons,
  lessonsForModule,
  moduleById,
  modules,
} from "../../lib/course";
import { ArticleMarkdown } from "./ArticleMarkdown";
import { useProgress, writeProgress } from "./useProgress";

function slugify(value: string) {
  return value
    .toLocaleLowerCase("ru-RU")
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .trim()
    .replace(/\s+/g, "-");
}

function articleToc(markdown: string) {
  return [...markdown.matchAll(/^##\s+(.+)$/gm)].map((match) => ({
    title: match[1].trim(),
    id: slugify(match[1]),
  }));
}

function CourseRail({
  lessonId,
  open,
  onClose,
}: {
  lessonId: string;
  open: boolean;
  onClose: () => void;
}) {
  const current = lessonById.get(lessonId);
  if (!current) return null;
  const gradeModules = modules.filter((module) => module.id.startsWith(`g${current.grade}-`));

  return (
    <>
      <button
        type="button"
        className={`book-rail-scrim${open ? " is-open" : ""}`}
        aria-label="Закрыть содержание"
        onClick={onClose}
      />
      <aside className={`book-rail${open ? " is-open" : ""}`}>
        <header>
          <Link href="/" onClick={onClose} aria-label="К оглавлению">
            <span>К</span>
            <strong>Контур</strong>
          </Link>
          <button type="button" onClick={onClose} aria-label="Закрыть содержание">
            ×
          </button>
        </header>
        <div className="book-rail__course">
          <span>Информатика и ИИ</span>
          <strong>{current.grade} класс</strong>
        </div>
        <nav aria-label={`Уроки ${current.grade} класса`}>
          {gradeModules.map((module) => {
            const activeModule = module.id === current.moduleId;
            return (
              <details key={module.id} open={activeModule}>
                <summary>
                  <span>{module.number}</span>
                  <strong>{module.shortTitle}</strong>
                </summary>
                <ol start={Number(lessonsForModule(module.id)[0]?.id ?? 1)}>
                  {lessonsForModule(module.id).map((lesson) => (
                    <li key={lesson.id}>
                      <Link
                        href={lessonHref(lesson.id)}
                        className={lesson.id === lessonId ? "is-active" : ""}
                        onClick={onClose}
                      >
                        <span>{lesson.id}</span>
                        <span>{lesson.title}</span>
                      </Link>
                    </li>
                  ))}
                </ol>
              </details>
            );
          })}
        </nav>
        <footer>
          <Link href={lessonHref(current.grade === 10 ? "41" : "01")}>
            Перейти в {current.grade === 10 ? "11" : "10"} класс
          </Link>
        </footer>
      </aside>
    </>
  );
}

export function LessonView({
  lessonId,
  markdown,
  connections,
}: {
  lessonId: string;
  markdown: string;
  connections: {
    outgoing: string[];
    incoming: string[];
  };
}) {
  const lesson = lessonById.get(lessonId);
  const courseModule = lesson ? moduleById.get(lesson.moduleId) : undefined;
  const index = lessons.findIndex((item) => item.id === lessonId);
  const previous = index > 0 ? lessons[index - 1] : undefined;
  const next = index < lessons.length - 1 ? lessons[index + 1] : undefined;
  const gradeLessons = lesson ? lessons.filter((item) => item.grade === lesson.grade) : [];
  const gradeIndex = gradeLessons.findIndex((item) => item.id === lessonId);
  const toc = useMemo(() => articleToc(markdown), [markdown]);
  const [railOpen, setRailOpen] = useState(false);
  const completed = useProgress();

  if (!lesson || !courseModule) return null;

  const complete = completed.includes(lessonId);
  const completedInGrade = gradeLessons.filter((item) => completed.includes(item.id)).length;

  function toggleComplete() {
    const updated = completed.includes(lessonId)
      ? completed.filter((item) => item !== lessonId)
      : [...completed, lessonId];
    writeProgress(updated);
  }

  return (
    <div className="book-shell">
      <CourseRail lessonId={lessonId} open={railOpen} onClose={() => setRailOpen(false)} />

      <header className="book-topbar">
        <button type="button" aria-label="Открыть содержание" onClick={() => setRailOpen(true)}>
          <span />
          <span />
          <span />
        </button>
        <Link href="/" className="book-topbar__mark" aria-label="К оглавлению">
          К
        </Link>
        <p>
          {lesson.grade} класс <span>/</span> {courseModule.shortTitle}
        </p>
        <div className="book-topbar__progress" aria-label={`Урок ${gradeIndex + 1} из ${gradeLessons.length}`}>
          <span>{gradeIndex + 1} / {gradeLessons.length}</span>
          <i>
            <b style={{ width: `${((gradeIndex + 1) / gradeLessons.length) * 100}%` }} />
          </i>
        </div>
        <nav aria-label="Соседние уроки">
          {previous?.grade === lesson.grade ? (
            <Link href={lessonHref(previous.id)} aria-label={`Предыдущий урок: ${previous.title}`}>
              ←
            </Link>
          ) : (
            <span />
          )}
          {next?.grade === lesson.grade ? (
            <Link href={lessonHref(next.id)} aria-label={`Следующий урок: ${next.title}`}>
              →
            </Link>
          ) : (
            <span />
          )}
        </nav>
      </header>

      <main className="book-page">
        <div className="article-head-grid">
          <header className="article-head">
            <nav className="breadcrumbs" aria-label="Путь к уроку">
              <Link href="/">Оглавление</Link>
              <span>›</span>
              <span>{courseModule.number}</span>
              <span>›</span>
              <span>Урок {lesson.id}</span>
            </nav>
            <p className="article-kicker">
              Урок {lesson.id} · {lesson.mode} · чтение + практика
            </p>
            <h1>{lesson.title}</h1>
            <p className="article-question">{lesson.question}</p>
          </header>

          <aside className="article-toc" aria-label="На этой странице">
            <strong>На этой странице</strong>
            <ol>
              {toc.map((item) => (
                <li key={item.id}>
                  <a href={`#${item.id}`}>{item.title}</a>
                </li>
              ))}
            </ol>
          </aside>
        </div>

        <article className="article-body">
          <ArticleMarkdown markdown={markdown} />
        </article>

        {connections.outgoing.length || connections.incoming.length ? (
          <aside className="article-connections" aria-label="Связи с другими уроками">
            <header>
              <span>Карта идей</span>
              <h2>Связи этой статьи</h2>
            </header>
            <div>
              {connections.outgoing.length ? (
                <section>
                  <h3>Продолжить мысль</h3>
                  <ul>
                    {connections.outgoing.map((id) => {
                      const connected = lessonById.get(id);
                      return connected ? (
                        <li key={id}>
                          <Link href={lessonHref(id)}>
                            <span>{id}</span>
                            {connected.title}
                          </Link>
                        </li>
                      ) : null;
                    })}
                  </ul>
                </section>
              ) : null}
              {connections.incoming.length ? (
                <section>
                  <h3>Сюда ссылаются</h3>
                  <ul>
                    {connections.incoming.map((id) => {
                      const connected = lessonById.get(id);
                      return connected ? (
                        <li key={id}>
                          <Link href={lessonHref(id)}>
                            <span>{id}</span>
                            {connected.title}
                          </Link>
                        </li>
                      ) : null;
                    })}
                  </ul>
                </section>
              ) : null}
            </div>
          </aside>
        ) : null}

        <section className="article-progress">
          <p>
            Пройдено в {lesson.grade} классе: {completedInGrade} из {gradeLessons.length}
          </p>
          <button type="button" className={complete ? "is-complete" : ""} onClick={toggleComplete}>
            {complete ? "Урок отмечен" : "Отметить урок"}
          </button>
        </section>

        <nav className="article-pagination" aria-label="Следующий и предыдущий урок">
          {previous ? (
            <Link href={lessonHref(previous.id)}>
              <span>← Урок {previous.id}</span>
              <strong>{previous.title}</strong>
            </Link>
          ) : (
            <span />
          )}
          {next ? (
            <Link href={lessonHref(next.id)}>
              <span>Урок {next.id} →</span>
              <strong>{next.title}</strong>
            </Link>
          ) : (
            <Link href="/">
              <span>Конец курса</span>
              <strong>Вернуться к оглавлению</strong>
            </Link>
          )}
        </nav>
      </main>
    </div>
  );
}
