"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { lessonHref, lessons, lessonsForModule, modules } from "../../lib/course";
import { useProgress } from "./useProgress";

export function CourseHome() {
  const [grade, setGrade] = useState<10 | 11>(10);
  const completed = useProgress();

  const gradeModules = useMemo(
    () => modules.filter((module) => module.id.startsWith(`g${grade}-`)),
    [grade],
  );
  const gradeLessons = lessons.filter((lesson) => lesson.grade === grade);
  const completedCount = gradeLessons.filter((lesson) => completed.includes(lesson.id)).length;
  const firstLesson = grade === 10 ? "01" : "41";

  return (
    <div className="catalog-shell">
      <header className="catalog-topbar">
        <Link href="/" className="catalog-brand">
          <span>К</span>
          <strong>Контур</strong>
        </Link>
        <nav aria-label="Разделы курса">
          <button type="button" className={grade === 10 ? "is-active" : ""} onClick={() => setGrade(10)}>
            10 класс
          </button>
          <button type="button" className={grade === 11 ? "is-active" : ""} onClick={() => setGrade(11)}>
            11 класс
          </button>
          <Link href={lessonHref(firstLesson)}>Начать чтение</Link>
        </nav>
      </header>

      <main className="catalog-page">
        <section className="catalog-intro" id="about">
          <p className="catalog-kicker">Учебник для 10–11 классов</p>
          <h1>Информатика и искусственный интеллект</h1>
          <p className="catalog-lead">
            Два учебных года, 90 уроков. Каждый урок устроен как самостоятельная
            статья с формулами, определениями, разобранными примерами,
            упражнениями и живой математической иллюстрацией.
          </p>
          <p>
            Курс начинается с данных и границы между заданным алгоритмом и
            обучаемой моделью. Дальше идут оптимизация, нейронные сети, линейная
            алгебра, статистика, обучение с подкреплением и генеративные модели.
            Математика вводится в том месте, где она нужна для задачи.
          </p>
          <div className="catalog-switch" aria-label="Выбор класса">
            <button type="button" className={grade === 10 ? "is-active" : ""} onClick={() => setGrade(10)}>
              <strong>10 класс</strong>
              <span>40 уроков</span>
            </button>
            <button type="button" className={grade === 11 ? "is-active" : ""} onClick={() => setGrade(11)}>
              <strong>11 класс</strong>
              <span>50 уроков</span>
            </button>
          </div>
          <p className="catalog-progress">
            На этом устройстве отмечено: {completedCount} из {gradeLessons.length}.
          </p>
        </section>

        <section className="catalog-course" id="course">
          <header>
            <h2>Программа {grade} класса</h2>
            <p>
              {grade === 10
                ? "12 уроков о задачах и данных, 14 о нейронах и оптимизации, 14 о зрении и линейной алгебре."
                : "16 уроков о статистике, 8 об основах машинного обучения, 8 об обучении с подкреплением, 18 о генеративных моделях."}
            </p>
          </header>

          {gradeModules.map((module) => {
            const moduleLessons = lessonsForModule(module.id);
            return (
              <section className="catalog-module" id={module.id} key={module.id}>
                <header>
                  <span>{module.number}</span>
                  <div>
                    <h3>{module.title}</h3>
                    <p>{module.description}</p>
                  </div>
                </header>
                <ol start={Number(moduleLessons[0]?.id ?? 1)}>
                  {moduleLessons.map((lesson) => (
                    <li key={lesson.id}>
                      <Link href={lessonHref(lesson.id)}>
                        <span>{lesson.id}</span>
                        <strong>{lesson.title}</strong>
                        <small>{lesson.mode}</small>
                        <b aria-label={completed.includes(lesson.id) ? "Урок отмечен" : undefined}>
                          {completed.includes(lesson.id) ? "✓" : "→"}
                        </b>
                      </Link>
                    </li>
                  ))}
                </ol>
              </section>
            );
          })}
        </section>

        <section className="catalog-reading" id="reading">
          <h2>Как читать урок</h2>
          <p>
            Основная колонка содержит связный конспект. Определения, утверждения
            и упражнения остаются внутри текста. Короткие возражения, исторические
            справки и вопросы автора вынесены на поля.
          </p>
          <p>
            Интерактивная фигура появляется после математической постановки.
            Перед ней объясняется модель, после неё разбираются наблюдения и
            ограничения. Статью можно прочитать без запуска фигуры.
          </p>
          <aside>
            <span>С полей</span>
            <strong>Хороший вопрос к любой модели</strong>
            <p>
              Какое предположение о данных мы сделали молча? Попробуйте найти
              его до того, как оно сломается на практике.
            </p>
          </aside>
        </section>

        <footer className="catalog-footer">
          <Link href={lessonHref(firstLesson)}>
            Открыть урок {firstLesson}: {lessons.find((lesson) => lesson.id === firstLesson)?.title}
          </Link>
        </footer>
      </main>
    </div>
  );
}
