import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

const lessonsRoot = new URL("../content/lessons/", import.meta.url);
const widgetsRoot = new URL("../public/interactive/widgets/", import.meta.url);

async function lessonEntries() {
  const names = (await readdir(lessonsRoot))
    .filter((name) => /^\d{2}\.md$/.test(name))
    .sort();
  return Promise.all(
    names.map(async (name) => ({
      name,
      source: await readFile(new URL(name, lessonsRoot), "utf8"),
    })),
  );
}

test("курс содержит 40 уроков 10 класса и 50 уроков 11 класса", async () => {
  const lessons = await lessonEntries();
  assert.deepEqual(
    lessons.map(({ name }) => name),
    Array.from({ length: 90 }, (_, index) => `${String(index + 1).padStart(2, "0")}.md`),
  );

  const { lessons: courseLessons } = await import("../lib/course.ts");
  assert.equal(courseLessons.length, 90);
  assert.equal(courseLessons.filter((lesson) => lesson.grade === 10).length, 40);
  assert.equal(courseLessons.filter((lesson) => lesson.grade === 11).length, 50);
});

test("каждый конспект имеет большую статью, поля, рисунки, математику, интерактив и задачи", async () => {
  for (const { name, source } of await lessonEntries()) {
    const headings = [...source.matchAll(/^## (.+)$/gm)].map((match) => match[1].trim());
    const words = source.match(/[\p{L}\p{N}]+/gu) ?? [];
    const articleSource = source.split(/^## Задачи$/m)[0];
    const articleWords = articleSource.match(/[\p{L}\p{N}]+/gu) ?? [];

    assert.ok(headings.length >= 6, `${name}: слишком мало разделов`);
    assert.equal(new Set(headings).size, headings.length, `${name}: повторяется заголовок`);
    assert.ok(words.length >= 1500, `${name}: конспект короче редакционного минимума`);
    assert.ok(
      articleWords.length >= 1250,
      `${name}: основной текст без домашней работы слишком короток`,
    );
    assert.ok((source.match(/(^|\n):::sidenote\{/gm) ?? []).length >= 4, `${name}: мало боковых заметок`);
    assert.ok((source.match(/(^|\n):::figure\{/gm) ?? []).length >= 3, `${name}: мало статических рисунков`);
    assert.match(source, /(^|\n):::widget\{/m, `${name}: нет интерактива`);
    assert.match(source, /\$[^$\n]+\$|\$\$[\s\S]+?\$\$/, `${name}: нет математики`);
    assert.match(source, /^## Задачи$/m, `${name}: нет листка задач`);
    assert.ok((source.match(/(^|\n):::problem\{points="\d+"\}/gm) ?? []).length >= 5, `${name}: мало полных задач`);
    assert.doesNotMatch(source, /^### (Просто|Средне|Сложно)$/m, `${name}: остались уровни сложности`);
    const internalTargets = [
      ...source.matchAll(/\]\(\/lesson\/(\d{2})(?:#[^)]+)?\)/g),
    ].map((match) => match[1]);
    assert.ok(
      new Set(internalTargets).size >= 4,
      `${name}: нужно не меньше четырёх разных внутренних связей`,
    );

    const problems = [...source.matchAll(/:::problem\{points="\d+"\}\n([\s\S]*?)\n:::/g)];
    for (const [, body] of problems) {
      const problemWords = body.match(/[\p{L}\p{N}]+/gu) ?? [];
      assert.ok(
        problemWords.length >= 20,
        `${name}: условие задачи слишком короткое для самодостаточной постановки`,
      );
      assert.doesNotMatch(
        body,
        /\b(?:из урока|из виджета|в виджете|график выше|рисунок выше|формул[ае] выше|лаборатории урока|предыдущ(?:ей|ую) задач(?:е|у))\b/iu,
        `${name}: задача ссылается на несамодостаточный контекст`,
      );
      assert.doesNotMatch(
        body,
        /^\s*(?:для|используя|возьмите)\s+(?:того|той|тех|этого|этой)\s+же\b/iu,
        `${name}: задача зависит от объекта из предыдущего условия`,
      );
    }
  }
});

test("сеть обратных ссылок охватывает каждую статью", async () => {
  const lessons = await lessonEntries();
  const incoming = new Map(lessons.map(({ name }) => [name.slice(0, 2), new Set()]));

  for (const { name, source } of lessons) {
    const sourceId = name.slice(0, 2);
    for (const match of source.matchAll(/\]\(\/lesson\/(\d{2})(?:#[^)]+)?\)/g)) {
      assert.ok(incoming.has(match[1]), `${name}: ссылка ведёт на неизвестный урок ${match[1]}`);
      if (match[1] !== sourceId) incoming.get(match[1]).add(sourceId);
    }
  }

  for (const [id, sources] of incoming) {
    assert.ok(sources.size >= 1, `${id}.md: на статью не ссылается ни один другой урок`);
  }
});

test("редакционные рисунки существуют, подписаны и не дублируют друг друга", async () => {
  const hashes = new Map();

  for (const { name, source } of await lessonEntries()) {
    const lessonId = name.slice(0, 2);
    const figures = [...source.matchAll(/:::figure\{([^}]+)\}/g)];
    for (const [, attributes] of figures) {
      const src = attributes.match(/\bsrc="([^"]+)"/)?.[1];
      const id = attributes.match(/\bid="([^"]+)"/)?.[1];
      const title = attributes.match(/\btitle="([^"]+)"/)?.[1];
      const alt = attributes.match(/\balt="([^"]+)"/)?.[1];

      assert.ok(
        src?.startsWith(`/figures/lessons/${lessonId}/`),
        `${name}: рисунок лежит не в папке своего урока`,
      );
      assert.match(id ?? "", new RegExp(`^fig-${lessonId}-\\d+$`), `${name}: неверный id рисунка`);
      assert.ok((title?.length ?? 0) >= 24, `${name}: подпись рисунка слишком короткая`);
      assert.ok((alt?.length ?? 0) >= 36, `${name}: alt рисунка не объясняет изображение`);

      const bytes = await readFile(new URL(`../public${src}`, import.meta.url));
      assert.ok(bytes.length >= 800, `${name}: рисунок ${src} подозрительно пуст`);
      const hash = createHash("sha256").update(bytes).digest("hex");
      const previous = hashes.get(hash);
      assert.equal(previous, undefined, `${name}: рисунок ${src} дублирует ${previous}`);
      hashes.set(hash, src);
    }
  }
});

test("карта статей полна, а каждый заявленный интерактив существует", async () => {
  const lessons = await lessonEntries();
  const articleMap = await readFile(new URL("../lib/articles.ts", import.meta.url), "utf8");
    const widgetNames = new Set(await readdir(widgetsRoot));
    const bundledNames = new Set([
      "model-residual-lab",
      "dataset-forensics",
      "turing-jury",
      "discipline-layers",
      "learning-signals",
      "classifier-studio",
      "regression-workbench",
      "clustering-lens",
      "label-budget-game",
      "galileo-lab",
      "dataset-passport-audit",
    ]);

  assert.equal((articleMap.match(/^import lesson\d{2} /gm) ?? []).length, 90);
  assert.equal((articleMap.match(/^\s*\["\d{2}", lesson\d{2}\],$/gm) ?? []).length, 90);
  assert.doesNotMatch(articleMap, /fallback/i);

  for (const { name, source } of lessons) {
    const widgets = [...source.matchAll(/:::widget\{[^}]*name="([^"]+)"/g)];
    assert.ok(widgets.length >= 1, `${name}: нужен хотя бы один основной интерактив`);
    for (const widget of widgets) {
      assert.ok(
        widgetNames.has(`${widget[1]}.js`) || bundledNames.has(widget[1]),
        `${name}: скрипт интерактива ${widget[1]} не найден`,
      );
    }
  }
});

test("каждый урок dispatchится в собственный интерактивный builder", async () => {
  const expectedBundles = [
    ["g10-neuron.js", 13, 26],
    ["g10-vision.js", 27, 40],
    ["g11-stats.js", 41, 56],
    ["g11-ml.js", 57, 64],
    ["g11-rl.js", 65, 72],
    ["g11-generative.js", 73, 90],
  ];

  for (const [file, first, last] of expectedBundles) {
    const source = await readFile(new URL(file, widgetsRoot), "utf8");
    const builderBlock = source.match(/var builders = \{([\s\S]*?)\n\s*\};/)?.[1] ?? "";
    const mappings = [...builderBlock.matchAll(/"(\d{2})":\s*(build\w+)/g)];
    const expectedIds = Array.from(
      { length: last - first + 1 },
      (_, index) => String(first + index).padStart(2, "0"),
    );

    assert.deepEqual(mappings.map((match) => match[1]), expectedIds, `${file}: неполный dispatch`);
    assert.equal(
      new Set(mappings.map((match) => match[2])).size,
      expectedIds.length,
      `${file}: несколько уроков используют один builder`,
    );
  }

  const dataBundle = await readFile(new URL("g10-data.js", widgetsRoot), "utf8");
  const dataBuilderBlock = dataBundle.match(/var builders = \{([\s\S]*?)\n\s*\};/)?.[1] ?? "";
  const dataMappings = [...dataBuilderBlock.matchAll(/"([^"]+)":\s*(build\w+)/g)];
  assert.equal(dataMappings.length, 11);
  assert.equal(new Set(dataMappings.map((match) => match[1])).size, 11);
  assert.equal(new Set(dataMappings.map((match) => match[2])).size, 11);
});

test("в пользовательской части нет следов стартового шаблона", async () => {
  const [packageJson, readme] = await Promise.all([
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../README.md", import.meta.url), "utf8"),
  ]);
  assert.match(packageJson, /"name": "kontur-ai-textbook"/);
  for (const obsoleteName of ["vinext" + "-starter", "site-creator-" + "vinext" + "-starter"]) {
    assert.doesNotMatch(`${packageJson}\n${readme}`, new RegExp(obsoleteName, "i"));
  }
});
