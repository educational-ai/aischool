import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { LessonView } from "../../components/LessonView";
import { articleConnections, articleForLesson } from "../../../lib/articles";
import { lessonById, lessons, moduleById } from "../../../lib/course";

export function generateStaticParams() {
  return lessons.map((lesson) => ({ id: lesson.id }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>;
}): Promise<Metadata> {
  const { id } = await params;
  const lesson = lessonById.get(id);
  if (!lesson) return {};
  return {
    title: `Урок ${lesson.id}. ${lesson.title}`,
    description: lesson.lead,
  };
}

export default async function LessonPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const lesson = lessonById.get(id);
  if (!lesson || !moduleById.has(lesson.moduleId)) notFound();
  return (
    <LessonView
      lessonId={id}
      markdown={articleForLesson(lesson)}
      connections={articleConnections(id)}
    />
  );
}
