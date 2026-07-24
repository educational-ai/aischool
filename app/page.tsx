import type { Metadata } from "next";
import { CourseHome } from "./components/CourseHome";

export const metadata: Metadata = {
  title: "Контур: информатика и искусственный интеллект",
  description:
    "90 уроков по искусственному интеллекту для 10–11 классов.",
};

export default function Home() {
  return <CourseHome />;
}
