import type { Metadata } from "next";
import { CourseHome } from "./components/CourseHome";

export const metadata: Metadata = {
  title: {
    absolute: "Информатика и искусственный интеллект: учебник для 10–11 классов",
  },
  description:
    "90 уроков по искусственному интеллекту для 10–11 классов.",
};

export default function Home() {
  return <CourseHome />;
}
