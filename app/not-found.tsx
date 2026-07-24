import Link from "next/link";

export default function NotFound() {
  return (
    <main className="not-found">
      <span>404</span>
      <h1>Такого урока пока нет</h1>
      <p>Вернись к карте: там собраны все 90 страниц двухгодичного курса.</p>
      <Link href="/">Открыть карту курса →</Link>
    </main>
  );
}
