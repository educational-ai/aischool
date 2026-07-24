"use client";

import { useSyncExternalStore } from "react";

const progressKey = "kontur-progress-v2";
const emptyProgress: string[] = [];
let cachedRaw: string | null | undefined;
let cachedProgress = emptyProgress;

function parseProgress(raw: string | null) {
  try {
    const value = JSON.parse(raw ?? "[]");
    return Array.isArray(value)
      ? value.filter((item): item is string => typeof item === "string")
      : emptyProgress;
  } catch {
    return emptyProgress;
  }
}

function progressSnapshot() {
  const raw = window.localStorage.getItem(progressKey);
  if (raw !== cachedRaw) {
    cachedRaw = raw;
    cachedProgress = parseProgress(raw);
  }
  return cachedProgress;
}

function serverProgressSnapshot() {
  return emptyProgress;
}

function subscribeProgress(onChange: () => void) {
  window.addEventListener("storage", onChange);
  window.addEventListener("kontur-progress", onChange);
  return () => {
    window.removeEventListener("storage", onChange);
    window.removeEventListener("kontur-progress", onChange);
  };
}

export function useProgress() {
  return useSyncExternalStore(
    subscribeProgress,
    progressSnapshot,
    serverProgressSnapshot,
  );
}

export function writeProgress(progress: string[]) {
  window.localStorage.setItem(progressKey, JSON.stringify(progress));
  cachedRaw = undefined;
  window.dispatchEvent(new Event("kontur-progress"));
}
