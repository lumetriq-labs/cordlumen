// GitHub Pages serves every file with `cache-control: max-age=600` and offers no way to change it,
// so a stale index.html can keep loading an old build. Compare against an uncached version.json and,
// if it differs, navigate to a new URL (the query string makes it a cache miss).
export async function ensureLatest(): Promise<void> {
  if (import.meta.env.DEV) return;
  try {
    const res = await fetch(`./version.json?t=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) return;
    const { build } = (await res.json()) as { build?: unknown };
    if (typeof build !== "string" || build === __BUILD_ID__) return;
    const url = new URL(location.href);
    if (url.searchParams.get("v") === build) return;
    url.searchParams.set("v", build);
    location.replace(url);
  } catch {
    // Offline or blocked: keep running the loaded build.
  }
}

export function watchForUpdates(): void {
  void ensureLatest();
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") void ensureLatest();
  });
}
