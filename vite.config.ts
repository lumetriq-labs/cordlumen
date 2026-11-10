import { defineConfig } from "vitest/config";

const buildId = process.env.GITHUB_SHA?.slice(0, 7) ?? "local";
const buildTime = new Date().toLocaleString("ja-JP", {
  timeZone: "Asia/Tokyo",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
});

export default defineConfig({
  // Relative asset URLs so the build works under the GitHub Pages sub-path (/cordlumen/).
  base: "./",
  build: {
    rollupOptions: {
      input: { main: "index.html", bench: "bench.html" },
    },
  },
  plugins: [
    {
      name: "version-json",
      generateBundle() {
        this.emitFile({ type: "asset", fileName: "version.json", source: JSON.stringify({ build: buildId }) });
      },
    },
  ],
  define: {
    __BUILD_ID__: JSON.stringify(buildId),
    __BUILD_TIME__: JSON.stringify(buildTime),
  },
  test: {
    include: ["src/**/*.test.ts"],
  },
});
