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
  define: {
    __BUILD_ID__: JSON.stringify(buildId),
    __BUILD_TIME__: JSON.stringify(buildTime),
  },
  test: {
    include: ["src/**/*.test.ts"],
  },
});
