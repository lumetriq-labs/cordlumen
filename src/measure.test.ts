import { describe, expect, it } from "vitest";
import { verdict } from "./draw";
import { POST_H, SPAN, homography, measure, targetY } from "./measure";

// A perspective view of the net plane: near post on the left, far post on the right.
const camera = homography(
  [[0, 0], [0, POST_H], [SPAN, 0], [SPAN, POST_H]],
  [[217, 2232], [200, 1203], [3888, 1309], [3900, 932]],
);

function bandTopFor(heightAt: (X: number) => number) {
  const samples = Array.from({ length: 2001 }, (_, i) => camera((SPAN * i) / 2000, heightAt((SPAN * i) / 2000)));
  return (u: number) => {
    const k = samples.findIndex(([su]) => su >= u);
    if (k <= 0) return samples[Math.max(0, k)]![1];
    const [u0, v0] = samples[k - 1]!;
    const [u1, v1] = samples[k]!;
    return v0 + ((v1 - v0) * (u - u0)) / (u1 - u0);
  };
}

const post = (X: number) => ({
  base: { x: camera(X, 0)[0], y: camera(X, 0)[1] },
  top: { x: camera(X, POST_H)[0], y: camera(X, POST_H)[1] },
});

describe("measure", () => {
  it("recovers the centre height of a regulation net", () => {
    const m = measure(bandTopFor(targetY), post(0), post(SPAN));
    expect(m.center).toBeCloseTo(0.914, 3);
  });

  it("recovers a sagging net", () => {
    const low = (X: number) => targetY(X) - 0.03 * (1 - ((X - SPAN / 2) / (SPAN / 2)) ** 2);
    const m = measure(bandTopFor(low), post(0), post(SPAN));
    expect(m.center).toBeCloseTo(0.884, 3);
  });
});

describe("verdict", () => {
  it("is ok within 1 cm", () => {
    expect(verdict(0.92).ok).toBe(true);
    expect(verdict(0.905).ok).toBe(true);
  });

  it("says low or high outside 1 cm", () => {
    expect(verdict(0.9).label).toBe("低すぎる");
    expect(verdict(0.93).label).toBe("高すぎる");
  });
});
