import { describe, expect, it } from "vitest";
import { cropBox, decode, IN_H, IN_W, letterbox } from "./keypoints";

describe("letterbox", () => {
  it("fits a 4:3 photo exactly", () => {
    const lb = letterbox(4032, 3024);
    expect(lb.w).toBe(IN_W);
    expect(lb.h).toBe(IN_H);
    expect(lb.ox).toBe(0);
    expect(lb.oy).toBe(0);
  });

  it("centres a portrait photo with grey bars at the sides", () => {
    const lb = letterbox(3024, 4032);
    expect(lb.h).toBe(IN_H);
    expect(lb.w).toBe(288);
    expect(lb.ox).toBe((IN_W - 288) / 2);
  });
});

describe("decode", () => {
  it("returns the peak centre in input pixels", () => {
    const hh = 48;
    const hw = 64;
    const data = new Float32Array(hh * hw);
    data[10 * hw + 20] = 1;
    const [p] = decode(data, 1, hh, hw);
    expect(p!.x).toBeCloseTo((20 + 0.5) * 8);
    expect(p!.y).toBeCloseTo((10 + 0.5) * 8);
    expect(p!.score).toBe(1);
  });

  it("moves toward a stronger neighbour", () => {
    const hh = 48;
    const hw = 64;
    const data = new Float32Array(hh * hw);
    data[10 * hw + 20] = 1;
    data[10 * hw + 21] = 1;
    const [p] = decode(data, 1, hh, hw);
    expect(p!.x).toBeCloseTo((20.5 + 0.5) * 8);
  });

  it("reads a chosen channel at the refine stride", () => {
    const n = 48;
    const data = new Float32Array(2 * n * n);
    data[n * n + 5 * n + 7] = 1;
    const [p] = decode(data, 1, n, n, 4, [1]);
    expect(p!.x).toBeCloseTo((7 + 0.5) * 4);
    expect(p!.y).toBeCloseTo((5 + 0.5) * 4);
  });
});

describe("cropBox", () => {
  it("centres a square of half the post length on the point", () => {
    const b = cropBox({ x: 1000, y: 500 }, 400);
    expect(b).toEqual({ x0: 900, y0: 400, w: 200, h: 200 });
  });

  it("keeps a minimum size for tiny posts", () => {
    expect(cropBox({ x: 100, y: 100 }, 10).w).toBe(32);
  });
});
