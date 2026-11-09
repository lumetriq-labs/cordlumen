import type { Pt } from "./band";

// ITF Rules of Tennis, Rule 1.
export const SPAN = 12.8;
export const POST_H = 1.07;
export const CENTER_H = 0.914;

const SAG = POST_H - CENTER_H;

/** Target height of the net top along the net, X metres from the left post. */
export const targetY = (X: number) => CENTER_H + SAG * ((X - SPAN / 2) / (SPAN / 2)) ** 2;

export interface Post {
  top: Pt;
  base: Pt;
}

export type Project = (X: number, Y: number) => [number, number];

function solve8(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]!]);
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(M[k]![i]!) > Math.abs(M[p]![i]!)) p = k;
    [M[i], M[p]] = [M[p]!, M[i]!];
    const piv = M[i]!;
    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const row = M[k]!;
      const f = row[i]! / piv[i]!;
      for (let j = i; j <= n; j++) row[j]! -= f * piv[j]!;
    }
  }
  return M.map((r, i) => r[n]! / r[i]!);
}

/** Homography from 4 point pairs (net plane in metres -> image pixels). */
export function homography(src: [number, number][], dst: [number, number][]): Project {
  const A: number[][] = [];
  const b: number[] = [];
  src.forEach(([x, y], i) => {
    const [u, v] = dst[i]!;
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  });
  const h = solve8(A, b);
  return (x, y) => {
    const w = h[6]! * x + h[7]! * y + 1;
    return [(h[0]! * x + h[1]! * y + h[2]!) / w, (h[3]! * x + h[4]! * y + h[5]!) / w];
  };
}

/**
 * Net-plane height of the band top at the centre. `bandTop(u)` gives the band's top edge row at image column u.
 * The post tops are taken as the cord at POST_H.
 */
export function measure(bandTop: (u: number) => number, left: Post, right: Post) {
  const toImg = homography(
    [[0, 0], [0, POST_H], [SPAN, 0], [SPAN, POST_H]],
    [[left.base.x, left.base.y], [left.top.x, left.top.y], [right.base.x, right.base.y], [right.top.x, right.top.y]],
  );
  const heightAt = (X: number) => {
    let lo = 0.3;
    let hi = 1.4;
    for (let k = 0; k < 50; k++) {
      const mid = (lo + hi) / 2;
      const [u, v] = toImg(X, mid);
      if (v > bandTop(u)) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  };
  return { toImg, center: heightAt(SPAN / 2), heightAt };
}
