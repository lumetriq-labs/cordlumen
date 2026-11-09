export interface Gray {
  w: number;
  h: number;
  L: Float32Array;
  S: Float32Array;
  factor: number;
}

export interface Pt {
  x: number;
  y: number;
}

interface Cand extends Pt {
  score: number;
}

interface Segment {
  f: (x: number) => number;
  pts: Pt[];
  xL: number;
  xR: number;
}

export interface Band {
  f: (x: number) => number;
  xL: number;
  xR: number;
  segments: number;
}

export function makeGray(image: ImageData, factor: number): Gray {
  const w = Math.floor(image.width / factor);
  const h = Math.floor(image.height / factor);
  const L = new Float32Array(w * h);
  const S = new Float32Array(w * h);
  const src = image.data;
  const W = image.width;
  const n = factor * factor;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let l = 0;
      let s = 0;
      for (let dy = 0; dy < factor; dy++) {
        let i = ((y * factor + dy) * W + x * factor) * 4;
        for (let dx = 0; dx < factor; dx++, i += 4) {
          const r = src[i]!;
          const g = src[i + 1]!;
          const b = src[i + 2]!;
          l += 0.299 * r + 0.587 * g + 0.114 * b;
          s += Math.max(r, g, b) - Math.min(r, g, b);
        }
      }
      L[y * w + x] = l / n;
      S[y * w + x] = s / n;
    }
  }
  return { w, h, L, S, factor };
}

function colMean(g: Gray, x: number, y0: number, y1: number): number {
  let sum = 0;
  let n = 0;
  for (let y = Math.max(0, y0); y < Math.min(g.h, y1); y++, n++) sum += g.L[y * g.w + x]!;
  return n ? sum / n : 0;
}

function colEnergy(g: Gray, x: number, y0: number, y1: number): number {
  let sum = 0;
  let n = 0;
  for (let y = Math.max(0, y0); y < Math.min(g.h - 1, y1); y++, n++) {
    sum += Math.abs(g.L[(y + 1) * g.w + x]! - g.L[y * g.w + x]!);
  }
  return n ? sum / n : 0;
}

function colSat(g: Gray, x: number, y0: number, y1: number): number {
  let sum = 0;
  let n = 0;
  for (let y = Math.max(0, y0); y < Math.min(g.h, y1); y++, n++) sum += g.S[y * g.w + x]!;
  return n ? sum / n : 0;
}

// Whiteness: bright and unsaturated. Holds up in evening light where the band is no brighter than the court.
function colWhite(g: Gray, x: number, y0: number, y1: number): number {
  return colMean(g, x, y0, y1) - 1.5 * colSat(g, x, y0, y1);
}

/** Candidate points on the top edge of the net band: a step up in whiteness with mesh texture below it. */
export function bandCandidates(g: Gray, { step = 2, win = 40, keep = 4 } = {}): Pt[] {
  const pts: Pt[] = [];
  for (let x = 2; x < g.w - 2; x += step) {
    const cands: Cand[] = [];
    for (let y = win + 2; y < g.h - win - 2; y++) {
      const edge = colWhite(g, x, y + 1, y + 4) - colWhite(g, x, y - 3, y);
      if (edge < 30) continue;
      if (colMean(g, x, y + 1, y + 4) < 130 || colWhite(g, x, y + 1, y + 4) < 110) continue;
      const eBelow = colEnergy(g, x, y + 3, y + 3 + win);
      const eAbove = colEnergy(g, x, y - 2 - win, y - 2);
      if (eBelow < 1.5 * eAbove + 2) continue;
      cands.push({ x, y: y + 0.5, score: edge * (eBelow - eAbove) });
    }
    cands.sort((a, b) => b.score - a.score);
    const kept: Cand[] = [];
    for (const c of cands) {
      if (kept.every((k) => Math.abs(k.y - c.y) > 8)) kept.push(c);
      if (kept.length >= keep) break;
    }
    pts.push(...kept.map(({ x, y }) => ({ x, y })));
  }
  return pts;
}

function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[p]![c]!)) p = r;
    [M[c], M[p]] = [M[p]!, M[c]!];
    const piv = M[c]!;
    if (Math.abs(piv[c]!) < 1e-12) return null;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const row = M[r]!;
      const f = row[c]! / piv[c]!;
      for (let k = c; k <= n; k++) row[k]! -= f * piv[k]!;
    }
  }
  return M.map((row, i) => row[n]! / row[i]!);
}

function polyFit(pts: Pt[], deg: number, xs: { c: number; s: number }): ((x: number) => number) | null {
  const n = deg + 1;
  const A = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const b = new Array<number>(n).fill(0);
  for (const p of pts) {
    const u = (p.x - xs.c) / xs.s;
    const pw = [1];
    for (let k = 1; k < 2 * n; k++) pw.push(pw[k - 1]! * u);
    for (let i = 0; i < n; i++) {
      b[i]! += pw[i]! * p.y;
      for (let j = 0; j < n; j++) A[i]![j]! += pw[i + j]!;
    }
  }
  const coef = solve(A, b);
  if (!coef) return null;
  return (x) => {
    const u = (x - xs.c) / xs.s;
    let v = 0;
    for (let k = n - 1; k >= 0; k--) v = v * u + coef[k]!;
    return v;
  };
}

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
}

function longestRun(inl: Pt[], gap: number): Pt[] {
  inl.sort((a, b) => a.x - b.x);
  const runs: Pt[][] = [];
  let cur: Pt[] = [];
  for (const p of inl) {
    if (cur.length && p.x - cur[cur.length - 1]!.x > gap) {
      runs.push(cur);
      cur = [];
    }
    cur.push(p);
  }
  if (cur.length) runs.push(cur);
  runs.sort((a, b) => b.length - a.length);
  return runs[0] ?? [];
}

interface FitConf {
  iters: number;
  thr: number;
  gap: number;
}

function ransacSegment(pts: Pt[], g: Gray, rand: () => number, anchor: Pt | null, { iters, thr, gap }: FitConf): Segment | null {
  const xs = { c: g.w / 2, s: g.w / 2 };
  let best: Pt[] = [];
  const need = anchor ? 2 : 3;
  for (let it = 0; it < iters && pts.length >= need; it++) {
    const sample = Array.from({ length: need }, () => pts[Math.floor(rand() * pts.length)]!);
    if (anchor) sample.push(anchor);
    const sx = sample.map((p) => p.x).sort((a, b) => a - b);
    if (sx.some((v, i) => i && v - sx[i - 1]! < g.w * 0.015)) continue;
    const f = polyFit(sample, 2, xs);
    if (!f) continue;
    let inl = pts.filter((p) => Math.abs(f(p.x) - p.y) < thr);
    if (anchor && !inl.some((p) => Math.abs(p.x - anchor.x) < gap)) continue;
    inl = longestRun(inl, gap);
    if (inl.length > best.length) best = inl;
  }
  if (best.length < 10) return null;
  const f = polyFit(anchor ? [...best, anchor, anchor, anchor] : best, 2, xs);
  if (!f) return null;
  return { f, pts: best, xL: best[0]!.x, xR: best[best.length - 1]!.x };
}

/**
 * Fit the band's top edge as a chain of quadratic segments (the centre strap puts a kink in it),
 * growing outward from the strongest segment.
 */
export function fitBand(pts: Pt[], g: Gray, { iters = 4000, thr = 2.5, step = 2 } = {}): Band | null {
  const conf: FitConf = { iters, thr, gap: g.w * 0.03 };
  const rand = rng(7);
  const dense = (s: Segment | null) => !!s && s.pts.length / ((s.xR - s.xL) / step + 1) >= 0.35;
  const first = ransacSegment(pts, g, rand, null, conf);
  if (!first) return null;
  const segs: Segment[] = [first];
  for (let round = 0; round < 4; round++) {
    const L = segs[0]!;
    const R = segs[segs.length - 1]!;
    let segL = ransacSegment(pts.filter((p) => p.x < L.xL - 2), g, rand, { x: L.xL, y: L.f(L.xL) }, conf);
    let segR = ransacSegment(pts.filter((p) => p.x > R.xR + 2), g, rand, { x: R.xR, y: R.f(R.xR) }, conf);
    if (!dense(segL)) segL = null;
    if (!dense(segR)) segR = null;
    if (segR && (!segL || segR.pts.length >= segL.pts.length)) segs.push(segR);
    else if (segL) segs.unshift(segL);
    else break;
  }
  const last = segs[segs.length - 1]!;
  return {
    f: (x) => (segs.find((s) => x <= s.xR) ?? last).f(x),
    xL: segs[0]!.xL,
    xR: last.xR,
    segments: segs.length,
  };
}
