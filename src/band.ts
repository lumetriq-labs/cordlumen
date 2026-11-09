export interface Gray {
  w: number;
  h: number;
  L: Float32Array;
  S: Float32Array;
  factor: number;
  /** Image pixel of the top-left corner (the image is u = ox + x * factor, v = oy + y * factor). */
  ox: number;
  oy: number;
}

export interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
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

/** Brightness and saturation, block-averaged by `factor`, of the whole image or of `rect` within it. */
export function makeGray(image: ImageData, factor: number, rect?: Rect): Gray {
  const ox = Math.max(0, Math.floor(rect?.x0 ?? 0));
  const oy = Math.max(0, Math.floor(rect?.y0 ?? 0));
  const w = Math.max(0, Math.floor((Math.min(image.width, Math.ceil(rect?.x1 ?? image.width)) - ox) / factor));
  const h = Math.max(0, Math.floor((Math.min(image.height, Math.ceil(rect?.y1 ?? image.height)) - oy) / factor));
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
        let i = ((oy + y * factor + dy) * W + ox + x * factor) * 4;
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
  return { w, h, L, S, factor, ox, oy };
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

/** Rows [y0, y1) to search at column x, or null to skip the column. */
export type RowRange = (x: number) => [number, number] | null;

export interface CandOpts {
  step?: number;
  win?: number;
  keep?: number;
  rows?: RowRange;
  /** Brightness and whiteness the band must reach; lower them when the search is already confined to the net. */
  minMean?: number;
  minWhite?: number;
  /** Mesh texture below must exceed `texture` times the texture above (plus 2). */
  texture?: number;
}

/** Candidate points on the top edge of the net band: a step up in whiteness with mesh texture below it. */
export function bandCandidates(g: Gray, { step = 2, win = 40, keep = 4, rows, minMean = 130, minWhite = 110, texture = 1.5 }: CandOpts = {}): Pt[] {
  const pts: Pt[] = [];
  for (let x = 2; x < g.w - 2; x += step) {
    const r: [number, number] | null = rows ? rows(x) : [0, g.h];
    if (!r) continue;
    const cands: Cand[] = [];
    for (let y = Math.max(win + 2, Math.floor(r[0])); y < Math.min(g.h - win - 2, Math.ceil(r[1])); y++) {
      const edge = colWhite(g, x, y + 1, y + 4) - colWhite(g, x, y - 3, y);
      if (edge < 30) continue;
      if (colMean(g, x, y + 1, y + 4) < minMean || colWhite(g, x, y + 1, y + 4) < minWhite) continue;
      const eBelow = colEnergy(g, x, y + 3, y + 3 + win);
      const eAbove = colEnergy(g, x, y - 2 - win, y - 2);
      if (eBelow < texture * eAbove + 2) continue;
      cands.push({ x, y: y + 0.5, score: edge * Math.max(eBelow - eAbove, 0.5) });
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

export interface NetFrame {
  /** Image column of the centre strap. */
  xC: number;
  /** Post tops (where the cord is fixed), left then right in the image, and how far the band may pass from each. */
  ends: [NetEnd, NetEnd];
  /** Largest allowed sag of either half below its chord, in pixels. */
  maxSag: number;
}

export interface NetEnd extends Pt {
  tol: number;
}

/** y = c + a t + b t², t = (x - xC) / scale. */
type Quad = [number, number, number];

/**
 * Fit the band's top edge as two quadratics (one per half, as a slack cord sags) that meet at the centre strap
 * and run out to the post tops. Each half is found on its own first; when one is hidden or the two disagree,
 * the centre comes from the better-supported half alone.
 */
export function fitNetBand(pts: Pt[], frame: NetFrame, { iters = 1500, thr = 2.5, minSide = 10 } = {}): Band | null {
  const { xC, ends, maxSag } = frame;
  const scale = Math.max(...ends.map((e) => Math.abs(e.x - xC)), 1);
  const tOf = (x: number) => (x - xC) / scale;
  const at = (q: Quad, x: number) => {
    const t = tOf(x);
    return q[0] + q[1] * t + q[2] * t * t;
  };
  const plausible = (q: Quad, side: number) => {
    const e = ends[side]!;
    const len = Math.abs(tOf(e.x));
    return (Math.abs(q[2]) * len * len) / 4 <= maxSag && Math.abs(at(q, e.x) - e.y) <= e.tol;
  };
  const lsq = (rows: { f: number[]; y: number }[]) => {
    const n = rows[0]!.f.length;
    const A = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    const b = new Array<number>(n).fill(0);
    for (const { f, y } of rows) {
      for (let i = 0; i < n; i++) {
        b[i]! += f[i]! * y;
        for (let j = 0; j < n; j++) A[i]![j]! += f[i]! * f[j]!;
      }
    }
    return solve(A, b);
  };
  const quad = (sp: Pt[]) => lsq(sp.map((p) => ({ f: [1, tOf(p.x), tOf(p.x) ** 2], y: p.y }))) as Quad | null;
  const near = (sp: Pt[], q: Quad) => sp.filter((p) => Math.abs(at(q, p.x) - p.y) < thr);

  const rand = rng(7);
  const half = (sp: Pt[], side: number): { q: Quad; inl: Pt[] } | null => {
    if (sp.length < minSide) return null;
    let best: Pt[] = [];
    for (let it = 0; it < iters; it++) {
      const sample = Array.from({ length: 3 }, () => sp[Math.floor(rand() * sp.length)]!);
      const q = quad(sample);
      if (!q || !plausible(q, side)) continue;
      const inl = near(sp, q);
      if (inl.length > best.length) best = inl;
    }
    if (best.length < minSide) return null;
    let q = quad(best);
    if (!q || !plausible(q, side)) return null;
    const inl = near(sp, q);
    const q2 = quad(inl);
    if (q2 && plausible(q2, side)) q = q2;
    return { q, inl: near(sp, q) };
  };

  const fits = [half(pts.filter((p) => p.x < xC), 0), half(pts.filter((p) => p.x >= xC), 1)];
  let qs: [Quad, Quad] | null = null;
  let used: Pt[] = [];
  if (fits[0] && fits[1]) {
    // Both halves share the centre value.
    const both = [...fits[0].inl, ...fits[1].inl];
    const sol = lsq(both.map((p) => {
      const t = tOf(p.x);
      return { f: p.x < xC ? [1, t, t * t, 0, 0] : [1, 0, 0, t, t * t], y: p.y };
    }));
    if (sol) {
      const L: Quad = [sol[0]!, sol[1]!, sol[2]!];
      const R: Quad = [sol[0]!, sol[3]!, sol[4]!];
      const inl = [...near(fits[0].inl, L), ...near(fits[1].inl, R)];
      if (plausible(L, 0) && plausible(R, 1) && inl.length >= 0.8 * both.length) {
        qs = [L, R];
        used = inl;
      }
    }
  }
  if (!qs) {
    const one = [fits[0], fits[1]].filter((f) => !!f).sort((a, b) => b!.inl.length - a!.inl.length)[0];
    if (!one) return null;
    qs = [one.q, one.q];
    used = one.inl;
  }
  const [L, R] = qs;
  const xs = used.map((p) => p.x);
  return {
    f: (x) => at(x < xC ? L : R, x),
    xL: Math.min(xC, ...xs),
    xR: Math.max(xC, ...xs),
    segments: L === R ? 1 : 2,
  };
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
