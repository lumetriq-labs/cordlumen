import { bandCandidates, fitNetBand, makeGray, type Band, type CandOpts, type Gray, type NetEnd, type NetFrame, type RowRange } from "./band";
import { homography, POST_H, SPAN, type Post, type Project } from "./measure";

// Net-plane window searched for the band top (metres): a slack net can sit well below the 0.914 m centre.
const BAND_LOW = 0.55;
const BAND_HIGH = 1.15;
// Keeps the post, winder and strap hardware near each post out of the search.
const POST_MARGIN = 0.4;
const MAX_SAG = 0.15;
// The cord is fixed at the post tops even when slack; allows for the cap and post-point error.
const END_TOL = 0.08;
// Confined to the net, a band in shade or overcast light (whiteness ~100) is kept.
export const CAND_IN_AREA: CandOpts = { minMean: 100, minWhite: 70, texture: 1.2 };
// Rows above and below the search area that the candidate test reads (its texture window plus the edge window).
const READ_MARGIN = 46;
export interface FoundBand {
  band: Band;
  /** Band top row at image column u. */
  bandTop: (u: number) => number;
  /** Image columns the fitted band spans. */
  u0: number;
  u1: number;
}

function netToImage(left: Post, right: Post): Project {
  return homography(
    [[0, 0], [0, POST_H], [SPAN, 0], [SPAN, POST_H]],
    [[left.base.x, left.base.y], [left.top.x, left.top.y], [right.base.x, right.base.y], [right.top.x, right.top.y]],
  );
}

/** Image-pixel search window per column: [u, top row, bottom row], sorted by u. */
function windowColumns(toImg: Project): [number, number, number][] {
  const cols: [number, number, number][] = [];
  for (let i = 0; i <= 256; i++) {
    const X = POST_MARGIN + ((SPAN - 2 * POST_MARGIN) * i) / 256;
    const [u, vHigh] = toImg(X, BAND_HIGH);
    const vLow = toImg(X, BAND_LOW)[1];
    cols.push([u, Math.min(vHigh, vLow), Math.max(vHigh, vLow)]);
  }
  return cols.sort((a, b) => a[0] - b[0]);
}

/**
 * The part of the image to read for the band given the two posts. Full-size photos are read at half resolution:
 * full resolution was tried and picked the wrong line more often, while the half-resolution fit is already
 * within ~0.6 px when it finds the band.
 */
export function readArea(image: ImageData, left: Post, right: Post): Gray {
  const cols = windowColumns(netToImage(left, right));
  const x0 = cols[0]![0];
  const x1 = cols[cols.length - 1]![0];
  const y0 = Math.min(...cols.map((c) => c[1]));
  const y1 = Math.max(...cols.map((c) => c[2]));
  const factor = image.width > 2500 ? 2 : 1;
  const m = READ_MARGIN * factor;
  // On the same pixel grid and column sampling (every 2nd column) as a whole-image read, so the result does not
  // depend on where the crop starts.
  const snap = (v: number) => Math.floor(v / (2 * factor)) * 2 * factor;
  return makeGray(image, factor, { x0: snap(x0 - 2 * factor), y0: snap(y0 - m), x1: x1 + 2 * factor, y1: y1 + m });
}

/** Where the band can be given the two posts, in the coordinates of `g`. */
export function searchArea(g: Gray, left: Post, right: Post): { rows: RowRange; frame: NetFrame } {
  const F = g.factor;
  const toImg = netToImage(left, right);
  const gx = (u: number) => (u - g.ox) / F;
  const gy = (v: number) => (v - g.oy) / F;
  const cols = windowColumns(toImg).map(([u, a, b]) => [gx(u), gy(a), gy(b)] as const);
  const rows: RowRange = (x) => {
    const i = cols.findIndex((c) => c[0] >= x);
    if (i <= 0) return null;
    const [x0, a0, b0] = cols[i - 1]!;
    const [x1, a1, b1] = cols[i]!;
    const k = (x - x0) / (x1 - x0 || 1);
    return [a0 + (a1 - a0) * k, b0 + (b1 - b0) * k];
  };
  const pxPerM = (X: number) => Math.abs(toImg(X, 1)[1] - toImg(X, 0)[1]) / F;
  const end = (X: number): NetEnd => {
    const [u, v] = toImg(X, POST_H);
    return { x: gx(u), y: gy(v), tol: END_TOL * pxPerM(X) };
  };
  const ends = [end(0), end(SPAN)].sort((a, b) => a.x - b.x) as [NetEnd, NetEnd];
  const frame = { xC: gx(toImg(SPAN / 2, 0.9)[0]), ends, maxSag: MAX_SAG * pxPerM(SPAN / 2) };
  return { rows, frame };
}

/** Band top in image pixels, searched only where the net can be given the two posts. */
export function findBand(image: ImageData, left: Post, right: Post): FoundBand | null {
  const g = readArea(image, left, right);
  const { rows, frame } = searchArea(g, left, right);
  const band = fitNetBand(bandCandidates(g, { rows, ...CAND_IN_AREA }), frame);
  if (!band) return null;
  const F = g.factor;
  return {
    band,
    bandTop: (u) => g.oy + band.f((u - g.ox) / F) * F,
    u0: g.ox + band.xL * F,
    u1: g.ox + band.xR * F,
  };
}
