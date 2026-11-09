import { detectPosts, type Detector } from "./keypoints";
import { log } from "./log";
import type { Pt } from "./band";
import { measure, SPAN, type Post, type Project } from "./measure";
import { findBand } from "./netband";

// Below this heatmap peak a point is treated as not found. Real net photos score >= 0.74 so far, a screenshot
// without a net scored 0.0-0.59; provisional until more negatives are checked.
const MIN_SCORE = 0.3;

export interface Timings {
  band: number;
  posts: number;
}

export type Result =
  | {
      ok: true;
      center: number;
      toImg: Project;
      posts: [Post, Post];
      scores: number[];
      /** Detected band top edge, sampled across the band, in image pixels. */
      bandLine: Pt[];
      /** Where the band top meets the net's centre line; its height is the result. */
      centerPt: Pt;
      timings: Timings;
    }
  | { ok: false; reason: string; timings: Partial<Timings> };

export async function analyze(image: ImageData, det: Detector, onStep: (label: string) => void = () => {}): Promise<Result> {
  onStep("ポストを探しています…");
  let t = performance.now();
  const { posts, scores } = await detectPosts(det, image);
  const timings: Partial<Timings> = { posts: performance.now() - t };
  log("ポスト", JSON.stringify(posts), "scores", scores.map((s) => s.toFixed(2)).join(" "));
  const leftOk = Math.min(scores[0]!, scores[1]!) >= MIN_SCORE;
  const rightOk = Math.min(scores[2]!, scores[3]!) >= MIN_SCORE;
  if (!leftOk || !rightOk) {
    const which = !leftOk && !rightOk ? "両方" : !leftOk ? "左" : "右";
    return { ok: false, reason: `ポスト（${which}）が見つかりませんでした`, timings };
  }

  onStep("白帯を検出中…");
  t = performance.now();
  const found = findBand(image, posts[0], posts[1]);
  timings.band = performance.now() - t;
  log("白帯", found ? `検出 x=${Math.round(found.u0)}..${Math.round(found.u1)}` : "なし");
  if (!found) return { ok: false, reason: "ネットの白帯が見つかりませんでした", timings };
  const { bandTop, u0: x0, u1: x1 } = found;
  const m = measure(bandTop, posts[0], posts[1]);
  const bandLine = Array.from({ length: 129 }, (_, i) => {
    const x = x0 + ((x1 - x0) * i) / 128;
    return { x, y: bandTop(x) };
  });
  const [cx, cy] = m.toImg(SPAN / 2, m.center);
  return {
    ok: true,
    center: m.center,
    toImg: m.toImg,
    posts,
    scores,
    bandLine,
    centerPt: { x: cx, y: cy },
    timings: timings as Timings,
  };
}
