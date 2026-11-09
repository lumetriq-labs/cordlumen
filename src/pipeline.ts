import { bandCandidates, fitBand, makeGray } from "./band";
import { detectPosts, type Detector } from "./keypoints";
import { log } from "./log";
import { measure, type Post, type Project } from "./measure";

// Below this heatmap peak a point is treated as not found. Real net photos score >= 0.74 so far, a screenshot
// without a net scored 0.0-0.59; provisional until more negatives are checked.
const MIN_SCORE = 0.3;

export interface Timings {
  band: number;
  posts: number;
}

export type Result =
  | { ok: true; center: number; toImg: Project; posts: [Post, Post]; scores: number[]; timings: Timings }
  | { ok: false; reason: string; timings: Partial<Timings> };

export async function analyze(image: ImageData, det: Detector, onStep: (label: string) => void = () => {}): Promise<Result> {
  onStep("白帯を検出中…");
  let t = performance.now();
  // Band search runs at half resolution for full-size photos; smaller images are used as they are.
  const g = makeGray(image, image.width > 2500 ? 2 : 1);
  const band = fitBand(bandCandidates(g), g);
  const timings: Partial<Timings> = { band: performance.now() - t };
  log("白帯", band ? `検出 x=${band.xL * g.factor}..${band.xR * g.factor}` : "なし");
  if (!band) return { ok: false, reason: "ネットの白帯が見つかりませんでした", timings };
  const F = g.factor;
  const bandTop = (u: number) => band.f(u / F) * F;

  onStep("ポストを探しています…");
  t = performance.now();
  const { posts, scores } = await detectPosts(det, image);
  timings.posts = performance.now() - t;
  log("ポスト", JSON.stringify(posts), "scores", scores.map((s) => s.toFixed(2)).join(" "));
  const leftOk = Math.min(scores[0]!, scores[1]!) >= MIN_SCORE;
  const rightOk = Math.min(scores[2]!, scores[3]!) >= MIN_SCORE;
  if (!leftOk || !rightOk) {
    const which = !leftOk && !rightOk ? "両方" : !leftOk ? "左" : "右";
    return { ok: false, reason: `ポスト（${which}）が見つかりませんでした`, timings };
  }
  const m = measure(bandTop, posts[0], posts[1]);
  return { ok: true, center: m.center, toImg: m.toImg, posts, scores, timings: timings as Timings };
}
