import { bandCandidates, fitBand, makeGray } from "./band";
import { measure, type Post, type Project } from "./measure";
import { embed, findPost, release, type Segmenter } from "./posts";

export interface Timings {
  band: number;
  encode: number;
  posts: number;
}

export type Result =
  | { ok: true; center: number; toImg: Project; posts: [Post, Post]; timings: Timings }
  | { ok: false; reason: string; timings: Partial<Timings> };

export async function analyze(image: ImageData, seg: Segmenter, onStep: (label: string) => void = () => {}): Promise<Result> {
  onStep("白帯を検出中…");
  let t = performance.now();
  // Band search runs at half resolution for full-size photos; smaller images are used as they are.
  const g = makeGray(image, image.width > 2500 ? 2 : 1);
  const band = fitBand(bandCandidates(g), g);
  const timings: Partial<Timings> = { band: performance.now() - t };
  if (!band) return { ok: false, reason: "ネットの白帯が見つかりませんでした", timings };
  const F = g.factor;
  const bandTop = (u: number) => band.f(u / F) * F;

  onStep("画像を解析中（SlimSAM）…");
  t = performance.now();
  const e = await embed(seg, image);
  timings.encode = performance.now() - t;

  onStep("ポストを探しています…");
  t = performance.now();
  try {
    const left = await findPost(e, { x: band.xL * F, y: bandTop(band.xL * F), out: -1 });
    const right = await findPost(e, { x: band.xR * F, y: bandTop(band.xR * F), out: 1 });
    timings.posts = performance.now() - t;
    if (!left || !right) {
      const which = !left && !right ? "両方" : !left ? "左" : "右";
      return { ok: false, reason: `ポスト（${which}）が見つかりませんでした`, timings };
    }
    const m = measure(bandTop, left, right);
    return { ok: true, center: m.center, toImg: m.toImg, posts: [left, right], timings: timings as Timings };
  } finally {
    release(e);
  }
}
