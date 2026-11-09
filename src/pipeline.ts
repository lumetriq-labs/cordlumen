import { bandCandidates, fitBand, makeGray } from "./band";
import { measure, type Post, type Project } from "./measure";
import { embed, findPost, type Segmenter } from "./posts";

export interface Timings {
  band: number;
  encode: number;
  posts: number;
}

export type Result =
  | { ok: true; center: number; toImg: Project; posts: [Post, Post]; timings: Timings }
  | { ok: false; reason: string; timings: Partial<Timings> };

export async function analyze(image: ImageData, seg: Segmenter): Promise<Result> {
  let t = performance.now();
  const g = makeGray(image, 2);
  const band = fitBand(bandCandidates(g), g);
  const timings: Partial<Timings> = { band: performance.now() - t };
  if (!band) return { ok: false, reason: "ネットの白帯が見つかりませんでした", timings };
  const F = g.factor;
  const bandTop = (u: number) => band.f(u / F) * F;

  t = performance.now();
  const e = await embed(seg, image);
  timings.encode = performance.now() - t;

  t = performance.now();
  const left = await findPost(e, { x: band.xL * F, y: bandTop(band.xL * F), out: -1 });
  const right = await findPost(e, { x: band.xR * F, y: bandTop(band.xR * F), out: 1 });
  timings.posts = performance.now() - t;
  if (!left || !right) {
    const which = !left && !right ? "両方" : !left ? "左" : "右";
    return { ok: false, reason: `ポスト（${which}）が見つかりませんでした`, timings };
  }

  const m = measure(bandTop, left, right);
  return { ok: true, center: m.center, toImg: m.toImg, posts: [left, right], timings: timings as Timings };
}
