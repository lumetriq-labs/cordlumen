import type { InferenceSession, Tensor } from "onnxruntime-web/wasm";
import { log } from "./log";
import type { Post } from "./measure";
import { resizeRegion } from "./resample";

// MobileNetV3-small + heatmap head trained on real photos and Blender renders: heatmaps for left top, left base,
// right top, right base (left/right by image x). Same file name across deploys, so tie the URL to the build.
const MODEL_URL = `./models/post_kp.onnx?v=${__BUILD_ID__}`;
// Stage 2: a square crop around each stage-1 point (side half the post length) gives top and base heatmaps at
// stride 4; the channel for the crop's own point is used. Trained by custom/train2.py on crops.py crops (training scripts kept outside this repo).
const REFINE_URL = `./models/post_refine.onnx?v=${__BUILD_ID__}`;
const CROP = 192;
const CROP_STRIDE = 4;
const CROP_SIDE = 0.5;

// Must match training (custom/train.py): letterbox into IN_W x IN_H on grey, ImageNet normalisation.
export const IN_W = 512;
export const IN_H = 384;
const STRIDE = 8;
const PAD = 114;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

type Ort = typeof import("onnxruntime-web/wasm");

export interface Detector {
  ort: Ort;
  session: InferenceSession;
  refine: InferenceSession;
}

export interface Letterbox {
  s: number;
  ox: number;
  oy: number;
  w: number;
  h: number;
}

export function letterbox(width: number, height: number): Letterbox {
  const s = Math.min(IN_W / width, IN_H / height);
  const w = Math.round(width * s);
  const h = Math.round(height * s);
  return { s, ox: (IN_W - w) / 2, oy: (IN_H - h) / 2, w, h };
}

export async function loadDetector(): Promise<Detector> {
  const ort = await import("onnxruntime-web/wasm");
  // The plain WebAssembly build; the asyncify/JSEP builds reportedly grow memory on iOS Safari (microsoft/onnxruntime#26827).
  const prefix = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
  ort.env.wasm.wasmPaths = { mjs: `${prefix}ort-wasm-simd-threaded.mjs`, wasm: `${prefix}ort-wasm-simd-threaded.wasm` };
  log("ポスト検出モデル読み込み開始", MODEL_URL, REFINE_URL);
  const open = (u: string) => ort.InferenceSession.create(u, { executionProviders: ["wasm"] });
  const [session, refine] = await Promise.all([open(MODEL_URL), open(REFINE_URL)]);
  log("ポスト検出モデル読み込み完了");
  return { ort, session, refine };
}

/** RGB bytes (w x h) into a CHW float canvas (cw x ch, starting at `offset`) with the top-left at (x, y). */
function normalise(rgb: Uint8Array, w: number, h: number, out: Float32Array, offset: number, cw: number, ch: number, x = 0, y = 0) {
  const n = cw * ch;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const i = (r * w + c) * 3;
      const o = offset + (r + y) * cw + c + x;
      for (let k = 0; k < 3; k++) out[o + k * n] = (rgb[i + k]! / 255 - MEAN[k]!) / STD[k]!;
    }
  }
}

function toInput(ort: Ort, image: ImageData, lb: Letterbox) {
  const n = IN_W * IN_H;
  const out = new Float32Array(3 * n);
  for (let k = 0; k < 3; k++) out.fill((PAD / 255 - MEAN[k]!) / STD[k]!, k * n, (k + 1) * n);
  const rgb = resizeRegion(image, 0, 0, image.width, image.height, lb.w, lb.h, PAD);
  normalise(rgb, lb.w, lb.h, out, 0, IN_W, IN_H, Math.round(lb.ox), Math.round(lb.oy));
  return new ort.Tensor("float32", out, [1, 3, IN_H, IN_W]);
}

export interface CropBox {
  x0: number;
  y0: number;
  /** Width and height in whole pixels; they can differ by one from rounding, as in crops.py. */
  w: number;
  h: number;
}

/** Crop square for a point (as crops.py): centred on it, side CROP_SIDE x the post length, whole pixels. */
export function cropBox(p: { x: number; y: number }, postLength: number): CropBox {
  const side = Math.max(32, CROP_SIDE * postLength);
  const x0 = Math.round(p.x - side / 2);
  const y0 = Math.round(p.y - side / 2);
  return { x0, y0, w: Math.round(p.x - side / 2 + side) - x0, h: Math.round(p.y - side / 2 + side) - y0 };
}

/** Crops (grey outside the photo) for the four points, as one [4, 3, CROP, CROP] batch. */
function cropInput(ort: Ort, image: ImageData, boxes: CropBox[]) {
  const n = CROP * CROP;
  const out = new Float32Array(boxes.length * 3 * n);
  boxes.forEach((b, i) => {
    const rgb = resizeRegion(image, b.x0, b.y0, b.w, b.h, CROP, CROP, PAD);
    normalise(rgb, CROP, CROP, out, i * 3 * n, CROP, CROP);
  });
  return new ort.Tensor("float32", out, [boxes.length, 3, CROP, CROP]);
}

export interface Peak {
  x: number;
  y: number;
  /** Heatmap peak value; the training target peaks at 1. */
  score: number;
}

/** Soft-argmax in a 5x5 window around each channel's peak, in model-input pixels (same as training's decode). */
export function decode(data: Float32Array, k: number, hh: number, hw: number, stride = STRIDE, channels?: number[]): Peak[] {
  const res: Peak[] = [];
  for (let j = 0; j < k; j++) {
    const off = (channels ? channels[j]! : j) * hh * hw;
    let best = 0;
    for (let i = 1; i < hh * hw; i++) if (data[off + i]! > data[off + best]!) best = i;
    const cy = Math.floor(best / hw);
    const cx = best % hw;
    let sx = 0;
    let sy = 0;
    let tot = 1e-6;
    for (let y = Math.max(cy - 2, 0); y < Math.min(cy + 3, hh); y++) {
      for (let x = Math.max(cx - 2, 0); x < Math.min(cx + 3, hw); x++) {
        const w = Math.max(0, data[off + y * hw + x]!);
        sx += x * w;
        sy += y * w;
        tot += w;
      }
    }
    res.push({ x: (sx / tot + 0.5) * stride, y: (sy / tot + 0.5) * stride, score: data[off + best]! });
  }
  return res;
}

export interface Detection {
  /** Stage-2 refined points. */
  posts: [Post, Post];
  /** Stage-1 points, before refinement. */
  coarse: [Post, Post];
  /** Stage-1 heatmap peaks (left top, left base, right top, right base). */
  scores: number[];
}

async function run(session: InferenceSession, input: Tensor) {
  let heat: Tensor | undefined;
  try {
    heat = (await session.run({ pixel_values: input })).heatmaps!;
    return { data: (heat.data as Float32Array).slice(), dims: heat.dims as number[] };
  } finally {
    input.dispose();
    heat?.dispose();
  }
}

type Pt = { x: number; y: number };

/** Stage 2 for points in the order left top, left base, right top, right base. */
export async function refinePoints(det: Detector, image: ImageData, p: Pt[]): Promise<Pt[]> {
  const len = [Math.hypot(p[1]!.x - p[0]!.x, p[1]!.y - p[0]!.y), Math.hypot(p[3]!.x - p[2]!.x, p[3]!.y - p[2]!.y)];
  const boxes = p.map((q, i) => cropBox(q, len[i >> 1]!));
  const h = await run(det.refine, cropInput(det.ort, image, boxes));
  const [, c, hh, hw] = h.dims;
  return boxes.map((b, i) => {
    // Crop i is a top (even i) or a base (odd i); read that channel of its own heatmaps.
    const [q] = decode(h.data.subarray(i * c! * hh! * hw!), 1, hh!, hw!, CROP_STRIDE, [i % 2]);
    // One scale for both axes, as eval_predict.py.
    const k = b.w / CROP;
    return { x: b.x0 + q!.x * k, y: b.y0 + q!.y * k };
  });
}

export async function detectPosts(det: Detector, image: ImageData): Promise<Detection> {
  const lb = letterbox(image.width, image.height);
  const h1 = await run(det.session, toInput(det.ort, image, lb));
  const [, k, hh, hw] = h1.dims;
  const peaks = decode(h1.data, k!, hh!, hw!);
  const p = peaks.map(({ x, y }) => ({ x: (x - lb.ox) / lb.s, y: (y - lb.oy) / lb.s }));
  const r = await refinePoints(det, image, p);
  return {
    posts: [{ top: r[0]!, base: r[1]! }, { top: r[2]!, base: r[3]! }],
    coarse: [{ top: p[0]!, base: p[1]! }, { top: p[2]!, base: p[3]! }],
    scores: peaks.map((q) => q.score),
  };
}
