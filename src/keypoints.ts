import type { InferenceSession, Tensor } from "onnxruntime-web/wasm";
import { log } from "./log";
import type { Post } from "./measure";

// MobileNetV3-small + heatmap head trained on real photos and Blender renders: heatmaps for left top, left base,
// right top, right base (left/right by image x). Same file name across deploys, so tie the URL to the build.
const MODEL_URL = `./models/post_kp.onnx?v=${__BUILD_ID__}`;

// Must match training (.tmp_net/custom/train.py): letterbox into IN_W x IN_H on grey, ImageNet normalisation.
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
  log("ポスト検出モデル読み込み開始", MODEL_URL);
  const session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  log("ポスト検出モデル読み込み完了");
  return { ort, session };
}

function toInput(ort: Ort, image: ImageData, lb: Letterbox) {
  const src = document.createElement("canvas");
  src.width = image.width;
  src.height = image.height;
  src.getContext("2d")!.putImageData(image, 0, 0);
  const c = document.createElement("canvas");
  c.width = IN_W;
  c.height = IN_H;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = `rgb(${PAD}, ${PAD}, ${PAD})`;
  ctx.fillRect(0, 0, IN_W, IN_H);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, Math.round(lb.ox), Math.round(lb.oy), lb.w, lb.h);
  src.width = src.height = 0;
  const { data } = ctx.getImageData(0, 0, IN_W, IN_H);
  c.width = c.height = 0;
  const n = IN_W * IN_H;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < 3; ch++) out[ch * n + i] = (data[i * 4 + ch]! / 255 - MEAN[ch]!) / STD[ch]!;
  }
  return new ort.Tensor("float32", out, [1, 3, IN_H, IN_W]);
}

export interface Peak {
  x: number;
  y: number;
  /** Heatmap peak value; the training target peaks at 1. */
  score: number;
}

/** Soft-argmax in a 5x5 window around each channel's peak, in model-input pixels (same as training's decode). */
export function decode(data: Float32Array, k: number, hh: number, hw: number): Peak[] {
  const res: Peak[] = [];
  for (let j = 0; j < k; j++) {
    const off = j * hh * hw;
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
    res.push({ x: (sx / tot + 0.5) * STRIDE, y: (sy / tot + 0.5) * STRIDE, score: data[off + best]! });
  }
  return res;
}

export interface Detection {
  posts: [Post, Post];
  scores: number[];
}

export async function detectPosts(det: Detector, image: ImageData): Promise<Detection> {
  const lb = letterbox(image.width, image.height);
  const input = toInput(det.ort, image, lb);
  let heat: Tensor | undefined;
  try {
    heat = (await det.session.run({ pixel_values: input })).heatmaps!;
    const [, k, hh, hw] = heat.dims as number[];
    const peaks = decode(heat.data as Float32Array, k!, hh!, hw!);
    const p = peaks.map(({ x, y }) => ({ x: (x - lb.ox) / lb.s, y: (y - lb.oy) / lb.s }));
    return { posts: [{ top: p[0]!, base: p[1]! }, { top: p[2]!, base: p[3]! }], scores: peaks.map((q) => q.score) };
  } finally {
    input.dispose();
    heat?.dispose();
  }
}
