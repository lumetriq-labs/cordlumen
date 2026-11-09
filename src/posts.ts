import { AutoProcessor, RawImage, SamModel, Tensor } from "@huggingface/transformers";
import type { Pt } from "./band";
import type { Post } from "./measure";

export const MODEL_ID = "Xenova/slimsam-77-uniform";

// SAM's low-resolution mask logits: 256x256 over the 1024px padded input.
const LOW = 256;

type Device = "webgpu" | "wasm";

export interface Segmenter {
  model: SamModel;
  processor: Awaited<ReturnType<typeof AutoProcessor.from_pretrained>>;
  device: Device;
}

export interface Embedded {
  seg: Segmenter;
  emb: Record<string, Tensor>;
  reshaped: [number, number];
  W: number;
  H: number;
}

export interface BandEnd extends Pt {
  /** -1 when the post is to the left of this end, +1 when to the right. */
  out: -1 | 1;
}

export async function loadSegmenter(onProgress?: (loaded: number, total: number) => void): Promise<Segmenter> {
  const device: Device = "gpu" in navigator ? "webgpu" : "wasm";
  const files = new Map<string, { loaded: number; total: number }>();
  const progress_callback = (p: { status: string; file?: string; loaded?: number; total?: number }) => {
    if (p.status !== "progress" || !p.file) return;
    files.set(p.file, { loaded: p.loaded ?? 0, total: p.total ?? 0 });
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    onProgress?.(loaded, total);
  };
  const [model, processor] = await Promise.all([
    SamModel.from_pretrained(MODEL_ID, { dtype: "fp32", device, progress_callback }),
    AutoProcessor.from_pretrained(MODEL_ID, {}),
  ]);
  return { model: model as SamModel, processor, device };
}

export async function embed(seg: Segmenter, image: ImageData): Promise<Embedded> {
  const raw = new RawImage(image.data, image.width, image.height, 4).rgb();
  const inputs = await seg.processor(raw);
  const emb = await seg.model.get_image_embeddings(inputs);
  return { seg, emb, reshaped: inputs.reshaped_input_sizes[0], W: image.width, H: image.height };
}

interface Stats {
  y0: number;
  y1: number;
  x0: number;
  x1: number;
  fill: number;
}

function maskStats(logits: Float32Array, off: number, toOrig: (j: number, i: number) => [number, number]): Stats | null {
  let n = 0;
  let ymin = Infinity;
  let ymax = -Infinity;
  let xmin = Infinity;
  let xmax = -Infinity;
  for (let i = 0; i < LOW; i++) {
    for (let j = 0; j < LOW; j++) {
      if (logits[off + i * LOW + j]! <= 0) continue;
      n++;
      if (i < ymin) ymin = i;
      if (i > ymax) ymax = i;
      if (j < xmin) xmin = j;
      if (j > xmax) xmax = j;
    }
  }
  if (!n) return null;
  const [x0, y0] = toOrig(xmin, ymin);
  const [x1, y1] = toOrig(xmax + 1, ymax + 1);
  return { x0, y0, x1, y1, fill: n / ((xmax - xmin + 1) * (ymax - ymin + 1)) };
}

function bilinear(logits: Float32Array, off: number, u: number, v: number): number {
  const j = Math.max(0, Math.min(LOW - 1.001, u));
  const i = Math.max(0, Math.min(LOW - 1.001, v));
  const j0 = Math.floor(j);
  const i0 = Math.floor(i);
  const a = j - j0;
  const b = i - i0;
  const p = (ii: number, jj: number) => logits[off + ii * LOW + jj]!;
  return (1 - a) * (1 - b) * p(i0, j0) + a * (1 - b) * p(i0, j0 + 1) + (1 - a) * b * p(i0 + 1, j0) + a * b * p(i0 + 1, j0 + 1);
}

/**
 * Find the net post beyond one end of the band: prompt SAM with a grid of points outside the end,
 * keep the tall, solid, roughly vertical mask whose top sits near band level.
 */
export async function findPost(e: Embedded, end: BandEnd, { cols = 10, rows = 4 } = {}): Promise<Post | null> {
  const { W, H } = e;
  const scale = e.reshaped[1] / W;
  const toOrig = (j: number, i: number): [number, number] => [(j * 4) / scale, (i * 4) / scale];
  const toLow = (x: number, y: number): [number, number] => [(x * scale) / 4 - 0.5, (y * scale) / 4 - 0.5];

  const prompts: [number, number][] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = end.x + end.out * (-0.01 + (c * 0.14) / (cols - 1)) * W;
      const y = end.y + (0.01 + r * 0.03) * H;
      if (x >= 0 && x < W && y >= 0 && y < H) prompts.push([x, y]);
    }
  }
  const N = prompts.length;
  if (!N) return null;
  const input_points = new Tensor("float32", Float32Array.from(prompts.flatMap(([x, y]) => [x * scale, y * scale])), [1, N, 1, 2]);
  const input_labels = new Tensor("int64", BigInt64Array.from(prompts.map(() => 1n)), [1, N, 1]);
  const out = await e.seg.model({ ...e.emb, input_points, input_labels });
  const logits = out.pred_masks.data as Float32Array;
  const iou = out.iou_scores.data as Float32Array;

  let best: { off: number; score: number } | null = null;
  for (let n = 0; n < N; n++) {
    for (let m = 0; m < 3; m++) {
      const off = (n * 3 + m) * LOW * LOW;
      const s = maskStats(logits, off, toOrig);
      if (!s) continue;
      const h = s.y1 - s.y0;
      if (h < 0.06 * H || s.x1 - s.x0 > 0.35 * h || s.fill < 0.5) continue;
      if (s.y0 < end.y - 0.08 * H || s.y0 > end.y + 0.02 * H || s.y1 < end.y + 0.05 * H) continue;
      if (s.y1 >= H - 4) continue;
      const score = h * s.fill * iou[n * 3 + m]!;
      if (!best || score > best.score) best = { off, score };
    }
  }
  if (!best) return null;

  const { off } = best;
  const rowsXY: [number, number][] = [];
  for (let i = 0; i < LOW; i++) {
    let sx = 0;
    let cnt = 0;
    for (let j = 0; j < LOW; j++) {
      if (logits[off + i * LOW + j]! > 0) {
        sx += j;
        cnt++;
      }
    }
    if (cnt) rowsXY.push([toOrig(0, i + 0.5)[1], toOrig(sx / cnt + 0.5, 0)[0]]);
  }
  const n = rowsXY.length;
  const my = rowsXY.reduce((a, r) => a + r[0], 0) / n;
  const mx = rowsXY.reduce((a, r) => a + r[1], 0) / n;
  const sxy = rowsXY.reduce((a, r) => a + (r[0] - my) * (r[1] - mx), 0);
  const syy = rowsXY.reduce((a, r) => a + (r[0] - my) ** 2, 0);
  const slope = syy ? sxy / syy : 0;
  const axisX = (y: number) => mx + slope * (y - my);
  const inside = (y: number) => bilinear(logits, off, ...toLow(axisX(y), y)) > 0;
  let top = my;
  while (top > 0 && inside(top - 1)) top--;
  let base = my;
  while (base < H - 1 && inside(base + 1)) base++;
  return { top: { x: axisX(top), y: top }, base: { x: axisX(base), y: base } };
}
