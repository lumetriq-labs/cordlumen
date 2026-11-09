import { log, logEnvironment } from "./log";

// Untrained MobileNetV3-small + heatmap head for 4 keypoints (post tops and bases). Only for memory/speed checks.
// Same file name across deploys and Pages allows caching for 10 minutes, so tie the URL to the build.
const MODEL_URL = `./models/post_kp_untrained.onnx?v=${__BUILD_ID__}`;
const RUNS = 5;

const params = new URLSearchParams(location.search);
const EP = params.get("ep") === "webgpu" ? "webgpu" : "wasm";
// Fully convolutional, so any size works; keep 4:3 and multiples of 32 (the backbone stride).
const IN_W = Math.max(32, Math.round(Number(params.get("size") ?? 512) / 32) * 32);
const IN_H = Math.round((IN_W * 3) / 4 / 32) * 32;

const ort: typeof import("onnxruntime-web/wasm") =
  EP === "webgpu" ? await import("onnxruntime-web/webgpu") : await import("onnxruntime-web/wasm");

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const fileInput = $<HTMLInputElement>("file");
const status = $<HTMLParagraphElement>("status");
const timingsEl = $<HTMLTableElement>("timings");

$("build").textContent = `${__BUILD_ID__} · ${__BUILD_TIME__}`;
void logEnvironment();
addEventListener("error", (e) => log("error", e.message));
addEventListener("unhandledrejection", (e) => log("unhandledrejection", e.reason));

if (EP === "wasm") {
  // The plain WebAssembly build; the asyncify/JSEP builds reportedly grow memory on iOS Safari (microsoft/onnxruntime#26827).
  const prefix = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
  ort.env.wasm.wasmPaths = { mjs: `${prefix}ort-wasm-simd-threaded.mjs`, wasm: `${prefix}ort-wasm-simd-threaded.wasm` };
}
$("mode").textContent = `${EP === "webgpu" ? "GPU（WebGPU）" : "CPU（WebAssembly）"} · 入力 ${IN_W}×${IN_H}`;
log("設定", EP, `${IN_W}x${IN_H}`, "crossOriginIsolated", crossOriginIsolated, "numThreads", ort.env.wasm.numThreads);

let session: Promise<import("onnxruntime-web/wasm").InferenceSession> | null = null;
function getSession() {
  session ??= ort.InferenceSession.create(MODEL_URL, { executionProviders: [EP] });
  return session;
}

/** Full-size photo into ImageData, as the real app does, so its memory is included. */
async function readImage(file: Blob) {
  const bmp = await createImageBitmap(file);
  const c = document.createElement("canvas");
  c.width = bmp.width;
  c.height = bmp.height;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const data = ctx.getImageData(0, 0, c.width, c.height);
  c.width = c.height = 0;
  return data;
}

function toTensor(image: ImageData) {
  const src = document.createElement("canvas");
  src.width = image.width;
  src.height = image.height;
  src.getContext("2d")!.putImageData(image, 0, 0);
  const c = document.createElement("canvas");
  c.width = IN_W;
  c.height = IN_H;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(src, 0, 0, IN_W, IN_H);
  src.width = src.height = 0;
  const { data } = ctx.getImageData(0, 0, IN_W, IN_H);
  const n = IN_W * IN_H;
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < 3; ch++) out[ch * n + i] = (data[i * 4 + ch]! / 255 - mean[ch]!) / std[ch]!;
  }
  return new ort.Tensor("float32", out, [1, 3, IN_H, IN_W]);
}

function argmaxPoints(t: import("onnxruntime-web/wasm").Tensor) {
  const [, k, h, w] = t.dims as number[];
  const d = t.data as Float32Array;
  const pts: string[] = [];
  for (let j = 0; j < k!; j++) {
    let best = 0;
    for (let i = 1; i < h! * w!; i++) if (d[j * h! * w! + i]! > d[j * h! * w! + best]!) best = i;
    pts.push(`(${best % w!}, ${Math.floor(best / w!)})`);
  }
  return pts.join(" ");
}

const sec = (ms: number) => `${(ms / 1000).toFixed(3)} 秒`;

async function run(file: Blob) {
  timingsEl.hidden = true;
  try {
    log("開始", file.type, `${(file.size / 1e6).toFixed(1)} MB`);
    status.textContent = "写真を読み込み中…";
    const image = await readImage(file);
    log("写真の読み込み完了", `${image.width}x${image.height}`);

    status.textContent = "モデルを準備中…";
    let t = performance.now();
    const s = await getSession();
    const load = performance.now() - t;
    log("モデル準備完了", sec(load));

    status.textContent = "実行中…";
    t = performance.now();
    const input = toTensor(image);
    const prep = performance.now() - t;
    const times: number[] = [];
    let points = "";
    for (let r = 0; r < RUNS; r++) {
      t = performance.now();
      const out = await s.run({ pixel_values: input });
      times.push(performance.now() - t);
      const heat = out.heatmaps!;
      points = argmaxPoints(heat);
      heat.dispose();
      log(`推論 ${r + 1}/${RUNS}`, sec(times[r]!));
    }
    input.dispose();
    log("完了", points);

    const body = timingsEl.tBodies[0]!;
    body.replaceChildren(
      ...(
        [
          ["モデルの準備", sec(load)],
          ["縮小と変換", sec(prep)],
          ["推論（1回目）", sec(times[0]!)],
          [`推論（2〜${RUNS}回目の平均）`, sec(times.slice(1).reduce((a, b) => a + b, 0) / (RUNS - 1))],
          ["実行先", EP === "webgpu" ? "GPU（WebGPU）" : "CPU（WebAssembly）"],
          ["入力サイズ", `${IN_W} × ${IN_H}`],
          ["写真のサイズ", `${image.width} × ${image.height}`],
          ["出力（意味なし）", points],
        ] as [string, string][]
      ).map(([k, v]) => {
        const tr = document.createElement("tr");
        for (const text of [k, v]) tr.append(Object.assign(document.createElement("td"), { textContent: text }));
        return tr;
      }),
    );
    timingsEl.hidden = false;
    status.textContent = "完了（落ちずに最後まで動きました）";
  } catch (err) {
    log("例外", err);
    console.error(err);
    status.textContent = `エラー: ${err instanceof Error ? err.message : String(err)}`;
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  fileInput.value = "";
  if (file) void run(file);
});

const src = params.get("src");
if (src) {
  fetch(src)
    .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(`${r.status} ${src}`))))
    .then(run, (err) => (status.textContent = `エラー: ${err.message}`));
}
