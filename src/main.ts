import { drawGuide, drawPosts, verdict } from "./draw";
import { log, logEnvironment } from "./log";
import { analyze } from "./pipeline";
import { loadSegmenter, type Dtype, type Segmenter } from "./posts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const fileInput = $<HTMLInputElement>("file");
const status = $<HTMLParagraphElement>("status");
const verdictEl = $<HTMLDivElement>("verdict");
const view = $<HTMLCanvasElement>("view");
const timingsEl = $<HTMLTableElement>("timings");

const DISPLAY_W = 1600;

$("build").textContent = `${__BUILD_ID__} · ${__BUILD_TIME__}`;
void logEnvironment();
addEventListener("error", (e) => log("error", e.message));
addEventListener("unhandledrejection", (e) => log("unhandledrejection", e.reason));

const params = new URLSearchParams(location.search);
// ?dtype=fp16 loads the half-precision model (for comparing memory use on devices).
const dtype: Dtype = params.get("dtype") === "fp16" ? "fp16" : "fp32";

let segmenter: Promise<{ seg: Segmenter; ms: number }> | null = null;

function getSegmenter() {
  if (!segmenter) {
    const t = performance.now();
    segmenter = loadSegmenter((loaded, total) => {
      if (total) status.textContent = `モデルを読み込み中… ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`;
    }, dtype).then((seg) => ({ seg, ms: performance.now() - t }));
    segmenter.catch(() => (segmenter = null));
  }
  return segmenter;
}

async function readImage(file: Blob): Promise<ImageData> {
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

function showTimings(rows: [string, string][]) {
  const body = timingsEl.tBodies[0]!;
  body.replaceChildren(
    ...rows.map(([k, v]) => {
      const tr = document.createElement("tr");
      for (const text of [k, v]) tr.append(Object.assign(document.createElement("td"), { textContent: text }));
      return tr;
    }),
  );
  timingsEl.hidden = false;
}

const sec = (ms: number | undefined) => (ms === undefined ? "-" : `${(ms / 1000).toFixed(2)} 秒`);

async function run(file: Blob) {
  verdictEl.hidden = true;
  timingsEl.hidden = true;
  try {
    log("判定開始", file.type, `${(file.size / 1e6).toFixed(1)} MB`);
    status.textContent = "写真を読み込み中…";
    const image = await readImage(file);
    log("写真の読み込み完了", `${image.width}x${image.height}`);
    const s = Math.min(1, DISPLAY_W / image.width);
    view.width = Math.round(image.width * s);
    view.height = Math.round(image.height * s);
    const ctx = view.getContext("2d")!;
    const bmp = await createImageBitmap(image);
    ctx.drawImage(bmp, 0, 0, view.width, view.height);
    bmp.close();

    status.textContent = "モデルを準備中…";
    const { seg, ms: loadMs } = await getSegmenter();
    const t = performance.now();
    const r = await analyze(image, seg, (label) => {
      log(label);
      status.textContent = label;
    });
    const total = performance.now() - t;
    log("判定終了", r.ok ? `中央 ${(r.center * 100).toFixed(1)} cm` : r.reason, r.timings);

    showTimings([
      ["実行環境", `${seg.device === "webgpu" ? "WebGPU" : "WebAssembly（CPU）"} · ${seg.dtype}`],
      ["モデルの準備（初回のみ）", sec(loadMs)],
      ["白帯の検出", sec(r.timings.band)],
      ["画像の解析（SlimSAM）", sec(r.timings.encode)],
      ["ポストの切り抜き", sec(r.timings.posts)],
      ["判定の合計", sec(total)],
      ["写真のサイズ", `${image.width} × ${image.height}`],
    ]);

    if (!r.ok) {
      status.textContent = r.reason;
      return;
    }
    const v = verdict(r.center);
    drawGuide(ctx, r.toImg, v, { scale: s });
    drawPosts(ctx, r.posts, s);
    verdictEl.className = `verdict ${v.ok ? "ok" : "ng"}`;
    verdictEl.innerHTML = "";
    verdictEl.append(v.label);
    const sign = v.diff >= 0 ? "+" : "";
    verdictEl.append(
      Object.assign(document.createElement("small"), {
        textContent: `中央 ${(r.center * 100).toFixed(1)} cm（${sign}${(v.diff * 100).toFixed(1)} cm）`,
      }),
    );
    verdictEl.hidden = false;
    status.textContent = "";
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

// ?src=<image URL> analyzes that image on load (for checking on a device without picking a file).
const src = params.get("src");
if (src) {
  fetch(src)
    .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(`${r.status} ${src}`))))
    .then(run, (err) => (status.textContent = `エラー: ${err.message}`));
}
