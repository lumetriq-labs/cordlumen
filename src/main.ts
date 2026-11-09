import { drawGuide, verdict } from "./draw";
import { analyze } from "./pipeline";
import { loadSegmenter, type Segmenter } from "./posts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const fileInput = $<HTMLInputElement>("file");
const status = $<HTMLParagraphElement>("status");
const verdictEl = $<HTMLDivElement>("verdict");
const view = $<HTMLCanvasElement>("view");
const timingsEl = $<HTMLTableElement>("timings");

const DISPLAY_W = 1600;

$("build").textContent = `${__BUILD_ID__} · ${__BUILD_TIME__}`;

let segmenter: Promise<{ seg: Segmenter; ms: number }> | null = null;

function getSegmenter() {
  if (!segmenter) {
    const t = performance.now();
    segmenter = loadSegmenter((loaded, total) => {
      if (total) status.textContent = `モデルを読み込み中… ${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`;
    }).then((seg) => ({ seg, ms: performance.now() - t }));
    segmenter.catch(() => (segmenter = null));
  }
  return segmenter;
}

const params = new URLSearchParams(location.search);

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
    status.textContent = "写真を読み込み中…";
    const image = await readImage(file);
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
    const r = await analyze(image, seg, (label) => (status.textContent = label));
    const total = performance.now() - t;

    showTimings([
      ["実行環境", seg.device === "webgpu" ? "WebGPU" : "WebAssembly（CPU）"],
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
