import { drawGuide, drawMeasure, drawPosts, verdict } from "./draw";
import { loadDetector, type Detector } from "./keypoints";
import { log, logEnvironment } from "./log";
import { analyze, type Result } from "./pipeline";

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

let detector: Promise<{ det: Detector; ms: number }> | null = null;

function getDetector() {
  if (!detector) {
    const t = performance.now();
    detector = loadDetector().then((det) => ({ det, ms: performance.now() - t }));
    detector.catch(() => (detector = null));
  }
  return detector;
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
    const { det, ms: loadMs } = await getDetector();
    const t = performance.now();
    const r = await analyze(image, det, (label) => {
      log(label);
      status.textContent = label;
    });
    const total = performance.now() - t;
    log("判定終了", r.ok ? `中央 ${(r.center * 100).toFixed(1)} cm` : r.reason, r.timings);

    showTimings([
      ["実行環境", "WebAssembly（CPU）"],
      ["モデルの準備（初回のみ）", sec(loadMs)],
      ["白帯の検出", sec(r.timings.band)],
      ["ポストの検出", sec(r.timings.posts)],
      ["判定の合計", sec(total)],
      ["写真のサイズ", `${image.width} × ${image.height}`],
      ...(r.ok ? ([["ポストの確信度", r.scores.map((s) => s.toFixed(2)).join(" / ")]] as [string, string][]) : []),
    ]);

    showResult(ctx, r, s);
  } catch (err) {
    log("例外", err);
    console.error(err);
    status.textContent = `エラー: ${errorText(err)}`;
  }
}

const errorText = (err: unknown) => (err instanceof Error ? err.message : String(err));

function showResult(ctx: CanvasRenderingContext2D, r: Result, s: number) {
  if (!r.ok) {
    verdictEl.hidden = true;
    status.textContent = r.reason;
    return;
  }
  const v = verdict(r.center);
  drawGuide(ctx, r.toImg, v, { scale: s });
  drawPosts(ctx, r.posts, s);
  drawMeasure(ctx, r.bandLine, r.centerPt, s);
  verdictEl.className = `verdict ${v.level}`;
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
}

// Live mode: take the newest camera frame, judge it, show it, repeat. Frames that arrive while a frame is being
// judged are never read, so they are dropped.
const liveBtn = $<HTMLButtonElement>("live");
const cam = $<HTMLVideoElement>("cam");
interface LiveSession {
  stop: boolean;
  stream?: MediaStream;
}
let live: LiveSession | null = null;

const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));

/** Stops the camera at once; the loop ends at its next check. */
function stopLive() {
  if (!live) return;
  live.stop = true;
  live.stream?.getTracks().forEach((t) => t.stop());
  cam.srcObject = null;
  live = null;
  liveBtn.textContent = "カメラで測る";
  liveBtn.classList.remove("on");
}

async function startLive() {
  const session: LiveSession = { stop: false };
  live = session;
  liveBtn.textContent = "停止";
  liveBtn.classList.add("on");
  verdictEl.hidden = true;
  timingsEl.hidden = true;
  const work = document.createElement("canvas");
  try {
    status.textContent = "カメラを起動中…";
    // Ask for 4K from the back camera; iPhone Safari gives 3840 x 2160 (2160 x 3840 held upright).
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 3840 }, height: { ideal: 2160 } },
    });
    session.stream = stream;
    if (session.stop) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    cam.srcObject = stream;
    await cam.play();
    const set = stream.getVideoTracks()[0]!.getSettings();
    log("カメラ", `${set.width}x${set.height}`, `${set.frameRate} fps`);
    status.textContent = "モデルを準備中…";
    const { det } = await getDetector();
    const wctx = work.getContext("2d", { willReadFrequently: true })!;
    while (!session.stop) {
      const w = cam.videoWidth;
      const h = cam.videoHeight;
      if (!w || !h) {
        await nextFrame();
        continue;
      }
      if (work.width !== w || work.height !== h) {
        work.width = w;
        work.height = h;
      }
      wctx.drawImage(cam, 0, 0);
      const image = wctx.getImageData(0, 0, w, h);
      const t = performance.now();
      const r = await analyze(image, det);
      const ms = performance.now() - t;
      if (session.stop) break;
      const s = Math.min(1, DISPLAY_W / w);
      view.width = Math.round(w * s);
      view.height = Math.round(h * s);
      const ctx = view.getContext("2d")!;
      ctx.drawImage(work, 0, 0, view.width, view.height);
      showResult(ctx, r, s);
      showTimings([
        ["カメラ", `${w} × ${h}`],
        ["1 回の判定", sec(ms)],
      ]);
      await nextFrame();
    }
  } catch (err) {
    if (!session.stop) {
      log("例外", err);
      console.error(err);
      status.textContent = `カメラ: ${errorText(err)}`;
    }
  } finally {
    if (live === session) stopLive();
    work.width = work.height = 0;
  }
}

liveBtn.addEventListener("click", () => (live ? stopLive() : void startLive()));
document.addEventListener("visibilitychange", () => document.hidden && stopLive());

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  fileInput.value = "";
  stopLive();
  if (file) void run(file);
});

// ?src=<image URL> analyzes that image on load (for checking on a device without picking a file).
const src = params.get("src");
if (src) {
  fetch(src)
    .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(`${r.status} ${src}`))))
    .then(run, (err) => (status.textContent = `エラー: ${err.message}`));
}
