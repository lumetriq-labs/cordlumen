import type { Pt } from "./band";
import { CENTER_H, SPAN, targetY, type Post, type Project } from "./measure";

export interface Verdict {
  ok: boolean;
  /** Guide colour: within `tol` green, within `near` yellow, otherwise red. */
  level: "ok" | "near" | "far";
  diff: number;
  label: string;
}

export function verdict(centerH: number, tol = 0.01, near = 0.03): Verdict {
  const diff = centerH - CENTER_H;
  const ok = Math.abs(diff) <= tol;
  const level = ok ? "ok" : Math.abs(diff) <= near ? "near" : "far";
  return { ok, level, diff, label: ok ? "ぴったり" : diff < 0 ? "低い" : "高い" };
}

const GUIDE_FILL = { ok: "rgba(40, 200, 80, 0.55)", near: "rgba(245, 200, 20, 0.6)", far: "rgba(230, 40, 40, 0.55)" };

/** Draw the target band (net top at regulation height, `depth` deep) in image coordinates scaled by `scale`. */
export function drawGuide(ctx: CanvasRenderingContext2D, toImg: Project, v: Verdict, { depth = 0.05, scale = 1 } = {}) {
  const N = 64;
  const top: [number, number][] = [];
  const bottom: [number, number][] = [];
  for (let i = 0; i <= N; i++) {
    const X = (SPAN * i) / N;
    top.push(toImg(X, targetY(X)));
    bottom.push(toImg(X, targetY(X) - depth));
  }
  ctx.save();
  ctx.scale(scale, scale);
  ctx.fillStyle = GUIDE_FILL[v.level];
  ctx.beginPath();
  top.forEach(([u, y], i) => (i ? ctx.lineTo(u, y) : ctx.moveTo(u, y)));
  for (let i = bottom.length - 1; i >= 0; i--) ctx.lineTo(...bottom[i]!);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/** Draw the detected band top edge and mark the centre point whose height is reported, in display pixels. */
export function drawMeasure(ctx: CanvasRenderingContext2D, bandLine: Pt[], center: Pt, scale = 1) {
  ctx.save();
  ctx.beginPath();
  bandLine.forEach(({ x, y }, i) => (i ? ctx.lineTo(x * scale, y * scale) : ctx.moveTo(x * scale, y * scale)));
  ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
  ctx.lineWidth = 3.5;
  ctx.stroke();
  ctx.strokeStyle = "#ff40ff";
  ctx.lineWidth = 1.5;
  ctx.stroke();
  const [x, y] = [center.x * scale, center.y * scale];
  ctx.beginPath();
  ctx.moveTo(x - 12, y);
  ctx.lineTo(x + 12, y);
  ctx.moveTo(x, y - 12);
  ctx.lineTo(x, y + 12);
  ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
  ctx.lineWidth = 5;
  ctx.stroke();
  ctx.strokeStyle = "#ff40ff";
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.restore();
}

/** Mark each detected post with a line from base to top and a dot at both ends, in display pixels. */
export function drawPosts(ctx: CanvasRenderingContext2D, posts: Post[], scale = 1) {
  ctx.save();
  ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillStyle = "#3cd2ff";
  for (const { top, base } of posts) {
    const [tx, ty, bx, by] = [top.x * scale, top.y * scale, base.x * scale, base.y * scale];
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.strokeStyle = "#3cd2ff";
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
    ctx.lineWidth = 2;
    for (const [x, y] of [[tx, ty], [bx, by]] as const) {
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
  ctx.restore();
}
