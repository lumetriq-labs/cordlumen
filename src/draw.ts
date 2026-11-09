import { CENTER_H, SPAN, targetY, type Project } from "./measure";

export interface Verdict {
  ok: boolean;
  diff: number;
  label: string;
}

export function verdict(centerH: number, tol = 0.01): Verdict {
  const diff = centerH - CENTER_H;
  const ok = Math.abs(diff) <= tol;
  return { ok, diff, label: ok ? "ぴったり" : diff < 0 ? "低すぎる" : "高すぎる" };
}

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
  ctx.fillStyle = v.ok ? "rgba(40, 200, 80, 0.55)" : "rgba(230, 40, 40, 0.55)";
  ctx.beginPath();
  top.forEach(([u, y], i) => (i ? ctx.lineTo(u, y) : ctx.moveTo(u, y)));
  for (let i = bottom.length - 1; i >= 0; i--) ctx.lineTo(...bottom[i]!);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}
