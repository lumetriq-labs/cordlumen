// Pillow's Image.resize(..., Image.BILINEAR), which the models were trained on: a triangle filter widened by the
// downscale factor (antialiasing), horizontal pass then vertical, 8-bit between passes (Pillow libImaging/Resample.c).
// Browser canvas scaling is implementation-defined, so it is not used for model inputs.

interface Coeffs {
  start: Int32Array;
  len: Int32Array;
  w: Float64Array;
  taps: number;
}

function coeffs(inSize: number, outSize: number): Coeffs {
  const scale = inSize / outSize;
  const fs = Math.max(scale, 1);
  const support = fs;
  const taps = Math.ceil(support) * 2 + 1;
  const start = new Int32Array(outSize);
  const len = new Int32Array(outSize);
  const w = new Float64Array(outSize * taps);
  for (let o = 0; o < outSize; o++) {
    const center = (o + 0.5) * scale;
    const lo = Math.max(Math.trunc(center - support + 0.5), 0);
    const n = Math.min(Math.trunc(center + support + 0.5), inSize) - lo;
    let tot = 0;
    for (let k = 0; k < n; k++) {
      const v = Math.max(0, 1 - Math.abs((k + lo - center + 0.5) / fs));
      w[o * taps + k] = v;
      tot += v;
    }
    if (tot) for (let k = 0; k < n; k++) w[o * taps + k]! /= tot;
    start[o] = lo;
    len[o] = n;
  }
  return { start, len, w, taps };
}

const clip8 = (v: number) => (v <= 0 ? 0 : v >= 255 ? 255 : Math.round(v));

/**
 * Resize the sw x sh region at (sx, sy) of `img` (pixels outside the image read as `pad`, as when Pillow crops past
 * the edge onto a grey canvas) to dw x dh. Returns RGB bytes, row-major.
 */
export function resizeRegion(img: ImageData, sx: number, sy: number, sw: number, sh: number, dw: number, dh: number, pad: number): Uint8Array {
  const { width: W, height: H, data } = img;
  const hc = coeffs(sw, dw);
  const vc = coeffs(sh, dh);
  const r0 = vc.start[0]!;
  const r1 = vc.start[dh - 1]! + vc.len[dh - 1]!;
  const mid = new Uint8Array((r1 - r0) * dw * 3);
  for (let r = r0; r < r1; r++) {
    const y = sy + r;
    const inside = y >= 0 && y < H;
    for (let o = 0; o < dw; o++) {
      let a = 0, b = 0, c = 0;
      const lo = hc.start[o]!;
      for (let k = 0; k < hc.len[o]!; k++) {
        const wk = hc.w[o * hc.taps + k]!;
        const x = sx + lo + k;
        if (inside && x >= 0 && x < W) {
          const i = (y * W + x) * 4;
          a += data[i]! * wk;
          b += data[i + 1]! * wk;
          c += data[i + 2]! * wk;
        } else {
          a += pad * wk;
          b += pad * wk;
          c += pad * wk;
        }
      }
      const m = ((r - r0) * dw + o) * 3;
      mid[m] = clip8(a);
      mid[m + 1] = clip8(b);
      mid[m + 2] = clip8(c);
    }
  }
  const out = new Uint8Array(dw * dh * 3);
  for (let o = 0; o < dh; o++) {
    const lo = vc.start[o]! - r0;
    for (let x = 0; x < dw; x++) {
      let a = 0, b = 0, c = 0;
      for (let k = 0; k < vc.len[o]!; k++) {
        const wk = vc.w[o * vc.taps + k]!;
        const m = ((lo + k) * dw + x) * 3;
        a += mid[m]! * wk;
        b += mid[m + 1]! * wk;
        c += mid[m + 2]! * wk;
      }
      const i = (o * dw + x) * 3;
      out[i] = clip8(a);
      out[i + 1] = clip8(b);
      out[i + 2] = clip8(c);
    }
  }
  return out;
}
