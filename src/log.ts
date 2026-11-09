const t0 = performance.now();

/** Console log with seconds since page load, so the last line before a crash shows how far the run got. */
export function log(...args: unknown[]) {
  console.log(`[cordlumen +${((performance.now() - t0) / 1000).toFixed(2)}s]`, ...args);
}

interface GpuAdapterLike {
  limits: Record<string, number>;
  features: { has(name: string): boolean };
  info?: { vendor?: string; architecture?: string };
}

export async function logEnvironment() {
  log(`build ${__BUILD_ID__} · ${__BUILD_TIME__}`);
  log("userAgent", navigator.userAgent);
  const gpu = (navigator as { gpu?: { requestAdapter(): Promise<GpuAdapterLike | null> } }).gpu;
  if (!gpu) {
    log("WebGPU: navigator.gpu なし（WASM で実行）");
    return;
  }
  const adapter = await gpu.requestAdapter().catch((e: unknown) => (log("WebGPU requestAdapter 失敗", e), null));
  if (!adapter) {
    log("WebGPU: アダプタを取得できませんでした");
    return;
  }
  const MB = (n: number | undefined) => (n === undefined ? "-" : `${(n / 1048576).toFixed(0)} MB`);
  log("WebGPU adapter", adapter.info?.vendor ?? "", adapter.info?.architecture ?? "");
  log("WebGPU maxBufferSize", MB(adapter.limits.maxBufferSize));
  log("WebGPU maxStorageBufferBindingSize", MB(adapter.limits.maxStorageBufferBindingSize));
  log("WebGPU shader-f16", adapter.features.has("shader-f16"));
}
