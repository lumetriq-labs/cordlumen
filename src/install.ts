/** Opened from the home screen (installed), rather than in a browser tab. */
export const isStandalone = (): boolean =>
  matchMedia("(display-mode: standalone)").matches ||
  (navigator as Navigator & { standalone?: boolean }).standalone === true;

/** iPhone/iPad. iPadOS reports itself as a Mac, so touch support tells them apart. */
const isIOS = (): boolean =>
  /iPhone|iPad|iPod/.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);

/** Chromium-only, non-standard: https://developer.mozilla.org/docs/Web/API/BeforeInstallPromptEvent */
interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<unknown>;
}

/** Wires up the "ホーム画面に追加" screen: platform order, installed state, and Chrome's install prompt. */
export function setupInstall(): void {
  const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
  const ios = $("installIos");
  const android = $("installAndroid");
  const button = $<HTMLButtonElement>("installBtn");
  const showInstalled = (text: string) => {
    const box = $("installed");
    box.querySelector("p")!.textContent = text;
    box.hidden = false;
  };

  if (!isIOS()) android.after(ios);
  if (isStandalone()) {
    showInstalled("いまはホーム画面から開いています。");
    $("installRow").hidden = true;
  }

  // The browser only offers this when the app can be installed and isn't yet; it can be used once.
  let deferred: BeforeInstallPromptEvent | null = null;
  addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    deferred = e as BeforeInstallPromptEvent;
    button.hidden = false;
  });
  button.addEventListener("click", async () => {
    if (!deferred) return;
    const prompt = deferred;
    deferred = null;
    button.hidden = true;
    await prompt.prompt();
  });
  addEventListener("appinstalled", () => {
    deferred = null;
    button.hidden = true;
    showInstalled("ホーム画面のアイコンから開けます。");
  });
}
