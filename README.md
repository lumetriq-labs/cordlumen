# Cordlumen

Check tennis net height from a single photo, in the browser.

Pick a photo that shows the whole net, down to the base of both posts. Cordlumen finds the white band and the two posts, works out the height at the centre strap, and overlays the regulation band: green within ±1 cm of 91.4 cm, red otherwise.

**https://lumetriq-labs.github.io/cordlumen/**

## How it works

1. **Band** — the top edge of the white band is found column by column (a step up in whiteness with net mesh below it) and fitted as a chain of quadratic segments, since the centre strap puts a kink in it.
2. **Posts** — [SlimSAM](https://huggingface.co/Xenova/slimsam-77-uniform) is prompted with a grid of points outside each end of the band; the tall, solid, vertical mask whose top sits near band level is taken as the post, giving its top and base.
3. **Height** — the post tops (cord at 1.07 m) and bases (0 m), 12.80 m apart, define a homography of the net plane (ITF Rules of Tennis, Rule 1). The band's top edge at the plane's centre line gives the centre height.

Everything runs on the device with [Transformers.js](https://huggingface.co/docs/transformers.js) (WebGPU when available, otherwise WebAssembly). The model (about 38 MB) is downloaded from Hugging Face on first use and cached by the browser. Photos are not sent anywhere.

WebGPU on iPhone needs iOS 26 (Safari 26) or later.

## Development

```sh
npm install
npm run dev          # local server
npm test             # geometry unit tests
npm run build        # type-check and build to dist/
```

`?src=<image URL>` analyzes that image on load, e.g. `http://localhost:5173/?src=samples/photo.jpeg` with photos in the git-ignored `samples/`.

## License

[MIT](LICENSE)
