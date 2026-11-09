# Cordlumen

Check tennis net height from a single photo, in the browser.

Pick a photo that shows the whole net, down to the base of both posts. Cordlumen finds the white band and the two posts, works out the height at the centre strap, and overlays the regulation band: green within ±1 cm of 91.4 cm, red otherwise.

**https://lumetriq-labs.github.io/cordlumen/**

## How it works

1. **Band** — the top edge of the white band is found column by column (a step up in whiteness with net mesh below it) and fitted as a chain of quadratic segments, since the centre strap puts a kink in it.
2. **Posts** — a small keypoint model (MobileNetV3-small with a heatmap head, trained on real photos and Blender renders) gives the top and base of each post from the photo letterboxed to 512×384.
3. **Height** — the post tops (cord at 1.07 m) and bases (0 m), 12.80 m apart, define a homography of the net plane (ITF Rules of Tennis, Rule 1). The band's top edge at the plane's centre line gives the centre height.

Everything runs on the device with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) on plain WebAssembly. The model (about 4 MB) is served with the app. Photos are not sent anywhere.

The earlier SlimSAM version is kept at the `slimsam` tag.

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
