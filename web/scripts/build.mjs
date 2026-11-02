// Bundle the worker (with onnxruntime-web) and collect the runtime's WebAssembly file.
import { build } from "esbuild";
import { copyFileSync, mkdirSync, rmSync } from "node:fs";

const dist = new URL("../dist/", import.meta.url);
rmSync(dist, { recursive: true, force: true });
mkdirSync(new URL("ort/", dist), { recursive: true });

await build({
  entryPoints: [new URL("../src/worker.ts", import.meta.url).pathname],
  outfile: new URL("worker.js", dist).pathname,
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  minify: true,
  legalComments: "none",
});

// onnxruntime-web/webgpu loads its runtime (WebGPU + WebAssembly backends) from wasmPaths:
// the .wasm and the JavaScript loader that instantiates it.
for (const file of ["ort-wasm-simd-threaded.asyncify.wasm", "ort-wasm-simd-threaded.asyncify.mjs"]) {
  copyFileSync(new URL(`../node_modules/onnxruntime-web/dist/${file}`, import.meta.url), new URL(`ort/${file}`, dist));
}
console.log("built dist/worker.js and dist/ort/");
