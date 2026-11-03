// Bundle the demo page's script (demo/demo.ts, with the client) into public/demo.js, which Vercel
// serves next to the API. The page loads the worker and models of this package version from
// jsDelivr, so publish the version before deploying a demo that points at it.
import { build } from "esbuild";
import { readFileSync } from "node:fs";

const web = new URL("../", import.meta.url);
const path = (p) => new URL(p, web).pathname;
const pkg = JSON.parse(readFileSync(path("package.json")));
const ort = JSON.parse(readFileSync(path("node_modules/onnxruntime-web/package.json")));

await build({
  entryPoints: [path("../demo/demo.ts")],
  outfile: path("../public/demo.js"),
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  minify: true,
  legalComments: "none",
  banner: { js: "// Built from demo/demo.ts by `npm run demo` in web/." },
  define: { __PACKAGE_VERSION__: JSON.stringify(pkg.version), __ORT_VERSION__: JSON.stringify(ort.version) },
});
console.log(`built public/demo.js (${pkg.name}@${pkg.version})`);
