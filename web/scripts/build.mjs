// Build the package: dist/worker.js (the pipeline with onnxruntime-web bundled), dist/client.js
// (the main-thread API, no dependencies) and its type declarations.
import { build } from "esbuild";
import { execFileSync } from "node:child_process";
import { copyFileSync, readFileSync, rmSync } from "node:fs";

const web = new URL("../", import.meta.url);
const pkg = JSON.parse(readFileSync(new URL("package.json", web)));
const ort = JSON.parse(readFileSync(new URL("node_modules/onnxruntime-web/package.json", web)));
const path = (p) => new URL(p, web).pathname;

rmSync(path("dist"), { recursive: true, force: true });
const common = { bundle: true, format: "esm", platform: "browser", target: "es2022", legalComments: "none" };
await build({ ...common, entryPoints: [path("src/worker.ts")], outfile: path("dist/worker.js"), minify: true });
await build({
  ...common,
  entryPoints: [path("src/client.ts")],
  outfile: path("dist/client.js"),
  // The client's default URLs: this version's files and the runtime release the worker bundles.
  define: { __PACKAGE_VERSION__: JSON.stringify(pkg.version), __ORT_VERSION__: JSON.stringify(ort.version) },
});
execFileSync("npx", ["tsc", "-p", path("tsconfig.build.json")], { stdio: "inherit" });

// npm only picks up files inside the package folder.
for (const file of ["LICENSE", "THIRD_PARTY_NOTICES.md"]) copyFileSync(path(`../${file}`), path(file));
console.log(`built dist/ for ${pkg.name}@${pkg.version} (onnxruntime-web ${ort.version})`);
