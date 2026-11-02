// Copy the browser build into play-graphic-tool (or the app given as the first argument):
//   public/team-recognizer/<hash>/  worker.js, ort/, models/   (served as static, cacheable files)
//   vendor/team-recognizer/         client.ts, types.ts, version.ts
// The folder name is a hash of everything in it, so a new build never reuses a cached file.
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = fileURLToPath(new URL("..", import.meta.url));
const app = resolve(process.argv[2] ?? join(web, "../../play-graphic-tool"));
const MODELS = ["config.json", "ocr-keys.json", "head.bin", "ocr-det.onnx", "ocr-rec.onnx", "dinov2.int8.onnx", "dinov2.fp16.onnx"];

const files = [
  ["worker.js", join(web, "dist/worker.js")],
  ...readdirSync(join(web, "dist/ort")).map((f) => [`ort/${f}`, join(web, "dist/ort", f)]),
  ...MODELS.map((f) => [`models/${f}`, join(web, "models", f)]),
];
const hash = createHash("sha256");
for (const [name, path] of files) hash.update(name).update(readFileSync(path));
const version = hash.digest("hex").slice(0, 12);

const publicRoot = join(app, "public/team-recognizer");
rmSync(publicRoot, { recursive: true, force: true }); // drop previous versions
for (const [name, path] of files) {
  const dest = join(publicRoot, version, name);
  mkdirSync(join(dest, ".."), { recursive: true });
  copyFileSync(path, dest);
}

const vendor = join(app, "vendor/team-recognizer");
mkdirSync(vendor, { recursive: true });
for (const f of ["client.ts", "types.ts"]) copyFileSync(join(web, "src", f), join(vendor, f));
writeFileSync(
  join(vendor, "version.ts"),
  `// Written by team-recognizer/web/scripts/sync.mjs; do not edit.\nexport const RECOGNIZER_BASE_URL = "/team-recognizer/${version}/";\n`
);
console.log(`synced ${files.length} files to ${publicRoot}/${version} and ${vendor}`);
