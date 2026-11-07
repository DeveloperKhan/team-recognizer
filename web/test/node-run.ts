/**
 * Run the browser pipeline under Node on every labeled screenshot and write its predictions,
 * so Python's evaluator can score them with the same metrics as the Python recognizer:
 *
 *   node --experimental-strip-types test/node-run.ts ~/Desktop/screenshots out.json [--model int8|fp16]
 *   uv run recognizer eval --predictions out.json
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import sharp from "sharp";
import type { RecognizerConfig } from "../src/config.ts";
import { createBackend, decodeHead } from "../src/models.ts";
import { Recognizer } from "../src/pipeline.ts";

const [screensDir, outFile] = process.argv.slice(2);
const variant = process.argv.includes("--model") ? process.argv[process.argv.indexOf("--model") + 1] : "int8";
const models = new URL("../models/", import.meta.url);
const read = (name: string) => readFileSync(new URL(name, models));

const config: RecognizerConfig = JSON.parse(read("config.json").toString());
const keys: string[] = JSON.parse(read("ocr-keys.json").toString());
const head = read("head.bin");
const options = { executionProviders: ["cpu"], intraOpNumThreads: 1 };
const backend = createBackend(ort, {
  embed: await ort.InferenceSession.create(read(`dinov2.${variant}.onnx`), options),
  det: await ort.InferenceSession.create(read("ocr-det.onnx"), options),
  rec: await ort.InferenceSession.create(read("ocr-rec.onnx"), options),
});
const recognizer = new Recognizer(config, keys, decodeHead(head.buffer.slice(head.byteOffset, head.byteOffset + head.length)), backend);

// The labels live in the data folder, found the way recognizer/paths.py finds it
const repo = fileURLToPath(new URL("../../", import.meta.url));
const dataDir =
  process.env.RECOGNIZER_DATA_DIR ??
  [join(repo, "../team-recognizer-data"), join(repo, "local")].find((dir) => existsSync(dir))!;
const labels: Record<string, unknown> = JSON.parse(readFileSync(join(dataDir, "labels/screens.json"), "utf8"));
const out: Record<string, unknown> = {};
const times: number[] = [];
for (const key of Object.keys(labels)) {
  const path = join(screensDir, key.normalize("NFD"));
  let file: Buffer;
  try {
    file = readFileSync(path);
  } catch {
    file = readFileSync(join(screensDir, key));
  }
  // Pixels as stored, ignoring any embedded color profile (macOS screenshots carry the display's),
  // the way OpenCV reads them and the worker decodes them (colorSpaceConversion: "none")
  const { data, info } = await sharp(file, { ignoreIcc: true })
    .flatten({ background: "#000000" })
    .raw()
    .toBuffer({ resolveWithObject: true });
  const start = performance.now();
  out[key] = await recognizer.recognize({ width: info.width, height: info.height, data: new Uint8Array(data) });
  times.push(performance.now() - start);
  process.stdout.write(".");
}
writeFileSync(outFile, JSON.stringify(out));
times.sort((a, b) => a - b);
console.log(`\n${times.length} screenshots, median ${(times[times.length >> 1] / 1000).toFixed(2)} s, max ${(times[times.length - 1] / 1000).toFixed(2)} s (single thread, ${variant})`);
