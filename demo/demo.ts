/**
 * The demo page (public/index.html): runs the browser build on dropped, pasted or chosen
 * screenshots and shows what it found. Built into public/demo.js by `npm run demo` in web/.
 */

import { createLocalRecognizer, DEFAULT_BASE_URL } from "../web/src/client.ts";
import type { RecognitionResult, RecognizedPokemon, RecognizedTeam, RelativeBox } from "../web/src/client.ts";

const SPRITE = "https://play.pokemonshowdown.com/sprites";

// On localhost, ?base=<url> runs a local build instead of the published one.
const local = ["localhost", "127.0.0.1"].includes(location.hostname);
const baseUrl = (local && new URLSearchParams(location.search).get("base")) || DEFAULT_BASE_URL;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const status = $("status");
const statusText = $("status-text");
const statusBar = $("status-bar");
const statusMeta = $("status-meta");
const results = $("results");
const fileInput = $<HTMLInputElement>("file");

function el<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, string> = {}, ...children: Array<Node | string>) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;

// --- the recognizer ---------------------------------------------------------------------------

const loadStart = performance.now();
const recognizer = createLocalRecognizer({
  baseUrl,
  onProgress(loaded, total) {
    statusText.textContent = "Downloading models…";
    statusMeta.textContent = total ? `${mb(loaded)} of ${mb(total)} (cached after the first visit)` : mb(loaded);
    statusBar.style.width = total ? `${Math.min(100, (100 * loaded) / total)}%` : "0";
  },
});
recognizer.ready.then(
  ({ backend, version }) => {
    status.dataset.state = "ready";
    statusText.textContent = `Ready. Running on ${backend === "webgpu" ? "WebGPU" : "WebAssembly (CPU)"}`;
    statusMeta.textContent = `models ${version} · loaded in ${((performance.now() - loadStart) / 1000).toFixed(1)} s`;
  },
  (error: Error) => {
    status.dataset.state = "error";
    statusText.textContent = `This browser couldn't load the recognizer: ${error.message}`;
    statusMeta.textContent = "";
  }
);

/** speciesId -> Showdown sprite name, from the models' config (the worker fetches it too). */
const slugs: Promise<Map<string, string>> = fetch(new URL("models/config.json", baseUrl))
  .then((r) => r.json())
  .then(
    (config: { species: { classes: Array<{ speciesId: string; showdownSlug: string | null }> } }) =>
      new Map(config.species.classes.filter((c) => c.showdownSlug).map((c) => [c.speciesId, c.showdownSlug!]))
  )
  .catch(() => new Map());

// --- input ------------------------------------------------------------------------------------

const drop = $("drop");
$("choose").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  addFiles(Array.from(fileInput.files ?? []));
  fileInput.value = "";
});
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
// Images dropped anywhere on the page count (rather than the browser navigating to them).
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  addFiles(Array.from(e.dataTransfer?.files ?? []));
});
document.addEventListener("paste", (e) => addFiles(Array.from(e.clipboardData?.files ?? [])));

function addFiles(files: File[]) {
  for (const file of files.filter((f) => f.type.startsWith("image/"))) void recognizeFile(file);
}

// --- one image --------------------------------------------------------------------------------

async function recognizeFile(file: File) {
  const url = URL.createObjectURL(file);
  const img = el("img", { src: url, alt: file.name || "screenshot" });
  const shot = el("div", { class: "shot scanning" }, img);
  const state = el("span", { class: "state" }, "Waiting for the models…");
  const panel = el("div", { class: "teams" });
  const foot = el("div", { class: "foot" });
  const card = el(
    "article",
    { class: "result" },
    el("div", { class: "result-head" }, el("span", { class: "file" }, file.name || "Pasted image"), state),
    el("div", { class: "result-body" }, shot, panel),
    foot
  );
  results.prepend(card);

  let result: RecognitionResult;
  try {
    await recognizer.ready;
    state.textContent = "Recognizing…";
    result = await recognizer.recognize(file);
  } catch (error) {
    shot.classList.remove("scanning");
    state.classList.add("error");
    state.textContent = `Failed: ${(error as Error).message}`;
    return;
  }
  shot.classList.remove("scanning");
  const found = result.teams.length;
  state.textContent = found ? `${found} team${found > 1 ? "s" : ""} in ${result.elapsedMs} ms` : `No team found (${result.elapsedMs} ms)`;

  const bitmap = await createImageBitmap(file);
  const slugMap = await slugs;
  if (!found) {
    panel.append(
      el("p", { class: "empty" }, "No team grid found. The recognizer only knows the 2027 broadcast overlay: a 2×3 grid of Pokémon tiles under a name bar.")
    );
  }
  result.teams.forEach((team, t) => {
    const teamBoxes = drawBoxes(shot, team, t);
    panel.append(renderTeam(team, bitmap, slugMap, teamBoxes));
  });
  bitmap.close();
  foot.append(renderTimings(result), renderActions(result, card));
}

function placeBox(parent: HTMLElement, [x0, y0, x1, y1]: RelativeBox, cls: string, label: string) {
  const box = el("div", { class: `box ${cls}` }, el("span", { class: "tag" }, label));
  Object.assign(box.style, {
    left: `${x0 * 100}%`,
    top: `${y0 * 100}%`,
    width: `${(x1 - x0) * 100}%`,
    height: `${(y1 - y0) * 100}%`,
  });
  parent.append(box);
  return box;
}

function drawBoxes(shot: HTMLElement, team: RecognizedTeam, t: number): HTMLElement[] {
  const prefix = team.side ? team.side[0].toUpperCase() : String(t + 1);
  placeBox(shot, team.nameBox, "name", `${prefix} name`);
  return team.pokemon.map((p, i) => placeBox(shot, p.box, "tile", `${prefix}${i + 1}`));
}

function crop(bitmap: ImageBitmap, [x0, y0, x1, y1]: RelativeBox, size = 112): string {
  const sx = x0 * bitmap.width;
  const sy = y0 * bitmap.height;
  const sw = (x1 - x0) * bitmap.width;
  const sh = (y1 - y0) * bitmap.height;
  const canvas = el("canvas");
  const scale = Math.min(1, size / Math.max(sw, sh));
  canvas.width = Math.max(1, Math.round(sw * scale));
  canvas.height = Math.max(1, Math.round(sh * scale));
  canvas.getContext("2d")?.drawImage(bitmap, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.9);
}

const label = (speciesId: string) =>
  speciesId
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");

function sprite(speciesId: string, slugMap: Map<string, string>): HTMLImageElement {
  const slug = slugMap.get(speciesId);
  const img = el("img", { class: "sprite", alt: label(speciesId) });
  if (!slug) return img;
  // HOME renders cover nearly everything; fall back to the animated sprite.
  const sources = [`${SPRITE}/home/${slug}.png`, `${SPRITE}/ani/${slug}.gif`];
  img.onerror = () => {
    sources.shift();
    if (sources.length) img.src = sources[0];
    else img.onerror = null;
  };
  img.src = sources[0];
  return img;
}

function renderTeam(team: RecognizedTeam, bitmap: ImageBitmap, slugMap: Map<string, string>, boxes: HTMLElement[]) {
  const head = el(
    "div",
    { class: "team-head" },
    el("span", { class: "side" }, team.side ? `${team.side} team` : "Team"),
    el("span", { class: team.player ? "player" : "player unread" }, team.player ?? "name not read")
  );
  if (team.country) head.append(el("span", { class: "country" }, team.country));
  const mons = el("div", { class: "mons" });
  team.pokemon.forEach((p, i) => {
    const row = renderPokemon(p, bitmap, slugMap);
    row.addEventListener("mouseenter", () => boxes[i]?.classList.add("hot"));
    row.addEventListener("mouseleave", () => boxes[i]?.classList.remove("hot"));
    mons.append(row);
  });
  return el("div", { class: "team" }, head, mons);
}

function renderPokemon(p: RecognizedPokemon, bitmap: ImageBitmap, slugMap: Map<string, string>) {
  const pct = Math.round(p.confidence * 100);
  const name = el("div", { class: "name" }, label(p.speciesId));
  if (p.shadow) name.append(el("span", { class: "pill shadow" }, "Shadow"));
  const meter = el("span", { class: "meter" }, el("i", { style: `width:${pct}%` }));
  const conf = el("div", { class: pct < 60 ? "conf low" : "conf" }, meter, `${pct}%`);
  const alts = p.alternatives.length
    ? `next: ${p.alternatives.map((a) => `${label(a.speciesId)} ${Math.round(a.confidence * 100)}%`).join(", ")}`
    : "";
  const detail = el(
    "div",
    { class: "detail", title: "Shadow check: violet fraction of the tile border / learned probe" },
    `${alts}${alts ? " · " : ""}shadow ${p.shadowViolet.toFixed(2)} / ${p.shadowProbe.toFixed(2)}`
  );
  return el(
    "div",
    { class: p.shadow ? "mon is-shadow" : "mon" },
    el("img", { class: "crop", src: crop(bitmap, p.box), alt: "tile from the screenshot" }),
    sprite(p.speciesId, slugMap),
    el("div", {}, name, conf, detail)
  );
}

function renderTimings(result: RecognitionResult) {
  const chips = el("div", { class: "timings" });
  for (const { stage, ms } of result.timings) chips.append(el("span", {}, `${stage} ${ms} ms`));
  chips.append(el("span", { class: "total" }, `total ${result.elapsedMs} ms`));
  return chips;
}

function renderActions(result: RecognitionResult, card: HTMLElement) {
  const json = JSON.stringify({ teams: result.teams }, null, 2);
  const copy = el("button", { type: "button" }, "Copy JSON");
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(json);
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "Copy failed";
    }
    setTimeout(() => (copy.textContent = "Copy JSON"), 1500);
  });
  const toggle = el("button", { type: "button" }, "Show JSON");
  let pre: HTMLElement | null = null;
  toggle.addEventListener("click", () => {
    if (pre) {
      pre.remove();
      pre = null;
      toggle.textContent = "Show JSON";
    } else {
      pre = el("pre", { class: "json" }, json);
      card.append(pre);
      toggle.textContent = "Hide JSON";
    }
  });
  return el("div", { class: "actions" }, toggle, copy);
}
