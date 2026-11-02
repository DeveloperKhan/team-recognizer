/** Result types, shared by the pipeline, the worker and consumers of client.ts. No imports. */

/** [x0, y0, x1, y1] as fractions of the image's width and height. */
export type RelativeBox = [number, number, number, number];

export interface RecognizedPokemon {
  /** Dracoviz speciesId, without "_shadow". */
  speciesId: string;
  shadow: boolean;
  confidence: number;
  alternatives: Array<{ speciesId: string; confidence: number }>;
  /** Where the tile is in the image. */
  box: RelativeBox;
  shadowViolet: number;
  shadowProbe: number;
}

export interface RecognizedTeam {
  side: "left" | "right" | null;
  /** The name as read from the overlay (not matched to any player list). */
  player: string | null;
  country: string | null;
  /** Where the name bar is in the image. */
  nameBox: RelativeBox;
  pokemon: RecognizedPokemon[];
}

export type Backend = "webgpu" | "wasm";

export type WorkerRequest =
  | { type: "init"; baseUrl: string }
  | { type: "recognize"; id: number; image: Blob };

export type WorkerResponse =
  | { type: "progress"; loaded: number; total: number }
  | { type: "log"; message: string }
  | { type: "ready"; backend: Backend; version: string }
  | { type: "result"; id: number; teams: RecognizedTeam[]; elapsedMs: number }
  | { type: "error"; id?: number; message: string };
