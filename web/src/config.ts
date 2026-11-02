/** Shape of models/config.json, written by `recognizer export-web` from the Python modules. */

export interface Style {
  name: string;
  aspect: number;
  gapX: number;
  gapY: number;
}

export interface GridConfig {
  styles: Style[];
  tilePx: number;
  refinePx: number;
  scaleStep: number;
  minTilePx: number;
  minTileFrac: number;
  /** Scales are sampled geometrically from here; those below minTileFrac are skipped. */
  scaleAnchorFrac: number;
  edgeSpan: [number, number];
  strip: number;
  contrastCap: number;
  textureScale: number;
  minScore: number;
  minGridScore: number;
  sizePrior: number;
  secondGridRatio: number;
  pairCandidates: number;
}

export interface SpeciesClass {
  sid: number;
  speciesId: string;
  shadowEligible: boolean;
}

export interface RecognizerConfig {
  version: string;
  grid: GridConfig;
  tileInset: number;
  embed: { inputPx: number; mean: number[]; std: number[]; dim: number };
  species: {
    scale: number;
    topK: number;
    classes: SpeciesClass[];
    /** [representative column, other member columns]: forms that look the same, scored as one. */
    formGroups: Array<[number, number[]]>;
  };
  shadow: {
    band: number;
    colorBand: number;
    colorThreshold: number;
    probeThreshold: number;
    probe: { mean: number[]; scale: number[]; coef: number[]; intercept: number };
  };
  ocr: {
    countryCodes: string[];
    minConfidence: number;
    countryMinConfidence: number;
    nameHeight: number;
    countryHeight: number;
    det: { thresh: number; boxThresh: number; unclipRatio: number; minSide: number };
    rec: { height: number; minWidth: number };
  };
}
