/**
 * Decode a whole team at once: six tiles, six different species (port of recognizer/assign.py).
 */

/**
 * For an (n x m) cost matrix with n <= m, the column assigned to each row, all distinct,
 * minimizing the total cost (Hungarian algorithm with potentials).
 */
export function minCostAssignment(cost: number[][]): number[] {
  const n = cost.length;
  const m = cost[0].length;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const owner = new Int32Array(m + 1); // owner[j] = row (1-based) assigned to column j; 0 = free
  const way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    owner[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(Infinity);
    const used = new Uint8Array(m + 1);
    for (;;) {
      used[j0] = 1;
      const i0 = owner[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[owner[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
      if (owner[j0] === 0) break;
    }
    while (j0) {
      const j1 = way[j0];
      owner[j0] = owner[j1];
      j0 = j1;
    }
  }
  const result = new Array<number>(n).fill(0);
  for (let j = 1; j <= m; j++) if (owner[j]) result[owner[j] - 1] = j - 1;
  return result;
}

/** Class index per tile, all distinct, maximizing the summed score over each tile's top-k. */
export function assignTeam(scores: Float32Array[], topK: number): number[] {
  const candidates = new Set<number>();
  for (const row of scores) {
    const order = Array.from(row.keys()).sort((a, b) => row[b] - row[a]);
    order.slice(0, topK).forEach((c) => candidates.add(c));
  }
  const cols = [...candidates].sort((a, b) => a - b);
  const picks = minCostAssignment(scores.map((row) => cols.map((c) => -row[c])));
  return picks.map((c) => cols[c]);
}
