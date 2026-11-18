// Interpolate network weights, then rebuild its exact piecewise-affine tiles.
function blend(a, b, t) {
  return Array.isArray(a) ? a.map((v, i) => blend(v, b[i], t)) : a + t * (b - a);
}

function interpolate(snaps, position) {
  if (snaps.length === 1) return snaps[0];
  const step = snaps[0].step + Math.max(0, Math.min(1, position)) * (snaps.at(-1).step - snaps[0].step);
  let hi = 1;
  while (hi < snaps.length - 1 && snaps[hi].step < step) hi++;
  const a = snaps[hi - 1], b = snaps[hi];
  const t = (step - a.step) / (b.step - a.step);
  return {
    W: blend(a.W, b.W, t), b: blend(a.b, b.b, t),
    a: blend(a.a, b.a, t), c: blend(a.c, b.c, t),
    slope: blend(a.slope || 0, b.slope || 0, t), step,
  };
}

function centroid(poly) {
  let area = 0, cx = 0, cy = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const cross = a[0] * b[1] - b[0] * a[1];
    area += cross; cx += (a[0] + b[0]) * cross; cy += (a[1] + b[1]) * cross;
  }
  return Math.abs(area) <= 2e-9 ? null : [cx / (3 * area), cy / (3 * area)];
}

function clip(poly, p, q, sign, layer) {
  const out = [];
  if (poly.edgeLayers) out.edgeLayers = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const da = p[0] * a[0] + p[1] * a[1] + q;
    const db = p[0] * b[0] + p[1] * b[1] + q;
    const ia = sign * da >= 0, ib = sign * db >= 0;
    if (ia) {
      out.push(a);
      if (out.edgeLayers) out.edgeLayers.push(poly.edgeLayers[i]);
    }
    if (ia !== ib) {
      const t = da / (da - db);
      out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
      // Provenance labels describe the outgoing edge at each vertex. An
      // exit intersection starts the new cut; an entry resumes an old edge.
      if (out.edgeLayers) out.edgeLayers.push(ia ? layer : poly.edgeLayers[i]);
    }
  }
  return out;
}

function meshFor(snap, includeEdges = false) {
  // Flat affine triples avoid allocating a row array for every unit in every tile.
  let pieces = [{poly: [[-3, -3], [3, -3], [3, 3], [-3, 3]], center: [0, 0], C: new Float64Array([1, 0, 0, 0, 1, 0])}];
  if (includeEdges) pieces[0].poly.edgeLayers = [-1, -1, -1, -1];
  for (let l = 0; l < snap.W.length; l++) {
    const next = [], width = snap.W[l].length, previous = snap.W[l][0].length;
    const weights = new Float64Array(width * previous);
    for (let u = 0; u < width; u++) weights.set(snap.W[l][u], u * previous);
    for (const piece of pieces) {
      const P = new Float64Array(width * 3), input = piece.C;
      for (let u = 0; u < width; u++) {
        let gx = 0, gy = 0, c = snap.b[l][u];
        const offset = u * previous;
        for (let i = 0; i < previous; i++) {
          const w = weights[offset + i], j = i * 3;
          gx += w * input[j]; gy += w * input[j + 1]; c += w * input[j + 2];
        }
        P[u * 3] = gx; P[u * 3 + 1] = gy; P[u * 3 + 2] = c;
      }
      let subs = [{poly: piece.poly, center: piece.center}];
      for (let u = 0; u < width; u++) {
        const grown = [], p = [P[u * 3], P[u * 3 + 1]], q = P[u * 3 + 2];
        for (const sub of subs) {
          let min = Infinity, max = -Infinity;
          for (const v of sub.poly) {
            const d = p[0] * v[0] + p[1] * v[1] + q;
            if (d < min) min = d;
            if (d > max) max = d;
          }
          if (min < -1e-10 && max > 1e-10) {
            for (const sign of [1, -1]) {
              const poly = clip(sub.poly, p, q, sign, l), center = centroid(poly);
              if (center) grown.push({poly, center});
            }
          } else grown.push(sub);
        }
        subs = grown;
      }
      for (const sub of subs) {
        const C = new Float64Array(P);
        for (let u = 0; u < width; u++) {
          const i = u * 3;
          if (P[i] * sub.center[0] + P[i + 1] * sub.center[1] + P[i + 2] <= 0) {
            C[i] *= snap.slope; C[i + 1] *= snap.slope; C[i + 2] *= snap.slope;
          }
        }
        next.push({poly: sub.poly, center: sub.center, C});
      }
    }
    pieces = next;
  }
  const mesh = {x: [], y: [], z: [], i: [], j: [], k: [], intensity: []};
  if (includeEdges) { mesh.edges = []; mesh.edgeLayers = []; }
  for (const piece of pieces) {
    let gx = 0, gy = 0, c = snap.c;
    for (let u = 0; u < snap.a.length; u++) {
      const w = snap.a[u], i = u * 3;
      gx += w * piece.C[i]; gy += w * piece.C[i + 1]; c += w * piece.C[i + 2];
    }
    const mean = gx * piece.center[0] + gy * piece.center[1] + c, base = mesh.x.length;
    for (const v of piece.poly) {
      mesh.x.push(v[0]); mesh.y.push(v[1]); mesh.z.push(gx * v[0] + gy * v[1] + c); mesh.intensity.push(mean);
    }
    if (includeEdges) for (let n = 0; n < piece.poly.length; n++) {
      mesh.edges.push(base + n, base + (n + 1) % piece.poly.length);
      mesh.edgeLayers.push(piece.poly.edgeLayers[n]);
    }
    for (let n = 1; n < piece.poly.length - 1; n++) {
      mesh.i.push(base); mesh.j.push(base + n); mesh.k.push(base + n + 1);
    }
  }
  return mesh;
}

if (typeof module !== 'undefined') module.exports = {interpolate, meshFor};
if (typeof self !== 'undefined') {
  let snapshots;
  self.onmessage = ({data}) => {
    if (data.type === 'init') { snapshots = data.snapshots; return; }
    try {
      const frame = (() => {
        const snap = interpolate(snapshots, data.position);
        return {mesh: meshFor(snap), step: snap.step};
      })();
      const mesh = frame.mesh;
      const buffers = [];
      for (const key of Object.keys(mesh)) {
        mesh[key] = ['i', 'j', 'k'].includes(key) ? new Uint32Array(mesh[key]) : new Float64Array(mesh[key]);
        buffers.push(mesh[key].buffer);
      }
      self.postMessage({id: data.id, position: data.position, step: frame.step, preview: !!data.preview, mesh}, buffers);
    } catch (error) { self.postMessage({id: data.id, error: String(error)}); }
  };
}
