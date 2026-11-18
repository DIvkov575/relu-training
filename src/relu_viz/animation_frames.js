// Offline: each output frame has its own interpolated weights and exact regions.
const fs = require('fs'), path = require('path');
const {interpolate, meshFor} = require('./animation_worker.js');
const [snapPath, folder, countArg, workerArg, workersArg, startArg, endArg, secondArg, stopArg, mode, edgesArg, alphaStartArg, alphaEndArg] = process.argv.slice(2);
const snapshots = JSON.parse(fs.readFileSync(snapPath, 'utf8'));
const count=+countArg, worker=+workerArg, workers=+workersArg;
const start=+(startArg || 0), end=+(endArg || 1), second=+(secondArg || 18);
const stop=+(stopArg || count);
const alphaStart=+(alphaStartArg ?? 0), alphaEnd=+(alphaEndArg ?? 1);
for(let frame=start+worker; frame<stop; frame+=workers) {
  // Bound queued geometry files for long exports.
  const old=path.join(folder,String(frame-64*workers).padStart(5,'0')+'.bin');
  while(frame>=start+64*workers && fs.existsSync(old)) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,50);
  const seconds=(18+second)*frame/(count-1);
  const position=mode==='direct' ? alphaStart+(alphaEnd-alphaStart)*(frame/(count-1)) : seconds <= 18 ? .005*seconds/18 : .005+(end-.005)*(seconds-18)/second;
  const mesh=meshFor(interpolate(snapshots,position), edgesArg==='edges');
  const vertices=new Float32Array(mesh.i.length*3*4);
  let offset=0;
  for(let t=0;t<mesh.i.length;t++)for(const v of [mesh.i[t],mesh.j[t],mesh.k[t]]) {
    vertices[offset++]=mesh.x[v];vertices[offset++]=mesh.y[v];
    vertices[offset++]=mesh.z[v];vertices[offset++]=mesh.intensity[v];
  }
  const target=path.join(folder,String(frame).padStart(5,'0')+'.bin');
  if (mesh.edges) {
    const edges=new Float32Array(mesh.edges.length*4);
    let at=0;
    for(const v of mesh.edges) { edges[at++]=mesh.x[v];edges[at++]=mesh.y[v];edges[at++]=mesh.z[v];edges[at++]=mesh.intensity[v]; }
    fs.writeFileSync(target+'.edges',Buffer.from(edges.buffer));
  }
  fs.writeFileSync(target+'.tmp',Buffer.from(vertices.buffer));fs.renameSync(target+'.tmp',target);
}
