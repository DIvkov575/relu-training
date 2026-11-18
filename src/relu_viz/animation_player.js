const gd = document.getElementById('animation');
const snapshots = __SNAPSHOTS__;
__TIMING__
__RENDERER__
const targetData = __POINTS__;
const pointHeights = new Float32Array(targetData.startZ);
const renderer = new SurfaceRenderer(gd, {xy:new Float32Array(targetData.xy),height:pointHeights,color:pointHeights}, ['#122b21','#6c39e8','#ff38cf','#33f5f1','#61ff2b','#f6ff35']);
renderer.pointCount=pointHeights.length;
gd._surfaceRenderer=renderer;
document.body.style.cssText = 'margin:0;background:#080d1a;color:#dbeafe;font:14px system-ui';
document.title = 'ReLU in Motion · MIG';
const header = document.createElement('header');
header.style.cssText = 'padding:20px 28px;display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap';
header.innerHTML = '<div><b style="font-size:26px;background:linear-gradient(90deg,#22d3ee,#c084fc,#fb7185,#facc15);background-clip:text;color:transparent">ReLU in Motion · MIG</b><div style="color:#94a3b8;margin-top:6px">__COUNT__ samples · ' + snapshots.length.toLocaleString() + ' saved steps · interpolated weights · exact ReLU planes</div></div><div style="display:flex;gap:18px;align-items:center"><label><input type="checkbox" id="orbit"> Orbit camera</label><label><input type="checkbox" id="target"> Target points</label><a href="/" target="_blank" rel="noopener" style="color:#22d3ee">Explorer ↗</a><a href="video.html" target="_blank" rel="noopener" style="color:#c4b5fd;font-weight:700">Watch 120 FPS video ↗</a></div>';
document.body.prepend(header);
const architecture = snapshots[0].b.map(layer => layer.length).join(' → ');
const parameterCount = snapshots[0].W.reduce((n, layer) => n + layer.reduce((m, row) => m + row.length, 0), 0)
  + snapshots[0].b.reduce((n, layer) => n + layer.length, 0) + snapshots[0].a.length + 1;
const modelLabel = document.createElement('div');
modelLabel.style.cssText = 'color:#c4b5fd;margin-top:5px;font-size:13px';
modelLabel.textContent = architecture + ' · ' + parameterCount.toLocaleString() + ' parameters';
header.firstElementChild.append(modelLabel);
const controls = document.createElement('div');
controls.style.cssText = 'display:flex;gap:16px;align-items:center;padding:18px 28px;background:#101a30;flex-wrap:wrap';
controls.innerHTML = '<button id="play" style="padding:14px 24px;font:600 16px system-ui;color:white;background:#7c3aed;border:0;border-radius:12px">❚❚ Pause</button><button id="restart" style="padding:14px 20px;font:600 16px system-ui;color:white;background:#253554;border:0;border-radius:12px">↺ Restart</button><label>Speed <select id="speed" style="padding:10px;background:#253554;color:white;border-radius:8px"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><input id="timeline" aria-label="Training progress" type="range" min="0" max="10000" value="0" style="flex:1;min-width:220px;accent-color:#c084fc"><span id="status" style="min-width:180px">Starting…</span>';
header.after(controls);
const stepButtons = document.createElement('span');
stepButtons.style.cssText = 'display:flex;gap:8px';
stepButtons.innerHTML = '<button id="prev-step" style="padding:12px 14px;font:600 14px system-ui;color:white;background:#253554;border:0;border-radius:10px">◀ Saved step</button><button id="next-step" style="padding:12px 14px;font:600 14px system-ui;color:white;background:#253554;border:0;border-radius:10px">Saved step ▶</button>';
controls.insertBefore(stepButtons, controls.querySelector('#timeline'));
const worker = new Worker(URL.createObjectURL(new Blob([__WORKER__], {type: 'text/javascript'})));
worker.postMessage({type: 'init', snapshots});
let position = 0, displayed = 0, displayedPreview = false, playing = true, speed = 1;
let started = performance.now() + 700, busy = false, failed = false, version = 0, requestVersion = 0, id = 0;
const timeline = document.getElementById('timeline');
const status = document.getElementById('status');
status.textContent = 'Saved state 1 / ' + snapshots.length + ' · step ' + snapshots[0].step.toLocaleString();
const play = document.getElementById('play');

const label = () => { play.textContent = playing ? '❚❚ Pause' : '▶ Play training'; };
function requestFrame() {
  if (busy || failed || (Math.abs(position - displayed) < 1e-6 && displayedPreview === playing)) return;
  busy = true; requestVersion = version;
  worker.postMessage({id: ++id, position, preview: playing});
}
worker.onmessage = async ({data}) => {
  try {
    if (data.error) throw new Error(data.error);
    if (requestVersion !== version) return;
    const m = data.mesh;
    if (data.preview) {
      renderer.setFrame(m);
      displayed=data.position;displayedPreview=true;
      return;
    }
    const update = {};
    for (const key of Object.keys(m)) update[key] = [m[key]];
    update.intensitymode = 'vertex';
    await Plotly.restyle(gd, update, [0]);
    let lo = -0.15, hi = 1.35;
    for (const z of m.z) { lo = Math.min(lo, z); hi = Math.max(hi, z); }
    const saved = snapshots.findIndex(s => Math.abs(s.step - data.step) < 1e-6);
    const description = saved >= 0 ? 'Saved state ' + (saved + 1) + ' / ' + snapshots.length : 'Interpolated weights';
    const title = 'Step ≈ ' + Math.round(data.step).toLocaleString() + ' / ' + snapshots.at(-1).step.toLocaleString() + ' · ' + description;
    {
      await Plotly.relayout(gd, {'title.text': title, 'scene.zaxis.range': [lo, hi]});
      renderer.hide();
    }
    displayed = data.position; displayedPreview = data.preview;
    status.textContent = description + ' · step ' + Math.round(data.step).toLocaleString();
  } catch (error) {
    failed = true; playing = false; label(); status.textContent = 'Animation error: ' + error.message;
    console.error(error);
  } finally { busy = false; }
};
worker.onerror = event => { failed = true; busy = false; playing = false; label(); status.textContent = 'Animation error: ' + event.message; };
play.onclick = () => {
  if (!playing && position >= 1) { position = 0; version++; }
  playing = !playing; version++;
  started = performance.now() - elapsedAt(position) / speed;
  label();
};
document.getElementById('restart').onclick = () => {
  position = 0; playing = true; version++; started = performance.now(); timeline.value = 0; label();
};
timeline.oninput = () => {
  position = +timeline.value / 10000; version++;
  started = performance.now() - elapsedAt(position) / speed;
  requestFrame();
};
function seekSaved(direction) {
  const first = snapshots[0].step, range = snapshots.at(-1).step - first;
  const current = first + position * range;
  let index;
  if (direction > 0) {
    index = snapshots.findIndex(s => s.step > current + 1e-6);
    if (index < 0) index = snapshots.length - 1;
  } else {
    index = snapshots.length - 1;
    while (index > 0 && snapshots[index].step >= current - 1e-6) index--;
  }
  position = range ? (snapshots[index].step - first) / range : 0;
  playing = false; version++; timeline.value = Math.round(position * 10000); label(); requestFrame();
}
document.getElementById('prev-step').onclick = () => seekSaved(-1);
document.getElementById('next-step').onclick = () => seekSaved(1);
document.getElementById('speed').onchange = function () {
  speed = +this.value; started = performance.now() - elapsedAt(position) / speed;
};
document.getElementById('target').onchange = function () { Plotly.restyle(gd, {visible: this.checked}, [1]); };
let angle = -Math.PI / 2, lastOrbit = 0, orbitBusy = false;
function tick(now) {
  if (playing) {
    position = positionAt((now - started) * speed);
    timeline.value = Math.round(position * 10000);
    if (position >= 1) { playing = false; label(); }
  }
  requestFrame();
  renderer.draw(now, document.getElementById('orbit').checked, document.getElementById('target').checked);
  if (playing) status.textContent='Interpolated weights · step ' + Math.round(snapshots[0].step+position*(snapshots.at(-1).step-snapshots[0].step)).toLocaleString();
  if (!playing && document.getElementById('orbit').checked && now - lastOrbit > 70 && !busy && !orbitBusy) {
    angle += 0.012; lastOrbit = now; orbitBusy = true;
    Plotly.relayout(gd, {'scene.camera.eye': {x: 1.8 * Math.cos(angle), y: 1.8 * Math.sin(angle), z: 2.2}})
      .finally(() => { orbitBusy = false; });
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
