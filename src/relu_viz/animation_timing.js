// Extend early training, then use a faster constant rate for the remainder.
const OPENING_MS = 18000;
const PLAYBACK_MS = 36000;
const OPENING_POSITION = 0.005;
function positionAt(milliseconds) {
  const t = Math.max(0, Math.min(PLAYBACK_MS, milliseconds));
  return t <= OPENING_MS
    ? OPENING_POSITION * t / OPENING_MS
    : OPENING_POSITION + (1 - OPENING_POSITION) * (t - OPENING_MS) / (PLAYBACK_MS - OPENING_MS);
}

function elapsedAt(position) {
  const p = Math.max(0, Math.min(1, position));
  return p <= OPENING_POSITION
    ? OPENING_MS * p / OPENING_POSITION
    : OPENING_MS + (PLAYBACK_MS - OPENING_MS) * (p - OPENING_POSITION) / (1 - OPENING_POSITION);
}

if (typeof module !== 'undefined') module.exports = {positionAt, elapsedAt};
