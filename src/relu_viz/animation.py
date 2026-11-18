"""A training animation with live interpolation and exact tile reconstruction."""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from relu_viz import ELEVATION_COLORS, ELEVATION_LIGHTING, _regions_mesh
from relu_viz.deep import deep_cells


def animation(pts: np.ndarray, snaps: list[dict], path: str = "animation.html") -> Path:
    """Export weights and an initial mesh; a worker builds interpolated tiles."""
    lo, hi = float(pts[:, 2].min()), float(pts[:, 2].max())
    cells, _ = deep_cells(snaps[0])
    mesh = _regions_mesh(cells, lo, hi, color_by="elevation")
    mesh.update(
        visible=True, showscale=True,
        colorbar=dict(title="Mean tile<br>elevation", thickness=15),
        lighting=ELEVATION_LIGHTING,
    )
    fig = go.Figure(mesh)
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", visible=False,
        marker=dict(size=1, color=pts[:, 2], colorscale=ELEVATION_COLORS, cmin=lo, cmax=hi, opacity=0.4),
        hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#080d1a", plot_bgcolor="#101a30",
        font=dict(color="#dbeafe"), height=720, margin=dict(l=10, r=10, t=65, b=10),
        title=f"Step {snaps[0]['step']:,} / {snaps[-1]['step']:,}",
        scene=dict(
            xaxis=dict(title="x", range=[-3, 3]),
            yaxis=dict(title="y", range=[-3, 3]),
            zaxis=dict(title="height", range=[min(-0.15, min(mesh.z)), max(1.35, max(mesh.z))]),
            camera=dict(eye=dict(x=0.25, y=-1.8, z=2.2)),
            aspectmode="manual", aspectratio=dict(x=1.5, y=1.5, z=0.55),
            uirevision="camera",
        ),
    )
    package = Path(__file__).parent
    worker = (package / "animation_worker.js").read_text()
    player = (package / "animation_player.js").read_text()
    timing = (package / "animation_timing.js").read_text()
    renderer = (package / "animation_renderer.js").read_text()
    points = {"xy": pts[:, :2].ravel().tolist(), "startZ": pts[:, 2].tolist()}
    serial = [{key: value.tolist() if isinstance(value, np.ndarray) else
               [v.tolist() for v in value] if key in {"W", "b"} else value
               for key, value in snap.items()} for snap in snaps]
    script = (player.replace("__SNAPSHOTS__", json.dumps(serial))
              .replace("__WORKER__", json.dumps(worker))
              .replace("__TIMING__", timing)
              .replace("__RENDERER__", renderer)
              .replace("__POINTS__", json.dumps(points))
              .replace("__COUNT__", f"{len(pts):,}"))
    html = fig.to_html(div_id="animation", include_plotlyjs="cdn", full_html=True,
                       auto_play=False, post_script=script)
    html = html.replace("<head>", "<head><title>ReLU in Motion · MIG</title>", 1)
    out = Path(path).resolve()
    out.write_text(html)
    return out
