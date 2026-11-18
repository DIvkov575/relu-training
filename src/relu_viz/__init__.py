import json
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from shapely import contains_xy, constrained_delaunay_triangles, distance, intersects_xy, points
from shapely.geometry import LineString, Polygon, box
from shapely.ops import split, unary_union

Unit = tuple[tuple[float, float], float]

ELEVATION_COLORS = [
    [
        0.0,
        "#122b21"
    ],
    [
        0.2,
        "#6c39e8"
    ],
    [
        0.4,
        "#ff38cf"
    ],
    [
        0.6,
        "#33f5f1"
    ],
    [
        0.8,
        "#61ff2b"
    ],
    [
        1.0,
        "#f6ff35"
    ]
]
ELEVATION_LIGHTING = dict(ambient=1.0, diffuse=0.25, specular=0.1, roughness=0.8)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def leaky(z: np.ndarray, slope: float = 0.01) -> np.ndarray:
    """Leaky relu. slope=0 is plain relu. Still piecewise linear either way."""
    return np.where(z > 0, z, slope * z)


def _random_params(count: int, scale: float, seed: int | None) -> list[Unit]:
    rng = np.random.default_rng(seed)
    return [
        ((float(a), float(c)), float(d))
        for a, c, d in rng.normal(0.0, scale, size=(count, 3))
    ]


def _axis_lines(extent: float, z_max: float) -> list[go.Scatter3d]:
    """The three coordinate axes, drawn through the origin."""
    spans = [
        ((-extent, extent), (0, 0), (0, 0)),
        ((0, 0), (-extent, extent), (0, 0)),
        ((0, 0), (0, 0), (0, max(z_max, 1e-9))),
    ]
    return [
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="black", width=4),
            showlegend=False,
            hoverinfo="skip",
        )
        for x, y, z in spans
    ]


def _eval(
    w: tuple[float, float], b: float, extent: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-extent, extent, n)
    x1, x2 = np.meshgrid(grid, grid)
    return grid, relu(w[0] * x1 + w[1] * x2 + b)


def surface(
    w: tuple[float, float] = (1.0, 1.0),
    b: float = 0.0,
    extent: float = 3.0,
    n: int = 200,
    axes: bool = True,
    points: np.ndarray | None = None,
) -> go.Figure:
    """relu(w . x + b) over a square patch of the input plane."""
    grid, z = _eval(w, b, extent, n)

    fig = go.Figure(go.Surface(x=grid, y=grid, z=z, colorscale="Viridis"))
    if axes:
        fig.add_traces(_axis_lines(extent, float(z.max())))
    if points is not None:
        fig.add_trace(_point_trace(points))
    fig.update_layout(
        title=f"relu({w[0]:.2f}x1 + {w[1]:.2f}x2 + {b:+.2f})",
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="y"),
    )
    return fig


def random_panel(
    count: int = 6,
    cols: int = 3,
    scale: float = 1.5,
    extent: float = 3.0,
    n: int = 80,
    seed: int | None = 0,
) -> go.Figure:
    """A grid of relu surfaces with weights and biases drawn from N(0, scale)."""
    rows = -(-count // cols)
    params = _random_params(count, scale, seed)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"}] * cols for _ in range(rows)],
        subplot_titles=[
            f"relu({w[0]:.2f}x1 {w[1]:+.2f}x2 {b:+.2f})" for w, b in params
        ],
    )

    for i, (w, b) in enumerate(params):
        grid, z = _eval(w, b, extent, n)
        row, col = divmod(i, cols)
        fig.add_trace(
            go.Surface(x=grid, y=grid, z=z, colorscale="Viridis", showscale=False),
            row=row + 1,
            col=col + 1,
        )
        for line in _axis_lines(extent, float(z.max())):
            fig.add_trace(line, row=row + 1, col=col + 1)

    fig.update_layout(title=f"{count} random ReLU half-planes", height=380 * rows)
    fig.update_scenes(xaxis_title="x1", yaxis_title="x2", zaxis_title="y")
    fig.update_annotations(font_size=11)
    return fig


def random_sum(
    count: int = 6,
    scale: float = 1.5,
    extent: float = 3.0,
    n: int = 200,
    seed: int | None = 0,
    show_units: bool = True,
    points: np.ndarray | None = None,
) -> go.Figure:
    """Sum of `count` random relu units, superimposed on the units themselves."""
    params = _random_params(count, scale, seed)

    grid = np.linspace(-extent, extent, n)
    total = np.zeros((n, n))
    units = []
    for w, b in params:
        _, z = _eval(w, b, extent, n)
        total += z
        units.append(z)

    fig = go.Figure()
    if show_units:
        for z in units:
            fig.add_trace(
                go.Surface(
                    x=grid,
                    y=grid,
                    z=z,
                    colorscale="Greys",
                    opacity=0.25,
                    showscale=False,
                    hoverinfo="skip",
                )
            )
    fig.add_trace(go.Surface(x=grid, y=grid, z=total, colorscale="Viridis"))
    fig.add_traces(_axis_lines(extent, float(total.max())))
    if points is not None:
        fig.add_trace(_point_trace(points))
    fig.update_layout(
        title=f"sum of {count} random relu units",
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="y"),
    )
    return fig


def _fold_segment(w: tuple[float, float], b: float, extent: float) -> LineString | None:
    """The line w . x + b = 0, as a segment long enough to cross the whole patch."""
    wx, wy = w
    norm2 = wx * wx + wy * wy
    if norm2 < 1e-12:
        return None
    norm = np.sqrt(norm2)
    foot = (-b * wx / norm2, -b * wy / norm2)
    step = (-wy / norm * 4 * extent, wx / norm * 4 * extent)
    return LineString(
        [
            (foot[0] - step[0], foot[1] - step[1]),
            (foot[0] + step[0], foot[1] + step[1]),
        ]
    )


class Region:
    """One cell of the fold-line arrangement, where the sum is a single plane."""

    def __init__(
        self,
        cell: Polygon,
        params: list[Unit],
        out: np.ndarray | None = None,
        out_bias: float = 0.0,
    ) -> None:
        self.cell = cell
        probe = cell.representative_point()
        self.pattern = tuple(
            wx * probe.x + wy * probe.y + b > 0 for (wx, wy), b in params
        )
        a = np.ones(len(params)) if out is None else np.asarray(out, dtype=float)
        live = [(u, ai) for u, ai, on in zip(params, a, self.pattern) if on]
        self.gx = sum(ai * w[0] for (w, _), ai in live)
        self.gy = sum(ai * w[1] for (w, _), ai in live)
        self.c = sum(ai * b for (_, b), ai in live) + out_bias

    @property
    def slope(self) -> float:
        return float(np.hypot(self.gx, self.gy))

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.gx * x + self.gy * y + self.c


def linear_regions(
    params: list[Unit],
    extent: float = 3.0,
    min_area: float = 1e-9,
    out: np.ndarray | None = None,
    out_bias: float = 0.0,
) -> list[Region]:
    """Exact cells of the arrangement, by cutting the patch with one fold at a time."""
    cells: list[Polygon] = [box(-extent, -extent, extent, extent)]
    for w, b in params:
        seg = _fold_segment(w, b, extent)
        if seg is None:
            continue
        nxt: list[Polygon] = []
        for cell in cells:
            pieces = split(cell, seg).geoms if seg.intersects(cell) else [cell]
            nxt.extend(p for p in pieces if p.area > min_area)
        cells = nxt
    return [Region(cell, params, out, out_bias) for cell in cells]


def region_figure(
    count: int = 6,
    scale: float = 1.5,
    extent: float = 3.0,
    seed: int | None = 0,
    show_edges: bool = True,
    points: np.ndarray | None = None,
    params: list[Unit] | None = None,
    out: np.ndarray | None = None,
    out_bias: float = 0.0,
    title: str | None = None,
) -> go.Figure:
    """A relu network's surface as exact flat panels, coloured by local gradient."""
    if params is None:
        params = _random_params(count, scale, seed)
    count = len(params)
    regions = linear_regions(params, extent, out=out, out_bias=out_bias)
    slopes = [r.slope for r in regions]
    lo, hi = min(slopes), max(slopes)

    fig = go.Figure()
    z_max = 0.0
    for r in regions:
        x, y = np.array(r.cell.exterior.coords[:-1]).T
        z = r.height(x, y)
        z_max = max(z_max, float(z.max()))
        t = 0.0 if hi == lo else (r.slope - lo) / (hi - lo)
        # cells are intersections of half-planes, so convex: a fan triangulation works
        fan = np.arange(1, len(x) - 1)
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=np.zeros_like(fan),
                j=fan,
                k=fan + 1,
                color=sample_colorscale("Viridis", float(t))[0],
                flatshading=True,
                hovertemplate=(
                    f"grad ({r.gx:.2f}, {r.gy:.2f})<br>"
                    f"{sum(r.pattern)}/{count} units on<extra></extra>"
                ),
            )
        )
        if show_edges:
            loop = np.append(np.arange(len(x)), 0)
            fig.add_trace(
                go.Scatter3d(
                    x=x[loop],
                    y=y[loop],
                    z=z[loop],
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.add_traces(_axis_lines(extent, z_max))
    if points is not None:
        fig.add_trace(_point_trace(points))
    fig.update_layout(
        title=title or f"{len(regions)} linear regions from {count} relu units",
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="y"),
    )
    return fig


def cylinder_points(
    count: int = 2000,
    radius: float = 0.3,
    height: float = 24.0,
    noise: float = 0.015,
    filled: bool = False,
    seed: int | None = 0,
) -> np.ndarray:
    """(count, 3) points on a cylinder of the given radius, rising along +z from 0."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2 * np.pi, count)
    z = rng.uniform(0.0, height, count)
    # sqrt keeps a filled cylinder uniform by area rather than crowding the axis
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, count)) if filled else np.full(count, radius)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return pts + rng.normal(0.0, noise, pts.shape) if noise else pts


def rhombus_points(
    count: int = 2000,
    half_x: float = 0.5,
    half_y: float = 0.25,
    height: float = 24.0,
    noise: float = 0.015,
    filled: bool = False,
    seed: int | None = 0,
) -> np.ndarray:
    """(count, 3) points on a rhombus prism rising along +z from 0.

    Cross section is |x| / half_x + |y| / half_y = 1. All four sides have equal
    length, so picking an edge uniformly is uniform along the perimeter.
    """
    rng = np.random.default_rng(seed)
    verts = np.array(
        [(half_x, 0.0), (0.0, half_y), (-half_x, 0.0), (0.0, -half_y)]
    )
    edge = rng.integers(0, 4, count)
    t = rng.uniform(0.0, 1.0, count)
    xy = verts[edge] + t[:, None] * (verts[(edge + 1) % 4] - verts[edge])
    if filled:
        # radial sqrt keeps a filled convex shape uniform by area
        xy = xy * np.sqrt(rng.uniform(0.0, 1.0, count))[:, None]
    pts = np.column_stack([xy, rng.uniform(0.0, height, count)])
    return pts + rng.normal(0.0, noise, pts.shape) if noise else pts


def triangle_points(
    count: int = 2000,
    radius: float = 0.5,
    height: float = 24.0,
    rotate: float = 90.0,
    noise: float = 0.015,
    filled: bool = False,
    seed: int | None = 0,
    axis: str = "z",
) -> np.ndarray:
    """(count, 3) points on an equilateral triangular prism.

    `radius` is the circumradius of the triangle and `rotate` turns it in its own
    plane, in degrees, with a vertex along +x at rotate=0. The three sides are
    equal, so picking an edge uniformly is uniform along the perimeter.

    `axis` is the extrusion direction. "z" stands the prism up, rising from 0, and
    the target height is then independent of the input. "x" or "y" tips it on its
    side and centres it, which makes height a real function of the input.
    """
    rng = np.random.default_rng(seed)
    angles = np.deg2rad(np.array([0.0, 120.0, 240.0]) + rotate)
    verts = radius * np.column_stack([np.cos(angles), np.sin(angles)])

    edge = rng.integers(0, 3, count)
    t = rng.uniform(0.0, 1.0, count)
    cross = verts[edge] + t[:, None] * (verts[(edge + 1) % 3] - verts[edge])
    if filled:
        cross = cross * np.sqrt(rng.uniform(0.0, 1.0, count))[:, None]
    u, v = cross.T
    run = rng.uniform(0.0, height, count)

    if axis == "z":
        pts = np.column_stack([u, v, run])
    elif axis == "x":
        pts = np.column_stack([run - height / 2, u, v + radius])
    elif axis == "y":
        pts = np.column_stack([u, run - height / 2, v + radius])
    else:
        raise ValueError(f"axis must be x, y or z, got {axis!r}")
    return pts + rng.normal(0.0, noise, pts.shape) if noise else pts


def trapezoid_points(
    count: int = 2000,
    bottom_width: float = 3.0,
    top_width: float = 1.5,
    height: float = 2.5,
    length: float = 5.0,
    noise: float = 0.015,
    seed: int | None = 0,
) -> np.ndarray:
    """Points on the roof and sloped sides of an isosceles trapezoidal prism.

    The prism runs along x, centred at zero. Its cross section in (y, z)
    has a wide base at z=0 and a narrower, centred top at z=height.
    The bottom is omitted so height is a single-valued function of (x, y).
    Sampling is uniform by area across the three remaining lateral faces.
    """
    if min(bottom_width, top_width, height, length) <= 0:
        raise ValueError("Prism dimensions must be positive")
    rng = np.random.default_rng(seed)
    verts = np.array([
        (-bottom_width / 2, 0.0), (bottom_width / 2, 0.0),
        (top_width / 2, height), (-top_width / 2, height),
    ])
    sides = np.roll(verts, -1, axis=0) - verts
    lengths = np.linalg.norm(sides, axis=1)
    lengths[0] = 0.0  # exclude the bottom face
    edge = rng.choice(4, count, p=lengths / lengths.sum())
    cross = verts[edge] + rng.uniform(size=(count, 1)) * sides[edge]
    pts = np.column_stack([rng.uniform(-length / 2, length / 2, count), cross])
    return pts + rng.normal(0.0, noise, pts.shape) if noise else pts


def m_points(
    count: int = 2000,
    width: float = 3.0,
    height: float = 2.5,
    valley: float = 0.5,
    length: float = 5.0,
    noise: float = 0.015,
    seed: int | None = 0,
) -> np.ndarray:
    """An open M profile in (y, z), extruded along x, without a bottom.

    Four sloped faces join two peaks and a central valley. The outer legs
    slope so each input location has exactly one target height.
    """
    if min(width, height, length) <= 0 or not 0 <= valley < height:
        raise ValueError("Dimensions must be positive and 0 <= valley < height")
    rng = np.random.default_rng(seed)
    verts = np.array([
        (-width / 2, 0.0), (-width / 4, height),
        (0.0, valley), (width / 4, height), (width / 2, 0.0),
    ])
    sides = np.diff(verts, axis=0)
    lengths = np.linalg.norm(sides, axis=1)
    edge = rng.choice(4, count, p=lengths / lengths.sum())
    cross = verts[edge] + rng.uniform(size=(count, 1)) * sides[edge]
    pts = np.column_stack([rng.uniform(-length / 2, length / 2, count), cross])
    return pts + rng.normal(0.0, noise, pts.shape) if noise else pts


def block_m_points(
    count: int = 5000,
    width: float = 3.0,
    height: float = 3.0,
    depth: float = 1.2,
    noise: float = 0.015,
    seed: int | None = 0,
) -> np.ndarray:
    """Surface points of a bold, upright block M in (x, z), extruded along y.

    Vertical stems and thick diagonal strokes form a closed, concave outline.
    Both letter faces and all extrusion faces are sampled uniformly by area.
    This orientation is a quarter turn from the M roof's (y, z) profile.
    """
    if min(width, height, depth) <= 0:
        raise ValueError("Letter dimensions must be positive")
    rng = np.random.default_rng(seed)
    verts = np.array([
        (-.5, 0), (-.5, 1), (-.28, 1), (0, .48),
        (.28, 1), (.5, 1), (.5, 0), (.28, 0),
        (.28, .62), (0, .10), (-.28, .62), (-.28, 0),
    ]) * [width, height]
    outline = Polygon(verts)
    sides = np.roll(verts, -1, axis=0) - verts
    lengths = np.linalg.norm(sides, axis=1)
    areas = np.r_[outline.area, outline.area, lengths * depth]
    face = rng.choice(len(areas), count, p=areas / areas.sum())
    pts = np.empty((count, 3))

    # Rejection sampling preserves the cutouts between the M's strokes.
    indices = np.flatnonzero(face < 2)
    remaining = indices
    while len(remaining):
        cross = rng.uniform([-width / 2, 0], [width / 2, height], (len(remaining), 2))
        keep = contains_xy(outline, cross[:, 0], cross[:, 1])
        pts[remaining[keep], 0] = cross[keep, 0]
        pts[remaining[keep], 2] = cross[keep, 1]
        remaining = remaining[~keep]
    pts[indices, 1] = np.where(face[indices] == 0, -depth / 2, depth / 2)

    indices = np.flatnonzero(face >= 2)
    edge = face[indices] - 2
    cross = verts[edge] + rng.uniform(size=(len(indices), 1)) * sides[edge]
    pts[indices, 0] = cross[:, 0]
    pts[indices, 2] = cross[:, 1]
    pts[indices, 1] = rng.uniform(-depth / 2, depth / 2, len(indices))
    return pts + rng.normal(0.0, noise, pts.shape) if noise else pts


def m_relief_triangles(height: float = 1.2, flare: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """A horizontal block M with a wider base and straight bevelled sides.

    Rotating the upright letter into (x, y) makes its roof and flared sides a
    single-valued height field. Triangulation keeps the target piecewise linear.
    Group IDs distinguish the roof, bevel and surrounding ground.
    """
    if height <= 0 or flare <= 0:
        raise ValueError("Height and flare must be positive")
    verts = np.array([
        (-.5, 0), (-.5, 1), (-.28, 1), (0, .48),
        (.28, 1), (.5, 1), (.5, 0), (.28, 0),
        (.28, .62), (0, .10), (-.28, .62), (-.28, 0),
    ]) * 3 - [0, 1.5]
    return _relief_triangles(Polygon(verts), height, flare)


def mig_relief_triangles(height: float = 1.2, flare: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal bold MIG lettering, with separate outward-flared bases."""
    m = np.array([
        (0, 0), (0, 1), (.22, 1), (.5, .48),
        (.78, 1), (1, 1), (1, 0), (.78, 0),
        (.78, .62), (.5, .10), (.22, .62), (.22, 0),
    ]) * [1.5, 2.7] + [-2.35, -1.35]
    # Serif I, with a wide cap and foot and a narrower central stem.
    i = np.array([
        (0, 0), (1, 0), (1, .18), (.66, .18), (.66, .82),
        (1, .82), (1, 1), (0, 1), (0, .82), (.34, .82),
        (.34, .18), (0, .18),
    ]) * [.8, 2.7] + [-.4, -1.35]
    # Angular G with an open upper-right aperture and an inward crossbar.
    g = np.array([
        (1, 1), (.18, 1), (0, .82), (0, .18), (.18, 0),
        (1, 0), (1, .55), (.52, .55), (.52, .34),
        (.78, .34), (.78, .22), (.28, .22), (.22, .28),
        (.22, .72), (.28, .78), (1, .78),
    ]) * [1.5, 2.7] + [.85, -1.35]
    return _relief_triangles(unary_union([Polygon(m), Polygon(i), Polygon(g)]), height, flare)


def dima_relief_triangles(height: float = 1.2, flare: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Bold angular DIMA roofs, with open D/A counters and flared bases.

    The lettering spans 5.28 by 2.5 input units. A 0.26-unit gap keeps
    neighbouring bases separate at the default 0.12-unit flare.
    """
    def letter(vertices, width, left, holes=()):
        scale = np.array([width, 2.5])
        offset = np.array([left, -1.25])
        return Polygon(np.asarray(vertices) * scale + offset,
                       [np.asarray(hole) * scale + offset for hole in holes])

    d = letter([
        (0, 0), (.7, 0), (1, .2), (1, .8), (.7, 1), (0, 1),
    ], 1.15, -2.64, holes=[[
        (.3, .22), (.64, .22), (.74, .3), (.74, .7),
        (.64, .78), (.3, .78),
    ]])
    i = letter([
        (0, 0), (1, 0), (1, .18), (.66, .18), (.66, .82),
        (1, .82), (1, 1), (0, 1), (0, .82), (.34, .82),
        (.34, .18), (0, .18),
    ], .65, -1.23)
    m = letter([
        (0, 0), (0, 1), (.22, 1), (.5, .48),
        (.78, 1), (1, 1), (1, 0), (.78, 0),
        (.78, .62), (.5, .10), (.22, .62), (.22, 0),
    ], 1.35, -.32)
    a = letter([
        (0, 0), (0, .08), (.38, 1), (.62, 1), (1, .08), (1, 0),
        (.74, 0), (.64, .24), (.36, .24), (.26, 0),
    ], 1.35, 1.29, holes=[[
        (.32, .42), (.68, .42), (.5, .83),
    ]])
    return _relief_triangles(unary_union([d, i, m, a]), height, flare)


def dima_relief_heights(xy: np.ndarray) -> np.ndarray:
    """Evaluate the DIMA roof, bevels and surrounding ground exactly."""
    return _relief_heights(xy, dima_relief_triangles()[0])


def dima_relief_points(count: int = 40000, seed: int | None = 0) -> np.ndarray:
    """Samples balanced across the four roofs, bevels and ground."""
    return _sample_relief(*dima_relief_triangles(), count, seed)


def _relief_triangles(roof, height: float, flare: float) -> tuple[np.ndarray, np.ndarray]:
    if height <= 0 or flare <= 0:
        raise ValueError("Height and flare must be positive")
    base = roof.buffer(flare, join_style="mitre")
    parts = [roof, base.difference(roof), box(-3, -3, 3, 3).difference(base)]
    triangles, groups = [], []
    for group, part in enumerate(parts):
        for triangle in constrained_delaunay_triangles(part).geoms:
            xy = np.asarray(triangle.exterior.coords)[:3]
            z = np.full(3, height) if group == 0 else np.zeros(3)
            if group == 1:
                # Vertices on the inner ring sit at roof height; outer ring at zero.
                z = np.where(distance(roof, points(xy)) < 1e-10, height, 0.0)
            triangles.append(np.column_stack([xy, z]))
            groups.append(group)
    return np.asarray(triangles), np.asarray(groups)


def m_relief_heights(xy: np.ndarray) -> np.ndarray:
    """Evaluate the exact flared M target, including its surrounding ground."""
    return _relief_heights(xy, m_relief_triangles()[0])


def mig_relief_heights(xy: np.ndarray) -> np.ndarray:
    return _relief_heights(xy, mig_relief_triangles()[0])


def _relief_heights(xy: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    z = np.full(len(xy), np.nan)
    for tri in triangles:
        mask = intersects_xy(Polygon(tri[:, :2]), xy[:, 0], xy[:, 1])
        plane = np.linalg.solve(np.column_stack([tri[:, :2], np.ones(3)]), tri[:, 2])
        z[mask] = np.column_stack([xy[mask], np.ones(mask.sum())]) @ plane
    return z


def m_relief_points(count: int = 4000, seed: int | None = 0) -> np.ndarray:
    """Surface samples balanced across the M roof, flared sides and ground."""
    return _sample_relief(*m_relief_triangles(), count, seed)


def mig_relief_points(count: int = 20000, seed: int | None = 0) -> np.ndarray:
    """Dense samples of all three letters, bevels and intervening ground."""
    return _sample_relief(*mig_relief_triangles(), count, seed)


def _sample_relief(triangles: np.ndarray, groups: np.ndarray, count: int, seed: int | None) -> np.ndarray:
    a, b = triangles[:, 1, :2] - triangles[:, 0, :2], triangles[:, 2, :2] - triangles[:, 0, :2]
    areas = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    weights = areas.copy()
    for group, mass in enumerate([0.4, 0.4, 0.2]):
        mask = groups == group
        weights[mask] *= mass / weights[mask].sum()
    rng = np.random.default_rng(seed)
    tri = triangles[rng.choice(len(triangles), count, p=weights)]
    u = np.sqrt(rng.uniform(size=(count, 1)))
    v = rng.uniform(size=(count, 1))
    return (1 - u) * tri[:, 0] + u * (1 - v) * tri[:, 1] + u * v * tri[:, 2]


def _point_trace(pts: np.ndarray, size: float = 2.0) -> go.Scatter3d:
    x, y, z = pts.T
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=min(size, 1.2) if len(pts) > 5000 else size,
            color="crimson", opacity=0.5 if len(pts) > 5000 else 1.0,
        ),
        name="data",
        hovertemplate="(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
    )


def points_figure(pts: np.ndarray, size: float = 2.0) -> go.Figure:
    """Scatter a point cloud on its own, with a true-to-scale aspect ratio."""
    fig = go.Figure(_point_trace(pts, size))
    fig.update_layout(
        title=f"{len(pts)} points",
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="y",
            aspectmode="data",
        ),
    )
    return fig


class Net:
    """2 -> hidden relu -> 1 linear, matching the topology plotted above."""

    def __init__(self, hidden: int = 6, seed: int | None = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 1.0, (hidden, 2))
        self.b = np.zeros(hidden)
        self.a = rng.normal(0.0, 1.0 / np.sqrt(hidden), hidden)
        self.c = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return relu(x @ self.W.T + self.b) @ self.a + self.c

    @property
    def units(self) -> list[Unit]:
        """Hidden layer as (w, b) pairs, for the region machinery."""
        return [((float(w[0]), float(w[1])), float(b)) for w, b in zip(self.W, self.b)]


def train(
    pts: np.ndarray,
    hidden: int = 6,
    steps: int = 4000,
    lr: float = 0.02,
    seed: int | None = 0,
    snapshots: int = 12,
) -> tuple[Net, list[float], list[dict]]:
    """Fit y = net(x1, x2) by Adam on RMSE.

    Returns the net, the loss history, and `snapshots` copies of the weights taken
    at even intervals (first is the initialisation, last is the trained net).
    """
    net = Net(hidden, seed)
    net.c = float(pts[:, 2].mean())
    x, y = pts[:, :2], pts[:, 2]
    n = len(pts)

    slots = ["W", "b", "a", "c"]
    m = {k: np.zeros_like(np.atleast_1d(getattr(net, k)), dtype=float) for k in slots}
    v = {k: np.zeros_like(np.atleast_1d(getattr(net, k)), dtype=float) for k in slots}
    history: list[float] = []
    snaps: list[dict] = []
    marks = set(np.linspace(1, steps - 1, snapshots - 1).round().astype(int).tolist())

    for step in range(1, steps + 1):
        pre = x @ net.W.T + net.b
        act = relu(pre)
        err = act @ net.a + net.c - y
        rmse = float(np.sqrt(np.mean(err**2)))
        history.append(rmse)
        if step in marks:
            snaps.append(
                dict(step=step, rmse=rmse, W=net.W.copy(), b=net.b.copy(),
                     a=net.a.copy(), c=net.c)
            )

        # d(rmse)/dp = (1 / (2 * rmse)) * d(mse)/dp
        chain = err / (n * max(rmse, 1e-12))
        grad = {
            "a": act.T @ chain,
            "c": np.atleast_1d(chain.sum()),
            "b": ((chain[:, None] * net.a) * (pre > 0)).sum(axis=0),
            "W": ((chain[:, None] * net.a) * (pre > 0)).T @ x,
        }
        for k in slots:
            m[k] = 0.9 * m[k] + 0.1 * grad[k]
            v[k] = 0.999 * v[k] + 0.001 * grad[k] ** 2
            mhat = m[k] / (1 - 0.9**step)
            vhat = v[k] / (1 - 0.999**step)
            upd = lr * mhat / (np.sqrt(vhat) + 1e-8)
            if k == "c":
                net.c -= float(upd[0])
            else:
                setattr(net, k, getattr(net, k) - upd)

    # the loop snapshots weights as they were when `rmse` was measured, so the
    # trained net itself is one update newer than the last of them
    final = float(np.sqrt(np.mean((net(x) - y) ** 2)))
    snaps.append(
        dict(step=steps, rmse=final, W=net.W.copy(), b=net.b.copy(),
             a=net.a.copy(), c=net.c)
    )
    return net, history, snaps


def _regions_mesh(
    regions: list[Region],
    cmin: float,
    cmax: float,
    color: str | None = None,
    opacity: float = 1.0,
    color_by: str = "slope",
) -> go.Mesh3d:
    """One flat color per region, based on slope or area-mean elevation.

    Each region is an affine plane, so its area-mean height is exactly its
    height at the polygon centroid (not the average of its boundary vertices).
    """
    if color_by not in {"slope", "elevation"}:
        raise ValueError("Color must be based on slope or elevation")
    xs, ys, zs, ii, jj, kk, inten = [], [], [], [], [], [], []
    for r in regions:
        x, y = np.array(r.cell.exterior.coords[:-1]).T
        z = r.height(x, y)
        base = len(xs)
        fan = np.arange(1, len(x) - 1)
        ii.extend([base] * len(fan))
        jj.extend(base + fan)
        kk.extend(base + fan + 1)
        xs.extend(x)
        ys.extend(y)
        zs.extend(z)
        center = r.cell.centroid
        value = float(r.height(center.x, center.y)) if color_by == "elevation" else r.slope
        inten.extend([value] * len(x))
    shading = (
        dict(color=color)
        if color
        else dict(
            intensity=inten,
            intensitymode="vertex",
            colorscale=ELEVATION_COLORS if color_by == "elevation" else "Viridis",
            cmin=cmin,
            cmax=cmax,
        )
    )
    if color_by == "elevation":
        shading["lighting"] = ELEVATION_LIGHTING
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
        opacity=opacity, showscale=False, flatshading=True,
        visible=False, hoverinfo="skip", **shading,
    )


def _regions_edges(regions: list[Region]) -> go.Scatter3d:
    """Every cell boundary as one trace, loops separated by gaps."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for r in regions:
        x, y = np.array(r.cell.exterior.coords[:-1]).T
        z = r.height(x, y)
        loop = np.append(np.arange(len(x)), 0)
        xs.extend([*x[loop], None])
        ys.extend([*y[loop], None])
        zs.extend([*z[loop], None])
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color="black", width=2),
        showlegend=False, hoverinfo="skip", visible=False,
    )


_EXPLORER_JS = """
const meta = __META__;
const gd = document.getElementById('plot');
const bar = document.createElement('div');
bar.style.cssText = 'font:13px system-ui;padding:10px 16px';

let boxes = '<label style="margin-right:18px"><input type="checkbox" id="ck-net"' +
            ' checked> <b>whole net</b></label>';
for (let u = 0; u < meta.units; u++) {
  boxes += '<label style="margin-right:14px;white-space:nowrap">' +
           '<input type="checkbox" class="ck-u" data-u="' + u + '"> ' +
           '<span style="color:' + meta.colors[u] + '">&#9632;</span> unit ' + u +
           ' <span id="aw-' + u + '" style="color:#666"></span></label>';
}
bar.innerHTML =
  '<div style="margin-bottom:8px">snapshot ' +
  '<input id="ctl-s" type="range" min="0" max="' + (meta.steps.length - 1) +
  '" value="' + (meta.steps.length - 1) + '" step="1" style="width:320px">' +
  ' <b id="lbl-s"></b></div><div>' + boxes + '</div>';
gd.parentNode.insertBefore(bar, gd);

function apply() {
  const s = +document.getElementById('ctl-s').value;
  const netOn = document.getElementById('ck-net').checked;
  const unitOn = new Array(meta.units).fill(false);
  document.querySelectorAll('.ck-u').forEach(function (el) {
    unitOn[+el.dataset.u] = el.checked;
  });

  document.getElementById('lbl-s').textContent =
      'step ' + meta.steps[s] + '   RMSE ' + meta.rmse[s].toFixed(4);
  for (let u = 0; u < meta.units; u++) {
    document.getElementById('aw-' + u).textContent =
        'a=' + meta.a[s][u].toFixed(2);
  }

  const shown = function (f) { return f < 0 ? netOn : unitOn[f]; };
  const vis = meta.always.map(function (keep, i) {
    return keep || (meta.snapOf[i] === s && shown(meta.focusOf[i]));
  });
  Plotly.restyle(gd, {visible: vis});

  let lo = meta.pz[0], hi = meta.pz[1];
  for (let f = -1; f < meta.units; f++) {
    if (!shown(f)) continue;
    lo = Math.min(lo, meta.zmin[s + ',' + f]);
    hi = Math.max(hi, meta.zmax[s + ',' + f]);
  }
  Plotly.relayout(gd, {'scene.zaxis.range': [lo - 1, hi + 1]});
  Plotly.restyle(gd, {x: [[meta.steps[s]]], y: [[meta.rmse[s]]]}, [meta.cursor]);
}
document.getElementById('ctl-s').addEventListener('input', apply);
document.getElementById('ck-net').addEventListener('change', apply);
document.querySelectorAll('.ck-u').forEach(function (el) {
  el.addEventListener('change', apply);
});
apply();
"""


UNIT_COLORS = [
    "#e4572e", "#17bebb", "#ffc914", "#2e86ab", "#a05195", "#76b041",
    "#d45087", "#665191",
]


def explorer_figure(
    pts: np.ndarray,
    snaps: list[dict],
    history: list[float],
    extent: float = 3.0,
) -> tuple[go.Figure, dict]:
    """The explorer figure plus the metadata its controls need.

    Trace groups are built for every (snapshot, view) pair, all hidden. A view is
    either the whole net or one hidden unit on its own, that unit's contribution
    a_i * relu(w_i . x + b_i) without the output bias.
    """
    units = len(snaps[0]["a"])
    pz = (float(pts[:, 2].min()), float(pts[:, 2].max()))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.74, 0.26],
        subplot_titles=("surface and data", "RMSE"),
    )
    always: list[bool] = []
    snap_of: list[int] = []
    focus_of: list[int] = []
    zmin: dict[str, float] = {}
    zmax: dict[str, float] = {}

    views = {
        (si, f): _view_regions(s, f, extent)
        for si, s in enumerate(snaps)
        for f in range(-1, units)
    }
    slopes = [r.slope for regions in views.values() for r in regions]
    cmin, cmax = min(slopes), max(slopes)

    for (si, f), regions in views.items():
        zs = np.concatenate(
            [r.height(*np.array(r.cell.exterior.coords[:-1]).T) for r in regions]
        )
        zmin[f"{si},{f}"] = float(zs.min())
        zmax[f"{si},{f}"] = float(zs.max())
        mesh = _regions_mesh(
            regions,
            cmin,
            cmax,
            color=None if f < 0 else UNIT_COLORS[f % len(UNIT_COLORS)],
            opacity=1.0 if f < 0 else 0.85,
        )
        for trace in (mesh, _regions_edges(regions)):
            fig.add_trace(trace, row=1, col=1)
            always.append(False)
            snap_of.append(si)
            focus_of.append(f)

    for trace in (_point_trace(pts), *_axis_lines(extent, pz[1])):
        fig.add_trace(trace, row=1, col=1)
        always.append(True)
        snap_of.append(-99)
        focus_of.append(-99)

    fig.add_trace(
        go.Scatter(
            y=history,
            x=np.arange(1, len(history) + 1),
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    always.append(True)
    snap_of.append(-99)
    focus_of.append(-99)

    cursor = len(always)
    fig.add_trace(
        go.Scatter(
            x=[snaps[-1]["step"]],
            y=[snaps[-1]["rmse"]],
            mode="markers",
            marker=dict(size=10, color="crimson"),
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    always.append(True)
    snap_of.append(-99)
    focus_of.append(-99)

    fig.update_layout(
        title="relu net fitted to the point column",
        showlegend=False,
        height=740,
        margin=dict(t=70, b=10, l=10, r=10),
    )
    fig.update_scenes(xaxis_title="x1", yaxis_title="x2", zaxis_title="y")
    fig.update_xaxes(title_text="step", row=1, col=2)
    fig.update_yaxes(title_text="RMSE", row=1, col=2)

    meta = dict(
        steps=[int(s["step"]) for s in snaps],
        rmse=[float(s["rmse"]) for s in snaps],
        a=[[float(v) for v in s["a"]] for s in snaps],
        units=units,
        colors=[UNIT_COLORS[u % len(UNIT_COLORS)] for u in range(units)],
        always=always,
        snapOf=snap_of,
        focusOf=focus_of,
        zmin=zmin,
        zmax=zmax,
        pz=list(pz),
        cursor=cursor,
    )
    return fig, meta


def explorer(
    pts: np.ndarray,
    snaps: list[dict],
    history: list[float],
    extent: float = 3.0,
    path: str = "relu_explorer.html",
) -> Path:
    """Write the explorer page: a snapshot slider plus one toggle per hidden unit."""
    fig, meta = explorer_figure(pts, snaps, history, extent)
    html = fig.to_html(
        div_id="plot",
        include_plotlyjs="cdn",
        full_html=True,
        post_script=_EXPLORER_JS.replace("__META__", json.dumps(meta)),
    )
    out = Path(path).resolve()
    out.write_text(html)
    return out


def _view_regions(snap: dict, focus: int, extent: float) -> list[Region]:
    """Regions for the whole net (focus < 0) or for a single hidden unit."""
    net_units = [
        ((float(w[0]), float(w[1])), float(b))
        for w, b in zip(snap["W"], snap["b"])
    ]
    if focus < 0:
        return linear_regions(net_units, extent, out=snap["a"], out_bias=snap["c"])
    return linear_regions(
        [net_units[focus]], extent, out=np.array([snap["a"][focus]])
    )


def main() -> None:
    # imported here so the deep module can import helpers from this one
    from relu_viz.deep import main as deep_main

    deep_main()


def main_single_layer() -> None:
    pts = cylinder_points()
    net, history, snaps = train(pts)
    print(f"RMSE {history[0]:.3f} -> {history[-1]:.3f}")
    out = explorer(pts, snaps, history)
    print(f"wrote {out}")
    webbrowser.open(out.as_uri())
