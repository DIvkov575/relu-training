# ReLU training and visualization archive

A complete snapshot of the ReLU/MIG training visualizations and experiments
served through the Reilu investment group tunnel, including source code,
interactive HTML, trained model data, plots, PNG exports, and rendered videos.

## Browse locally

```sh
python3 -m http.server 8003 --directory .
```

Open <http://localhost:8003/> for the latest five-layer model gallery.
Other entry points include `animation.html`, `relu_deep.html`, `flow.html`,
`flow-density.html`, `flow-spatial.html`, `flow-study.html`, `flow-art.html`,
`topology.html`, and `dima.html`. Serve the files over HTTP so the viewers
can fetch their data. GitHub shows the files but does not run interactive HTML.

- `src/relu_viz/`: training, geometry, visualization, and rendering code.
- `experiments/`: saved experiment results, models, plots, and flow videos.
- `viewer-data/`: data used by the interactive viewers.
- `exports/`: downloadable PNG collections.
- Root `*.mp4`: rendered training animations and model transitions.
- `tests/`: existing project checks.

The original working directory and running tunnel are preserved. Local Python
environments, caches, render progress files, and an incomplete video are omitted.

## Original project documentation

# relu-viz

Visualizing piecewise-linear functions built from ReLU networks: the 3D surface,
its linear regions, and the fold lines where activations switch sign.

## Setup

```sh
uv sync
```

Generate the visualization with `BROWSER=true OPENBLAS_NUM_THREADS=1 uv run relu-viz`.
The generated page is `relu_deep.html`. Layer widths can be set explicitly, for
example `uv run relu-viz --target mig --samples 40000 --hidden 48 48 32`.
Generation also writes `animation.html`, titled **ReLU in Motion · MIG**. The
explorer's **Open animation** link opens it in a separate tab. During playback,
a browser worker interpolates every weight and bias between the neighboring
saved training states at the requested time, then rebuilds that network's exact
ReLU regions and planes. A dedicated WebGL canvas draws these new meshes during
playback; Plotly handles paused inspection. Surface vertices and heights are not
morphed between different networks. Geometry frame rate depends on region-rebuild
time; repeated canvas draws are not additional model frames.
Intermediates are not additional optimizer steps. At 1× speed, the first 500 steps (0.5% of training)
play over 18 seconds, followed by the remaining 99,500 steps over 18 seconds.
Each phase advances at a constant rate, with 36 seconds total playback. Scrubbing
and pause/resume use the matching inverse timing function.
Controls
include play/pause, restart, a scrubbable timeline, 0.5×/1×/2× speed, optional target
points, and camera orbit. Exact views on both pages color each region tile by its
**area-average elevation**, using a rainbow scale fixed to the target's height
range. For an affine tile this average is its height at the polygon centroid;
it is not generally the average of its corner heights. Target points use their
own elevations on the same scale.

The reference-inspired elevation palette runs from dark emerald through violet,
hot pink, cyan, lime and lemon yellow. Mean tile elevation still determines
color; the neon upper range contrasts with darker low regions.

The current target is horizontal **MIG** lettering with outward-flared bases. Only its
roof, bevels, and surrounding ground are sampled, so every input has one height.
The target has a height of 1.2 and a base flare of 0.12, keeping the letters separate.
The default is 40,000 samples, up from 20,000 for the first MIG version and 4,000
for the original M. Training uses
1,024-point minibatches for **100,000 Adam steps**, up from 20,000. The learning
rate stays at 0.003 for the first 20,000 steps, then decays exponentially to
0.0003 for late refinement. Snapshots are denser in the first 10,000 steps so the
animation preserves early optimizer changes. The loss curve
shows minibatch RMSE; snapshot labels show error over the full training cloud.
Use the **target points** checkbox to hide the dense cloud and inspect the model.

The MIG network now uses **48 + 48 + 32 hidden neurons (4,097 parameters)** to resolve the narrower strokes
and three separate letters. Deeper layers can bend
and gate folds into finite edges, rather than extending every hinge
across the full input plane. This configuration was selected by measured error,
not by a claim that three layers are always best.

With the same 40,000 samples, independent 20,000-point validation cloud, seed 1,
100,000 Adam steps, minibatch size 1,024 and learning-rate schedule:

| Hidden layers | Parameters | Training RMSE | Held-out RMSE |
| --- | ---: | ---: | ---: |
| 24 + 24 + 16 | 1,089 | 0.0836 | 0.0876 |
| **48 + 48 + 32** | **4,097** | **0.0304** | **0.0340** |

Added width reduces held-out error by about 61% in this matched run. This supports
using more capacity here, while not proving the smaller network's theoretical
representation limit. The explorer focuses on the final model and its individual
neurons; the separate animation retains all 500 training states. Earlier-layer
partitions are shared while exporting neuron views, avoiding repeated polygon cuts.
Individual neuron meshes load when selected, keeping the initial explorer page
small even for the wider model. Their trace indices and exact geometry are retained.

Earlier MIG comparison with the same 20,000 training samples, 20,000 independently sampled validation
points (seed 123), initialization seed 1, 20,000 steps and 1,024-point minibatches:

| MIG hidden layers | Held-out RMSE | 95th-percentile absolute error |
| --- | ---: | ---: |
| 10 + 10 + 8 | 0.2188 | 0.4654 |
| 16 + 16 + 16 | 0.1271 | 0.2803 |
| **24 + 24 + 16** | **0.1149** | **0.2541** |

More samples improve target coverage; these matched runs separately demonstrate
the effect of added model capacity. The learned MIG surface remains approximate,
particularly around narrow strokes and sharp corners.

The earlier 40,000-sample, 20,000-step run had held-out RMSE 0.1332 on the same validation
cloud. Denser sampling covers more of the target; it does not guarantee lower
error under a fixed stochastic training budget.

The earlier 1,089-parameter **100,000-step** run used the same 40,000 training samples and the
same 20,000 held-out samples. Training RMSE is **0.0836** and held-out RMSE is
**0.0876**, about **34% lower** than the 20,000-step run. The animation now records
**500 saved states**, up from 68: 350 through step 10,000 and 150 across the tail
including the final trained state. **Saved step** buttons pause playback and move
between actual recorded states, with a counter distinguishing them from interpolated frames.

The earlier, single-M experiment selected **10 + 10 + 8** as follows (these metrics
apply to the single M and do not transfer to MIG):

Matched comparison: 4,000 training surface samples (seed 0), 4,000 independent
held-out surface samples (seed 123), initialization seed 1, 16,000 Adam steps,
learning rate 0.005, and ordinary ReLU activations. Parameters include all weights
and biases. Samples are balanced across roof, bevels, and ground.

| Hidden layers | Parameters | Held-out RMSE | 95th-percentile absolute error |
| --- | ---: | ---: | ---: |
| 12 + 12 | 205 | 0.0919 | 0.2009 |
| 8 + 8 + 8 | 177 | 0.0747 | 0.1621 |
| **10 + 10 + 8** | **237** | **0.0391** | **0.0741** |
| 8 + 8 + 8 + 8 | 249 | 0.0534 | 0.1206 |

The selected single-M model cuts RMSE by about 57% versus 12 + 12 for 16% more parameters.
The four-layer model costs more and fits less accurately in this run. These are
single-initialization comparisons on this target, not a proof of a globally
optimal architecture or exact reconstruction; sharp corners remain approximate.

## MP4 export

Render the complete 36-second animation as a 1080p, 60 FPS H.264 MP4:

```sh
uv run --with moderngl --with imageio-ffmpeg --with pillow python -m relu_viz.video_export --snapshots /path/to/snapshots.pkl --output mig-training-60fps.mp4
```

Each of the 2,160 frames interpolates network weights and biases at its own
time, rebuilds the exact ReLU planes, and renders their average-elevation colors.
The first 500 steps occupy the first 18 seconds. Eight offline geometry workers
prepare frames while EGL renders and FFmpeg encodes them. Playback uses the MP4
and performs no network evaluation or region reconstruction.

FPS is controlled with `--fps`: `frame_count = (18 + second_seconds) × fps`.
The default `--second-seconds 18` gives a 36-second video. Frame timestamps are
evenly spaced across the full training playback timeline, including both endpoints;
each timestamp produces newly interpolated weights and a newly reconstructed mesh.
For the smoother top-down version, use `--fps 120 --view top --output mig-training-top-120fps.mp4`.
This generates 4,320 distinct model frames, with an orthographic top-down camera
and 4× multisample anti-aliasing (`--samples 4`). The viewing screen's refresh
rate sets the limit on how many video frames can be displayed per second.

Use `--end-step 75000` to omit the final quarter of the 100,000-step run.
The opening remains 18 seconds, and steps 500–75,000 receive the remaining
18 seconds, slowing the later phase. `--start-time 18` exports just that
second half when reusing an existing identical opening.

Video speed is piecewise linear in training step, independently of FPS.
`--second-seconds 24 --end-step 65000` preserves the first 18 seconds for the
first 500 steps, then gives steps 500–65,000 another 24 seconds (42 seconds total).
`--fps 120` controls temporal frame density, giving 5,040 frames for this version.

`--regime-speed 2 --end-step 55000` makes the second phase twice the opening
step rate: about 55.6 training steps/second. That produces a 981-second second
phase and a 999-second (16:39) total video, containing 119,880 frames at 120 FPS.
`--regime-speed` overrides `--second-seconds`. Long exports can be split with
`--start-time` and `--stop-time`; `--threads` controls encoder parallelism.

The current cutoff is step 30,000 at 2× opening speed: 531 seconds for the later phase, 549 seconds total (9:09). Existing rendered frames can be reused when only the cutoff changes, since the step rate remains the same.

The linear variant ends at step 10,000 with speed increasing linearly from
500/18 steps/sec to twice that rate across 240 seconds. Step versus time is
quadratic. `video_retime` reuses the existing 2×-regime video and changes frame
timestamps; 120 FPS resampling may repeat/drop cached frames rather than
creating new model states or morphing plane geometry.

```sh
uv run --with imageio-ffmpeg python -m relu_viz.video_retime mig-training-top-120fps-30k-2x.mp4 --end-step 10000 --end-speed 2
```

The current default uses a quadratic increase in speed from 1× to 5×; use `--end-speed 5 --speed-power 2`.
Speed is `v(t) = (500/18) × [1 + 4(t/T)^2]`; integrating this gives a cubic
training-step curve. At the 10,000-step cutoff, `T ≈ 154.286 seconds` (2:34).
The midpoint speed is 2×; the final speed is 5×.

The current video uses the neon palette. `video_palette` projects cached frame
colors back to elevation and applies the new scale using a 65³ RGB lookup table.
Only the fixed top-down model area is recolored; frame labels and page text are
preserved. This reuses the video frames and leaves timing and geometry intact.

The neon version uses `--end-speed 10 --speed-bias 6`, keeping the 10,000-step
cutoff. Speed bias is the exponent in `v(t)=v0×[1+(max_speed−1)(t/T)^bias]`;
higher values hold the initial speed longer and move acceleration toward the end.
At bias 6, speed is 1.14× halfway through, reaches 2× around 1:49, and ends at
10× after 157.5 seconds. Cached ramp videos are supported with
`--source-speed-power`, `--source-end-speed`, and `--source-end-step`, so their
frames can be retimed correctly without rerendering model geometry.

### Direct initial-to-final weight variation

```sh
uv run --with moderngl --with imageio-ffmpeg --with pillow python -m relu_viz.video_export --snapshots /path/to/snapshots.pkl --weight-path direct --duration 30 --fps 120 --view top --output mig-direct-weights-120fps.mp4
```

This keeps only the first and last weight sets, ignoring all intermediate
training snapshots. Every frame evaluates θ(α) = (1 − α)θinitial + αθfinal,
with α uniformly spaced from 0 to 1, and reconstructs the exact planes.
`--duration` controls direct playback length; training speed curves do not apply.
The label reports α rather than implying these networks are optimizer steps.

### DIMA — original video architecture

The DIMA tab (`dima.html`) preserves the original video network:
**2 → 48 → 48 → 32 → 1**, three ReLU hidden layers, linear output,
**4,097 parameters**. It trains on 40,000 relief samples for 100,000 Adam
steps, batch size 1,024, model seed 1, sample seed 0. Learning rate starts
at 0.003, stays there for 20,000 steps, then decreases exponentially to
0.0003. Settings and final errors are saved in `experiments/dima/config.json`.

The three DIMA color versions share the same geometry: interpolate directly
from the initial weights to the final trained weights, α = 0.1 to 1 over
20 seconds, then reverse for 20 seconds. Videos are 1920 × 1080 at 60 FPS,
with top-down view and region edges. The original MIG page and video remain
available from the main page.

### Gaussian → MIG flow matching

`flow-spatial.html` shows the final trained velocity field as a speed heat map with
arrowed streamlines. Its time slider changes the model input while keeping
all weights fixed. The original density animation is at `flow-density.html`.
Run `OPENBLAS_NUM_THREADS=1 uv run --with matplotlib python -m relu_viz.flow_field`
to rebuild the field views from the existing weights. The color scale is
linear and fixed across all times, capped at their combined 99th percentile.

The density animation is a separate interactive visualization. This trains a new
two-dimensional velocity model; it does not interpolate ReLU model weights.
The prior is an isotropic Gaussian (standard deviation 1.625 in plot units).
The target is uniform over the filled roof outlines of the existing MIG relief,
excluding its bevel and background. Conditional flow-matching examples use
`x(t)=(1-t)x0+t*x1` and target velocity `x1-x0`.

```sh
OPENBLAS_NUM_THREADS=1 uv run --with scipy python -m relu_viz.flow_matching
OPENBLAS_NUM_THREADS=1 uv run --with scipy --with pillow --with imageio-ffmpeg python -m relu_viz.flow_media
```

The network has three tanh hidden layers of width 96, with position/time
sine and cosine features. It trains for 40,000 steps with Adam, batch size
512. Half the sampled times are uniform; the other half emphasize the end
using `t=1-u³`. New Gaussian points are integrated through the learned field
with 240 midpoint steps. The heat maps use 32,768 points, a fixed 180² grid,
1.1-cell display blur, and one fixed logarithmic color scale. The page allows
scrubbing, playback speed changes, and an optional learned-motion arrow field.
`experiments/flow-mig/config.json` records the training configuration and losses;
`flow.json` records sampling settings and the fraction ending inside MIG.

The primary flow page now matches the time–position reference format, with separate
x and y projections. Rebuild with `OPENBLAS_NUM_THREADS=1 uv run --with matplotlib
python -m relu_viz.flow_spacetime`. It shows signed mean velocity with actual
projected paths, and a second figure of path density, both with endpoint
histograms. The spatial field remains at `flow-spatial.html`.

The primary `flow.html` is now an art-directed composition: a Gaussian cloud
flows into the full MIG footprint, with jade/sand and ink/pearl PNGs. The
paths use designed interpolation and exact endpoints inside the MIG roof
outlines; they do not claim to represent the trained velocity model. Rebuild
with `OPENBLAS_NUM_THREADS=1 uv run --with pillow --with scipy python -m
relu_viz.flow_art`. The scientific time-position figures remain at
`flow-study.html`.

The active flow page restores the supplied-reference layout: Gaussian
histograms left, time-position heat maps center, and literal MIG lettering
in place of the posterior histograms on the right. Page metadata is reduced
to short captions and PNG links. Rebuild with `relu_viz.flow_spacetime`;
cached paths are reused. The separate art composition is at `flow-art.html`.

The flow page has an Export images button that downloads a ZIP containing
only its two visible PNG figures. `relu_viz.topology.export_flow_images`
rebuilds that archive. `topology.html` is a separate warm contour composition
from a Gaussian-shaped prior through gray intermediate outlines to MIG.
Rebuild with `OPENBLAS_NUM_THREADS=1 uv run --with matplotlib --with scipy
python -m relu_viz.topology`. This uses actual cached trajectories of the trained flow model, shown at
times 0, 0.55, 0.86, and 1. The known Gaussian prior is evaluated analytically;
the other contours use density estimates of 16,384 model-driven samples.
The full scene is rotated 90° clockwise. No manually placed intermediate
blobs or polygon-based final contours remain.
