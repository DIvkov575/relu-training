"""Plot recorded training statistics and fixed-set checks of saved networks.

No training is performed. Run with BLAS thread counts set to one before Python
starts; independent checkpoint evaluations use separate worker processes.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle

import numpy as np

from relu_viz import mig_relief_points
from relu_viz.deep import eval_snap


_HELDOUT = None


def _initialize():
    global _HELDOUT
    _HELDOUT = mig_relief_points(count=20000, seed=7)


def _score(snap):
    residual = eval_snap(snap, _HELDOUT[:, :2]) - _HELDOUT[:, 2]
    return float(np.sqrt(np.mean(residual**2)))


def make_stats(checkpoint: Path, output: Path, workers: int = 16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with checkpoint.open("rb") as handle:
        pts, net, history, snaps = pickle.load(handle)
    output.mkdir(parents=True, exist_ok=True)
    # Intermediate checkpoints were captured before the update bearing their
    # recorded step number. The final checkpoint was captured after all updates.
    recorded = np.array([s["step"] for s in snaps], dtype=int)
    steps = recorded.copy()
    steps[:-1] -= 1
    train = np.array([s["rmse"] for s in snaps])
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize) as pool:
        heldout = np.array(list(pool.map(_score, snaps, chunksize=4)))
    heldout_points = mig_relief_points(count=20000, seed=7)
    target = heldout_points[:, 2]
    prediction = eval_snap(snaps[-1], heldout_points[:, :2])
    residual = prediction - target
    final_train = float(np.sqrt(np.mean((eval_snap(snaps[-1], pts[:, :2]) - pts[:, 2])**2)))
    assert np.isclose(final_train, train[-1], rtol=1e-12)
    assert np.isclose(heldout[-1], np.sqrt(np.mean(residual**2)), rtol=1e-12)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_hash
    history = np.asarray(history)
    rates = .003 * (.0003 / .003) ** np.maximum(0., (np.arange(1, len(history)+1) / len(history) - .2) / .8)
    changes = []
    intervals = np.diff(steps)
    for layer in range(len(snaps[0]["W"])):
        distances = [np.sqrt(np.sum((b["W"][layer] - a["W"][layer])**2)
                            + np.sum((b["b"][layer] - a["b"][layer])**2))
                     for a, b in zip(snaps[:-1], snaps[1:])]
        changes.append(np.array(distances) / intervals)
    output_change = np.array([np.sqrt(np.sum((b["a"]-a["a"])**2)+(b["c"]-a["c"])**2)
                              for a,b in zip(snaps[:-1],snaps[1:])]) / intervals

    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":12,
        "axes.labelsize":12,"axes.titlesize":16,"axes.titleweight":"medium",
        "figure.facecolor":"#f7f7f3","axes.facecolor":"#f7f7f3",
        "axes.edgecolor":"#aab2b5","axes.labelcolor":"#26343d",
        "text.color":"#26343d","xtick.color":"#56636a","ytick.color":"#56636a",
        "savefig.facecolor":"#f7f7f3","axes.spines.top":False,"axes.spines.right":False})
    plots = []

    def figure(title, xlabel="Training steps", ylabel="Prediction error (RMSE)"):
        fig, ax = plt.subplots(figsize=(9, 5.8), layout="constrained")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.16, linewidth=.7)
        ax.set_axisbelow(True)
        if xlabel == "Training steps":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k" if x else "0"))
        return fig, ax

    def save(fig, name, title, description):
        fig.savefig(output/name, dpi=170)
        plt.close(fig)
        plots.append(dict(image=name,title=title,description=description))

    fig, ax = figure("Prediction error during training")
    ax.plot(np.arange(len(history)), history, color="#9eaab9", alpha=.3, lw=.5, label="Recorded batch error")
    rolling = np.convolve(history, np.ones(200)/200, mode="valid")
    ax.plot(np.arange(199,len(history)), rolling, color="#566d8f", lw=1.4, label="Batch error · 200-step average")
    ax.plot(steps, train, color="#138579", lw=2, label="All 40,000 training examples")
    ax.plot(steps, heldout, color="#d17b3d", lw=1.7, label="Same 20,000 unseen examples")
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper right")
    save(fig,"error.png","Training and unseen-example error",
         "Recorded batch error and its 200-step average, compared with saved-checkpoint error on all training examples and a fixed set of 20,000 unseen examples. Vertical axis uses a logarithmic scale.")

    fig, ax = figure("Unseen error minus training error", ylabel="Difference in RMSE")
    gap = heldout-train
    ax.axhline(0,color="#76828c",lw=1)
    ax.plot(steps,gap,color="#9c658d",lw=1.8)
    ax.fill_between(steps,0,gap,color="#9c658d",alpha=.12)
    save(fig,"gap.png","Gap between unseen and training examples",
         "Positive values mean the saved network has more error on the fixed unseen examples. The same examples are used at every checkpoint.")

    fig, ax = figure("Learning rate used by the training schedule", ylabel="Learning rate")
    ax.plot(np.arange(1,len(history)+1),rates,color="#397e9d",lw=2)
    ax.axvline(4000,color="#8d9699",lw=.8,ls="--")
    ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    save(fig,"learning-rate.png","Learning-rate schedule",
         "Reconstructed from the verified training settings: 0.003 for the first 4,000 steps, then a gradual decrease to 0.0003 at step 20,000. This value was not logged separately.")

    fig, ax = figure("How far each layer changed between saved states",ylabel="Weight-change distance / training step")
    for index,change in enumerate(changes):
        ax.plot(steps[1:],change,lw=1.4,label=f"Hidden layer {index+1}")
    ax.plot(steps[1:],output_change,lw=1.3,color="#39444d",ls="--",label="Output layer")
    ax.set_yscale("log")
    ax.legend(fontsize=9,ncol=2)
    save(fig,"weight-change.png","Changes in each layer",
         "Straight-line distance between consecutive saved weight and offset values, divided by elapsed steps. This is checkpoint movement, not gradient size or the full distance traveled between checkpoints.")

    fig, ax = figure("Final predictions on unseen examples",xlabel="Correct elevation",ylabel="Predicted elevation")
    density = ax.hexbin(target,prediction,gridsize=65,mincnt=1,bins="log",cmap="viridis")
    lo,hi = min(target.min(),prediction.min()),max(target.max(),prediction.max())
    ax.plot([lo,hi],[lo,hi],color="#b34f4b",ls="--",lw=1.3,label="Perfect prediction")
    ax.legend(fontsize=10)
    fig.colorbar(density,ax=ax,label="Examples in each cell")
    save(fig,"predictions.png","Predicted versus correct elevation",
         "Final network predictions on the same 20,000 unseen examples. The dashed line marks exact agreement; color counts examples in each small cell on a logarithmic scale.")

    fig, ax = figure("Final prediction errors",xlabel="Predicted elevation − correct elevation",ylabel="Number of unseen examples")
    ax.hist(residual,bins=90,color="#348a87",alpha=.85)
    ax.axvline(0,color="#334554",ls="--",lw=1.4)
    save(fig,"residuals.png","Distribution of final errors",
         "All 20,000 unseen examples. Negative values are predictions that are too low; positive values are predictions that are too high.")

    metrics = dict(final_training_rmse=final_train,final_heldout_rmse=float(heldout[-1]),
                   heldout_mean_error=float(residual.mean()),heldout_mean_absolute_error=float(np.mean(np.abs(residual))),
                   heldout_absolute_error_95th_percentile=float(np.quantile(np.abs(residual),.95)),
                   final_gap=float(gap[-1]),training_steps=len(history),snapshot_count=len(snaps))
    methods = dict(checkpoint_sha256=original_hash,heldout_seed=7,heldout_count=20000,
                   training_count=len(pts),hidden_widths=[len(w) for w in snaps[-1]["W"]],
                   no_retraining=True,checkpoint_error_evaluation="Same fixed held-out examples for every saved state; computed after training.",
                   snapshot_timing="Intermediate recorded step numbers precede that update; x-axis uses completed updates (recorded step minus one). Final snapshot follows all 20,000 updates.",
                   history_timing="Batch RMSE is recorded before each update; plotted against completed updates starting at zero.",
                   learning_rate="Reconstructed from verified trainer configuration: .003, exponential decay after 20% of steps to .0003.",
                   weight_changes="Euclidean distance of weights and offsets between snapshots divided by completed-update interval; not gradients or optimizer update magnitudes.")
    data = dict(metrics=metrics,methods=methods,raw_batch_rmse=history.tolist(),
                recorded_snapshot_steps=recorded.tolist(),snapshot_completed_updates=steps.tolist(),
                full_training_rmse=train.tolist(),heldout_rmse=heldout.tolist(),
                heldout_minus_training_rmse=gap.tolist(),learning_rates=rates.tolist(),
                layer_checkpoint_change_per_step=[x.tolist() for x in changes],
                output_checkpoint_change_per_step=output_change.tolist(),
                final_heldout_targets=target.tolist(),final_heldout_predictions=prediction.tolist())
    (output/"data.json").write_text(json.dumps(data,indent=2)+"\n")
    manifest = dict(plots=plots,metrics=metrics,methods=methods,data="data.json")
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=16)
    args=parser.parse_args()
    make_stats(args.checkpoint,args.output,args.workers)


if __name__ == "__main__":
    main()
