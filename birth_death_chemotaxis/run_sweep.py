"""Repeat the paired simulation across particle counts and random seeds.

Example: python run_sweep.py --counts 20,50,100,200,400 --repeats 5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from chemotaxis import SimulationConfig, simulate


def parse_counts(value: str) -> list[int]:
    try:
        counts = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("counts must be comma-separated integers") from error
    if not counts or any(count < 1 for count in counts) or len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("counts must be distinct positive integers")
    return counts


def plot_sweep(
    times: np.ndarray,
    counts: list[int],
    means: np.ndarray,
    standard_deviations: np.ndarray,
    wasserstein2_means: np.ndarray,
    wasserstein2_standard_deviations: np.ndarray,
    repeats: int,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_curve, ax_final) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(counts)))
    for index, (count, color) in enumerate(zip(counts, colors)):
        ax_curve.plot(times, means[index], color=color, label=f"N={count}")
        if repeats > 1:
            ax_curve.fill_between(
                times,
                np.maximum(0, means[index] - standard_deviations[index]),
                means[index] + standard_deviations[index],
                color=color,
                alpha=0.12,
            )
    ax_curve.set(xlabel="time", ylabel="torus RMS distance", title="Mean pathwise difference")
    ax_curve.legend(fontsize=8)
    ax_curve.grid(alpha=0.25)

    final_mean = means[:, -1]
    final_se = standard_deviations[:, -1] / np.sqrt(repeats)
    wasserstein2_se = wasserstein2_standard_deviations / np.sqrt(repeats)
    ax_final.errorbar(
        counts, final_mean, yerr=final_se, fmt="o-", color="#295d95",
        capsize=4, label="paired RMS",
    )
    ax_final.errorbar(
        counts, wasserstein2_means, yerr=wasserstein2_se, fmt="s-",
        color="#b64936", capsize=4, label=r"empirical $W_2$",
    )
    ax_final.set(
        xlabel="number of particles N",
        ylabel="final torus distance",
        title=f"Final mean ± standard error ({repeats} seeds)",
    )
    ax_final.set_xticks(counts)
    ax_final.grid(alpha=0.25)
    ax_final.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=parse_counts, default=[20, 50, 100, 200, 400])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2023, help="first seed; repeats use successive seeds")
    parser.add_argument("--output-dir", type=Path, default=Path("output/sweep"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    all_curves = []
    all_wasserstein2_finals = []
    for count in args.counts:
        curves = []
        wasserstein2_finals = []
        for repeat in range(args.repeats):
            config = SimulationConfig(
                n_particles=count,
                steps=args.steps,
                dt=args.dt,
                grid_size=args.grid_size,
                seed=args.seed + repeat,
                snapshot_stride=args.steps,
            )
            result = simulate(config)
            curves.append(result.rms_distance)
            wasserstein2_finals.append(result.wasserstein2_distance[-1])
            print(
                f"N={count:4d}, seed={config.seed}: "
                f"final torus RMS={result.rms_distance[-1]:.6f}, "
                f"empirical W2={result.wasserstein2_distance[-1]:.6f}"
            )
        all_curves.append(np.stack(curves))
        all_wasserstein2_finals.append(wasserstein2_finals)

    runs = np.stack(all_curves)
    wasserstein2_runs = np.asarray(all_wasserstein2_finals)
    means = runs.mean(axis=1)
    standard_deviations = runs.std(axis=1, ddof=1) if args.repeats > 1 else np.zeros_like(means)
    wasserstein2_means = wasserstein2_runs.mean(axis=1)
    wasserstein2_standard_deviations = (
        wasserstein2_runs.std(axis=1, ddof=1)
        if args.repeats > 1 else np.zeros_like(wasserstein2_means)
    )
    times = result.times
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "particle_count_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "n_particles", "repeats", "final_mean_rms", "final_sd_rms", "final_se_rms",
            "final_mean_wasserstein2", "final_sd_wasserstein2", "final_se_wasserstein2",
        ))
        for index, count in enumerate(args.counts):
            writer.writerow(
                (
                    count,
                    args.repeats,
                    means[index, -1],
                    standard_deviations[index, -1],
                    standard_deviations[index, -1] / np.sqrt(args.repeats),
                    wasserstein2_means[index],
                    wasserstein2_standard_deviations[index],
                    wasserstein2_standard_deviations[index] / np.sqrt(args.repeats),
                )
            )

    curves_path = args.output_dir / "particle_count_curves.npz"
    np.savez_compressed(
        curves_path,
        counts=np.array(args.counts),
        seeds=np.arange(args.seed, args.seed + args.repeats),
        times=times,
        rms_runs=runs,
        rms_means=means,
        rms_standard_deviations=standard_deviations,
        final_wasserstein2_runs=wasserstein2_runs,
        final_wasserstein2_means=wasserstein2_means,
        final_wasserstein2_standard_deviations=wasserstein2_standard_deviations,
    )
    print(f"Saved {summary_path} and {curves_path}")
    if not args.no_plot:
        figure_path = args.output_dir / "particle_count_comparison.png"
        plot_sweep(
            times, args.counts, means, standard_deviations,
            wasserstein2_means, wasserstein2_standard_deviations,
            args.repeats, figure_path,
        )
        print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
