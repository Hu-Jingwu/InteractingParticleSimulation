"""Run and visualize the birth-death chemotaxis comparison.

Example: python run_simulation.py --steps 450 --seed 2023
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chemotaxis import SimulationConfig, SimulationResult, simulate


def expected_chemical_count(result: SimulationResult) -> np.ndarray:
    """Immigration-death expectation for fixed particle count."""
    config = result.config
    start = int(result.chemical_counts[0])
    if config.death_rate == 0:
        return start + config.birth_rate * config.n_particles * result.times
    equilibrium = config.birth_rate * config.n_particles / config.death_rate
    decay = np.exp(-config.death_rate * result.times)
    return start * decay + equilibrium * (1.0 - decay)


def plot_result(result: SimulationResult, path: Path) -> None:
    """Create a summary of the two models and both cell-distance metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = result.config
    explicit = result.stochastic_particles[-1]
    chemicals = result.chemical_snapshots[-1]
    field_particles = result.field_particles[-1]
    field_density = result.field_snapshots[-1].T * config.grid_size**2

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(
        f"Birth-death chemotaxis on the unit torus  |  N={config.n_particles}, "
        f"t={result.times[-1]:g}, seed={config.seed}"
    )

    ax = axes[0, 0]
    if len(chemicals):
        ax.scatter(chemicals[:, 0], chemicals[:, 1], s=7, c="#e7aa2b", alpha=0.55, label="chemicals")
    ax.scatter(explicit[:, 0], explicit[:, 1], s=20, c="#1769aa", label="particles")
    ax.set_title(f"Individual chemicals ({len(chemicals)} alive)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    heatmap = ax.imshow(
        field_density,
        origin="lower",
        extent=(0, 1, 0, 1),
        interpolation="nearest",
        cmap="magma",
    )
    ax.scatter(field_particles[:, 0], field_particles[:, 1], s=18, c="#6de1ff", edgecolors="black", linewidths=0.25)
    ax.set_title("Chemical field approximation + particles")
    fig.colorbar(heatmap, ax=ax, label="chemical mass / unit area", shrink=0.84)

    for ax in axes[0]:
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", aspect="equal")

    ax = axes[1, 0]
    ax.plot(result.times, result.rms_distance, color="#295d95", lw=2, label="paired RMS")
    ax.plot(
        result.snapshot_times,
        result.wasserstein2_distance,
        "o",
        color="#b64936",
        ms=4,
        label=r"empirical $W_2$ (snapshots)",
    )
    ax.set(xlabel="time", ylabel="torus distance", title="Cell population comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(result.times, result.chemical_counts, color="#ba7920", alpha=0.8, label="individual chemicals")
    ax.plot(result.times, result.field_total_mass / config.mass, color="#8b47a6", lw=2, label="field mass / chemical mass")
    ax.plot(result.times, expected_chemical_count(result), color="#444444", ls="--", lw=1, label="count expectation")
    ax.set(xlabel="time", ylabel="equivalent chemical count", title="Birth-death mass balance")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_animation(result: SimulationResult, path: Path, fps: int = 8) -> None:
    """Save a side-by-side animation from the sampled simulation states."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    config = result.config
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax in (ax_left, ax_right):
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", aspect="equal")
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.set_yticks(np.linspace(0, 1, 5))
    ax_left.set_title("Individual chemicals and particles")
    ax_right.set_title("Chemical field approximation + particles")
    chemicals_artist = ax_left.scatter([], [], s=7, c="#e7aa2b", alpha=0.55)
    explicit_artist = ax_left.scatter([], [], s=20, c="#1769aa")
    density = result.field_snapshots * config.grid_size**2
    vmax = float(np.quantile(density[-1], 0.99))
    heatmap = ax_right.imshow(
        density[0].T,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="magma",
        vmin=0,
        vmax=max(vmax, 1e-12),
    )
    field_artist = ax_right.scatter([], [], s=18, c="#6de1ff", edgecolors="black", linewidths=0.25)
    fig.colorbar(heatmap, ax=ax_right, label="chemical mass / unit area", shrink=0.8)
    title = fig.suptitle("")

    def update(index: int):
        chemicals_artist.set_offsets(result.chemical_snapshots[index])
        explicit_artist.set_offsets(result.stochastic_particles[index])
        heatmap.set_data(density[index].T)
        field_artist.set_offsets(result.field_particles[index])
        title.set_text(
            f"t={result.snapshot_times[index]:.3f}  |  "
            f"chemicals={len(result.chemical_snapshots[index])}  |  seed={config.seed}"
        )
        return chemicals_artist, explicit_artist, heatmap, field_artist, title

    animation = FuncAnimation(fig, update, frames=len(result.snapshot_times), blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="number of particles")
    parser.add_argument("--steps", type=int, default=450)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--snapshot-stride", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--alpha-particle", type=float, default=0.5)
    parser.add_argument("--alpha-chemical", type=float, default=0.4)
    parser.add_argument("--birth-rate", type=float, default=8.0)
    parser.add_argument("--death-rate", type=float, default=2.0)
    parser.add_argument("--chemical-mass", type=float, default=None)
    parser.add_argument("--interaction-radius", type=float, default=0.25)
    parser.add_argument("--initial-particles", choices=["split", "single"], default="split")
    parser.add_argument(
        "--initial-chemicals",
        choices=["at_particles", "one_uniform", "none"],
        default="at_particles",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--no-plot", action="store_true", help="save data only (NumPy required)")
    parser.add_argument("--gif", action="store_true", help="also save an animated GIF from snapshots")
    args = parser.parse_args()
    if args.no_plot and args.gif:
        parser.error("--gif cannot be combined with --no-plot")

    config = SimulationConfig(
        n_particles=args.n,
        steps=args.steps,
        dt=args.dt,
        alpha_particle=args.alpha_particle,
        alpha_chemical=args.alpha_chemical,
        birth_rate=args.birth_rate,
        death_rate=args.death_rate,
        chemical_mass=args.chemical_mass,
        interaction_radius=args.interaction_radius,
        grid_size=args.grid_size,
        snapshot_stride=args.snapshot_stride,
        seed=args.seed,
        initial_particles=args.initial_particles,
        initial_chemicals=args.initial_chemicals,
    )
    result = simulate(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "birth_death_simulation.npz"
    csv_path = args.output_dir / "birth_death_metrics.csv"
    wasserstein_path = args.output_dir / "birth_death_wasserstein.csv"
    result.save(data_path)
    np.savetxt(
        csv_path,
        np.column_stack(
            (result.times, result.rms_distance, result.chemical_counts, result.field_total_mass)
        ),
        delimiter=",",
        header="time,torus_rms_distance,chemical_count,field_total_mass",
        comments="",
    )
    np.savetxt(
        wasserstein_path,
        np.column_stack((result.snapshot_times, result.wasserstein2_distance)),
        delimiter=",",
        header="time,cell_empirical_wasserstein2",
        comments="",
    )
    print(f"Saved {data_path}, {csv_path}, and {wasserstein_path}")
    print(
        f"Final time={result.times[-1]:.3f}, chemicals={result.chemical_counts[-1]}, "
        f"field mass={result.field_total_mass[-1]:.6f}, "
        f"paired torus RMS={result.rms_distance[-1]:.6f}, "
        f"cell empirical W2={result.wasserstein2_distance[-1]:.6f}"
    )
    if not args.no_plot:
        figure_path = args.output_dir / "birth_death_simulation.png"
        plot_result(result, figure_path)
        print(f"Saved {figure_path}")
        if args.gif:
            gif_path = args.output_dir / "birth_death_simulation.gif"
            plot_animation(result, gif_path)
            print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
