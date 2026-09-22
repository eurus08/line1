#!/usr/bin/env python3
"""
bench/mpi/plot_scaling.py — Generate strong/weak scaling plots for
blas_mpi_dot (Phase 6, step 6.8)

Reads the CSV that run_scaling.sh writes:

    bench/mpi/results/scaling_results.csv

columns: study,ranks,n_local,n_global,time_us,gflops,gbs

and produces two PNG files in the same directory:

    strong_scaling.png — time vs ranks, and speedup vs ranks (with an
                         ideal linear-speedup reference line)
    weak_scaling.png   — time vs ranks, and efficiency vs ranks (with
                         an ideal efficiency=1.0 reference line)

Deliberately uses only the standard library's csv module plus
matplotlib -- no pandas or numpy dependency, since this is a small,
one-off analysis script, not something that needs a full data-science
stack. The C library itself has zero Python dependency anywhere; this
script exists purely to visualise what run_scaling.sh already
measured and wrote to disk.

Usage:
    python3 bench/mpi/plot_scaling.py
    python3 bench/mpi/plot_scaling.py --csv path/to/other.csv

Requires: matplotlib (pip install matplotlib --break-system-packages,
or via your distro's package manager, e.g.
`sudo zypper install python3-matplotlib` on openSUSE).
"""

import argparse
import csv
import sys
from pathlib import Path


def load_results(csv_path: Path):
    """Read the CSV into two lists of dicts, one per study, each
    sorted by rank count ascending. Returns (strong_rows, weak_rows).
    """
    if not csv_path.exists():
        print(f"error: {csv_path} not found.", file=sys.stderr)
        print(
            "Run the benchmark first: bash bench/mpi/run_scaling.sh",
            file=sys.stderr,
        )
        sys.exit(1)

    strong_rows = []
    weak_rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "ranks": int(row["ranks"]),
                "n_local": int(row["n_local"]),
                "n_global": int(row["n_global"]),
                "time_us": float(row["time_us"]),
                "gflops": float(row["gflops"]),
                "gbs": float(row["gbs"]),
            }
            if row["study"] == "strong":
                strong_rows.append(parsed)
            elif row["study"] == "weak":
                weak_rows.append(parsed)
            else:
                print(
                    f"warning: unrecognised study '{row['study']}', skipping row",
                    file=sys.stderr,
                )

    strong_rows.sort(key=lambda r: r["ranks"])
    weak_rows.sort(key=lambda r: r["ranks"])
    return strong_rows, weak_rows


def plot_strong_scaling(rows, out_path: Path, physical_cores: int):
    if len(rows) < 2:
        print(
            f"skipping strong scaling plot: need at least 2 data points, got {len(rows)}",
            file=sys.stderr,
        )
        return

    import matplotlib.pyplot as plt

    ranks = [r["ranks"] for r in rows]
    times_ms = [r["time_us"] / 1000.0 for r in rows]

    # Speedup is computed relative to the SMALLEST rank count actually
    # measured -- if that happens not to be 1 (e.g. you only ran
    # ranks=2 upward), the axis label says so rather than silently
    # assuming a ranks=1 baseline exists.
    ref_ranks = ranks[0]
    ref_time = times_ms[0]
    speedup = [ref_time / t for t in times_ms]
    ideal_speedup = [r / ref_ranks for r in ranks]

    fig, (ax_time, ax_speedup) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_time.plot(ranks, times_ms, marker="o", color="tab:blue")
    ax_time.set_xlabel("MPI ranks")
    ax_time.set_ylabel("Time (ms)")
    ax_time.set_title("Strong scaling — time vs ranks\n(fixed global problem size)")
    ax_time.set_xscale("log", base=2)
    ax_time.set_yscale("log")
    ax_time.grid(True, which="both", alpha=0.3)

    ax_speedup.plot(
        ranks, speedup, marker="o", color="tab:blue", label="measured"
    )
    ax_speedup.plot(
        ranks,
        ideal_speedup,
        linestyle="--",
        color="gray",
        label="ideal (linear)",
    )
    ax_speedup.set_xlabel("MPI ranks")
    ax_speedup.set_ylabel(f"Speedup (relative to {ref_ranks} rank{'s' if ref_ranks != 1 else ''})")
    ax_speedup.set_title("Strong scaling — speedup vs ranks")
    ax_speedup.set_xscale("log", base=2)

    # Physical-core reference line: past this point, additional ranks
    # share a physical core via SMT/hyperthreading rather than getting
    # one of their own. Memory-bandwidth-bound work (which is exactly
    # what LINE1 is -- see the Phase 4 bench notes) typically shows a
    # visible knee right here, since hyperthread siblings on the same
    # core compete for the same memory bandwidth path rather than
    # adding real throughput. Drawn on BOTH panels, not just one, so
    # it's visible whether you're looking at raw time or speedup.
    if physical_cores > 0 and min(ranks) <= physical_cores <= max(ranks):
        for ax in (ax_time, ax_speedup):
            ax.axvline(
                physical_cores,
                linestyle=":",
                color="tab:red",
                alpha=0.7,
                label=f"physical cores ({physical_cores})",
            )

    ax_speedup.legend()
    ax_speedup.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_weak_scaling(rows, out_path: Path, physical_cores: int):
    if len(rows) < 2:
        print(
            f"skipping weak scaling plot: need at least 2 data points, got {len(rows)}",
            file=sys.stderr,
        )
        return

    import matplotlib.pyplot as plt

    ranks = [r["ranks"] for r in rows]
    times_ms = [r["time_us"] / 1000.0 for r in rows]

    ref_ranks = ranks[0]
    ref_time = times_ms[0]
    # Weak-scaling efficiency: ideally time stays constant as ranks
    # grow (work per rank is fixed), so efficiency = ref_time / time.
    # 1.0 is perfect; below 1.0 reflects communication overhead
    # growing with rank count.
    efficiency = [ref_time / t for t in times_ms]

    fig, (ax_time, ax_eff) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_time.plot(ranks, times_ms, marker="o", color="tab:orange")
    ax_time.set_xlabel("MPI ranks")
    ax_time.set_ylabel("Time (ms)")
    ax_time.set_title("Weak scaling — time vs ranks\n(fixed per-rank problem size)")
    ax_time.set_xscale("log", base=2)
    ax_time.grid(True, which="both", alpha=0.3)

    ax_eff.plot(ranks, efficiency, marker="o", color="tab:orange", label="measured")
    ax_eff.axhline(1.0, linestyle="--", color="gray", label="ideal (constant time)")

    # Same physical-core reference line as the strong scaling plot --
    # see the comment there for why the knee is expected right at this
    # rank count for memory-bandwidth-bound work.
    if physical_cores > 0 and min(ranks) <= physical_cores <= max(ranks):
        for ax in (ax_time, ax_eff):
            ax.axvline(
                physical_cores,
                linestyle=":",
                color="tab:red",
                alpha=0.7,
                label=f"physical cores ({physical_cores})",
            )

    ax_eff.set_xlabel("MPI ranks")
    ax_eff.set_ylabel(f"Efficiency (relative to {ref_ranks} rank{'s' if ref_ranks != 1 else ''})")
    ax_eff.set_title("Weak scaling — efficiency vs ranks")
    ax_eff.set_xscale("log", base=2)
    ax_eff.set_ylim(0, max(1.1, max(efficiency) * 1.1))
    ax_eff.legend()
    ax_eff.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to scaling_results.csv (default: results/scaling_results.csv "
        "next to this script)",
    )
    parser.add_argument(
        "--physical-cores",
        type=int,
        default=6,
        help="Physical core count, for a reference line marking where ranks "
        "start sharing a core via SMT/hyperthreading rather than getting "
        "one of their own (default: 6, matching a 6-core/12-thread CPU; "
        "pass 0 to disable the line entirely)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    csv_path = args.csv if args.csv is not None else results_dir / "scaling_results.csv"

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "error: matplotlib is not installed.\n"
            "Install it with: pip install matplotlib --break-system-packages\n"
            "or your distro's package manager, e.g.:\n"
            "  sudo zypper install python3-matplotlib   (openSUSE)",
            file=sys.stderr,
        )
        sys.exit(1)

    strong_rows, weak_rows = load_results(csv_path)

    out_dir = csv_path.parent
    plot_strong_scaling(strong_rows, out_dir / "strong_scaling.png", args.physical_cores)
    plot_weak_scaling(weak_rows, out_dir / "weak_scaling.png", args.physical_cores)


if __name__ == "__main__":
    main()
