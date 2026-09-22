#!/usr/bin/env bash
#
# bench/mpi/run_scaling.sh — Strong and weak scaling studies for
# blas_mpi_dot (Phase 6, step 6.8)
#
# bench_mpi_dot measures exactly ONE (rank count, n_local) data point
# per invocation (see bench_mpi_dot.c for why). This script drives it
# across several rank counts and assembles the two studies the build
# plan asks for:
#
#   STRONG scaling: fixed GLOBAL size, n_local = N_GLOBAL / ranks.
#                   Time should DECREASE as ranks increase.
#   WEAK scaling:   fixed LOCAL size (same on every rank), so global
#                   size grows with rank count. Time should stay
#                   roughly CONSTANT -- any growth is communication
#                   overhead, since per-rank work never changes.
#
# In addition to printing human-readable tables to stdout (as before),
# every data point is now also appended to a CSV file checked into
# the repo:
#
#   bench/mpi/results/scaling_results.csv
#
# columns: study,ranks,n_local,n_global,time_us,gflops,gbs
#
# This file is what plot_scaling.py reads to generate the scaling
# plots -- committing it means the actual numbers your machine
# produced live in the repo, not just a plot image derived from them
# that nobody could regenerate or double-check.
#
# Usage:
#   bash bench/mpi/run_scaling.sh [path-to-bench_mpi_dot]
#
# If no path is given, defaults to ./build/bench/mpi/bench_mpi_dot
# (i.e. run this from the repository root after a normal
# `cmake -B build -DLINE1_BUILD_MPI=ON -DLINE1_BUILD_BENCH=ON` build).
#
# RANK_COUNTS and the two fixed sizes below are the obvious things to
# edit for your own machine -- e.g. set RANK_COUNTS to match your
# actual core count instead of the default guesses. Each run
# OVERWRITES the CSV from scratch (not appends across runs), so the
# committed file always reflects one clean, complete, internally
# consistent study -- not a patchwork of numbers from different days
# or different RANK_COUNTS settings.

set -euo pipefail

BENCH="${1:-./build/bench/mpi/bench_mpi_dot}"

if [ ! -x "$BENCH" ]; then
    echo "error: $BENCH not found or not executable." >&2
    echo "Build with: cmake -B build -DLINE1_BUILD_MPI=ON -DLINE1_BUILD_BENCH=ON && cmake --build build --parallel" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
CSV_FILE="$RESULTS_DIR/scaling_results.csv"
mkdir -p "$RESULTS_DIR"

# Edit these to fit your machine's actual core count. Defaults here
# assume a 6 physical-core / 12-thread CPU (divisors of 6, so strong-
# scaling partitions land on whole numbers, plus two points past the
# physical core count to show the SMT/hyperthreading knee).
RANK_COUNTS=(1 2 3 4 6 8 12)

# Strong scaling: total problem size held fixed across all rank counts.
STRONG_N_GLOBAL=64000000

# Weak scaling: per-rank size held fixed across all rank counts.
WEAK_N_LOCAL=8000000

# MPIRUN_EXTRA_FLAGS: the exact combination matters, not just whether
# hwthreads are enabled at all. Confirmed by hand with
# `mpirun --report-bindings` at ranks 6, 8, and 12 on a 6-core/12-
# thread CPU:
#
#   --use-hwthread-cpus  tells OpenMPI SMT threads are usable slots
#                        at all (without this, np > 6 refuses to run
#                        or silently oversubscribes real cores)
#   --map-by core        makes OpenMPI prefer an UNUSED core over
#                        doubling up on a busy one -- this is the
#                        one that actually matters. Without it (e.g.
#                        --map-by hwthread instead), OpenMPI packs
#                        BOTH SMT threads of core 0 before ever
#                        touching core 1, so an 8-rank run can end
#                        up using only 4 of your 6 cores while 2
#                        sit completely idle -- silently wrong
#                        scaling data with no error or warning.
#   --bind-to hwthread   pins each rank to the specific hwthread
#                        --map-by chose, rather than letting it
#                        float across a wider set
#
# At rank counts <= your physical core count, this reduces to one
# rank per distinct core (verified: 6 ranks -> cores 0-5, no
# doubling). Above it, ranks spill onto second SMT threads only
# after every core already has one -- at rank counts that aren't a
# multiple of your core count (e.g. 8 ranks / 6 cores), a few cores
# unavoidably end up with 2 ranks while the rest have 1; that's the
# mathematically minimal imbalance, not a misconfiguration.
MPIRUN_EXTRA_FLAGS="${MPIRUN_EXTRA_FLAGS:---use-hwthread-cpus --map-by core --bind-to hwthread}"

HEADER_FMT="%8s %14s %14s %14s %12s %12s\n"
ROW_HEADER=$(printf "$HEADER_FMT" "ranks" "n_local" "n_global" "time_us" "GFlop/s" "GB/s")
SEPARATOR="------------------------------------------------------------------------"

echo "study,ranks,n_local,n_global,time_us,gflops,gbs" > "$CSV_FILE"

# run_one <study_label> <ranks> <n_local>
# Runs bench_mpi_dot once, prints its row (for the human-readable
# table), and appends the same fields as a CSV row tagged with
# study_label ("strong" or "weak").
run_one() {
    local study="$1"
    local ranks="$2"
    local n_local="$3"
    local row

    # shellcheck disable=SC2086
    row="$(mpirun -np "$ranks" $MPIRUN_EXTRA_FLAGS "$BENCH" "$n_local")"
    echo "$row"

    # bench_mpi_dot's row is whitespace-separated:
    #   ranks  n_local  n_global  time_us  gflops  gbs
    read -r r nl ng t gf gb <<< "$row"
    echo "$study,$r,$nl,$ng,$t,$gf,$gb" >> "$CSV_FILE"
}

echo "=========================================================================="
echo " blas_mpi_dot -- STRONG scaling (fixed global N = $STRONG_N_GLOBAL)"
echo " Time should DECREASE as ranks increase -- more ranks sharing the same work."
echo "=========================================================================="
echo "$ROW_HEADER"
echo "$SEPARATOR"
for ranks in "${RANK_COUNTS[@]}"; do
    n_local=$(( STRONG_N_GLOBAL / ranks ))
    if [ "$n_local" -lt 1 ]; then
        echo "  (skipping ranks=$ranks: n_local would be < 1)"
        continue
    fi
    run_one "strong" "$ranks" "$n_local"
done

echo
echo "=========================================================================="
echo " blas_mpi_dot -- WEAK scaling (fixed per-rank n_local = $WEAK_N_LOCAL)"
echo " Time should stay roughly CONSTANT as ranks increase -- growth reveals"
echo " communication overhead, since per-rank work never changes."
echo "=========================================================================="
echo "$ROW_HEADER"
echo "$SEPARATOR"
for ranks in "${RANK_COUNTS[@]}"; do
    run_one "weak" "$ranks" "$WEAK_N_LOCAL"
done

echo
echo "Notes:"
echo "  - Strong-scaling speedup at ranks=P, relative to ranks=1, is"
echo "    time(1) / time(P). Near-linear speedup (speedup ~ P) means the"
echo "    problem is compute/communication-bound rather than already"
echo "    memory-bandwidth-saturated on a single rank."
echo "  - Weak-scaling efficiency at ranks=P is time(1) / time(P) (NOT"
echo "    the same formula's meaning as strong scaling -- here staying"
echo "    close to 1.0 is the good outcome, since the work per rank is"
echo "    constant by construction)."
echo "  - blas_mpi_dot's MPI_Allreduce cost typically grows like log(P),"
echo "    so some falloff in weak-scaling efficiency at high rank counts"
echo "    is expected and is not a bug."
echo
echo "Results written to: $CSV_FILE"
echo "Generate plots with: python3 bench/mpi/plot_scaling.py"
