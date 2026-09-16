#!/usr/bin/env bash
#
# bench/mpi/run_scaling.sh — Strong and weak scaling studies for
# blas_mpi_dot (Phase 6, step 6.8)
#
# bench_mpi_dot measures exactly ONE (rank count, n_local) data point
# per invocation (see bench_mpi_dot.c for why). This script drives it
# across several rank counts and assembles the two tables the build
# plan asks for:
#
#   STRONG scaling: fixed GLOBAL size, n_local = N_GLOBAL / ranks.
#                   Time should DECREASE as ranks increase.
#   WEAK scaling:   fixed LOCAL size (same on every rank), so global
#                   size grows with rank count. Time should stay
#                   roughly CONSTANT -- any growth is communication
#                   overhead, since per-rank work never changes.
#
# Usage:
#   bash bench/mpi/run_scaling.sh [path-to-bench_mpi_dot]
#
# If no path is given, defaults to ./build/bench/mpi/bench_mpi_dot
# (i.e. run this from the repository root after a normal
# `cmake -B build -DBLAS1_BUILD_MPI=ON -DBLAS1_BUILD_BENCH=ON` build).
#
# RANK_COUNTS and the two fixed sizes below are the obvious things to
# edit for your own machine -- e.g. set RANK_COUNTS to match your
# actual core count instead of the default guesses.

set -euo pipefail

BENCH="${1:-./build/bench/mpi/bench_mpi_dot}"

if [ ! -x "$BENCH" ]; then
    echo "error: $BENCH not found or not executable." >&2
    echo "Build with: cmake -B build -DBLAS1_BUILD_MPI=ON -DBLAS1_BUILD_BENCH=ON && cmake --build build --parallel" >&2
    exit 1
fi

# Edit these to fit your machine's actual core count.
RANK_COUNTS=(1 2 4 8)

# Strong scaling: total problem size held fixed across all rank counts.
STRONG_N_GLOBAL=64000000

# Weak scaling: per-rank size held fixed across all rank counts.
WEAK_N_LOCAL=8000000

# MPIRUN_EXTRA_FLAGS lets you pass e.g. --oversubscribe on a machine
# with fewer physical cores than the largest RANK_COUNTS entry.
MPIRUN_EXTRA_FLAGS="${MPIRUN_EXTRA_FLAGS:-}"

HEADER_FMT="%8s %14s %14s %14s %12s %12s\n"
ROW_HEADER=$(printf "$HEADER_FMT" "ranks" "n_local" "n_global" "time_us" "GFlop/s" "GB/s")
SEPARATOR="------------------------------------------------------------------------"

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
    # shellcheck disable=SC2086
    mpirun -np "$ranks" $MPIRUN_EXTRA_FLAGS "$BENCH" "$n_local"
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
    # shellcheck disable=SC2086
    mpirun -np "$ranks" $MPIRUN_EXTRA_FLAGS "$BENCH" "$WEAK_N_LOCAL"
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
