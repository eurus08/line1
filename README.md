# blas1

A production-quality implementation of **BLAS Level 1** (vector-vector operations) in C, built from scratch across seven phases: serial implementation, correctness testing, benchmarking, SIMD kernels, and an MPI-parallel layer.

BLAS (Basic Linear Algebra Subprograms) is the specification underneath nearly every scientific computing library — NumPy, MATLAB, and PyTorch all eventually call into a BLAS implementation. Level 1 covers the simplest building block: operations on 1D vectors, no matrices yet. This project implements all six standard Level 1 operations, correctly and with real attention to the numerical and performance details that separate a working implementation from a *careful* one — then parallelizes them with MPI and measures exactly how, and how far, that parallelism actually helps.

## The six operations

| Function | What it computes | Math |
|---|---|---|
| `dot` | Dot product of two vectors | `Σ xᵢ·yᵢ` |
| `axpy` | Scale + add | `y ← αx + y` |
| `scal` | Scale a vector in place | `x ← αx` |
| `nrm2` | Euclidean norm | `√(Σ xᵢ²)` |
| `asum` | Sum of absolute values | `Σ |xᵢ|` |
| `iamax` | Index of the largest-magnitude element | `argmax |xᵢ|` |

Every function supports non-unit strides (`incx`/`incy`), matching the real BLAS calling convention where vector elements aren't always contiguous in memory.

## Requirements

- CMake 3.18+
- A C11 compiler (developed and tested with GCC; C11 is enforced strictly — `-std=c11`, not `-std=gnu11`)
- **Optional**, for the MPI layer: an MPI implementation (OpenMPI or MPICH)
- **Optional**, for the scaling plots: Python 3 + matplotlib (`pip install matplotlib` or your distro's package)

## Building

Quick start (serial library + test suite):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build
```

### Build options

| Option | Default | Description |
|---|---|---|
| `BLAS1_BUILD_TESTS` | `ON` | Build the correctness test suite |
| `BLAS1_BUILD_BENCH` | `OFF` | Build the benchmark suite (`bench_dot`, `bench_axpy`, and the MPI scaling benchmark) |
| `BLAS1_BUILD_MPI` | `OFF` | Build the MPI parallel layer, its tests, and its benchmark |
| `BLAS1_USE_FLOAT` | `OFF` | Build in single precision (`float`) instead of `double` |
| `BLAS1_STRICT_IEEE` | `OFF` | Disable `-ffast-math`, for callers who need guaranteed IEEE-compliant rounding (e.g. Kahan compensation to survive exactly as written) |
| `BLAS1_BUILD_SHARED` | `OFF` | Also build `libblas1.so` (properly versioned, `libblas1.so.1.0.0` with `.so.1`/`.so` symlinks) alongside the always-built static `libblas1.a` |

With the MPI layer and benchmarks both enabled:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBLAS1_BUILD_MPI=ON -DBLAS1_BUILD_BENCH=ON
cmake --build build --parallel
ctest --test-dir build          # runs serial + MPI tests + benchmark smoke tests
```

Architecture-specific SIMD kernels (AVX2 on x86, NEON on ARM) are detected and selected automatically at configure time — no flag needed. If neither is available, a portable scalar fallback is used, and this is reported in the configure output either way.

## Installing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBLAS1_BUILD_MPI=ON
cmake --build build --parallel
cmake --install build --prefix /your/install/path   # defaults to /usr/local
```

This installs the static library, public headers, and a [pkg-config](https://en.wikipedia.org/wiki/Pkg-config) file, so other projects can find and link this library with:

```bash
gcc myprogram.c $(pkg-config --cflags --libs blas1) -o myprogram
```

If built with `-DBLAS1_BUILD_MPI=ON`, a second file, `blas1-mpi`, covers the MPI layer separately (`Requires: blas1`, so its own flags chain in `blas1`'s automatically):

```bash
mpicc myprogram.c $(pkg-config --cflags --libs blas1-mpi) -o myprogram
```

`make install` is a shorthand for the same thing.

## Testing

```bash
ctest --test-dir build            # everything
ctest --test-dir build -R mpi_    # just the MPI tests
make test                         # equivalent, via the convenience Makefile
```

Testing philosophy: floating-point results are never compared with `==`. Every test checks that the absolute or relative error is below an explicit tolerance, with known-answer cases, edge cases (empty vectors, single-element vectors, non-unit strides), and cross-checks against reference implementations computed independently.

## Benchmarks

### Serial

Measured with `bench_dot`/`bench_axpy` (100 trials per size, median reported; full output in `results/`). `dot` is 2 FLOPs and 16 bytes per element; `axpy` is 2 FLOPs and 24 bytes:

> **Re-measure before trusting these:** a bug in `cmake/DetectArch.cmake` (fixed — see commit) meant `BLAS1_HAS_AVX2`/`BLAS1_HAS_NEON` were pre-set before the compiler-support check ran, which silently skipped the check and forced the generic scalar fallback on every build, this one included. The table below was almost certainly measured on the scalar path, not AVX2. Re-run `make bench` now that detection actually works and replace these numbers.

| n | dot warm GB/s | dot warm GFlop/s | axpy warm GB/s | axpy warm GFlop/s |
|---:|---:|---:|---:|---:|
| 1,000 | 61.1 | 7.63 | 180.5 | 15.04 |
| 16,000 | 66.2 | 8.27 | 122.0 | 10.17 |
| 1,000,000 | 20.7 | 2.59 | 40.7 | 3.39 |
| 16,000,000 | 22.5 | 2.81 | 23.5 | 1.96 |
| 100,000,000 | 24.0 | 3.00 | 28.7 | 2.39 |

**Key insight, and the reason the numbers plateau instead of climbing:** BLAS Level 1 is almost always memory-bandwidth-limited, not compute-limited. There's very little arithmetic per element — the bottleneck is how fast data can move from DRAM, not how fast the CPU can multiply. This single fact shapes most of the design decisions in this library (the SIMD kernels, for instance, help most at small-to-medium sizes still resident in cache; at large `n`, moving to AVX2 barely matters because the memory bus, not the ALU, is already the limit).

### MPI scaling

Measured with `bench_mpi_dot` across MPI rank counts 1–12 on a 6-core/12-thread CPU, both strong scaling (fixed global problem size, more ranks sharing the same work) and weak scaling (fixed per-rank size, so the global problem grows with rank count):

![Strong scaling](bench/mpi/results/strong_scaling.png)
![Weak scaling](bench/mpi/results/weak_scaling.png)

| ranks | strong: time (ms) | strong: speedup | weak: time (ms) | weak: efficiency |
|---:|---:|---:|---:|---:|
| 1 | 40.4 | 1.00× | 5.0 | 1.00 |
| 2 | 33.3 | 1.21× | 8.3 | 0.60 |
| 3 | 31.0 | 1.30× | 11.4 | 0.44 |
| 4 | 29.8 | **1.36×** | 14.8 | 0.34 |
| 6 | 30.9 | 1.31× | 23.2 | 0.22 |
| 8 | 30.4 | 1.33× | 30.6 | 0.16 |
| 12 | 36.0 | 1.12× | 48.8 | 0.10 |

**This is the same memory-bandwidth story, now at the MPI layer.** `dot`'s local computation is essentially pure memory traffic with almost no arithmetic to hide it behind, so once a handful of ranks are pulling data from DRAM simultaneously, the memory bus — not the number of ranks — is the bottleneck. Strong-scaling speedup peaks at just 4 ranks (~1.36×) and never approaches linear; weak-scaling efficiency falls to ~10% by 12 ranks. Both are the expected signature of a memory-bandwidth-bound kernel on a CPU with a small number of memory channels relative to its core count, not a flaw in the MPI implementation — `blas_mpi_dot`'s correctness was verified independently (see `tests/mpi/`) before any of this scaling data was gathered.

**A real gotcha worth documenting**, since it silently produced wrong data before being caught: OpenMPI's default rank-to-core mapping packs both SMT/hyperthread siblings of a core before moving to the next core, rather than spreading ranks across distinct physical cores first. On a 6-core/12-thread CPU, an unqualified 8-rank run can end up using only 4 of the 6 physical cores while 2 sit idle — with no error or warning. Verified by hand with `mpirun --report-bindings`; the fix is `--map-by core --bind-to hwthread` alongside `--use-hwthread-cpus`, which forces breadth-first placement across distinct cores before ever doubling up on SMT threads. See `bench/mpi/run_scaling.sh` for the full explanation and the exact flags used.

Reproduce with:
```bash
cmake -B build -DBLAS1_BUILD_MPI=ON -DBLAS1_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
bash bench/mpi/run_scaling.sh          # edit RANK_COUNTS to match your core count first
python3 bench/mpi/plot_scaling.py
```

## API reference

### Serial (`include/blas1/*.h`)

| Function | Signature | Notes |
|---|---|---|
| `blas_dot` | `BLAS_REAL blas_dot(blas_int n, const BLAS_REAL *x, blas_int incx, const BLAS_REAL *y, blas_int incy)` | |
| `blas_dot_kahan` | same signature as `blas_dot` | Kahan-compensated summation, for callers who need reduced rounding error at some extra cost |
| `blas_axpy` | `void blas_axpy(blas_int n, BLAS_REAL alpha, const BLAS_REAL *x, blas_int incx, BLAS_REAL *y, blas_int incy)` | `alpha == 0` is a fast no-op path |
| `blas_scal` | `void blas_scal(blas_int n, BLAS_REAL alpha, BLAS_REAL *x, blas_int incx)` | `alpha == 0` explicitly zeroes (never multiplies), avoiding NaN propagation |
| `blas_nrm2` | `BLAS_REAL blas_nrm2(blas_int n, const BLAS_REAL *x, blas_int incx)` | Scaled two-pass algorithm (Blue 1978 / LAPACK `dnrm2`) — avoids overflow/underflow that a naive `sqrt(sum(x[i]*x[i]))` would hit on extreme-magnitude input |
| `blas_asum` | `BLAS_REAL blas_asum(blas_int n, const BLAS_REAL *x, blas_int incx)` | |
| `blas_iamax` | `blas_int blas_iamax(blas_int n, const BLAS_REAL *x, blas_int incx)` | 1-based index (BLAS/Fortran convention); returns `0` for `n <= 0` |

`BLAS_REAL` is `double` by default, `float` if built with `-DBLAS1_USE_FLOAT=ON`. `blas_int` is `int64_t`, to support vectors past 2 billion elements.

### MPI (`include/blas1/mpi/*.h`, requires `-DBLAS1_BUILD_MPI=ON`)

Every MPI function takes the calling rank's **local slice** of the vector — partitioning the global vector across ranks is the caller's responsibility, matching how real distributed BLAS layers are used.

| Function | Signature | Collective? |
|---|---|---|
| `blas_mpi_dot` | `BLAS_REAL blas_mpi_dot(blas_int n_local, const BLAS_REAL *x, blas_int incx, const BLAS_REAL *y, blas_int incy, MPI_Comm comm)` | Yes — one `MPI_Allreduce` |
| `blas_mpi_axpy` | `void blas_mpi_axpy(blas_int n_local, BLAS_REAL alpha, const BLAS_REAL *x, blas_int incx, BLAS_REAL *y, blas_int incy, MPI_Comm comm)` | No — embarrassingly parallel, `comm` kept only for a uniform call signature |
| `blas_mpi_nrm2` | `BLAS_REAL blas_mpi_nrm2(blas_int n_local, const BLAS_REAL *x, blas_int incx, MPI_Comm comm)` | Yes — two `MPI_Allreduce` calls (global scale, then scaled sum of squares) — see `mpi_nrm2.h` for why one collective isn't enough to stay overflow-safe |
| `blas_mpi_iamax` | `blas_mpi_iamax_result_t blas_mpi_iamax(blas_int n_local, const BLAS_REAL *x, blas_int incx, blas_int global_offset, MPI_Comm comm)` | Yes — `MPI_Allreduce(MPI_MAXLOC)` plus one `MPI_Bcast`, to get a correct 64-bit global index (`MPI_MAXLOC`'s location field is only 32 bits — see `mpi_iamax.h`) |

## Project structure

```
blas1/
├── CMakeLists.txt
├── Makefile                    # convenience wrapper: make build/test/bench
├── cmake/
│   ├── CompilerFlags.cmake     # -O3/-ffast-math/-Wall etc., per build type
│   └── DetectArch.cmake        # picks AVX2 / NEON / generic at configure time
├── include/blas1/              # public headers
│   ├── types.h  blas1.h        # shared types + umbrella header
│   ├── dot.h  axpy.h  scal.h  nrm2.h  asum.h  iamax.h
│   └── mpi/                    # MPI public headers (requires BLAS1_BUILD_MPI)
├── src/                        # serial implementations
│   ├── dot.c  axpy.c           # thin dispatchers -> kernel/{generic,x86,arm}/
│   ├── scal.c  nrm2.c  asum.c  iamax.c
│   └── mpi/                    # MPI wrappers around the serial kernels
├── kernel/
│   ├── generic/                # portable scalar fallback
│   ├── x86/                    # AVX2 + FMA intrinsics
│   └── arm/                    # NEON intrinsics
├── tests/                      # CTest-registered correctness tests
│   └── mpi/                    # MPI correctness tests, run via mpirun
├── bench/                      # throughput benchmarks (GFlop/s, GB/s)
│   └── mpi/                    # MPI scaling benchmark + plotting scripts
└── results/                    # captured benchmark output
```

## Design principles

A few decisions worth knowing about if you're reading the source:

- **`restrict` + `const` correctness everywhere.** Input pointers are `const` and marked non-aliasing (`BLAS_RESTRICT`), unlocking auto-vectorization the compiler couldn't otherwise safely apply.
- **Kahan summation is opt-in, not silently applied.** `blas_dot_kahan` exists alongside plain `blas_dot` because compensated summation costs real performance — callers who don't need the extra precision shouldn't pay for it unknowingly.
- **`nrm2` uses a scaled two-pass algorithm**, never a naive `sqrt(sum(x[i]*x[i]))`, specifically to avoid overflow on large values and underflow on tiny ones.
- **The MPI layer is wrappers, not reimplementations.** Every `blas_mpi_*` function calls the existing serial kernel on its local slice, then does the minimum necessary MPI collective(s) to combine results — no parallel algorithm is written twice.
- **Correct → Measured → Optimized, always in that order.** Every phase of this project followed that sequence; the MPI scaling results above are a direct example of measuring before concluding, since the first scaling run had a core-binding bug that produced misleading numbers until it was caught and fixed.

## Roadmap

| Phase | Work | Status |
|---|---|---|
| 1 | Project skeleton, CMake, shared types | ✅ |
| 2 | Serial implementations (all 6 operations) | ✅ |
| 3 | Correctness test suite | ✅ |
| 4 | Benchmarking infrastructure | ✅ |
| 5 | SIMD kernels (AVX2 / NEON) | ✅ |
| 6 | MPI parallel layer | ✅ |
| 7 | Docs, packaging, CI | ✅ |
| 8 | BLAS Level 2 (matrix-vector operations) | ⏳ in progress |
| 9 | BLAS Level 3 (matrix-matrix operations) | ⏳ in progress |

## License

MIT — see [LICENSE](LICENSE).
