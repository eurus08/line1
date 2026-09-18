# Makefile — Convenience wrapper around CMake
#
# Lets you type short commands instead of raw CMake invocations:
#
#   make           -> configure + build (Release)
#   make build     -> same as above
#   make debug     -> configure + build (Debug, with sanitizers)
#   make test      -> run the test suite
#   make bench     -> build + run benchmarks
#   make docs      -> generate API documentation (requires doxygen)
#   make clean     -> remove the build folder
#   make rebuild   -> clean + build
#   make info      -> print detected architecture and flags

BUILD_DIR   := build
BUILD_DEBUG := build-debug

.PHONY: all build debug test bench docs clean rebuild info

# Default target
all: build

# ── Release build ───────────────────────────────────────────────────
build:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel

# ── Debug build (sanitizers on) ─────────────────────────────────────
debug:
	cmake -B $(BUILD_DEBUG) -DCMAKE_BUILD_TYPE=Debug
	cmake --build $(BUILD_DEBUG) --parallel

# ── Run tests ───────────────────────────────────────────────────────
test: build
	cmake --build $(BUILD_DIR) --target test -- CTEST_OUTPUT_ON_FAILURE=1

# ── Run benchmarks ──────────────────────────────────────────────────
bench:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DBLAS1_BUILD_BENCH=ON
	cmake --build $(BUILD_DIR) --parallel
	cd $(BUILD_DIR) && ctest -R bench -V

# ── Generate API documentation ───────────────────────────────────────
# Reads Doxyfile at the repo root; output goes to docs/html/index.html.
# Requires doxygen on PATH -- not otherwise a build dependency of the
# library itself, so this is deliberately its own target rather than
# folded into `build`.
docs:
	doxygen Doxyfile
	@echo "Generated docs/html/index.html"

# ── Clean ───────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR) $(BUILD_DEBUG)

# ── Rebuild from scratch ─────────────────────────────────────────────
rebuild: clean build

# ── Print config info ───────────────────────────────────────────────
info:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -E "Status|SIMD|Precision|Compiler|fast math"