# tests/CMakeLists.txt
#
# Phase 3 — Testing Infrastructure
#
# Each test is a standalone executable linked against the blas1 static
# library. Tests are registered with CTest via add_test() so the whole
# suite can be run with `ctest` or `cmake --build build -t test`.
#
# Pattern for adding a new test file:
#   add_executable(test_X test_X.c)
#   target_link_libraries(test_X PRIVATE blas1)
#   target_include_directories(test_X PRIVATE ${CMAKE_SOURCE_DIR}/include)
#   add_test(NAME X COMMAND test_X)

message(STATUS "Configuring test suite (Phase 3)")

# ---- test_dot ------------------------------------------------------
add_executable(test_dot test_dot.c)
target_link_libraries(test_dot PRIVATE blas1)
target_include_directories(test_dot PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME dot COMMAND test_dot)