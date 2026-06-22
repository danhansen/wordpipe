#!/usr/bin/env bash
set -euo pipefail

# Build a local ONNX Runtime shared library tuned for Ivy Bridge class CPUs.
#
# The Python/Rust ORT package binaries may be built on newer x86_64 targets and
# can execute AVX2 instructions during initialization on this machine. This
# build uses Ivy Bridge codegen, which allows AVX/SSE4/F16C but not AVX2.
#
# Usage:
#   scripts/build_onnxruntime_ivybridge.sh
#   ORT_DYLIB_PATH="$PWD/build/onnxruntime-ivybridge/install/lib/libonnxruntime.so" \
#     scripts/wordpipe-dev stream-file-test ...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT_VERSION="${ORT_VERSION:-v1.24.4}"
SOURCE_DIR="${ORT_SOURCE_DIR:-$ROOT_DIR/build/onnxruntime-src-$ORT_VERSION}"
BUILD_DIR="${ORT_BUILD_DIR:-$ROOT_DIR/build/onnxruntime-ivybridge}"
INSTALL_DIR="${ORT_INSTALL_DIR:-$BUILD_DIR/install}"
PARALLEL="${PARALLEL:-$(nproc)}"

mkdir -p "$ROOT_DIR/build"

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git clone --recursive --branch "$ORT_VERSION" --depth 1 \
    https://github.com/microsoft/onnxruntime.git "$SOURCE_DIR"
else
  git -C "$SOURCE_DIR" submodule update --init --recursive
fi

COMMON_FLAGS="-O3 -march=ivybridge -mtune=ivybridge"

python3 "$SOURCE_DIR/tools/ci_build/build.py" \
  --config Release \
  --build_dir "$BUILD_DIR" \
  --update \
  --build \
  --parallel "$PARALLEL" \
  --skip_tests \
  --build_shared_lib \
  --cmake_generator Ninja \
  --cmake_extra_defines \
    CMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    CMAKE_C_FLAGS_RELEASE="$COMMON_FLAGS -DNDEBUG" \
    CMAKE_CXX_FLAGS_RELEASE="$COMMON_FLAGS -DNDEBUG"

cmake --install "$BUILD_DIR/Release"

echo "Built: $INSTALL_DIR/lib/libonnxruntime.so"
echo "Use: ORT_DYLIB_PATH=$INSTALL_DIR/lib/libonnxruntime.so"
