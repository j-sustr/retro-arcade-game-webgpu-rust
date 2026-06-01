#!/usr/bin/env sh
set -eu

cargo build --target wasm32-unknown-unknown --release
wasm-bindgen \
  --target web \
  --out-dir pkg \
  target/wasm32-unknown-unknown/release/pristine_grid_webgpu.wasm
