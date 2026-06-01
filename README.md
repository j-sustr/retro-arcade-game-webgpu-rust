# Pristine Grid WebGPU

A Rust/WASM WebGPU implementation of the "Pristine Grid" technique described in this wonderful little blog post: https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8

Nothing fancy to see here, just a very direct port of the shader to WGSL and a minimal Rust render loop to display it in the browser.

## Build

Install the Rust WebAssembly target and the `wasm-bindgen` CLI once:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.122
```

Then build the demo:

```sh
./build.sh
```

Serve the directory with any static file server:

```sh
python3 -m http.server 8080
```

Then open http://127.0.0.1:8080/.
