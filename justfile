set shell := ["sh", "-cu"]

port := "8080"

default:
    just --list

build:
    ./build.sh

serve:
    python3 -m http.server {{port}}

dev: build serve

clean:
    cargo clean
    rm -rf pkg
