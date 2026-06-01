set shell := ["sh", "-cu"]

port := "8080"

alias b := build
alias s := serve
alias d := dev
alias c := clean
alias x := stop

default:
    just --list

build:
    @./build.sh

serve:
    @caddy file-server --listen :{{port}} --root .

dev: build serve

stop:
    @pids="$(lsof -ti tcp:{{port}} || true)"; \
    if [ -n "$pids" ]; then \
      echo "Stopping server on port {{port}}: $pids"; \
      kill $pids; \
    else \
      echo "No server listening on port {{port}}"; \
    fi

clean:
    @cargo clean
    @rm -rf pkg
