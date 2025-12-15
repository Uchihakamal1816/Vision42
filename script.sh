#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BIN=${PY_BIN:-python}
PIP_BIN=${PIP_BIN:-pip}
NPM_BIN=${NPM_BIN:-npm}
HF_CLI_BIN=${HF_CLI_BIN:-huggingface-cli}

BACKEND_PORT=${BACKEND_PORT:-15200}
FRONTEND_PORT=${FRONTEND_PORT:-5173}
QWEN_PORT=${QWEN_PORT:-8000}

BEST_PT_PATH="$ROOT_DIR/best.pt"
BEST_PT_URL=${BEST_PT_URL:-""}
QWEN_MODEL_DIR="$ROOT_DIR/qwen"
QWEN_REPO=${QWEN_REPO:-"Qwen/Qwen3-VL-8B-Instruct"}
VLLM_ARGS=${VLLM_ARGS:-"--max-model-len 8192 --dtype bfloat16"}

LOG_DIR="$ROOT_DIR/.logs"
PID_DIR="$ROOT_DIR/.pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

log() { printf '[%s] %s\n' "$1" "$2"; }

ensure_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log ERROR "Required command '$1' not found. Install it first."
        exit 1
    fi
}

install_python_requirements() {
    ensure_command "$PIP_BIN"
    log INFO "Installing Python dependencies"
    "$PIP_BIN" install --upgrade pip
    "$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"
    "$PIP_BIN" install --upgrade vllm huggingface-hub
}

install_frontend_dependencies() {
    ensure_command "$NPM_BIN"
    log INFO "Installing frontend dependencies"
    (cd "$ROOT_DIR/frontend" && "$NPM_BIN" install)
}

download_best_pt() {
    if [ -f "$BEST_PT_PATH" ]; then
        log INFO "best.pt already present"
        return
    fi
    if [ -z "$BEST_PT_URL" ]; then
        log ERROR "best.pt missing and BEST_PT_URL not set"
        exit 1
    fi
    log INFO "Downloading best.pt from $BEST_PT_URL"
    curl -L "$BEST_PT_URL" -o "$BEST_PT_PATH"
}

download_qwen_model() {
    if [ -d "$QWEN_MODEL_DIR" ] && [ -n "$(ls -A "$QWEN_MODEL_DIR" 2>/dev/null)" ]; then
        log INFO "Qwen weights already exist"
        return
    fi
    ensure_command "$HF_CLI_BIN"
    log INFO "Fetching Qwen model ($QWEN_REPO)"
    "$HF_CLI_BIN" download "$QWEN_REPO" --local-dir "$QWEN_MODEL_DIR" --local-dir-use-symlinks False
}

install_all() {
    download_best_pt
    download_qwen_model
    install_python_requirements
    install_frontend_dependencies
    log INFO "Installation complete"
}

start_process() {
    local name="$1"
    local cmd="$2"
    local pid_file="$PID_DIR/${name}.pid"
    local log_file="$LOG_DIR/${name}.log"

    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1; then
        log WARN "$name already running (PID $(cat "$pid_file"))"
        return
    fi

    log INFO "Starting $name"
    nohup bash -c "cd '$ROOT_DIR' && $cmd" >"$log_file" 2>&1 &
    echo $! >"$pid_file"
    log INFO "$name PID $(cat "$pid_file"), logs at $log_file"
}

stop_process() {
    local name="$1"
    local pid_file="$PID_DIR/${name}.pid"
    if [ ! -f "$pid_file" ]; then
        log WARN "$name not running"
        return
    fi
    local pid=$(cat "$pid_file")
    if kill -0 "$pid" >/dev/null 2>&1; then
        log INFO "Stopping $name (PID $pid)"
        kill "$pid"
    else
        log WARN "$name PID $pid already dead"
    fi
    rm -f "$pid_file"
}

start_qwen() {
    local cmd="$PY_BIN -m vllm.entrypoints.openai.api_server --model '$QWEN_MODEL_DIR' --host 0.0.0.0 --port $QWEN_PORT $VLLM_ARGS"
    start_process "qwen" "$cmd"
}

start_backend() {
    local cmd="$PY_BIN -m uvicorn backend.server:app --host 0.0.0.0 --port $BACKEND_PORT"
    start_process "backend" "$cmd"
}

start_frontend() {
    local cmd="$NPM_BIN run dev -- --host 0.0.0.0 --port $FRONTEND_PORT"
    start_process "frontend" "cd '$ROOT_DIR/frontend' && $cmd"
}

start_all() {
    start_qwen
    start_backend
    start_frontend
    log INFO "Services running: Qwen=$QWEN_PORT Backend=$BACKEND_PORT Frontend=$FRONTEND_PORT"
}

stop_all() {
    stop_process "frontend"
    stop_process "backend"
    stop_process "qwen"
    log INFO "All services stopped"
}

status_all() {
    for svc in qwen backend frontend; do
        local pid_file="$PID_DIR/${svc}.pid"
        if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1; then
            log INFO "$svc running (PID $(cat "$pid_file"))"
        else
            log INFO "$svc not running"
        fi
    done
}

show_logs() {
    local name="$1"
    local log_file="$LOG_DIR/${name}.log"
    if [ -f "$log_file" ]; then
        tail -n 100 "$log_file"
    else
        log WARN "No log file for $name"
    fi
}

usage() {
    cat <<EOF
Usage: ./script.sh <command>

Commands:
  install        Install Python + Node deps and required weights
  start          Launch Qwen, backend, and frontend
  stop           Stop all services
  status         Show service status
  logs <service> Tail the last 100 lines of logs (service: qwen|backend|frontend)
  help           Show this message

Environment overrides:
  BEST_PT_URL   Direct download URL for best.pt if missing
  QWEN_REPO     Hugging Face repo slug (default Qwen/Qwen3-VL-8B-Instruct)
  VLLM_ARGS     Extra flags for vLLM server
  PY_BIN / PIP_BIN / NPM_BIN / HF_CLI_BIN for custom tool paths
  BACKEND_PORT / FRONTEND_PORT / QWEN_PORT to change listen ports
EOF
}

cmd=${1:-help}
case "$cmd" in
    install) install_all ;;
    start) start_all ;;
    stop) stop_all ;;
    status) status_all ;;
    logs)
        if [ -z "${2:-}" ]; then
            usage
            exit 1
        fi
        show_logs "$2"
        ;;
    help|*) usage ;;
esac
