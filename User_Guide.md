# GeoNLI Deployment & Inference Guide

This document can be exported directly to **User_Guide.pdf** (for example, `pandoc User_Guide.md -o User_Guide.pdf`). It summarizes how to install, run, and interact with the GeoNLI application, including the automated deployment script and inference workflows.

---

## 1. System Overview

The project provides:
- **Backend API** (`backend/server.py`) powered by FastAPI + YOLO OBB detector + Qwen multimodal LLM.
- **Frontend UI** (`frontend/`) built with Vite/React for interactive querying.
- **Inference utilities** (`inference.py`, `test.py`) for standalone batch processing and prompt-level checks.
- **Deployment script** (`script.sh`) to automate installation, startup, and shutdown across all services.

All components run locally but expect the Qwen OpenAI-compatible server (via vLLM) to be active on `localhost:8000`.

---

## 2. Prerequisites

1. **Operating System**: Linux (tested on Ubuntu 22.04+).
2. **Python**: 3.10+ (conda environment `vllm_main` recommended).
3. **Node.js & npm**: Node 18+ / npm 9+ for the Vite frontend.
4. **huggingface-cli**: Required if weights must be downloaded from Hugging Face.
5. **GPU**: Strongly recommended for running vLLM (CUDA-ready drivers installed).
6. **Network access**: Needed only if model files (`best.pt`, Qwen weights) are not already present locally.

Optional but recommended:
- `conda` or `mamba` for environment isolation.
- `pm2` or systemd if you plan to daemonize services beyond the provided script.

---

## 3. Directory Highlights

```
/data1/MP3
├── backend/             # FastAPI + YOLO logic
├── frontend/            # React/Vite UI
├── inference.py         # Batch JSON inference script
├── script.sh            # Deployment/automation helper
├── test.py              # Quick Qwen endpoint sanity check
├── best.pt              # YOLO OBB checkpoint (required)
├── qwen/                # Qwen3-VL-8B-Instruct weights (required)
└── sample_dataset_...   # Example query/response payloads
```

---

## 4. Deployment Script Usage (`script.sh`)

> Ensure the file is executable: `chmod +x script.sh`

### 4.1 Environment Variables

Set these **before** running `script.sh install` if needed:
- `BEST_PT_URL`: Direct download URL for `best.pt` (skip if file already exists).
- `QWEN_REPO`: Hugging Face repo slug (default `Qwen/Qwen3-VL-8B-Instruct`).
- `VLLM_ARGS`: Extra flags for the vLLM server (default `--max-model-len 2048 --dtype bfloat16`).
- `PY_BIN`, `PIP_BIN`, `NPM_BIN`, `HF_CLI_BIN`: Override executables if not on PATH.

### 4.2 Installation

```
./script.sh install
```

What it does:
1. Downloads `best.pt`.
2. Downloads the Qwen model directory.
3. Installs Python dependencies from `requirements.txt`, plus `vllm` and `huggingface-hub`.
4. Installs frontend dependencies via `npm install` in `frontend/`.

### 4.3 Launching Services

```
./script.sh start
```

Starts three background processes (logs under `.logs/`, PIDs under `.pids/`):
1. **Qwen Server**: `python -m vllm.entrypoints.openai.api_server ... --port 8000`
2. **Backend API**: `uvicorn backend.server:app --port 15200`
3. **Frontend UI**: `npm run dev -- --port 5173`

Endpoints after startup:
- Qwen API: `http://localhost:8000/v1/chat/completions`
- Backend REST API: `http://localhost:15200`
- Frontend UI: `http://localhost:5173`

### 4.4 Status & Shutdown

```
./script.sh status   # Show which services are running
./script.sh stop     # Gracefully stop frontend, backend, and Qwen server
```

---

## 5. Manual Operation (without script)

If you prefer to run each service manually:

1. **Python deps**: `pip install -r requirements.txt && pip install vllm huggingface-hub`
2. **Frontend deps**: `cd frontend && npm install`
3. **Start Qwen server** (new terminal, inside `vllm_main`):
   ```
   python -m vllm.entrypoints.openai.api_server \
     --model /data1/MP3/qwen \
     --host 0.0.0.0 --port 8000 \
     --max-model-len 8192 --dtype bfloat16
   ```
4. **Start backend**: `uvicorn backend.server:app --host 0.0.0.0 --port 15200`
5. **Start frontend**: `cd frontend && npm run dev -- --host 0.0.0.0 --port 5173`

---

## 6. Using the Frontend Website

1. Open `http://localhost:5173` in your browser.
2. Upload an image or drag-and-drop onto the drop zone.
3. Enter a natural language query (caption, counting, detection, etc.).
4. Submit; the UI calls `/geoNLI/eval` on the backend, which internally routes to YOLO + Qwen.
5. Review the answer in the chat panel; for grounding queries you’ll see textual descriptions (current UI does not render boxes automatically).

> **Note:** The backend uses Qwen for semantic reasoning and YOLO for object detection. If Qwen is offline, you’ll see “Unable to process this query” responses.

---

## 7. Structured Inference from Terminal

### 7.1 Batch JSON Inference

```
python inference.py \
  sample_dataset_inter_iit_v1_3/sample1_query.json \
  sample_dataset_inter_iit_v1_3/sample1_response.json
```

- Reads input image URL + queries from the JSON file.
- Calls Qwen for caption, binary, numeric, and semantic answers.
- Runs YOLO OBB for grounding targets and injects physical `metrics` (width/height/area using `spatial_resolution_m`).
- Saves a visualization PNG alongside the response JSON.

### 7.2 Quick Prompt Test

```
python test.py
```

Sends a single prompt + image link to `http://localhost:8000/v1/chat/completions` and prints the raw response—useful for sanity-checking Qwen availability.

---

## 8. Model Asset Management

- **YOLO weights (`best.pt`)**: Either keep the file in repo root or set `BEST_PT_URL` so `script.sh install` can download it.
- **Qwen weights**: Expect the `qwen/` folder to contain Hugging Face files (`config.json`, `model-*.safetensors`, tokenizer, etc.). If missing, ensure you are logged into `huggingface-cli login` before running the installer.
- **Environment variables** can be persisted via `.env` or export statements in your shell profile.

---

## 9. Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `Connection refused` when calling `http://localhost:8000` | Qwen/vLLM server not running | Run `./script.sh start` or manual vLLM command; check `.logs/qwen.log` |
| `ModuleNotFoundError: cv2` | Python deps missing in current env | Activate `vllm_main` and run `pip install -r requirements.txt` |
| Frontend can’t reach backend (`Network Error`) | Backend not running or CORS blocked | Ensure backend is up on port 15200; restart via script |
| YOLO returns zero boxes | Target class normalization failed (Qwen returned `None`) | Confirm Qwen is healthy; review backend logs for prompt failures |
| `huggingface-cli download` fails | Auth or rate limit issues | Run `huggingface-cli login` and retry `./script.sh install` |

Log locations: `.logs/qwen.log`, `.logs/backend.log`, `.logs/frontend.log`.

---

For further assistance or customization, reach out to the development team or consult the project README files.
