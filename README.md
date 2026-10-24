# PVEL-AD Solar Cell Defect Inspection

A local inspection application for electroluminescence (EL) images of photovoltaic cells. A custom YOLO26-GhostLite detector proposes defect bounding boxes; an optional Ollama vision model turns those detector results into a cautious written explanation. The React/Vite frontend uploads images to a FastAPI backend and displays results and health status.

Detector output is a model prediction, not a verified physical defect assessment. Do not use a single image or this prototype to make safety or repair decisions.

## Architecture

- `frontend/`: React interface built with Vite. Its API client defaults to `http://localhost:8000`.
- `backend.py`: FastAPI endpoints for health, image inspection, result images, and in-memory recent history. Models load lazily.
- `custom_yolo.py` and `modules/ghost_blocks.py`: register custom Ghost layers and load a model from its architecture YAML.
- `models/yolo26_ghostlite.yaml` and `models/yolo26_ghost.yaml`: model architecture definitions; keep these in source control.
- `train_ghostlite.py`, `continue_ghostlite.py`, `evaluate_ghostlite.py`, and related scripts: training, continuation, evaluation, repair, and audit workflows.
- `solar_inspector.py`: command-line inspection flow with optional Ollama explanation.

The API currently expects the local repaired dataset manifest at `yolo_dataset_repaired_8class/data.yaml` and detector weights at `PVEL_Detection_Repaired/ghostlite/weights/best.pt`. The dataset is used to read class names; inference requires the model YAML and trained weights. The application can run without Ollama, but the explanatory report will be unavailable.

## Setup

Use Python 3.10+ (the installed PyTorch build must support your Python version and hardware) and Node.js 18+. Install PyTorch using the command recommended for your CPU/CUDA setup, then install the remaining API dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-api.txt
```

Install and launch the frontend in a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Start the backend from the repository root in another terminal:

```powershell
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

Open the Vite URL printed in the frontend terminal (normally `http://localhost:5173`). The API health endpoint is `http://localhost:8000/api/health`.

For Ollama explanations, install and start Ollama separately and pull the configured vision model (default `qwen3-vl:4b-instruct`). To override settings in PowerShell, set `$env:OLLAMA_URL` and `$env:OLLAMA_MODEL` before starting the backend. `VITE_API_URL` is a Vite build-time setting; set it in `frontend/.env.local` if the API is hosted elsewhere. `.env.example` contains example values only; the application does not automatically load a root `.env` file.

## Local datasets and weights

Do not commit datasets, repaired dataset copies, labels, trained weights, or inspection images. Place authorized local data in the expected ignored directories:

- Dataset images and labels: `yolo_dataset_repaired_8class/images/{train,val}` and `yolo_dataset_repaired_8class/labels/{train,val}`.
- Dataset manifest: create `yolo_dataset_repaired_8class/data.yaml` with the eight class names and paths relative to the dataset root. The expected class ID order is `crack`, `finger`, `black_core`, `thick_line`, `star_crack`, `horizontal_dislocation`, `vertical_dislocation`, `short_circuit`.
- Weights: place the trained checkpoint at `PVEL_Detection_Repaired/ghostlite/weights/best.pt` (and `last.pt` if needed).
- Additional original or repaired datasets can stay in `solar_cell_EL_image/`, `yolo_dataset/`, or another ignored local location. Update the local dataset YAML/script path when training against a different copy.

The current training scripts expect `yolo_dataset_repaired_8class/data.yaml`. The repository's existing local dataset YAMLs use machine-specific absolute paths, so create a local manifest with paths for your machine rather than copying those absolute paths to a shared configuration. Keep the directory structure and local manifests out of Git; `.gitignore` excludes the known dataset roots, weights, runs, caches, evaluation output, and inspection uploads/results.

## Repository hygiene

`requirements-api.txt`, `frontend/package.json`, and `frontend/package-lock.json` are dependency manifests and should remain tracked. Architecture YAMLs under `models/`, Python source, scripts, frontend source, and documentation are also intended for Git. Dataset contents and model artifacts are local inputs and are ignored. Before a first commit, inspect `git status --short` and `git status --short --ignored`; this workspace currently has no `.git` repository, so no tracked files or staged files can be audited until Git is initialized.

If unwanted files were added to the Git index in a repository, remove only their index entries while preserving local files, for example:

```powershell
git rm -r --cached -- yolo_dataset yolo_dataset_repaired_8class solar_cell_EL_image PVEL_Detection_Repaired evaluation_results inspection_test label_audit
git rm --cached -- yolo26n.pt
```

Review the paths before running those commands and include any other tracked artifact paths found by `git ls-files`. If such files were already committed, `.gitignore` alone does not remove them from history; coordinate history cleanup before publishing.
