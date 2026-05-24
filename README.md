# Haiku Studio

Haiku Studio is a local training workspace for small GPT-style language models. The repository combines the `h2` Python training engine with an optional Electron desktop interface for managing tokenizers, project folders, checkpoints, pretraining, supervised fine-tuning, Direct Preference Optimization, and local checkpoint chat testing.

The core training path is Python-first. The desktop app is an optional control surface that stages the selected project into the runtime `data/` and `config/` folders, starts the same Python scripts used from the command line, and streams process output into the in-app kernel logger.

## Features

- GPT-style transformer training with RoPE, RMSNorm, SwiGLU MLPs, and configurable grouped-query attention.
- Deterministic pretokenized cache workflow for resumable pretraining.
- Supervised fine-tuning on `user:` / `bot:` dialogue data with completion-only loss masking.
- Direct Preference Optimization for preference-pair alignment.
- Built-in tokenizer builder for either a single text file or a corpus folder.
- Project-based artifact management for tokenizers, configs, checkpoints, datasets, caches, and logs.
- Optional Electron desktop UI that runs as a local app rather than a Flask browser window.
- Local Chat Lab for testing checkpoints from the active project.
- Hugging Face export tooling for project artifacts.

## Repository layout

```text
.
├─ pretrain.py                 # pretraining engine
├─ sft.py                      # supervised fine-tuning engine
├─ dpo.py                      # preference optimization engine
├─ tokenizer_bpe.py            # tokenizer implementation
├─ config/
│  └─ gpt_config.json          # runtime config staged from the active project
├─ data/
│  └─ tokenizer.json           # runtime tokenizer staged from the active project
├─ projects/
│  └─ haiku_studio/            # default project
│     ├─ tokenizer.json
│     ├─ config/gpt_config.json
│     ├─ checkpoints/
│     ├─ logs/
│     ├─ cache/pretokenized/
│     └─ datasets/
│        ├─ corpus/
│        ├─ sft/
│        └─ dpo/
├─ tools/
│  ├─ train_tokenizer.py
│  └─ chat_once.py
└─ studio/                     # optional Electron UI
   ├─ server.ts
   ├─ electron.cjs
   ├─ preload.cjs
   ├─ public/
   └─ src/
```

## Project model

Projects are the persistent source of truth. The global `data/` and `config/` folders are runtime staging areas.

When a project is loaded, Haiku Studio syncs the project tokenizer and config into:

```text
data/tokenizer.json
config/gpt_config.json
```

When training runs, checkpoints, logs, and cache files are written back to the active project folder. This keeps separate experiments isolated while preserving compatibility with command-line scripts that expect runtime paths.

A project folder contains:

```text
projects/<project_name>/
├─ tokenizer.json
├─ config/gpt_config.json
├─ checkpoints/
├─ logs/
├─ cache/
│  └─ pretokenized/
└─ datasets/
   ├─ corpus/
   ├─ sft/
   └─ dpo/
```

## Requirements

### Python

Recommended:

- Python 3.10 or newer
- PyTorch with CUDA support for GPU training
- NumPy
- tokenizers

Install the Python dependencies from the repository root:

```bat
pip install -r requirements.txt
```

Install the correct PyTorch build for your CUDA version from the official PyTorch install selector when GPU training is required.

On linux systems you can use the nix package manager to install all dependencies.
```bash
nix-shell
```

### Desktop UI

Recommended:

- Node.js 20 LTS or newer
- npm

The launcher installs UI dependencies inside `studio/` when required.

## Launching the desktop app

From the repository root:

Windows:
```bat
launch_haiku_studio.bat
```

Linux:
```bash
bash linux-run.sh
```

The app opens as an Electron desktop window. The local API runs on `127.0.0.1` and starts the Python training processes behind the UI.

If a dependency install is interrupted, repair the UI dependencies with:

```bat
repair_studio_deps.bat
```

## Command-line workflows

The UI and command line use the same Python scripts and config file. You can use either workflow.

### Train or restore the tokenizer

The desktop app can build a tokenizer from a single text file or a corpus folder. It saves the tokenizer to both:

```text
data/tokenizer.json
projects/<active_project>/tokenizer.json
```

The bundled fallback tokenizer is stored at:

```text
studio/prebuilt/default_tokenizer.json
```

### Build the pretokenized cache

```bat
python pretrain.py --config config/gpt_config.json --build-cache --cache-only
```

### Start pretraining

```bat
python pretrain.py --config config/gpt_config.json
```

Resume from the latest continuation checkpoint:

```bat
python pretrain.py --config config/gpt_config.json --resume
```

### Run supervised fine-tuning

```bat
python sft.py --config config/gpt_config.json
```

Resume SFT:

```bat
python sft.py --config config/gpt_config.json --resume
```

### Run DPO alignment

```bat
python dpo.py --config config/gpt_config.json
```

Resume DPO:

```bat
python dpo.py --config config/gpt_config.json --resume
```

## Dataset formats

### Pretraining corpus

Pretraining data can be organized as one or more plain text files in the active project's corpus folder:

```text
projects/<project>/datasets/corpus/
```

The pretokenized cache builder creates deterministic train/validation splits and stores cache files under the active project.

### SFT data

SFT files use `user:` / `bot:` dialogue blocks separated by blank lines:

```text
user: Explain GPUs simply.
bot: A GPU is a processor designed to run many math operations in parallel.

user: What is your name?
bot: I am Haiku.
```

Only assistant reply content is supervised. User text and role tags remain context but are excluded from the loss target.

### DPO data

DPO accepts JSONL preference pairs:

```json
{"prompt":"user: Explain gravity.\nbot:","chosen":"Gravity is the attraction between masses.","rejected":"Gravity is when objects randomly fall."}
```

It also supports text blocks using `chosen:` and `rejected:` fields. The Chat Lab can create preference pairs interactively from thumbs-up and thumbs-down feedback.

## Checkpoints and logs

By default, active project artifacts are stored under:

```text
projects/<project>/checkpoints/
projects/<project>/logs/
projects/<project>/cache/
```

Typical checkpoint names include:

```text
model.pt
model.best.pt
model.sft.pt
model.sft.best.pt
model.dpo.pt
model.dpo.best.pt
```

The Chat Lab loads the best available checkpoint from the active project, prioritizing DPO checkpoints first, then SFT checkpoints, then base pretraining checkpoints.

## Configuration

The main configuration file is:

```text
config/gpt_config.json
```

In the desktop workflow, this file is staged from the selected project. Important sections include:

- `paths`: tokenizer, datasets, checkpoints, caches, and logs.
- `model`: architecture size, attention layout, RoPE, dropout, and embedding settings.
- `data`: context length, validation split, cache behavior, and document handling.
- `train`: pretraining batch size, epochs, accumulation, save cadence, and evaluation cadence.
- `optimizer`: AdamW settings, learning rate schedule, weight decay, and gradient clipping.
- `sft`: supervised fine-tuning settings.
- `dpo`: preference optimization settings.

## Runtime logs

Desktop app logs are written to:

```text
studio/logs/
```

Training metrics are written to the active project:

```text
projects/<project>/logs/
```

Large generated files, checkpoints, cache files, `node_modules`, and desktop logs are intentionally ignored by Git.

## Development notes

Run the UI in development mode:

```bat
launch_haiku_studio_dev.bat
```

Or manually:

```bat
cd studio
npm install
npm run dev
```

Build the frontend bundle:

```bat
cd studio
npm run build
```

Run TypeScript checks:

```bat
cd studio
npm run lint
```

## Scope

This repository is designed for local experimentation and training of small to mid-sized GPT-style models on consumer or workstation hardware. It does not include distributed training, tensor parallelism, FSDP, ZeRO sharding, or a managed cloud training backend.
