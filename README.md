# Haiku Training Upgrade

This repository contains a safer pretraining/SFT workflow for the Haiku/Tanka-style GPT architecture.

## Scope

Designed and tested for single-GPU training of models from roughly 100M to ~1B parameters on consumer hardware (8–24 GB VRAM). The architecture (GQA + SwiGLU + RoPE + RMSNorm) generalizes upward, but training a 7B-class model in this script is impractical: at that scale you need FSDP / tensor parallelism / ZeRO sharding and a sharded data layout, none of which are included here. If you want to push past ~1B on a single card, you will hit either VRAM limits (params + grads + optimizer state) or a 2-TB-class bin file that mmap will tolerate but is operationally annoying.

## Files

```text
pretrain.py
sft.py
dpo.py
config/gpt_config.json
README.md
```

The code expects your existing `tokenizer_bpe.py` and `data/tokenizer.json` to be present beside the scripts or in the configured paths.

## What this fixes

### Resume correctness

The old trainer restored model weights and optimizer state, but it did not restore the active data position. That meant a resumed run could restart epoch streaming from the beginning while keeping the old `global_step`.

The new `pretrain.py` stores:

```text
model weights
optimizer state
AMP scaler state
global_step
epoch_idx
sample_cursor
best_val
tokenizer hash
pretokenized cache signature
RNG states
```

The console ETA and steps/s now use:

```text
session_steps = global_step - session_start_step
```

so a resumed run no longer divides lifetime steps by only the resumed process runtime.

### Stable validation split

The validation split is deterministic. Documents are assigned to train/val using:

```text
seed + file index + document index
```

The pretokenized cache stores a manifest containing the tokenizer hash, corpus fingerprints, split seed, val ratio, document-splitting settings, and EOT marker. If any split-affecting setting changes, the trainer refuses to reuse the old cache.

### Pretokenized cache

`pretrain.py` can build:

```text
data/pretokenized/train.bin
data/pretokenized/val.bin
data/pretokenized/manifest.json
```

Training then runs from the `.bin` files instead of live text streaming. This makes resume simple and exact from the last checkpoint made by this script.

Incomplete cache builds are not trusted. Cache files are written as temporary files and atomically renamed only after successful completion. If cache settings change, rebuild the cache.

### Per-epoch window order

Training windows in the pretokenized bin are walked in a deterministic permuted order, reseeded each epoch from `(runtime.seed, epoch_idx)`. Two epochs of the same data never present windows in the same sequence, which avoids the failure mode where the loss curve mirrors the macro-order of the cached corpus. Resuming mid-epoch preserves the cursor inside the current epoch's permutation; on epoch wrap the next permutation is generated automatically.

## Directory layout

Recommended layout:

```text
repo/
  pretrain.py
  sft.py
  dpo.py
  tokenizer_bpe.py
  README.md
  config/
    gpt_config.json
  data/
    tokenizer.json
  corpus/
    Arxiv_Corpus.txt
    CommonNews.txt
    FineWeb.txt
    ...
  sft/
    sft_samples.txt
  dpo/
    preference_pairs.jsonl
```

## Configuration

Everything important is configured in:

```text
config/gpt_config.json
```

Important sections:

### `paths`

Controls tokenizer, corpus, checkpoints, caches, logs, SFT dataset paths, and DPO preference dataset paths.

### `model`

Controls architecture:

```json
{
  "block_size": 1024,
  "n_layer": 26,
  "n_head": 8,
  "n_kv_head": 8,
  "n_embd": 1024,
  "dropout": 0.0,
  "rope_base": 50000.0,
  "grad_checkpoint": true,
  "hidden_mult": 2.67,
  "tied_embeddings": false,
  "residual_init_scale": true
}
```

Set `"tied_embeddings": true` if you want the output head to share weights with the token embedding table. Leave it `false` for compatibility with your untied Tanka-style checkpoints.

### `data`

Controls pretokenization, validation split, corpus document boundaries, shuffle buffer, and stride.

### `train`

Controls pretraining epochs, batch size, grad accumulation, and logit chunking.

### `optimizer`

Controls AdamW, LR range, weight decay, betas, grad clipping, and foreach mode.

### `sft`

Controls supervised fine-tuning settings, including lower LR, SFT batch size, completion-only masking, eval/save cadence, and SFT split ratio.

### `dpo`

Controls Direct Preference Optimization settings: policy checkpoint, frozen reference checkpoint, beta, DPO LR, batch size, grad accumulation, preference split ratio, and checkpoint cadence.

## Pretraining workflow

### 1. Build the pretokenized cache

```bat
python pretrain.py --config config/gpt_config.json --build-cache --cache-only
```

This creates:

```text
data/pretokenized/train.bin
data/pretokenized/val.bin
data/pretokenized/manifest.json
```

### 2. Start pretraining

```bat
python pretrain.py --config config/gpt_config.json --resume
```

If no checkpoint exists yet, omit `--resume`:

```bat
python pretrain.py --config config/gpt_config.json
```

### 3. Resume pretraining

```bat
python pretrain.py --config config/gpt_config.json --resume
```

The console should show the restored step, epoch, and sample cursor.

### 4. Rebuild cache intentionally

Only do this when you intentionally changed tokenizer, corpus, split settings, or document-boundary settings:

```bat
python pretrain.py --config config/gpt_config.json --build-cache --force-rebuild-cache --cache-only
```

## Important checkpoint behavior

`data/model.pt` is the continuation checkpoint. It includes optimizer/scaler/training state.

`data/model.best.pt` is also saved with full state by this upgraded trainer, but the safest continuation target is still `data/model.pt`.

If you resume from an older `pretrain_v2.py` checkpoint that does not have `sample_cursor`, the script approximates cursor position from:

```text
global_step * batch_size * grad_accum
```

After the first new checkpoint from this upgraded trainer, future resumes are exact from that checkpoint.

## SFT dataset format

SFT input is plain text.

Each dialog/document is separated by a blank line:

```text
user: Hello.
bot: Hi. What can I help with?

user: What is your name?
bot: I am Haiku.
```

Multi-turn dialogs are allowed:

```text
user: Explain GPUs simply.
bot: A GPU is a processor built to do many small math operations in parallel.
user: Why does that help AI?
bot: Neural networks use many matrix operations, so GPUs can run them much faster than CPUs.
```

Only `bot:` reply content is supervised. `user:` content and role tags are used as context but ignored in the loss.

## SFT workflow

### 1. Put SFT files in the configured folder

Default:

```text
sft/
  sft_samples.txt
```

### 2. Run SFT from the best pretrained checkpoint

```bat
python sft.py --config config/gpt_config.json
```

By default, this loads:

```text
data/model.best.pt
```

and saves:

```text
data/model.sft.pt
data/model.sft.best.pt
```

### 3. Resume SFT

```bat
python sft.py --config config/gpt_config.json --resume
```

SFT resume stores:

```text
global_step
epoch_idx
batch_cursor
optimizer state
AMP scaler state
dataset signature
tokenizer hash
```

## Loss masking details for SFT

For a dialog like:

```text
user: Say hi.
bot: Hi.
```

the model sees the whole sequence as context, but labels are set to ignore index for the user prompt and the `bot:` tag. Only the actual bot reply tokens are used in the cross-entropy loss.

This avoids training the model to copy user prompts and focuses fine-tuning on the assistant response distribution.

## Recommended commands

Pretraining on an 8 GB / 11 GB single GPU:

```bat
python pretrain.py --config config/gpt_config.json --build-cache --cache-only
python pretrain.py --config config/gpt_config.json
```

Resume:

```bat
python pretrain.py --config config/gpt_config.json --resume
```

SFT:

```bat
python sft.py --config config/gpt_config.json
```

Resume SFT:

```bat
python sft.py --config config/gpt_config.json --resume
```

## Safety checks

The trainer refuses to resume if:

```text
tokenizer hash changed
pretokenized cache signature changed
SFT dataset signature changed
checkpoint path is missing while --resume is requested
```

The trainer also guards against validation sets smaller than one full context window.

## Notes

- Pretokenized training uses sequential token windows from an already interleaved cache. The interleaving happens during cache construction.
- The cache is deterministic under the same tokenizer, corpus files, seed, validation ratio, document split settings, and EOT marker.
- Exact resume means exact from the last checkpoint written by this upgraded trainer. A hard process crash can still lose progress since the last `save_every` checkpoint.
- For maximum safety during long runs, periodically copy `data/model.pt`, `data/model.best.pt`, the log file, `data/tokenizer.json`, and `data/pretokenized/manifest.json` to an external backup folder.

---

# Optional Haiku Studio UI

This build includes `studio/`, extracted from the Haiku Studio UI archive and rewired so **h2 remains the working engine**.

The UI no longer launches the old duplicated Flask Python engine. The desktop/backend layer uses Node/Electron for the application shell and runs the h2 Python scripts behind it:

```text
studio/server.ts              local app API + process manager
studio/electron.cjs           desktop window/container launcher
pretrain.py                   h2 pretraining engine
sft.py                        h2 SFT engine
dpo.py                        h2 DPO preference-alignment engine
tools/train_tokenizer.py      h2 tokenizer training bridge
tools/chat_once.py            h2 checkpoint chat bridge
config/gpt_config.json        single source of truth for model/training config
```

## Launch as a desktop-style app on Windows

From the repo root:

```bat
launch_haiku_studio.bat
```

Or manually:

```bat
cd studio
npm install
npm run studio
```

This opens Haiku Studio in its own Electron window. It is not a Flask app in a browser tab. The UI API listens only on `127.0.0.1` and spawns h2 Python processes as needed.

## Optional browser/dev mode

For UI debugging only:

```bat
cd studio
npm install
npm run dev
```

Then open the printed local URL. Normal use should be `npm run studio` or `launch_haiku_studio.bat`.

## What the UI controls

The UI writes to the same h2 config file used by command-line training:

```text
config/gpt_config.json
```

Pretraining in the UI launches:

```bat
python pretrain.py --config config/gpt_config.json
```

If `data/model.pt` already exists, the UI automatically adds `--resume`.

SFT in the UI launches:

```bat
python sft.py --config config/gpt_config.json
```

If `data/model.sft.pt` already exists, the UI automatically adds `--resume`.

DPO in the UI launches:

```bat
python dpo.py --config config/gpt_config.json
```

If `data/model.dpo.pt` already exists, the UI automatically adds `--resume`. DPO reads prompt/chosen/rejected pairs from the configured `paths.dpo_dataset`, creates a frozen reference checkpoint from the configured SFT policy checkpoint if needed, and saves aligned checkpoints to `data/model.dpo.pt` / `data/model.dpo.best.pt`.

Tokenizer training launches:

```bat
python tools/train_tokenizer.py --input corpus --output data/tokenizer.json
```

Chat Lab uses `tools/chat_once.py`, loading the best available checkpoint in this order:

```text
data/model.dpo.best.pt
data/model.dpo.pt
data/model.sft.best.pt
data/model.sft.pt
data/model.best.pt
data/model.pt
```

## DPO preference format

DPO accepts JSONL records:

```json
{"prompt":"user: Explain X.\nbot:","chosen":"Clear preferred answer.","rejected":"Weak rejected answer."}
```

It also accepts blank-line-separated text blocks:

```text
user: Explain X.
chosen: Clear preferred answer.
rejected: Weak rejected answer.
```

The Chat Lab thumbs-up/down actions can add DPO pairs, but DPO requires both sides of the comparison. If you mark the current answer as chosen, the UI asks for a rejected alternative. If you mark it as rejected, the UI asks for a preferred replacement.

## Dependency notes

Python dependencies are the h2 dependencies: PyTorch, NumPy, tokenizers, and the packages already needed by `pretrain.py` / `sft.py`.

UI dependencies are installed inside `studio/` with npm. Electron is used only as the optional desktop container.
