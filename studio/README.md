# Haiku Studio for h2

This folder is an optional UI shell for the h2 repo. It should live directly inside the h2 repo root.

## Desktop launch

```bat
cd studio
npm install
npm run studio
```

Or from the repo root:

```bat
launch_haiku_studio.bat
```

## Development launch

```bat
cd studio
npm run dev
```

## Architecture

- `electron.cjs` opens a desktop window.
- `server.ts` serves the React UI and exposes local API endpoints.
- `server.ts` does not run the old Flask engine.
- Training endpoints spawn `../pretrain.py`, `../sft.py`, `../dpo.py`, and `../tools/train_tokenizer.py`.
- Chat Lab uses `../tools/chat_once.py` for checkpoint inference.
- Chat Lab feedback can append prompt/chosen/rejected preference pairs to `../dpo/studio_feedback.jsonl`.
- All persistent training/model settings come from `../config/gpt_config.json`.


## Desktop renderer diagnostics

The desktop launcher writes backend logs to `studio/logs/backend.log` and renderer console errors to `studio/logs/renderer.log`. The normal launcher builds and serves the React bundle from `studio/dist/` so Electron does not depend on a live Vite dev transform to show the UI.
