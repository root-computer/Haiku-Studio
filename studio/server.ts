import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import fs from "fs";
import { spawn, ChildProcessWithoutNullStreams } from "child_process";
import axios from "axios";
import { fileURLToPath } from "url";
import { listDatasets, commit, createRepo } from "@huggingface/hub";

const __filename = fileURLToPath(import.meta.url);
const STUDIO_DIR = path.dirname(__filename);
const REPO_ROOT = path.resolve(STUDIO_DIR, "..");
const SETTINGS_PATH = path.join(REPO_ROOT, ".haiku_studio.json");
const CONFIG_PATH = path.join(REPO_ROOT, "config", "gpt_config.json");
const PORT = Number(process.env.HAIKU_STUDIO_PORT || 3000);

type JsonObject = Record<string, any>;

type RunningJob = {
  name: string;
  child: ChildProcessWithoutNullStreams;
  startedAt: number;
};

let runningJob: RunningJob | null = null;
let logCursor = 0;
const logLines: Array<[number, string]> = [];
const MAX_LOG_LINES = 2000;

function pushLog(line: string) {
  const clean = line.replace(/\r/g, "").trimEnd();
  if (!clean) return;
  for (const part of clean.split("\n")) {
    const text = part.trimEnd();
    if (!text) continue;
    logLines.push([logCursor++, text]);
    while (logLines.length > MAX_LOG_LINES) logLines.shift();
    console.log(text);
  }
}

function readJson(filePath: string, fallback: JsonObject = {}): JsonObject {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (err: any) {
    pushLog(`[settings] Failed to read ${filePath}: ${err.message}`);
    return fallback;
  }
}

function writeJsonAtomic(filePath: string, data: JsonObject) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tmp = `${filePath}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2), "utf8");
  fs.renameSync(tmp, filePath);
}

function deepGet(obj: JsonObject, dotted: string, fallback: any = undefined): any {
  let cur: any = obj;
  for (const part of dotted.split(".")) {
    if (cur == null || typeof cur !== "object" || !(part in cur)) return fallback;
    cur = cur[part];
  }
  return cur;
}

function deepSet(obj: JsonObject, dotted: string, value: any) {
  const parts = dotted.split(".");
  let cur = obj;
  for (const part of parts.slice(0, -1)) {
    if (!cur[part] || typeof cur[part] !== "object") cur[part] = {};
    cur = cur[part];
  }
  cur[parts[parts.length - 1]] = value;
}

function resolveRepoPath(p: string | undefined, fallback: string): string {
  const raw = p && p.length ? p : fallback;
  return path.isAbsolute(raw) ? raw : path.join(REPO_ROOT, raw);
}

function relToRepo(absPath: string): string {
  const rel = path.relative(REPO_ROOT, absPath);
  return rel && !rel.startsWith("..") ? rel.replace(/\\/g, "/") : absPath;
}

function pythonCmd(): string {
  return process.env.HAIKU_PYTHON || (process.platform === "win32" ? "python" : "python3");
}

function loadConfig(): JsonObject {
  return readJson(CONFIG_PATH, {});
}

function saveConfig(config: JsonObject) {
  writeJsonAtomic(CONFIG_PATH, config);
}

function loadUiSettings(): JsonObject {
  const defaults = { theme: "dark", show_tooltips: true, project_dir: REPO_ROOT };
  return { ...defaults, ...readJson(SETTINGS_PATH, {}) };
}

function saveUiSettings(settings: JsonObject) {
  writeJsonAtomic(SETTINGS_PATH, settings);
}

function normalizePreferenceObject(obj: JsonObject): boolean {
  const prompt = obj.prompt || obj.user || obj.input || obj.instruction;
  const chosen = obj.chosen || obj.preferred || obj.good || obj.accepted || obj.winner || obj.response_chosen;
  const rejected = obj.rejected || obj.dispreferred || obj.bad || obj.declined || obj.loser || obj.response_rejected;
  return Boolean(String(prompt || "").trim() && String(chosen || "").trim() && String(rejected || "").trim() && String(chosen) !== String(rejected));
}

function countPreferencePairs(): number {
  const config = loadConfig();
  const dpoPath = resolveRepoPath(deepGet(config, "paths.dpo_dataset", "dpo"), "dpo");
  if (!fs.existsSync(dpoPath)) return 0;
  const files = fs.statSync(dpoPath).isDirectory()
    ? fs.readdirSync(dpoPath).map(f => path.join(dpoPath, f)).filter(f => fs.statSync(f).isFile() && path.basename(f) !== ".gitkeep")
    : [dpoPath];
  let count = 0;
  for (const file of files) {
    const text = fs.readFileSync(file, "utf8");
    if (file.toLowerCase().endsWith(".jsonl")) {
      for (const line of text.split(/\r?\n/)) {
        const clean = line.trim();
        if (!clean) continue;
        try { if (normalizePreferenceObject(JSON.parse(clean))) count += 1; } catch {}
      }
    } else {
      for (const block of text.split(/\n\s*\n/g)) {
        const low = block.toLowerCase();
        if ((low.includes("user:") || low.includes("prompt:")) && low.includes("chosen:") && low.includes("rejected:")) count += 1;
      }
    }
  }
  return count;
}

function latestStepFromLog(logPathRaw: string | undefined, fallback: string): number {
  const logPath = resolveRepoPath(logPathRaw, fallback);
  if (!fs.existsSync(logPath)) return 0;
  let best = 0;
  for (const line of fs.readFileSync(logPath, "utf8").split(/\r?\n/).filter(Boolean).slice(-1000)) {
    try {
      const row = JSON.parse(line);
      if (row.step !== undefined) best = Math.max(best, Number(row.step) || 0);
    } catch {}
  }
  return best;
}

function studioFeedbackPath(): string {
  const config = loadConfig();
  const raw = String(deepGet(config, "paths.dpo_dataset", "dpo"));
  const target = resolveRepoPath(raw, "dpo");
  const looksLikeFile = Boolean(path.extname(target));
  if (looksLikeFile) {
    fs.mkdirSync(path.dirname(target), { recursive: true });
    return target;
  }
  fs.mkdirSync(target, { recursive: true });
  return path.join(target, "studio_feedback.jsonl");
}

function getSettingsPayload(): JsonObject {
  const config = loadConfig();
  const ui = loadUiSettings();
  const model = config.model || {};
  const paths = config.paths || {};
  return {
    ...ui,
    project_dir: REPO_ROOT,
    projects: [path.basename(REPO_ROOT)],
    active_project: path.basename(REPO_ROOT),
    h2_repo_root: REPO_ROOT,
    h2_config_path: CONFIG_PATH,
    device: deepGet(config, "runtime.device", "auto"),
    model_params: estimateParams(model, tokenizerVocabSize(paths.tokenizer_path)),
    model_layers: model.n_layer ?? 0,
    model_dim: model.n_embd ?? 0,
    model_heads: model.n_head ?? 0,
    model_kv_heads: model.n_kv_head ?? model.n_head ?? 0,
    block_size: model.block_size ?? 0,
    vocab_size: tokenizerVocabSize(paths.tokenizer_path),
    dpo_ready: fs.existsSync(path.join(REPO_ROOT, "dpo.py")),
    dpo_buffer: countPreferencePairs(),
    dpo_global_step: latestStepFromLog(paths.dpo_log, "data/dpo_loss.jsonl"),
    dpo_dataset: deepGet(config, "paths.dpo_dataset", "dpo"),
    dpo_checkpoint: deepGet(config, "paths.dpo_checkpoint", "data/model.dpo.pt"),
    dpo_beta: deepGet(config, "dpo.beta", 0.1),
    dpo_lr: deepGet(config, "dpo.lr", 0.000003),
    dpo_epochs: deepGet(config, "dpo.epochs", 1),
    dpo_batch_size: deepGet(config, "dpo.batch_size", 1),
    corpus_dir: deepGet(config, "paths.corpus_dir", "corpus"),
    sft_dataset: deepGet(config, "paths.sft_dataset", "sft"),
    pretrain_checkpoint: deepGet(config, "paths.pretrain_checkpoint", "data/model.pt"),
    sft_checkpoint: deepGet(config, "paths.sft_checkpoint", "data/model.sft.pt"),
    is_training: Boolean(runningJob),
    current_task: runningJob?.name ?? null,
  };
}

function tokenizerVocabSize(tokenizerPathRaw: string | undefined): number {
  try {
    const tokPath = resolveRepoPath(tokenizerPathRaw, "data/tokenizer.json");
    if (!fs.existsSync(tokPath)) return 0;
    const tok = readJson(tokPath, {});
    const vocab = tok.model?.vocab;
    if (vocab && typeof vocab === "object") return Object.keys(vocab).length;
  } catch {}
  return 0;
}

function estimateParams(model: JsonObject, vocabSize: number): number {
  const nLayer = Number(model.n_layer || 0);
  const nEmb = Number(model.n_embd || 0);
  const nHead = Number(model.n_head || 1);
  const nKv = Number(model.n_kv_head || nHead || 1);
  const hiddenMult = Number(model.hidden_mult || 2.67);
  if (!nLayer || !nEmb || !vocabSize) return 0;
  const headDim = nEmb / nHead;
  const attn = nEmb * (nHead * headDim + 2 * nKv * headDim + nEmb);
  const mlpHidden = Math.floor(nEmb * hiddenMult);
  const mlp = nEmb * mlpHidden * 2 + mlpHidden * nEmb;
  const norms = 2 * nEmb;
  const blocks = nLayer * (attn + mlp + norms);
  const embeddings = vocabSize * nEmb;
  const head = model.tied_embeddings ? 0 : vocabSize * nEmb;
  return Math.round(blocks + embeddings + head + nEmb);
}

function updateConfigFromTrainingRequest(kind: "pretrain" | "sft" | "dpo", body: JsonObject): JsonObject {
  const config = loadConfig();
  if (kind === "pretrain") {
    if (body.corpus_dir) deepSet(config, "paths.corpus_dir", body.corpus_dir);
    if (body.epochs !== undefined) deepSet(config, "train.epochs", Number(body.epochs));
    if (body.lr !== undefined) deepSet(config, "optimizer.lr", Number(body.lr));
    if (body.batch_size !== undefined) deepSet(config, "train.batch_size", Number(body.batch_size));
    if (body.grad_accum !== undefined) deepSet(config, "train.grad_accum", Number(body.grad_accum));
    if (body.val_split !== undefined) deepSet(config, "data.val_ratio", Number(body.val_split));
    if (body.weight_decay !== undefined) deepSet(config, "optimizer.weight_decay", Number(body.weight_decay));
    if (body.min_lr_ratio !== undefined) {
      const lr = Number(deepGet(config, "optimizer.lr", 3e-4));
      deepSet(config, "optimizer.lr_min", lr * Number(body.min_lr_ratio));
    }
  } else if (kind === "sft") {
    if (body.data_path) deepSet(config, "paths.sft_dataset", body.data_path);
    if (body.epochs !== undefined) deepSet(config, "sft.epochs", Number(body.epochs));
    if (body.lr !== undefined) deepSet(config, "sft.lr", Number(body.lr));
    if (body.batch_size !== undefined) deepSet(config, "sft.batch_size", Number(body.batch_size));
    if (body.grad_accum !== undefined) deepSet(config, "sft.grad_accum", Number(body.grad_accum));
  } else {
    if (body.data_path) deepSet(config, "paths.dpo_dataset", body.data_path);
    if (body.epochs !== undefined) deepSet(config, "dpo.epochs", Number(body.epochs));
    if (body.lr !== undefined) deepSet(config, "dpo.lr", Number(body.lr));
    if (body.lr_min !== undefined) deepSet(config, "dpo.lr_min", Number(body.lr_min));
    if (body.beta !== undefined) deepSet(config, "dpo.beta", Number(body.beta));
    if (body.batch_size !== undefined) deepSet(config, "dpo.batch_size", Number(body.batch_size));
    if (body.grad_accum !== undefined) deepSet(config, "dpo.grad_accum", Number(body.grad_accum));
  }
  saveConfig(config);
  return config;
}

function updateConfigFromSettings(body: JsonObject) {
  const config = loadConfig();
  const ui = loadUiSettings();

  const uiKeys = ["theme", "show_tooltips"];
  for (const key of uiKeys) if (key in body) ui[key] = body[key];

  if (body.model_layers !== undefined) deepSet(config, "model.n_layer", Number(body.model_layers));
  if (body.model_dim !== undefined) deepSet(config, "model.n_embd", Number(body.model_dim));
  if (body.model_heads !== undefined) deepSet(config, "model.n_head", Number(body.model_heads));
  if (body.model_kv_heads !== undefined) deepSet(config, "model.n_kv_head", Number(body.model_kv_heads));
  if (body.block_size !== undefined) deepSet(config, "model.block_size", Number(body.block_size));
  if (body.device !== undefined) deepSet(config, "runtime.device", String(body.device));

  saveUiSettings(ui);
  saveConfig(config);
}

function spawnJob(name: string, args: string[], extraEnv: JsonObject = {}) {
  if (runningJob) throw new Error(`${runningJob.name} is already running`);
  const child = spawn(pythonCmd(), args, {
    cwd: REPO_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: "1", ...extraEnv },
    shell: false,
  });
  runningJob = { name, child, startedAt: Date.now() };
  pushLog(`[studio] Started ${name}: ${pythonCmd()} ${args.join(" ")}`);
  child.stdout.on("data", chunk => pushLog(String(chunk)));
  child.stderr.on("data", chunk => pushLog(String(chunk)));
  child.on("error", err => pushLog(`[studio] ${name} failed to start: ${err.message}`));
  child.on("close", code => {
    pushLog(`[studio] ${name} exited with code ${code}`);
    if (runningJob?.child === child) runningJob = null;
  });
  return child;
}

function stopRunningJob(): boolean {
  if (!runningJob) return false;
  pushLog(`[studio] Stop requested for ${runningJob.name}`);
  if (process.platform === "win32") {
    spawn("taskkill", ["/pid", String(runningJob.child.pid), "/T", "/F"]);
  } else {
    runningJob.child.kill("SIGINT");
  }
  return true;
}

function latestMetrics(): JsonObject {
  const config = loadConfig();
  const paths = config.paths || {};
  const files = [
    resolveRepoPath(paths.pretrain_log, "data/pretrain_loss.jsonl"),
    resolveRepoPath(paths.sft_log, "data/sft_loss.jsonl"),
    resolveRepoPath(paths.dpo_log, "data/dpo_loss.jsonl"),
  ];
  const steps: number[] = [];
  const trainLoss: number[] = [];
  const valLoss: number[] = [];
  const points: JsonObject[] = [];
  for (const f of files) {
    if (!fs.existsSync(f)) continue;
    const lines = fs.readFileSync(f, "utf8").split(/\r?\n/).filter(Boolean).slice(-500);
    for (const line of lines) {
      try {
        const row = JSON.parse(line);
        if (row.step === undefined) continue;
        points.push({ ...row, source: path.basename(f) });
        steps.push(Number(row.step));
        trainLoss.push(Number(row.train_loss ?? row.loss ?? NaN));
        valLoss.push(Number(row.val_loss ?? NaN));
      } catch {}
    }
  }
  return {
    steps,
    train_loss: trainLoss,
    val_loss: valLoss,
    meta: {
      repo_root: REPO_ROOT,
      running: Boolean(runningJob),
      current_task: runningJob?.name ?? null,
      last_points: points.slice(-50),
    },
  };
}

async function detectHardware(): Promise<JsonObject> {
  const script = [
    "import json",
    "info={'device':'cpu','name':'CPU','vram_total':0,'vram_free':0,'cuda_version':None,'compute_capability':None,'supports_bf16':False,'supports_flash_attn':False}",
    "try:",
    "    import torch",
    "    info['device']='cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu')",
    "    if torch.cuda.is_available():",
    "        idx=torch.cuda.current_device(); p=torch.cuda.get_device_properties(idx)",
    "        info.update({'name':p.name,'vram_total':p.total_memory/(1024**3),'vram_free':(p.total_memory-torch.cuda.memory_reserved(idx))/(1024**3),'cuda_version':torch.version.cuda,'compute_capability':f'{p.major}.{p.minor}','supports_bf16':p.major>=8,'supports_flash_attn':p.major>=8})",
    "except Exception as e:",
    "    info['error']=str(e)",
    "print(json.dumps(info))",
  ].join("\n");

  const out = await new Promise<string>((resolve) => {
    const child = spawn(pythonCmd(), ["-c", script], { cwd: REPO_ROOT });
    let buf = "";
    child.stdout.on("data", d => (buf += String(d)));
    child.stderr.on("data", d => pushLog(String(d)));
    child.on("close", () => resolve(buf.trim()));
    child.on("error", () => resolve("{}"));
  });
  let hardware: JsonObject = {};
  try { hardware = JSON.parse(out || "{}"); } catch { hardware = {}; }

  const vram = Number(hardware.vram_total || 0);
  let rec;
  if (vram >= 24) rec = { n_layer: 24, n_embd: 1536, n_head: 16, n_kv_head: 8, batch_size: 2, grad_accum: 8, tier: "24GB+" };
  else if (vram >= 12) rec = { n_layer: 20, n_embd: 1024, n_head: 8, n_kv_head: 4, batch_size: 1, grad_accum: 8, tier: "12GB" };
  else if (vram >= 8) rec = { n_layer: 20, n_embd: 832, n_head: 8, n_kv_head: 4, batch_size: 1, grad_accum: 8, tier: "8GB" };
  else rec = { n_layer: 8, n_embd: 512, n_head: 8, n_kv_head: 4, batch_size: 1, grad_accum: 8, tier: "CPU/Lite" };
  return { hardware, recommendation: rec };
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "20mb" }));

  app.get("/api/health", (_req, res) => {
    res.json({ ok: true, repo_root: REPO_ROOT, running: Boolean(runningJob), task: runningJob?.name ?? null });
  });

  app.get("/api/settings", (_req, res) => res.json(getSettingsPayload()));
  app.post("/api/settings", (req, res) => {
    updateConfigFromSettings(req.body || {});
    res.json({ status: "ok", settings: getSettingsPayload() });
  });

  app.get("/api/logs", (req, res) => {
    const since = Number(req.query.since || 0);
    const lines = logLines.filter(([idx]) => idx >= since);
    res.json({ lines, cursor: logCursor });
  });

  app.get("/api/metrics", (_req, res) => res.json(latestMetrics()));
  app.get("/api/hardware/detect", async (_req, res) => res.json(await detectHardware()));

  app.post("/api/train/pretrain", (req, res) => {
    try {
      const config = updateConfigFromTrainingRequest("pretrain", req.body || {});
      const args = ["pretrain.py", "--config", relToRepo(CONFIG_PATH)];
      const ckpt = resolveRepoPath(deepGet(config, "paths.pretrain_checkpoint", "data/model.pt"), "data/model.pt");
      if (fs.existsSync(ckpt) || req.body?.resume) args.push("--resume");
      if (req.body?.cache_only) args.push("--cache-only");
      if (req.body?.force_rebuild_cache) args.push("--force-rebuild-cache");
      spawnJob("pretrain", args);
      res.json({ status: "Pretraining started", config_path: CONFIG_PATH });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.post("/api/train/sft", (req, res) => {
    try {
      const config = updateConfigFromTrainingRequest("sft", req.body || {});
      const args = ["sft.py", "--config", relToRepo(CONFIG_PATH)];
      const ckpt = resolveRepoPath(deepGet(config, "paths.sft_checkpoint", "data/model.sft.pt"), "data/model.sft.pt");
      if (fs.existsSync(ckpt) || req.body?.resume) args.push("--resume");
      if (req.body?.force_rebuild_cache) args.push("--force-rebuild-cache");
      spawnJob("sft", args);
      res.json({ status: "SFT started", config_path: CONFIG_PATH });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.post("/api/train/tokenizer", (req, res) => {
    try {
      const body = req.body || {};
      const inputPath = body.path || deepGet(loadConfig(), "paths.corpus_dir", "corpus");
      const outPath = body.out_path || deepGet(loadConfig(), "paths.tokenizer_path", "data/tokenizer.json");
      const args = [
        path.join("tools", "train_tokenizer.py"),
        "--input", inputPath,
        "--output", outPath,
        "--vocab-size", String(body.vocab_size || 50000),
        "--min-freq", String(body.min_freq || 2),
      ];
      spawnJob("tokenizer", args);
      res.json({ status: "Tokenizer training started", output: outPath });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.post("/api/train/dpo", (req, res) => {
    try {
      const config = updateConfigFromTrainingRequest("dpo", req.body || {});
      const args = ["dpo.py", "--config", relToRepo(CONFIG_PATH)];
      const ckpt = resolveRepoPath(deepGet(config, "paths.dpo_checkpoint", "data/model.dpo.pt"), "data/model.dpo.pt");
      if (fs.existsSync(ckpt) || req.body?.resume) args.push("--resume");
      if (req.body?.force_rebuild_cache) args.push("--force-rebuild-cache");
      spawnJob("dpo", args);
      res.json({ status: "DPO started", config_path: CONFIG_PATH });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.post("/api/dpo/add-pair", (req, res) => {
    try {
      const { prompt, chosen, rejected, source } = req.body || {};
      if (!String(prompt || "").trim() || !String(chosen || "").trim() || !String(rejected || "").trim()) {
        return res.status(400).json({ error: "prompt, chosen, and rejected are required" });
      }
      const row = { prompt: String(prompt), chosen: String(chosen), rejected: String(rejected), source: source || "studio", created_at: new Date().toISOString() };
      fs.appendFileSync(studioFeedbackPath(), JSON.stringify(row) + "\n", "utf8");
      res.json({ status: "Preference pair added", dpo_buffer: countPreferencePairs() });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.post("/api/dpo/flush-feedback", (_req, res) => {
    try {
      const file = studioFeedbackPath();
      if (fs.existsSync(file)) {
        const archive = file.replace(/\.jsonl$/i, `.${Date.now()}.bak.jsonl`);
        fs.renameSync(file, archive);
        return res.json({ status: "Studio feedback buffer archived", archive: relToRepo(archive), dpo_buffer: countPreferencePairs() });
      }
      res.json({ status: "No studio feedback buffer to flush", dpo_buffer: countPreferencePairs() });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.post("/api/train/stop", (_req, res) => {
    const stopped = stopRunningJob();
    res.json({ status: stopped ? "Stop signal sent" : "No active job" });
  });

  app.post("/api/chat", (req, res) => {
    try {
      const body = req.body || {};
      const payload = JSON.stringify(body);
      const child = spawn(pythonCmd(), [path.join("tools", "chat_once.py"), "--config", relToRepo(CONFIG_PATH), "--json"], {
        cwd: REPO_ROOT,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", d => (stdout += String(d)));
      child.stderr.on("data", d => (stderr += String(d)));
      child.on("close", code => {
        if (stderr.trim()) pushLog(stderr);
        if (code !== 0) return res.status(500).json({ reply: `Chat inference failed. ${stderr.trim()}`, prompt: "" });
        try { res.json(JSON.parse(stdout)); }
        catch { res.status(500).json({ reply: stdout.trim() || "No response from chat engine.", prompt: "" }); }
      });
      child.stdin.write(payload);
      child.stdin.end();
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.get("/api/hub/datasets", async (req, res) => {
    try {
      const search = String(req.query.search || "text");
      const datasets: any[] = [];
      const iterator = listDatasets({ search: { query: search }, limit: 20 });
      for await (const dataset of iterator) datasets.push(dataset);
      res.json(datasets);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch datasets" });
    }
  });

  app.post("/api/hub/download-dataset", async (req, res) => {
    const { datasetId, filename } = req.body || {};
    if (!datasetId || !filename) return res.status(400).json({ error: "Missing datasetId or filename" });
    try {
      const url = `https://huggingface.co/datasets/${datasetId}/resolve/main/${filename}`;
      const response = await axios({ method: "get", url, responseType: "stream" });
      const config = loadConfig();
      const corpusDir = resolveRepoPath(deepGet(config, "paths.corpus_dir", "corpus"), "corpus");
      fs.mkdirSync(corpusDir, { recursive: true });
      const destPath = path.join(corpusDir, path.basename(filename));
      const writer = fs.createWriteStream(destPath);
      response.data.pipe(writer);
      writer.on("finish", () => res.json({ status: "Downloaded", path: destPath }));
      writer.on("error", () => res.status(500).json({ error: "File write error" }));
    } catch (error) {
      res.status(500).json({ error: "Download failed" });
    }
  });

  app.post("/api/hub/export-model", async (req, res) => {
    const { repoName, hfToken } = req.body || {};
    if (!repoName || !hfToken) return res.status(400).json({ error: "Missing repoName or hfToken" });
    try {
      try {
        await createRepo({ repo: repoName, credentials: { accessToken: hfToken } });
      } catch (e: any) {
        if (!String(e.message || "").includes("already exists")) throw e;
      }
      const dataDir = resolveRepoPath(deepGet(loadConfig(), "paths.data_dir", "data"), "data");
      const files = fs.readdirSync(dataDir).filter(f => /\.(pt|json|bin)$/.test(f));
      if (!files.length) return res.status(404).json({ error: "No model artifacts found in data/." });
      const operations = files.map(filename => ({
        path: filename,
        content: new Blob([fs.readFileSync(path.join(dataDir, filename))]),
        operation: "addOrUpdate" as const,
      }));
      await commit({ repo: repoName, credentials: { accessToken: hfToken }, title: "Exported from Haiku Studio", operations });
      res.json({ status: "Model exported successfully" });
    } catch (error: any) {
      res.status(500).json({ error: error.message || "Export failed" });
    }
  });


  app.post("/api/shutdown", (_req, res) => {
    if (process.env.HAIKU_STUDIO_ALLOW_SHUTDOWN !== "1") {
      return res.status(403).json({ error: "Shutdown endpoint disabled" });
    }
    pushLog("[studio] Shutdown requested by Electron shell");
    res.json({ status: "shutting_down" });
    setTimeout(() => process.exit(0), 150);
  });

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({ server: { middlewareMode: true }, appType: "spa", root: STUDIO_DIR });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(STUDIO_DIR, "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => res.sendFile(path.join(distPath, "index.html")));
  }

  app.listen(PORT, "127.0.0.1", () => {
    pushLog(`[studio] Haiku Studio API/UI listening on http://127.0.0.1:${PORT}`);
    pushLog(`[studio] h2 repo root: ${REPO_ROOT}`);
  });
}

startServer().catch(err => {
  console.error(err);
  process.exit(1);
});
