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
const DEFAULT_PROJECT_NAME = "haiku_studio";
const PROJECTS_DIR = path.join(REPO_ROOT, "projects");
const PREBUILT_TOKENIZER_PATH = path.join(STUDIO_DIR, "prebuilt", "default_tokenizer.json");
const DATA_TOKENIZER_PATH = path.join(REPO_ROOT, "data", "tokenizer.json");

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
  const clean = line.replace(/\r/g, "\n").trimEnd();
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

function safeProjectName(name: string | undefined): string {
  const raw = String(name || DEFAULT_PROJECT_NAME).trim();
  const safe = raw.replace(/[^a-zA-Z0-9._-]/g, "_").replace(/^\.+/, "").slice(0, 80);
  return safe || DEFAULT_PROJECT_NAME;
}

function ensureProjectDir(projectName?: string): string {
  const name = safeProjectName(projectName);
  const dir = path.join(PROJECTS_DIR, name);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function listProjects(): string[] {
  fs.mkdirSync(PROJECTS_DIR, { recursive: true });
  const names = new Set<string>([DEFAULT_PROJECT_NAME]);
  for (const entry of fs.readdirSync(PROJECTS_DIR, { withFileTypes: true })) {
    if (entry.isDirectory()) names.add(safeProjectName(entry.name));
  }
  return Array.from(names).sort();
}

function activeProjectName(): string {
  const ui = readJson(SETTINGS_PATH, {});
  return safeProjectName(ui.active_project || DEFAULT_PROJECT_NAME);
}

function projectTokenizerPath(projectName?: string): string {
  return path.join(ensureProjectDir(projectName || activeProjectName()), "tokenizer.json");
}


function projectAbs(projectName: string | undefined, ...parts: string[]): string {
  return path.join(ensureProjectDir(projectName || activeProjectName()), ...parts);
}

function projectRel(projectName: string | undefined, ...parts: string[]): string {
  return relToRepo(projectAbs(projectName, ...parts));
}

function projectConfigPath(projectName?: string): string {
  return projectAbs(projectName || activeProjectName(), "config", "gpt_config.json");
}

function ensureProjectFolders(projectName?: string): JsonObject {
  const name = safeProjectName(projectName || activeProjectName());
  const root = ensureProjectDir(name);
  const folders = {
    root,
    config_dir: path.join(root, "config"),
    checkpoints_dir: path.join(root, "checkpoints"),
    logs_dir: path.join(root, "logs"),
    cache_dir: path.join(root, "cache"),
    pretokenized_cache_dir: path.join(root, "cache", "pretokenized"),
    datasets_dir: path.join(root, "datasets"),
    corpus_dir: path.join(root, "datasets", "corpus"),
    sft_dir: path.join(root, "datasets", "sft"),
    dpo_dir: path.join(root, "datasets", "dpo"),
  };
  for (const dir of Object.values(folders)) fs.mkdirSync(String(dir), { recursive: true });
  for (const keep of [folders.corpus_dir, folders.sft_dir, folders.dpo_dir]) {
    const f = path.join(keep, ".gitkeep");
    if (!fs.existsSync(f)) fs.writeFileSync(f, "", "utf8");
  }
  return folders;
}

function defaultProjectPaths(projectName?: string): JsonObject {
  const name = safeProjectName(projectName || activeProjectName());
  return {
    data_dir: "data",
    tokenizer_path: "data/tokenizer.json",
    corpus_dir: projectRel(name, "datasets", "corpus"),
    sft_dataset: projectRel(name, "datasets", "sft"),
    dpo_dataset: projectRel(name, "datasets", "dpo"),
    pretokenized_cache_dir: projectRel(name, "cache", "pretokenized"),
    sft_cache: projectRel(name, "cache", "sft_cache.pt"),
    dpo_cache: projectRel(name, "cache", "dpo_cache.pt"),
    pretrain_checkpoint: projectRel(name, "checkpoints", "model.pt"),
    pretrain_best_checkpoint: projectRel(name, "checkpoints", "model.best.pt"),
    sft_checkpoint: projectRel(name, "checkpoints", "model.sft.pt"),
    sft_best_checkpoint: projectRel(name, "checkpoints", "model.sft.best.pt"),
    dpo_checkpoint: projectRel(name, "checkpoints", "model.dpo.pt"),
    dpo_best_checkpoint: projectRel(name, "checkpoints", "model.dpo.best.pt"),
    pretrain_log: projectRel(name, "logs", "pretrain_loss.jsonl"),
    sft_log: projectRel(name, "logs", "sft_loss.jsonl"),
    dpo_log: projectRel(name, "logs", "dpo_loss.jsonl"),
  };
}

function isLegacyDefaultPath(value: any, defaults: string[]): boolean {
  const v = String(value || "").replace(/\\/g, "/");
  return !v || defaults.includes(v);
}

function applyProjectPaths(configInput: JsonObject, projectName?: string): JsonObject {
  const name = safeProjectName(projectName || activeProjectName());
  ensureProjectFolders(name);
  const config = JSON.parse(JSON.stringify(configInput || {}));
  const p = defaultProjectPaths(name);
  const oldPaths = config.paths || {};
  config.paths = { ...oldPaths };

  config.paths.data_dir = "data";
  config.paths.tokenizer_path = "data/tokenizer.json";
  config.paths.pretokenized_cache_dir = p.pretokenized_cache_dir;
  config.paths.sft_cache = p.sft_cache;
  config.paths.dpo_cache = p.dpo_cache;
  config.paths.pretrain_checkpoint = p.pretrain_checkpoint;
  config.paths.pretrain_best_checkpoint = p.pretrain_best_checkpoint;
  config.paths.sft_checkpoint = p.sft_checkpoint;
  config.paths.sft_best_checkpoint = p.sft_best_checkpoint;
  config.paths.dpo_checkpoint = p.dpo_checkpoint;
  config.paths.dpo_best_checkpoint = p.dpo_best_checkpoint;
  config.paths.pretrain_log = p.pretrain_log;
  config.paths.sft_log = p.sft_log;
  config.paths.dpo_log = p.dpo_log;

  if (isLegacyDefaultPath(oldPaths.corpus_dir, ["corpus", "data/corpus"])) config.paths.corpus_dir = p.corpus_dir;
  else config.paths.corpus_dir = oldPaths.corpus_dir;

  if (isLegacyDefaultPath(oldPaths.sft_dataset, ["sft", "data/sft"])) config.paths.sft_dataset = p.sft_dataset;
  else config.paths.sft_dataset = oldPaths.sft_dataset;

  if (isLegacyDefaultPath(oldPaths.dpo_dataset, ["dpo", "data/dpo"])) config.paths.dpo_dataset = p.dpo_dataset;
  else config.paths.dpo_dataset = oldPaths.dpo_dataset;

  config.sft = { ...(config.sft || {}) };
  if (isLegacyDefaultPath(config.sft.base_checkpoint, ["data/model.best.pt", "model.best.pt"])) {
    config.sft.base_checkpoint = p.pretrain_best_checkpoint;
  }

  config.dpo = { ...(config.dpo || {}) };
  if (isLegacyDefaultPath(config.dpo.policy_checkpoint, ["data/model.sft.best.pt", "model.sft.best.pt"])) {
    config.dpo.policy_checkpoint = p.sft_best_checkpoint;
  }
  if (isLegacyDefaultPath(config.dpo.reference_checkpoint, ["data/model.sft.reference.pt", "model.sft.reference.pt"])) {
    config.dpo.reference_checkpoint = projectRel(name, "checkpoints", "model.sft.reference.pt");
  }

  config.project = {
    ...(config.project || {}),
    name,
    root: projectRel(name),
    tokenizer_path: projectRel(name, "tokenizer.json"),
    config_path: projectRel(name, "config", "gpt_config.json"),
  };
  return config;
}

function loadProjectConfig(projectName?: string): JsonObject {
  const name = safeProjectName(projectName || activeProjectName());
  const cfgPath = projectConfigPath(name);
  if (fs.existsSync(cfgPath)) return readJson(cfgPath, loadConfig());
  return loadConfig();
}

function saveProjectConfig(projectName: string | undefined, config: JsonObject) {
  const name = safeProjectName(projectName || activeProjectName());
  const scoped = applyProjectPaths(config, name);
  writeJsonAtomic(projectConfigPath(name), scoped);
  writeJsonAtomic(CONFIG_PATH, scoped);
}

function copyIfExists(src: string, dest: string, label: string, overwrite = true): boolean {
  try {
    if (!fs.existsSync(src)) return false;
    if (!overwrite && fs.existsSync(dest)) return false;
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.copyFileSync(src, dest);
    pushLog(`[project] ${label}: ${relToRepo(src)} -> ${relToRepo(dest)}`);
    return true;
  } catch (err: any) {
    pushLog(`[project] Failed to copy ${label}: ${err.message}`);
    return false;
  }
}

function seedProjectFromRuntime(projectName?: string, overwrite = false) {
  const name = safeProjectName(projectName || activeProjectName());
  ensureProjectFolders(name);
  copyIfExists(DATA_TOKENIZER_PATH, projectAbs(name, "tokenizer.json"), "tokenizer", overwrite);
  const checkpointMap: Array<[string, string]> = [
    ["data/model.pt", "model.pt"],
    ["data/model.best.pt", "model.best.pt"],
    ["data/model.sft.pt", "model.sft.pt"],
    ["data/model.sft.best.pt", "model.sft.best.pt"],
    ["data/model.sft.reference.pt", "model.sft.reference.pt"],
    ["data/model.dpo.pt", "model.dpo.pt"],
    ["data/model.dpo.best.pt", "model.dpo.best.pt"],
  ];
  for (const [srcRel, destName] of checkpointMap) copyIfExists(path.join(REPO_ROOT, srcRel), projectAbs(name, "checkpoints", destName), `checkpoint ${destName}`, overwrite);
  const logMap: Array<[string, string]> = [
    ["data/pretrain_loss.jsonl", "pretrain_loss.jsonl"],
    ["data/sft_loss.jsonl", "sft_loss.jsonl"],
    ["data/dpo_loss.jsonl", "dpo_loss.jsonl"],
  ];
  for (const [srcRel, destName] of logMap) copyIfExists(path.join(REPO_ROOT, srcRel), projectAbs(name, "logs", destName), `metrics ${destName}`, overwrite);
}

function ensureProjectLayout(projectName?: string): JsonObject {
  const name = safeProjectName(projectName || activeProjectName());
  ensureProjectFolders(name);
  ensurePrebuiltTokenizerCopy();

  const projectTok = projectAbs(name, "tokenizer.json");
  if (!fs.existsSync(projectTok)) {
    if (fs.existsSync(DATA_TOKENIZER_PATH)) copyIfExists(DATA_TOKENIZER_PATH, projectTok, "seed project tokenizer", false);
    else if (fs.existsSync(PREBUILT_TOKENIZER_PATH)) copyIfExists(PREBUILT_TOKENIZER_PATH, projectTok, "seed project tokenizer", false);
  }

  const cfgPath = projectConfigPath(name);
  if (!fs.existsSync(cfgPath)) {
    const scoped = applyProjectPaths(loadConfig(), name);
    writeJsonAtomic(cfgPath, scoped);
  }

  return {
    name,
    root: projectRel(name),
    config_path: projectRel(name, "config", "gpt_config.json"),
    tokenizer_path: projectRel(name, "tokenizer.json"),
    checkpoints_dir: projectRel(name, "checkpoints"),
    logs_dir: projectRel(name, "logs"),
    cache_dir: projectRel(name, "cache"),
    corpus_dir: projectRel(name, "datasets", "corpus"),
    sft_dir: projectRel(name, "datasets", "sft"),
    dpo_dir: projectRel(name, "datasets", "dpo"),
  };
}

function syncProjectToData(projectName?: string): JsonObject {
  const name = safeProjectName(projectName || activeProjectName());
  const layout = ensureProjectLayout(name);
  const projectTok = projectAbs(name, "tokenizer.json");
  if (fs.existsSync(projectTok)) {
    fs.mkdirSync(path.dirname(DATA_TOKENIZER_PATH), { recursive: true });
    fs.copyFileSync(projectTok, DATA_TOKENIZER_PATH);
    pushLog(`[project] Loaded tokenizer into runtime data folder: ${relToRepo(projectTok)} -> ${relToRepo(DATA_TOKENIZER_PATH)}`);
  }
  const scoped = applyProjectPaths(loadProjectConfig(name), name);
  writeJsonAtomic(CONFIG_PATH, scoped);
  writeJsonAtomic(projectConfigPath(name), scoped);
  return layout;
}

function saveRuntimeToProject(projectName?: string): JsonObject {
  const name = safeProjectName(projectName || activeProjectName());
  const layout = ensureProjectLayout(name);
  seedProjectFromRuntime(name, true);
  const scoped = applyProjectPaths(loadConfig(), name);
  writeJsonAtomic(projectConfigPath(name), scoped);
  pushLog(`[project] Runtime data artifacts saved back to ${projectRel(name)}`);
  return layout;
}

function activateProject(projectName?: string, seedCurrent = false): JsonObject {
  if (runningJob) throw new Error("Cannot switch projects while a trainer is running.");
  const name = safeProjectName(projectName || DEFAULT_PROJECT_NAME);
  if (seedCurrent) seedProjectFromRuntime(name, false);
  const layout = syncProjectToData(name);
  const ui = loadUiSettings();
  ui.active_project = name;
  ui.project_dir = projectAbs(name);
  saveUiSettings(ui);
  pushLog(`[project] Active project set to ${name}. Runtime data/tokenizer.json and config/gpt_config.json are synced from the project.`);
  return layout;
}

function prepareProjectRuntime(projectName?: string): JsonObject {
  const name = safeProjectName(projectName || activeProjectName());
  syncProjectToData(name);
  const config = applyProjectPaths(loadProjectConfig(name), name);
  saveProjectConfig(name, config);
  return config;
}

function ensurePrebuiltTokenizerCopy() {
  try {
    fs.mkdirSync(path.dirname(PREBUILT_TOKENIZER_PATH), { recursive: true });
    if (!fs.existsSync(PREBUILT_TOKENIZER_PATH) && fs.existsSync(DATA_TOKENIZER_PATH)) {
      fs.copyFileSync(DATA_TOKENIZER_PATH, PREBUILT_TOKENIZER_PATH);
      pushLog(`[tokenizer] Created prebuilt tokenizer copy: ${relToRepo(PREBUILT_TOKENIZER_PATH)}`);
    }
    const projectTok = projectTokenizerPath(DEFAULT_PROJECT_NAME);
    if (!fs.existsSync(projectTok) && fs.existsSync(DATA_TOKENIZER_PATH)) {
      fs.copyFileSync(DATA_TOKENIZER_PATH, projectTok);
      pushLog(`[tokenizer] Created default project tokenizer copy: ${relToRepo(projectTok)}`);
    }
  } catch (err: any) {
    pushLog(`[tokenizer] Failed to initialize prebuilt tokenizer copy: ${err.message}`);
  }
}

function restorePrebuiltTokenizerToProject(projectName?: string): JsonObject {
  ensurePrebuiltTokenizerCopy();
  if (!fs.existsSync(PREBUILT_TOKENIZER_PATH)) throw new Error(`Prebuilt tokenizer is missing: ${PREBUILT_TOKENIZER_PATH}`);
  fs.mkdirSync(path.dirname(DATA_TOKENIZER_PATH), { recursive: true });
  fs.copyFileSync(PREBUILT_TOKENIZER_PATH, DATA_TOKENIZER_PATH);
  const projectPath = projectTokenizerPath(projectName);
  fs.copyFileSync(PREBUILT_TOKENIZER_PATH, projectPath);
  pushLog(`[tokenizer] Restored prebuilt tokenizer to ${relToRepo(DATA_TOKENIZER_PATH)} and ${relToRepo(projectPath)}`);
  return { data_tokenizer: relToRepo(DATA_TOKENIZER_PATH), project_tokenizer: relToRepo(projectPath) };
}

function loadUiSettings(): JsonObject {
  const defaults = {
    theme: "dark",
    show_tooltips: true,
    active_project: DEFAULT_PROJECT_NAME,
    project_dir: ensureProjectDir(DEFAULT_PROJECT_NAME),
  };
  const loaded = { ...defaults, ...readJson(SETTINGS_PATH, {}) };
  loaded.active_project = safeProjectName(loaded.active_project);
  loaded.project_dir = ensureProjectDir(loaded.active_project);
  return loaded;
}

function saveUiSettings(settings: JsonObject) {
  if (settings.active_project !== undefined) {
    settings.active_project = safeProjectName(settings.active_project);
    settings.project_dir = ensureProjectDir(settings.active_project);
  }
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

function listDataFiles(target: string): string[] {
  if (!fs.existsSync(target)) return [];
  const stat = fs.statSync(target);
  if (stat.isFile()) return [target];
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === ".gitkeep" || entry.name.startsWith(".")) continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile()) out.push(full);
    }
  };
  walk(target);
  return out.sort();
}

function countPreferencePairsInFile(file: string): number {
  if (!fs.existsSync(file) || !fs.statSync(file).isFile()) return 0;
  const text = fs.readFileSync(file, "utf8");
  let count = 0;
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
  return count;
}

function latestJsonlRecord(file: string): JsonObject | null {
  if (!fs.existsSync(file)) return null;
  const lines = fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean).slice(-250).reverse();
  for (const line of lines) {
    try {
      const row = JSON.parse(line);
      if (row && typeof row === "object") return row;
    } catch {}
  }
  return null;
}

function fileSummary(file: string): JsonObject {
  const stat = fs.statSync(file);
  return {
    path: relToRepo(file),
    size_bytes: stat.size,
    modified_at: stat.mtime.toISOString(),
    preference_pairs: countPreferencePairsInFile(file),
  };
}

function checkpointSummary(pathRaw: string | undefined, fallback: string): JsonObject {
  const abs = resolveRepoPath(pathRaw, fallback);
  if (!fs.existsSync(abs)) return { path: relToRepo(abs), exists: false, size_bytes: 0, modified_at: null };
  const stat = fs.statSync(abs);
  return { path: relToRepo(abs), exists: true, size_bytes: stat.size, modified_at: stat.mtime.toISOString() };
}

function dpoStatsPayload(): JsonObject {
  const active = activeProjectName();
  ensureProjectLayout(active);
  const config = loadConfig();
  const datasetRaw = String(deepGet(config, "paths.dpo_dataset", projectRel(active, "datasets", "dpo")) || projectRel(active, "datasets", "dpo"));
  const datasetAbs = resolveRepoPath(datasetRaw, projectRel(active, "datasets", "dpo"));
  const files = listDataFiles(datasetAbs);
  const fileRows = files.map(fileSummary);
  const feedbackFile = studioFeedbackPath();
  const logRaw = String(deepGet(config, "paths.dpo_log", projectRel(active, "logs", "dpo_loss.jsonl")) || projectRel(active, "logs", "dpo_loss.jsonl"));
  const logAbs = resolveRepoPath(logRaw, projectRel(active, "logs", "dpo_loss.jsonl"));
  const lastMetrics = latestJsonlRecord(logAbs);

  return {
    active_project: active,
    ready: fs.existsSync(path.join(REPO_ROOT, "dpo.py")),
    dataset_path: relToRepo(datasetAbs),
    dataset_exists: fs.existsSync(datasetAbs),
    file_count: fileRows.length,
    preference_pairs: fileRows.reduce((total, row) => total + Number(row.preference_pairs || 0), 0),
    feedback_file: relToRepo(feedbackFile),
    feedback_pairs: countPreferencePairsInFile(feedbackFile),
    files: fileRows.slice(0, 25),
    latest_step: latestStepFromLog(logRaw, projectRel(active, "logs", "dpo_loss.jsonl")),
    latest_metrics: lastMetrics,
    log_path: relToRepo(logAbs),
    checkpoints: {
      policy: checkpointSummary(deepGet(config, "dpo.policy_checkpoint", projectRel(active, "checkpoints", "model.sft.best.pt")), projectRel(active, "checkpoints", "model.sft.best.pt")),
      reference: checkpointSummary(deepGet(config, "dpo.reference_checkpoint", projectRel(active, "checkpoints", "model.sft.reference.pt")), projectRel(active, "checkpoints", "model.sft.reference.pt")),
      output: checkpointSummary(deepGet(config, "paths.dpo_checkpoint", projectRel(active, "checkpoints", "model.dpo.pt")), projectRel(active, "checkpoints", "model.dpo.pt")),
      best: checkpointSummary(deepGet(config, "paths.dpo_best_checkpoint", projectRel(active, "checkpoints", "model.dpo.best.pt")), projectRel(active, "checkpoints", "model.dpo.best.pt")),
    },
    beta: deepGet(config, "dpo.beta", 0.1),
    lr: deepGet(config, "dpo.lr", 0.000003),
    epochs: deepGet(config, "dpo.epochs", 1),
    batch_size: deepGet(config, "dpo.batch_size", 1),
  };
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
  const active = activeProjectName();
  ensureProjectLayout(active);
  const config = loadConfig();
  const defaultTarget = projectAbs(active, "datasets", "dpo");
  const raw = String(deepGet(config, "paths.dpo_dataset", projectRel(active, "datasets", "dpo")) || projectRel(active, "datasets", "dpo"));
  const target = isLegacyDefaultPath(raw, ["dpo", "data/dpo"]) ? defaultTarget : resolveRepoPath(raw, projectRel(active, "datasets", "dpo"));
  const looksLikeFile = Boolean(path.extname(target));
  if (looksLikeFile) {
    fs.mkdirSync(path.dirname(target), { recursive: true });
    return target;
  }
  fs.mkdirSync(target, { recursive: true });
  return path.join(target, "studio_feedback.jsonl");
}

function getSettingsPayload(): JsonObject {
  ensurePrebuiltTokenizerCopy();
  const ui = loadUiSettings();
  const activeProject = safeProjectName(ui.active_project);
  const layout = ensureProjectLayout(activeProject);
  const config = runningJob ? loadConfig() : applyProjectPaths(loadProjectConfig(activeProject), activeProject);
  if (!runningJob) {
    writeJsonAtomic(CONFIG_PATH, config);
    writeJsonAtomic(projectConfigPath(activeProject), config);
  }
  const model = config.model || {};
  const paths = config.paths || {};
  return {
    ...ui,
    ...layout,
    project_dir: projectAbs(activeProject),
    project_layout: layout,
    projects: listProjects(),
    active_project: activeProject,
    h2_repo_root: REPO_ROOT,
    h2_config_path: CONFIG_PATH,
    project_config_path: relToRepo(projectConfigPath(activeProject)),
    tokenizer_path: relToRepo(DATA_TOKENIZER_PATH),
    prebuilt_tokenizer_path: relToRepo(PREBUILT_TOKENIZER_PATH),
    project_tokenizer_path: relToRepo(projectTokenizerPath(activeProject)),
    project_checkpoint_dir: projectRel(activeProject, "checkpoints"),
    project_log_dir: projectRel(activeProject, "logs"),
    project_cache_dir: projectRel(activeProject, "cache"),
    project_corpus_dir: projectRel(activeProject, "datasets", "corpus"),
    project_sft_dir: projectRel(activeProject, "datasets", "sft"),
    project_dpo_dir: projectRel(activeProject, "datasets", "dpo"),
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
    dpo_global_step: latestStepFromLog(paths.dpo_log, projectRel(activeProject, "logs", "dpo_loss.jsonl")),
    dpo_dataset: deepGet(config, "paths.dpo_dataset", projectRel(activeProject, "datasets", "dpo")),
    dpo_checkpoint: deepGet(config, "paths.dpo_checkpoint", projectRel(activeProject, "checkpoints", "model.dpo.pt")),
    dpo_beta: deepGet(config, "dpo.beta", 0.1),
    dpo_lr: deepGet(config, "dpo.lr", 0.000003),
    dpo_epochs: deepGet(config, "dpo.epochs", 1),
    dpo_batch_size: deepGet(config, "dpo.batch_size", 1),
    corpus_dir: deepGet(config, "paths.corpus_dir", projectRel(activeProject, "datasets", "corpus")),
    sft_dataset: deepGet(config, "paths.sft_dataset", projectRel(activeProject, "datasets", "sft")),
    pretrain_checkpoint: deepGet(config, "paths.pretrain_checkpoint", projectRel(activeProject, "checkpoints", "model.pt")),
    sft_checkpoint: deepGet(config, "paths.sft_checkpoint", projectRel(activeProject, "checkpoints", "model.sft.pt")),
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
  const active = activeProjectName();
  let config = prepareProjectRuntime(active);
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
  config = applyProjectPaths(config, active);
  saveProjectConfig(active, config);
  pushLog(`[project] ${kind} outputs will save under ${projectRel(active, "checkpoints")} and ${projectRel(active, "logs")}`);
  return config;
}

function updateConfigFromSettings(body: JsonObject) {
  if (body.active_project !== undefined && safeProjectName(body.active_project) !== activeProjectName()) {
    activateProject(body.active_project, false);
  }
  const active = safeProjectName(body.active_project || activeProjectName());
  let config = prepareProjectRuntime(active);
  const ui = loadUiSettings();

  const uiKeys = ["theme", "show_tooltips", "active_project"];
  for (const key of uiKeys) if (key in body) ui[key] = body[key];
  ui.active_project = active;
  ui.project_dir = projectAbs(active);

  if (body.model_layers !== undefined) deepSet(config, "model.n_layer", Number(body.model_layers));
  if (body.model_dim !== undefined) deepSet(config, "model.n_embd", Number(body.model_dim));
  if (body.model_heads !== undefined) deepSet(config, "model.n_head", Number(body.model_heads));
  if (body.model_kv_heads !== undefined) deepSet(config, "model.n_kv_head", Number(body.model_kv_heads));
  if (body.block_size !== undefined) deepSet(config, "model.block_size", Number(body.block_size));
  if (body.device !== undefined) deepSet(config, "runtime.device", String(body.device));

  saveUiSettings(ui);
  saveProjectConfig(active, config);
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

  ensureProjectLayout(activeProjectName());
  if (!runningJob) syncProjectToData(activeProjectName());

  app.get("/api/health", (_req, res) => {
    res.json({ ok: true, repo_root: REPO_ROOT, running: Boolean(runningJob), task: runningJob?.name ?? null });
  });

  app.get("/api/settings", (_req, res) => res.json(getSettingsPayload()));
  app.post("/api/settings", (req, res) => {
    updateConfigFromSettings(req.body || {});
    res.json({ status: "ok", settings: getSettingsPayload() });
  });


  app.get("/api/projects", (_req, res) => {
    res.json({ projects: listProjects(), active_project: activeProjectName(), settings: getSettingsPayload() });
  });

  app.post("/api/projects/create", (req, res) => {
    try {
      const name = safeProjectName(req.body?.name);
      if (!name) return res.status(400).json({ error: "Project name is required." });
      ensureProjectLayout(name);
      if (req.body?.seed_current !== false) seedProjectFromRuntime(name, false);
      const config = applyProjectPaths(loadProjectConfig(name), name);
      writeJsonAtomic(projectConfigPath(name), config);
      pushLog(`[project] Created project: ${name}`);
      res.json({ status: "Project created", project: name, projects: listProjects(), layout: ensureProjectLayout(name) });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.post("/api/projects/load", (req, res) => {
    try {
      const layout = activateProject(req.body?.name, false);
      res.json({ status: "Project loaded", project: safeProjectName(req.body?.name), layout, settings: getSettingsPayload() });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.post("/api/projects/sync-to-data", (_req, res) => {
    try {
      const layout = syncProjectToData(activeProjectName());
      res.json({ status: "Project loaded into runtime data folder", layout, settings: getSettingsPayload() });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.post("/api/projects/save-runtime", (_req, res) => {
    try {
      const layout = saveRuntimeToProject(activeProjectName());
      res.json({ status: "Runtime tokenizer/config saved back to project", layout, settings: getSettingsPayload() });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.get("/api/logs", (req, res) => {
    const since = Number(req.query.since || 0);
    const lines = logLines.filter(([idx]) => idx >= since);
    res.json({ lines, cursor: logCursor });
  });

  app.get("/api/metrics", (_req, res) => res.json(latestMetrics()));
  app.get("/api/dpo/stats", (_req, res) => {
    try {
      res.json(dpoStatsPayload());
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });
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
      const sourceKind = String(body.source_type || body.source_kind || "auto").trim();
      const inputPath = String(body.input_path || body.path || "").trim();
      if (!inputPath) {
        return res.status(400).json({ error: "No tokenizer source selected. Choose a .txt file or corpus folder first." });
      }
      if (!["file", "corpus", "auto"].includes(sourceKind)) {
        return res.status(400).json({ error: "Tokenizer source type must be file, corpus, or auto." });
      }

      const outPath = "data/tokenizer.json";
      const activeProject = activeProjectName();
      ensureProjectLayout(activeProject);
      const projectCopy = relToRepo(projectTokenizerPath(activeProject));
      const config = applyProjectPaths(loadProjectConfig(activeProject), activeProject);
      deepSet(config, "paths.tokenizer_path", outPath);
      saveProjectConfig(activeProject, config);

      const args = [
        path.join("tools", "train_tokenizer.py"),
        "--source-kind", sourceKind,
        "--input", inputPath,
        "--output", outPath,
        "--project-copy", projectCopy,
        "--vocab-size", String(body.vocab_size || 50000),
        "--min-freq", String(body.min_freq || 2),
        "--max-input-mb", String(body.max_input_mb || 0),
        "--chunk-mb", String(body.chunk_mb || 8),
      ];
      if (body.no_ram_guard) args.push("--no-ram-guard");

      pushLog(`[tokenizer] Requested ${sourceKind} tokenizer source: ${inputPath}`);
      pushLog(`[tokenizer] Output will be saved to data/tokenizer.json and ${projectCopy}`);
      spawnJob("tokenizer", args);
      res.json({ status: "Tokenizer training started", output: outPath, project_copy: projectCopy });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.post("/api/tokenizer/restore-prebuilt", (_req, res) => {
    try {
      const paths = restorePrebuiltTokenizerToProject(activeProjectName());
      res.json({ status: "Prebuilt tokenizer restored", ...paths, vocab_size: tokenizerVocabSize("data/tokenizer.json") });
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
      prepareProjectRuntime(activeProjectName());
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
      const active = activeProjectName();
      const config = prepareProjectRuntime(active);
      const corpusDir = resolveRepoPath(deepGet(config, "paths.corpus_dir", projectRel(active, "datasets", "corpus")), projectRel(active, "datasets", "corpus"));
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
      const active = activeProjectName();
      const projectDir = projectAbs(active);
      const exportFiles: Array<{ source: string; dest: string }> = [];
      const checkpointDir = path.join(projectDir, "checkpoints");
      if (fs.existsSync(checkpointDir)) {
        for (const filename of fs.readdirSync(checkpointDir).filter(f => /\.(pt|bin)$/.test(f))) {
          exportFiles.push({ source: path.join(checkpointDir, filename), dest: `checkpoints/${filename}` });
        }
      }
      const tok = projectAbs(active, "tokenizer.json");
      if (fs.existsSync(tok)) exportFiles.push({ source: tok, dest: "tokenizer.json" });
      const cfg = projectConfigPath(active);
      if (fs.existsSync(cfg)) exportFiles.push({ source: cfg, dest: "config/gpt_config.json" });
      if (!exportFiles.length) return res.status(404).json({ error: `No project artifacts found for ${active}.` });
      const operations = exportFiles.map(file => ({
        path: file.dest,
        content: new Blob([fs.readFileSync(file.source)]),
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
