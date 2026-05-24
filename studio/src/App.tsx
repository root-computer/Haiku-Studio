/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  BookOpen, 
  Send, 
  Terminal, 
  Settings, 
  HelpCircle, 
  Zap, 
  Brain, 
  BarChart3, 
  Globe,
  ChevronDown,
  ChevronRight, 
  Cpu, 
  Database, 
  MessagesSquare,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Save,
  Trash2,
  X,
  FileText,
  Bot,
  Plus,
  Sun,
  Moon,
  Lock,
  Unlock,
  Info,
  AlertTriangle
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  ResponsiveContainer 
} from 'recharts';
import { cn } from './lib/utils';
import axios from 'axios';
import HuggingFaceHub from './components/HuggingFaceHub';
import { HuggingFaceIcon } from './components/HuggingFaceIcon';

declare global {
  interface Window {
    haikuStudio?: {
      pickTokenizerFile: () => Promise<string>;
      pickCorpusFolder: () => Promise<string>;
    };
  }
}

// --- Types ---
interface Message {
  role: 'user' | 'bot';
  content: string;
  meta?: { prompt: string; shown: string };
}

interface Metrics {
  steps: number[];
  train_loss: number[];
  val_loss: number[];
  meta: any;
}

interface AppSettings {
  project_dir: string;
  projects?: string[];
  active_project?: string;
  theme?: 'light' | 'dark';
  show_tooltips?: boolean;
  device: string;
  model_params: number;
  model_layers: number;
  model_dim: number;
  model_heads: number;
  model_kv_heads: number;
  block_size: number;
  vocab_size: number;
  dpo_ready: boolean;
  dpo_buffer: number;
  dpo_global_step: number;
  dpo_dataset?: string;
  dpo_checkpoint?: string;
  dpo_beta?: number;
  dpo_lr?: number;
  dpo_epochs?: number;
  dpo_batch_size?: number;
  tokenizer_path?: string;
  prebuilt_tokenizer_path?: string;
  project_tokenizer_path?: string;
  is_training?: boolean;
  current_task?: string | null;
  project_config_path?: string;
  project_checkpoint_dir?: string;
  project_log_dir?: string;
  project_cache_dir?: string;
  project_corpus_dir?: string;
  project_sft_dir?: string;
  project_dpo_dir?: string;
  corpus_dir?: string;
  sft_dataset?: string;
  pretrain_checkpoint?: string;
  sft_checkpoint?: string;
}

interface DpoStats {
  active_project?: string;
  ready?: boolean;
  dataset_path?: string;
  dataset_exists?: boolean;
  file_count?: number;
  preference_pairs?: number;
  feedback_file?: string;
  feedback_pairs?: number;
  latest_step?: number;
  latest_metrics?: any;
  log_path?: string;
  checkpoints?: Record<string, { path: string; exists: boolean; size_bytes?: number; modified_at?: string | null }>;
  files?: Array<{ path: string; size_bytes: number; modified_at: string; preference_pairs: number }>;
}

// --- Context ---
const ThemeContext = React.createContext<{ theme: 'light' | 'dark', showTooltips: boolean }>({ theme: 'light', showTooltips: true });
const useTheme = () => React.useContext(ThemeContext);

type KernelSeverity = 'normal' | 'warning' | 'error';

type KernelLogEntry = {
  id: number;
  line: string;
  time: string;
  severity: KernelSeverity;
};

const severityRank: Record<KernelSeverity, number> = { normal: 0, warning: 1, error: 2 };

const maxSeverity = (a: KernelSeverity, b: KernelSeverity): KernelSeverity => severityRank[b] > severityRank[a] ? b : a;

const classifyKernelLine = (line: string): KernelSeverity => {
  const lower = line.toLowerCase();

  const nonFatalPattern = /\b(warn|warning|deprecated|fallback|retry|skipping|skip|nonfatal|non-fatal|recoverable|restored|no checkpoint|checkpoint not found|missing optional|not configured|slow|ram guard|low ram)\b/;
  const explicitlyNonFatal = /failed to persist theme|local ui theme changed|using fallback|falling back|optional dependency|missing optional/.test(lower);

  if (!explicitlyNonFatal && /\b(fatal|critical|traceback|exception|uncaught|runtimeerror|syntaxerror|typeerror|valueerror|cuda out of memory|out of memory|oom|crash|crashed|npm error|errno|econnrefused|etimedout|spawn einval|spawn eperm|permission denied|access denied|cannot find module|module not found|file not found|no such file|failed to start|failed to train|failure|backend exited code=[1-9]|exited code=[1-9]|process exited with code [1-9])\b/.test(lower)) {
    return 'error';
  }

  if (!explicitlyNonFatal && /\b(error|failed)\b/.test(lower)) {
    return 'error';
  }

  if (explicitlyNonFatal || nonFatalPattern.test(lower)) {
    return 'warning';
  }

  return 'normal';
};

const kernelSeverityClass = (severity: KernelSeverity, theme: 'light' | 'dark') => {
  if (severity === 'error') {
    return theme === 'dark'
      ? 'bg-rose-950/55 border-l-2 border-rose-400 text-rose-100 hover:bg-rose-950/70'
      : 'bg-rose-50 border-l-2 border-rose-500 text-rose-950 hover:bg-rose-100/70';
  }
  if (severity === 'warning') {
    return theme === 'dark'
      ? 'bg-amber-950/45 border-l-2 border-amber-400 text-amber-100 hover:bg-amber-950/60'
      : 'bg-amber-50 border-l-2 border-amber-500 text-amber-950 hover:bg-amber-100/70';
  }
  return theme === 'dark'
    ? 'border-zinc-800 hover:bg-zinc-800/30 text-zinc-400'
    : 'border-zinc-50 border-b hover:bg-zinc-50/50 text-zinc-500';
};

const kernelSeverityLabelClass = (severity: KernelSeverity, theme: 'light' | 'dark') => {
  if (severity === 'error') return theme === 'dark' ? 'bg-rose-500 text-white' : 'bg-rose-600 text-white';
  if (severity === 'warning') return theme === 'dark' ? 'bg-amber-400 text-black' : 'bg-amber-500 text-black';
  return theme === 'dark' ? 'bg-zinc-800 text-zinc-400' : 'bg-zinc-100 text-zinc-500';
};

const formatBytes = (bytes?: number) => {
  const value = Number(bytes || 0);
  if (!value) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const idx = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / Math.pow(1024, idx)).toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
};

const formatShortDate = (value?: string | null) => {
  if (!value) return 'not created';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'unknown';
  return date.toLocaleString();
};

// --- Components ---

const SidebarItem = ({ icon: Icon, label, active, onClick }: { icon: any, label: string, active?: boolean, onClick: () => void }) => {
  const { theme } = useTheme();
  return (
      <button 
        onClick={onClick}
        className={cn(
          "w-full flex items-center gap-3 px-4 py-2.5 rounded-xl transition-all duration-200 text-sm font-semibold capitalize",
          active 
            ? (theme === 'dark' ? "bg-white text-black" : "bg-zinc-900 text-white shadow-lg shadow-zinc-200/50")
            : "text-zinc-500 hover:text-zinc-900 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-800/50"
        )}
      >
      <Icon className="w-4 h-4 opacity-70" />
      {label}
    </button>
  );
};

const Card = ({ children, className, title, subtitle }: { children: React.ReactNode, className?: string, title?: string, subtitle?: string }) => {
  const { theme } = useTheme();
  return (
    <div className={cn(
      "rounded-xl border transition-all animate-in fade-in slide-in-from-bottom-2 duration-500 overflow-hidden", 
      "bg-[var(--card)] border-[var(--border)]",
      theme === 'dark' ? "shadow-none" : "shadow-sm shadow-zinc-100",
      className
    )}>
      {title && (
        <div className={cn("px-6 py-4 border-b flex flex-col", "border-[var(--border)]")}>
          <h3 className={cn("text-base font-bold tracking-tight", "text-[var(--foreground)]")}>{title}</h3>
          {subtitle && <p className="text-[10px] font-semibold capitalize tracking-wider mt-1 text-zinc-400 dark:text-zinc-500">{subtitle}</p>}
        </div>
      )}
      <div className="p-6">
        {children}
      </div>
    </div>
  );
};

const TutorialBox = ({ title, description, icon: Icon, colorClass }: { title: string, description: string, icon: any, colorClass?: string }) => {
  const { theme } = useTheme();
  return (
    <div className={cn(
      "p-6 rounded-xl border flex gap-6 mb-8 transition-all duration-500 animate-in fade-in slide-in-from-top-4", 
      theme === 'dark' 
        ? (colorClass?.includes('emerald') ? "bg-emerald-500/5 border-emerald-500/20 text-emerald-100" : "bg-zinc-900/50 border-zinc-800 text-zinc-100") 
        : (colorClass || "bg-zinc-900 text-white border-zinc-900 shadow-xl shadow-zinc-200/50"),
    )}>
      <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center shrink-0", theme === 'dark' ? "bg-zinc-800 text-zinc-400 shadow-none" : "bg-white/10 text-white shadow-lg")}>
        <Icon className="w-6 h-6" />
      </div>
      <div className="flex-1">
        <h4 className={cn("text-lg font-bold mb-1 tracking-tight", theme === 'dark' ? "text-white" : "text-white")}>{title}</h4>
        <p className={cn("text-xs leading-relaxed font-medium capitalize tracking-wide opacity-70", theme === 'dark' ? "text-zinc-400" : "text-white")}>{description}</p>
      </div>
    </div>
  );
};


const fieldClass = (theme: 'light' | 'dark', disabled?: boolean, className?: string) => cn(
  "w-full rounded-xl px-4 py-3 text-sm font-mono border transition-all outline-none placeholder:text-zinc-400 dark:placeholder:text-zinc-600",
  theme === 'dark'
    ? "bg-zinc-900 border-zinc-800 text-zinc-200 focus:border-zinc-600 focus:bg-zinc-900"
    : "bg-zinc-50 border-zinc-200 text-zinc-900 focus:bg-white focus:border-zinc-900",
  disabled && "opacity-60 cursor-not-allowed bg-zinc-50 dark:bg-zinc-900/30 border-transparent text-zinc-400 dark:text-zinc-500",
  className
);

type TextFieldProps = Omit<React.InputHTMLAttributes<HTMLInputElement>, 'className'> & { className?: string };
const TextField = ({ className, disabled, ...props }: TextFieldProps) => {
  const { theme } = useTheme();
  return <input {...props} disabled={disabled} className={fieldClass(theme, disabled, className)} />;
};

type NumberFieldProps = {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  integer?: boolean;
  disabled?: boolean;
  className?: string;
  ariaLabel?: string;
};
const NumberField = ({ value, onChange, min, max, step, integer, disabled, className, ariaLabel }: NumberFieldProps) => {
  const { theme } = useTheme();
  const inputRef = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState(String(value ?? 0));

  useEffect(() => {
    if (document.activeElement !== inputRef.current) {
      setDraft(String(value ?? 0));
    }
  }, [value]);

  const commit = (raw: string, force = false) => {
    const next = raw.trim();
    setDraft(raw);
    if (!force && (next === '' || next === '-' || next === '.' || next === '-.')) return;
    let parsed = Number(next);
    if (!Number.isFinite(parsed)) {
      if (force) setDraft(String(value ?? 0));
      return;
    }
    if (integer) parsed = Math.round(parsed);
    if (typeof min === 'number') parsed = Math.max(min, parsed);
    if (typeof max === 'number') parsed = Math.min(max, parsed);
    onChange(parsed);
    if (force) setDraft(String(parsed));
  };

  return (
    <input
      ref={inputRef}
      type="text"
      inputMode={integer ? "numeric" : "decimal"}
      value={draft}
      disabled={disabled}
      aria-label={ariaLabel}
      data-step={step}
      onChange={(e) => commit(e.target.value)}
      onBlur={(e) => commit(e.target.value, true)}
      onKeyDown={(e) => {
        if (e.key === 'Enter') e.currentTarget.blur();
        if (e.key === 'Escape') {
          setDraft(String(value ?? 0));
          e.currentTarget.blur();
        }
      }}
      className={fieldClass(theme, disabled, className)}
    />
  );
};

const RunButton = ({ children, onClick, disabled, icon: Icon = Zap }: { children: React.ReactNode; onClick: () => void; disabled?: boolean; icon?: any }) => {
  const { theme } = useTheme();
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "w-full py-4 font-bold rounded-xl transition-all flex items-center justify-center gap-2 disabled:cursor-not-allowed disabled:opacity-50",
        theme === 'dark' ? "bg-white text-black hover:bg-zinc-100 disabled:bg-zinc-700 disabled:text-zinc-400" : "bg-black text-white hover:bg-zinc-800 shadow-xl shadow-zinc-300 disabled:bg-zinc-300 disabled:text-white disabled:shadow-none"
      )}
    >
      <Icon className="w-4 h-4" /> {children}
    </button>
  );
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'home' | 'chat' | 'pretrain' | 'sft' | 'dpo' | 'help' | 'hub'>('home');
  const [deployMenuOpen, setDeployMenuOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [logs, setLogs] = useState<KernelLogEntry[]>([]);
  const [logCursor, setLogCursor] = useState(0);
  const kernelLogIdRef = React.useRef(0);
  const [metrics, setMetrics] = useState<Metrics>({ steps: [], train_loss: [], val_loss: [], meta: {} });
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [dpoStats, setDpoStats] = useState<DpoStats | null>(null);
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);
  const [kernelAttention, setKernelAttention] = useState<KernelSeverity>('normal');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [hardwareInfo, setHardwareInfo] = useState<any>(null);
  const [autoConfigPending, setAutoConfigPending] = useState(false);
  const [projects, setProjects] = useState<string[]>(['haiku_studio']);
  const [activeProject, setActiveProject] = useState('haiku_studio');
  const [isNewProjectOpen, setIsNewProjectOpen] = useState(false);
  const [newProjectName, setNewProjectName] = useState('haiku_experiment');
  const [newProjectSeedCurrent, setNewProjectSeedCurrent] = useState(true);
  const [isCreatingProject, setIsCreatingProject] = useState(false);

  // Pretraining Form State
  const [corpusDir, setCorpusDir] = useState('corpus');
  const [pretrainEpochs, setPretrainEpochs] = useState(1);
  const [pretrainLR, setPretrainLR] = useState(0.0003);
  const [pretrainBatchSize, setPretrainBatchSize] = useState(1);
  const [pretrainGradAccum, setPretrainGradAccum] = useState(8);
  const [enableGradAccum, setEnableGradAccum] = useState(true);
  const [pretrainMinLRRatio, setPretrainMinLRRatio] = useState(0.1);
  const [pretrainWeightDecay, setPretrainWeightDecay] = useState(0.01);
  const [pretrainValSplit, setPretrainValSplit] = useState(2);
  const [pretrainWarmupSteps, setPretrainWarmupSteps] = useState(100);
  const [isTraining, setIsTraining] = useState(false);
  
  // SFT Training State
  const [sftDataPath, setSftDataPath] = useState('sft');
  const [sftEpochs, setSftEpochs] = useState(3);
  const [sftBatchSize, setSftBatchSize] = useState(1);
  const [sftLR, setSftLR] = useState(0.00005);
  
  // DPO Training State
  const [dpoDataPath, setDpoDataPath] = useState('dpo');
  const [dpoEpochs, setDpoEpochs] = useState(1);
  const [dpoBatchSize, setDpoBatchSize] = useState(1);
  const [dpoBeta, setDpoBeta] = useState(0.1);
  const [dpoLR, setDpoLR] = useState(0.000003);
  
  // Tokenizer Training State
  const [tokSourceType, setTokSourceType] = useState<'file' | 'corpus'>('corpus');
  const [tokVocabSize, setTokVocabSize] = useState(50000);
  const [tokMinFreq, setTokMinFreq] = useState(5);
  const [tokPath, setTokPath] = useState('');
  const [tokMaxInputMb, setTokMaxInputMb] = useState(0);
  const [isTrainingTok, setIsTrainingTok] = useState(false);
  
  // Chat configuration
  const [temperature, setTemperature] = useState(0.9);
  const [presencePenalty, setPresencePenalty] = useState(0.0);
  
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');
  const [showTooltips, setShowTooltips] = useState(true);
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const logBoxRef = useRef<HTMLDivElement>(null);

  const [isArchitectureLocked, setIsArchitectureLocked] = useState(true);

  const appendKernelLogs = React.useCallback((incoming: string | string[]) => {
    const lines = (Array.isArray(incoming) ? incoming : [incoming])
      .map(line => String(line ?? ''))
      .filter(line => line.length > 0);
    if (lines.length === 0) return;

    const now = new Date();
    const stamp = now.toLocaleTimeString();
    const entries: KernelLogEntry[] = lines.map(line => ({
      id: ++kernelLogIdRef.current,
      line,
      time: stamp,
      severity: classifyKernelLine(line),
    }));
    const incomingSeverity = entries.reduce<KernelSeverity>((level, entry) => maxSeverity(level, entry.severity), 'normal');

    if (incomingSeverity === 'error') {
      setKernelAttention('error');
      setIsTerminalOpen(true);
    } else if (incomingSeverity === 'warning') {
      setKernelAttention(prev => prev === 'error' ? prev : 'warning');
    }

    setLogs(prev => [...prev, ...entries].slice(-1000));
  }, []);

  const fetchDpoStats = React.useCallback(async () => {
    try {
      const res = await axios.get('/api/dpo/stats');
      setDpoStats(res.data);
    } catch (e) {
      // DPO stats are supplemental; trainer failures still appear in the kernel logger.
    }
  }, []);

  // Poll logs and metrics
  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const res = await axios.get(`/api/logs?since=${logCursor}`);
        if (res.data.lines?.length > 0) {
          appendKernelLogs(res.data.lines.map((l: any) => l[1]));
          setLogCursor(res.data.cursor);
        }
      } catch (e) {}
    };

    const fetchMetrics = async () => {
      try {
        const res = await axios.get('/api/metrics');
        setMetrics(res.data);
        if (res.data?.meta?.running !== undefined) {
          const running = Boolean(res.data.meta.running);
          setIsTraining(running);
          setIsTrainingTok(running && res.data.meta.current_task === 'tokenizer');
        }
      } catch (e) {}
    };

    const interval = setInterval(() => {
      fetchLogs();
      fetchMetrics();
      if (activeTab === 'dpo') fetchDpoStats();
    }, 2000);

    return () => clearInterval(interval);
  }, [logCursor, appendKernelLogs, activeTab, fetchDpoStats]);

  const fetchSettings = async () => {
    try {
      const res = await axios.get('/api/settings');
      setSettings(res.data);
      if (res.data.theme) setTheme(res.data.theme);
      if (res.data.show_tooltips !== undefined) setShowTooltips(res.data.show_tooltips);
      if (res.data.projects) setProjects(res.data.projects);
      if (res.data.active_project) setActiveProject(res.data.active_project);
      if (res.data.corpus_dir) setCorpusDir(res.data.corpus_dir);
      if (res.data.sft_dataset) setSftDataPath(res.data.sft_dataset);
      if (res.data.dpo_dataset) setDpoDataPath(res.data.dpo_dataset);
      if (res.data.dpo_beta !== undefined) setDpoBeta(Number(res.data.dpo_beta));
      if (res.data.dpo_lr !== undefined) setDpoLR(Number(res.data.dpo_lr));
      if (res.data.dpo_epochs !== undefined) setDpoEpochs(Number(res.data.dpo_epochs));
      if (res.data.dpo_batch_size !== undefined) setDpoBatchSize(Number(res.data.dpo_batch_size));
      if (res.data.is_training !== undefined) setIsTraining(Boolean(res.data.is_training));
      fetchDpoStats();
    } catch (e) {}
  };

  useEffect(() => {
    fetchSettings();
  }, []);

  const detectHardware = async () => {
    setAutoConfigPending(true);
    try {
      const res = await axios.get('/api/hardware/detect');
      setHardwareInfo(res.data);
    } catch (e) {
      alert("Hardware detection failed.");
    } finally {
      setAutoConfigPending(false);
    }
  };

  const applyAutoRecommendation = () => {
    if (!hardwareInfo?.recommendation) return;
    const rec = hardwareInfo.recommendation;
    // Update local state or trigger a save settings call
    setSettings(prev => prev ? ({
      ...prev,
      model_layers: rec.n_layer,
      model_dim: rec.n_embd,
      model_heads: rec.n_head,
    }) : null);
    alert(`Applied ${rec.tier} settings for ${hardwareInfo.hardware.name}`);
  };

  const toggleTheme = async () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    const payload = {
      ...(settings || {}),
      theme: newTheme,
      show_tooltips: showTooltips,
      active_project: activeProject,
      projects
    };
    setSettings(payload as AppSettings);
    try {
      await axios.post('/api/settings', payload);
    } catch (e) {
      appendKernelLogs('[theme] Warning: failed to persist theme selection; local UI theme changed for this session.');
    }
  };

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (logBoxRef.current) {
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
    }
  }, [logs]);

  const openNewProjectDialog = (suggestedName = 'haiku_experiment') => {
    if (isTraining) return;
    setNewProjectName(suggestedName);
    setNewProjectSeedCurrent(true);
    setIsNewProjectOpen(true);
  };

  const createProject = async () => {
    if (isTraining || isCreatingProject) return;
    const name = newProjectName.trim();
    if (!name) {
      alert('Project name is required.');
      return;
    }
    try {
      setIsCreatingProject(true);
      setIsTerminalOpen(true);
      const res = await axios.post('/api/projects/create', { name, seed_current: newProjectSeedCurrent });
      const createdName = res.data.project || name;
      setProjects(res.data.projects || []);
      appendKernelLogs(`[project] ${res.data.status}: ${createdName}`);
      setIsNewProjectOpen(false);
      await loadProject(createdName);
    } catch (e: any) {
      alert(e?.response?.data?.error || 'Failed to create project.');
    } finally {
      setIsCreatingProject(false);
    }
  };

  const loadProject = async (name: string) => {
    if (!name || isTraining) return;
    try {
      setIsTerminalOpen(true);
      const res = await axios.post('/api/projects/load', { name });
      const next = res.data.settings || {};
      if (next.projects) setProjects(next.projects);
      if (next.active_project) setActiveProject(next.active_project);
      setSettings(next);
      if (next.corpus_dir) setCorpusDir(next.corpus_dir);
      if (next.sft_dataset) setSftDataPath(next.sft_dataset);
      if (next.dpo_dataset) setDpoDataPath(next.dpo_dataset);
      appendKernelLogs(`[project] Loaded ${name}. Project tokenizer/config are now staged into data/.`);
      fetchSettings();
    } catch (e: any) {
      alert(e?.response?.data?.error || 'Failed to load project.');
    }
  };

  const syncProjectToData = async () => {
    try {
      setIsTerminalOpen(true);
      const res = await axios.post('/api/projects/sync-to-data');
      appendKernelLogs(`[project] ${res.data.status}`);
      fetchSettings();
    } catch (e: any) {
      alert(e?.response?.data?.error || 'Failed to sync project into data/.');
    }
  };

  const saveRuntimeToProject = async () => {
    try {
      setIsTerminalOpen(true);
      const res = await axios.post('/api/projects/save-runtime');
      appendKernelLogs(`[project] ${res.data.status}`);
      fetchSettings();
    } catch (e: any) {
      alert(e?.response?.data?.error || 'Failed to save runtime files to project.');
    }
  };

  const startPretraining = async () => {
    if (isTraining) return;
    setIsTraining(true);
    try {
      await axios.post('/api/train/pretrain', { 
        corpus_dir: corpusDir, 
        epochs: pretrainEpochs,
        lr: pretrainLR,
        batch_size: pretrainBatchSize,
        grad_accum: enableGradAccum ? pretrainGradAccum : 1,
        val_split: pretrainValSplit / 100,
        warmup_steps: pretrainWarmupSteps,
        min_lr_ratio: pretrainMinLRRatio,
        weight_decay: pretrainWeightDecay
      });
    } catch (e) {
      alert("Failed to start pretraining. Check console.");
      setIsTraining(false);
    }
  };

  const stopTraining = async () => {
    try {
      await axios.post('/api/train/stop');
      setIsTraining(false);
    } catch (e) {}
  };

  const browseTokenizerSource = async () => {
    try {
      const picker = tokSourceType === 'file'
        ? window.haikuStudio?.pickTokenizerFile
        : window.haikuStudio?.pickCorpusFolder;
      if (!picker) {
        alert("Native picker is unavailable. Paste the local path manually.");
        return;
      }
      const selected = await picker();
      if (selected) setTokPath(selected);
    } catch (e: any) {
      alert(e?.message || "Failed to open picker.");
    }
  };

  const startTokenizerTraining = async () => {
    if (isTraining || isTrainingTok) return;
    const cleanPath = tokPath.trim();
    if (!cleanPath) {
      alert("No tokenizer source selected. Choose a .txt file or corpus folder first.");
      return;
    }
    setIsTerminalOpen(true);
    setIsTrainingTok(true);
    try {
      const res = await axios.post('/api/train/tokenizer', {
        source_type: tokSourceType,
        input_path: cleanPath,
        vocab_size: tokVocabSize,
        min_freq: tokMinFreq,
        max_input_mb: tokMaxInputMb
      });
      appendKernelLogs(`[tokenizer] ${res.data.status}. Saving to data/tokenizer.json and project tokenizer copy.`);
    } catch (e: any) {
      const message = e?.response?.data?.error || "Failed to start tokenizer training.";
      alert(message);
      setIsTrainingTok(false);
    }
  };

  const restorePrebuiltTokenizer = async () => {
    setIsTerminalOpen(true);
    try {
      const res = await axios.post('/api/tokenizer/restore-prebuilt');
      appendKernelLogs(`[tokenizer] ${res.data.status}: data/tokenizer.json restored from bundled prebuilt tokenizer.`);
      fetchSettings();
    } catch (e: any) {
      alert(e?.response?.data?.error || "Failed to restore prebuilt tokenizer.");
    }
  };

  const startSFT = async () => {
    if (isTraining) return;
    setIsTraining(true);
    try {
      const res = await axios.post('/api/train/sft', {
        data_path: sftDataPath,
        epochs: sftEpochs,
        batch_size: sftBatchSize,
        lr: sftLR
      });
      alert(res.data.status);
    } catch (e) {
      alert("Failed to start SFT.");
      setIsTraining(false);
    }
  };

  const startDPO = async () => {
    if (isTraining) return;
    setIsTraining(true);
    try {
      const res = await axios.post('/api/train/dpo', {
        data_path: dpoDataPath,
        epochs: dpoEpochs,
        batch_size: dpoBatchSize,
        beta: dpoBeta,
        lr: dpoLR
      });
      alert(res.data.status);
      fetchSettings();
      fetchDpoStats();
    } catch (e: any) {
      alert(e?.response?.data?.error || "Failed to start DPO.");
      setIsTraining(false);
    }
  };

  const flushDPOFeedback = async () => {
    try {
      const res = await axios.post('/api/dpo/flush-feedback');
      alert(res.data.status);
      fetchSettings();
      fetchDpoStats();
    } catch (e: any) {
      alert(e?.response?.data?.error || "Failed to flush DPO feedback buffer.");
    }
  };

  const promptForBotMessage = (botIndex: number) => {
    const previousUser = [...messages.slice(0, botIndex)].reverse().find(m => m.role === 'user');
    return previousUser?.content || messages[botIndex]?.meta?.prompt || '';
  };

  const addChatPreferencePair = async (botIndex: number, currentIs: 'chosen' | 'rejected') => {
    const botMsg = messages[botIndex];
    if (!botMsg || botMsg.role !== 'bot') return;
    const prompt = promptForBotMessage(botIndex);
    if (!prompt.trim()) {
      alert('No user prompt was found for this response.');
      return;
    }
    let chosen = '';
    let rejected = '';
    if (currentIs === 'chosen') {
      chosen = botMsg.content;
      rejected = window.prompt('Paste the weaker/rejected response for this same prompt. DPO needs both a chosen and rejected answer.') || '';
    } else {
      rejected = botMsg.content;
      chosen = window.prompt('Paste the preferred replacement response for this same prompt. DPO needs both a chosen and rejected answer.') || '';
    }
    if (!chosen.trim() || !rejected.trim()) return;
    try {
      const res = await axios.post('/api/dpo/add-pair', { prompt, chosen, rejected, source: 'chat_lab' });
      alert(`${res.data.status}. Buffer: ${res.data.dpo_buffer} pair(s).`);
      fetchSettings();
    } catch (e: any) {
      alert(e?.response?.data?.error || 'Failed to add DPO preference pair.');
    }
  };

  const retryBotMessage = async (botIndex: number) => {
    const history = messages.slice(0, botIndex);
    if (!history.length || isGenerating) return;
    setMessages(history);
    setIsGenerating(true);
    try {
      const res = await axios.post('/api/chat', {
        history,
        temperature,
        presence_penalty: presencePenalty,
        max_new_tokens: 128
      });
      setMessages([...history, { role: 'bot', content: res.data.reply, meta: { prompt: res.data.prompt, shown: res.data.reply } }]);
    } catch (e) {
      setMessages([...history, { role: 'bot', content: "Error communicating with the model engine." }]);
    } finally {
      setIsGenerating(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isGenerating) return;
    
    const userMsg: Message = { role: 'user', content: inputMessage };
    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setIsGenerating(true);

    try {
      const res = await axios.post('/api/chat', { 
        history: [...messages, userMsg],
        temperature: temperature,
        presence_penalty: presencePenalty,
        max_new_tokens: 128
      });
      setMessages(prev => [...prev, { role: 'bot', content: res.data.reply, meta: { prompt: res.data.prompt, shown: res.data.reply } }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'bot', content: "Error communicating with the model engine." }]);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <ThemeContext.Provider value={{ theme, showTooltips }}>
      <div className={cn(
        "flex h-screen overflow-hidden font-sans transition-colors duration-500",
        "bg-[var(--background)] text-[var(--foreground)]",
        theme === 'dark' ? "dark" : ""
      )}>
      {/* Sidebar */}
      <div className={cn(
        "w-72 border-r flex flex-col p-6 gap-8 shrink-0 transition-all",
        "bg-[var(--card)] border-[var(--border)]"
      )}>
        <div className="flex items-center gap-3 px-1">
          <img
            src="/haiku-logo.png"
            alt="Haiku Studio"
            className="w-9 h-9 shrink-0 select-none"
            draggable={false}
          />
          <div className="flex flex-col">
            <span className={cn("font-bold text-base tracking-tight leading-none", theme === 'dark' ? "text-white" : "text-zinc-900")}>Haiku Studio</span>
            <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider mt-1 opacity-70">BY ROOTCOMPUTER</span>
          </div>
        </div>

        <div className="space-y-6">
          <div className="space-y-1">
            <h4 className="text-[10px] font-bold text-zinc-400 capitalize tracking-widest px-3 mb-2 opacity-50">Operation modes</h4>
            <nav className="flex flex-col gap-0.5">
              <SidebarItem icon={Zap} label="Home" active={activeTab === 'home'} onClick={() => setActiveTab('home')} />
              <SidebarItem icon={MessagesSquare} label="Chat Lab" active={activeTab === 'chat'} onClick={() => setActiveTab('chat')} />
              <SidebarItem icon={Database} label="Pretraining" active={activeTab === 'pretrain'} onClick={() => setActiveTab('pretrain')} />
              <SidebarItem icon={Brain} label="Instruct (SFT)" active={activeTab === 'sft'} onClick={() => setActiveTab('sft')} />
              <SidebarItem icon={Zap} label="Alignment (DPO)" active={activeTab === 'dpo'} onClick={() => setActiveTab('dpo')} />
              <SidebarItem icon={HuggingFaceIcon} label="Hugging Face" active={activeTab === 'hub'} onClick={() => setActiveTab('hub')} />
            </nav>
          </div>

            <div className="space-y-1">
              <h4 className="text-[10px] font-bold text-zinc-400 capitalize tracking-widest px-3 mb-2 opacity-50">Training resources</h4>
              <SidebarItem icon={Settings} label="Settings" onClick={() => setIsSettingsOpen(true)} />
              <SidebarItem icon={BookOpen} label="Field Manual" active={activeTab === 'help'} onClick={() => setActiveTab('help')} />
            </div>
        </div>

        <div className="mt-auto pt-6 border-t border-[var(--border)]">
          <div className={cn(
            "p-4 rounded-xl border transition-all duration-300",
            theme === 'dark' 
              ? "bg-zinc-900 border-zinc-800 shadow-none" 
              : "bg-white border-[#F1F1F1] shadow-sm shadow-zinc-100"
          )}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[9px] text-zinc-500 dark:text-zinc-400 font-bold uppercase tracking-wider">Inference Device</span>
              <span className={cn("w-1.5 h-1.5 rounded-full", settings?.device?.includes('cuda') ? "bg-emerald-500" : "bg-amber-500")} />
            </div>
            <div className="text-[10px] font-bold text-zinc-800 dark:text-zinc-300 truncate flex items-center gap-2 uppercase tracking-wide">
               <Cpu className="w-3 h-3 text-zinc-500" />
               {settings?.device?.toUpperCase() || 'INITIALIZING...'}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col relative min-w-0 bg-[var(--background)]">
        
        <header className={cn(
          "h-16 border-b flex items-center justify-between px-8 backdrop-blur-xl sticky top-0 z-10",
          theme === 'dark' ? "bg-zinc-900/80 border-zinc-800" : "bg-white/80 border-zinc-200"
        )}>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <div className={cn("w-1.5 h-6 rounded-full", theme === 'dark' ? "bg-white" : "bg-zinc-900")} />
              <h2 className={cn("text-sm font-bold tracking-tight capitalize", theme === 'dark' ? "text-white" : "text-zinc-900")}>{activeTab} Studio</h2>
            </div>
            <div className="h-6 w-px bg-zinc-200 dark:bg-zinc-800" />
            <div className="flex items-center gap-2">
               <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Active Project:</span>
               <select
                 value={activeProject}
                 disabled={isTraining}
                 onChange={(e) => loadProject(e.target.value)}
                 className={cn(
                   "px-3 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider outline-none border",
                   theme === 'dark' ? "bg-zinc-800 text-zinc-200 border-zinc-700" : "bg-zinc-100 text-zinc-700 border-zinc-200"
                 )}
               >
                 {projects.map(project => <option key={project} value={project}>{project}</option>)}
               </select>
               <button
                 onClick={() => openNewProjectDialog()}
                 disabled={isTraining}
                 className={cn(
                   "px-2 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider border disabled:opacity-40",
                   theme === 'dark' ? "bg-zinc-900 text-zinc-300 border-zinc-700 hover:bg-zinc-800" : "bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-50"
                 )}
               >
                 New
               </button>
            </div>
          </div>
          <div className="flex items-center gap-6">
             <div className="flex items-center gap-4 text-[11px] font-bold text-zinc-500 capitalize tracking-wider">
                <span className="flex items-center gap-1.5"><Save className="w-3.5 h-3.5" /> Auto-checkpoint: Active</span>
             </div>
             <div className="relative">
                <button 
                  onClick={() => setDeployMenuOpen(!deployMenuOpen)}
                  className={cn(
                    "px-4 py-2 text-xs font-bold rounded-lg transition-all capitalize flex items-center gap-2",
                    theme === 'dark' ? "bg-white text-black hover:bg-zinc-100 shadow-none border-none" : "bg-black hover:bg-zinc-800 text-white shadow-md shadow-zinc-200"
                  )}
                >
                    Export / Deploy
                    <ChevronDown className={cn("w-3.5 h-3.5 transition-transform", deployMenuOpen && "rotate-180")} />
                </button>
                
                <AnimatePresence>
                  {deployMenuOpen && (
                    <motion.div 
                      initial={{ opacity: 0, y: 10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 10, scale: 0.95 }}
                      className={cn(
                        "absolute right-0 mt-2 w-56 rounded-xl border p-1 shadow-2xl backdrop-blur-xl z-50",
                        theme === 'dark' ? "bg-zinc-900/95 border-zinc-800" : "bg-white/95 border-zinc-100"
                      )}
                    >
                       <button 
                        onClick={() => {
                          setDeployMenuOpen(false);
                          appendKernelLogs([`[SYSTEM] h2 trainers save checkpoints and metrics into the active project folder.`, `[SYSTEM] Use Hugging Face Push to export saved artifacts.`]);
                        }}
                        className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left text-[10px] font-bold uppercase tracking-wider text-zinc-500 hover:text-black dark:hover:text-white hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-all group"
                       >
                          <Save className="w-4 h-4 text-emerald-500" />
                          Checkpoint Status
                       </button>
                       <button 
                        onClick={() => {
                          setDeployMenuOpen(false);
                          setActiveTab('hub');
                        }}
                        className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left text-[10px] font-bold uppercase tracking-wider text-zinc-500 hover:text-black dark:hover:text-white hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-all group"
                       >
                          <HuggingFaceIcon className="w-4 h-4 text-blue-500" />
                          Hugging Face Push
                       </button>
                    </motion.div>
                  )}
                </AnimatePresence>
             </div>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-8 scroll-smooth custom-scrollbar">
          <AnimatePresence mode="wait">
            {activeTab === 'home' && (
              <motion.div 
                key="home"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="max-w-6xl mx-auto space-y-16 py-8"
              >
                <section className="space-y-4">
                  <div className="flex items-center gap-2 mb-6">
                    <span className="px-2 py-0.5 bg-emerald-500 text-white text-[9px] font-bold rounded uppercase tracking-widest">Welcome</span>
                  </div>
                  <h1 className={cn("text-4xl font-bold tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>
                    Welcome to Haiku Studio.
                  </h1>
                  <p className="text-zinc-500 text-lg max-w-2xl font-medium leading-relaxed">
                    The professional environment for neural weight synthesis and behavioral alignment. Start by choosing a training protocol or continue your research in the evaluation labs.
                  </p>
                </section>

                <Card title="Project Workspace" subtitle="Project files are persistent; data/ is runtime staging">
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <div className={cn("rounded-xl border p-4", theme === 'dark' ? "bg-zinc-950 border-zinc-800" : "bg-zinc-50 border-zinc-100")}>
                      <p className="text-[9px] font-bold uppercase tracking-widest text-zinc-500 mb-1">Active Project</p>
                      <p className={cn("text-sm font-mono font-bold truncate", theme === 'dark' ? "text-zinc-200" : "text-zinc-900")}>{activeProject}</p>
                      <p className="text-[10px] text-zinc-500 dark:text-zinc-400 mt-2 break-all font-mono leading-relaxed">{settings?.project_dir}</p>
                    </div>
                    <div className={cn("rounded-xl border p-4", theme === 'dark' ? "bg-zinc-950 border-zinc-800" : "bg-zinc-50 border-zinc-100")}>
                      <p className="text-[9px] font-bold uppercase tracking-widest text-zinc-500 mb-1">Checkpoint Target</p>
                      <p className="text-[10px] font-mono text-zinc-500 truncate">{settings?.project_checkpoint_dir || 'projects/haiku_studio/checkpoints'}</p>
                      <p className="text-[9px] text-zinc-600 mt-2">Pretrain, SFT, DPO weights save here.</p>
                    </div>
                    <div className={cn("rounded-xl border p-4", theme === 'dark' ? "bg-zinc-950 border-zinc-800" : "bg-zinc-50 border-zinc-100")}>
                      <p className="text-[9px] font-bold uppercase tracking-widest text-zinc-500 mb-1">Metrics Target</p>
                      <p className="text-[10px] font-mono text-zinc-500 truncate">{settings?.project_log_dir || 'projects/haiku_studio/logs'}</p>
                      <p className="text-[9px] text-zinc-600 mt-2">Loss JSONL files save here.</p>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-3 pt-4">
                    <button onClick={syncProjectToData} disabled={isTraining} className={cn("px-4 py-2 rounded-xl text-[10px] font-bold uppercase tracking-wider border disabled:opacity-40", theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-200 hover:bg-zinc-800" : "bg-white border-zinc-200 text-zinc-900 hover:bg-zinc-50")}>Load Project Into Data</button>
                    <button onClick={saveRuntimeToProject} disabled={isTraining} className={cn("px-4 py-2 rounded-xl text-[10px] font-bold uppercase tracking-wider border disabled:opacity-40", theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-200 hover:bg-zinc-800" : "bg-white border-zinc-200 text-zinc-900 hover:bg-zinc-50")}>Save Runtime To Project</button>
                    <button onClick={() => openNewProjectDialog()} disabled={isTraining} className={cn("px-4 py-2 rounded-xl text-[10px] font-bold uppercase tracking-wider border disabled:opacity-40", theme === 'dark' ? "bg-white border-white text-black hover:bg-zinc-200" : "bg-zinc-900 border-zinc-900 text-white hover:bg-black")}>Create Project</button>
                  </div>
                </Card>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                  <div className="space-y-8">
                    <h3 className="text-sm font-bold text-zinc-400 capitalize tracking-widest flex items-center gap-3">
                      <div className="w-1 h-3 bg-zinc-400 rounded-full" />
                      Quick start
                    </h3>
                    <div className="grid grid-cols-1 gap-4">
                      {[
                        { 
                          title: "Model Evaluation", 
                          desc: "Test current weights in natural dialogue.", 
                          icon: MessagesSquare, 
                          tab: 'chat' as const,
                          color: "emerald"
                        },
                        { 
                          title: "Base Pretraining", 
                          desc: "Incorporate new knowledge into the transformer core.", 
                          icon: Database, 
                          tab: 'pretrain' as const,
                          color: "blue"
                        },
                        { 
                          title: "Instruction Tuning", 
                          desc: "Align the model to follow specific protocols.", 
                          icon: Brain, 
                          tab: 'sft' as const,
                          color: "purple"
                        }
                      ].map((item, i) => (
                        <button 
                          key={i}
                          onClick={() => setActiveTab(item.tab)}
                          className={cn(
                            "flex items-center gap-6 p-6 rounded-2xl border text-left transition-all group",
                            theme === 'dark' 
                              ? "bg-zinc-900 border-zinc-800 hover:bg-zinc-800/80" 
                              : "bg-white border-zinc-100 hover:border-zinc-300 shadow-sm"
                          )}
                        >
                          <div className={cn(
                            "w-12 h-12 rounded-xl flex items-center justify-center transition-all",
                            theme === 'dark' ? "bg-zinc-800 text-zinc-400" : "bg-zinc-50 text-zinc-900 group-hover:scale-110"
                          )}>
                            <item.icon className="w-6 h-6" />
                          </div>
                          <div>
                            <h4 className={cn("font-bold text-sm tracking-tight", theme === 'dark' ? "text-zinc-200" : "text-zinc-900")}>
                              {item.title}
                            </h4>
                            <p className="text-xs text-zinc-500 font-medium">{item.desc}</p>
                          </div>
                          <ChevronRight className="w-5 h-5 ml-auto text-zinc-300 group-hover:text-zinc-900 transition-colors" />
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-8">
                    <h3 className="text-sm font-bold text-zinc-400 capitalize tracking-widest flex items-center gap-3">
                      <div className="w-1 h-3 bg-zinc-400 rounded-full" />
                      What's new in v2.4.0
                    </h3>
                    <div className={cn(
                      "p-8 rounded-2xl border space-y-8",
                      theme === 'dark' ? "bg-zinc-900/50 border-zinc-800" : "bg-zinc-50/50 border-zinc-100"
                    )}>
                      <div className="space-y-4">
                        <div className="flex items-center gap-3">
                          <div className="w-6 h-6 rounded-full bg-emerald-500/10 text-emerald-500 flex items-center justify-center">
                            <Zap className="w-3.5 h-3.5" />
                          </div>
                          <h4 className={cn("text-xs font-bold", theme === 'dark' ? "text-zinc-300" : "text-zinc-900")}>DPO Alignment Engine</h4>
                        </div>
                        <p className="text-[11px] text-zinc-500 leading-relaxed font-medium">
                          Direct Preference Optimization is now fully integrated. Align your models without needing a secondary reward model. High-fidelity KL-divergence control implemented.
                        </p>
                      </div>

                      <div className="space-y-4">
                        <div className="flex items-center gap-3">
                          <div className="w-6 h-6 rounded-full bg-blue-500/10 text-blue-500 flex items-center justify-center">
                            <Database className="w-3.5 h-3.5" />
                          </div>
                          <h4 className={cn("text-xs font-bold", theme === 'dark' ? "text-zinc-300" : "text-zinc-900")}>Multi-directory Ingestion</h4>
                        </div>
                        <p className="text-[11px] text-zinc-500 leading-relaxed font-medium">
                          The pretraining kernel can now recursively scan and stream multi-file directories. Point the engine to any folder and it will mix documents into an balanced training stream.
                        </p>
                      </div>

                      <div className="space-y-4">
                        <div className="flex items-center gap-3">
                          <div className="w-6 h-6 rounded-full bg-amber-500/10 text-amber-500 flex items-center justify-center">
                            <Cpu className="w-3.5 h-3.5" />
                          </div>
                          <h4 className={cn("text-xs font-bold", theme === 'dark' ? "text-zinc-300" : "text-zinc-900")}>Hardware Auto-Configuration</h4>
                        </div>
                        <p className="text-[11px] text-zinc-500 leading-relaxed font-medium">
                          New hardware discovery system detected: {settings?.device}. The system now suggests optimal tiers based on detected CUDA cores and available VRAM.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'chat' && (
              <motion.div 
                key="chat"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="max-w-4xl mx-auto h-full flex flex-col gap-8"
              >
                 <div className="flex-1 space-y-8">
                   {messages.length === 0 && (
                     <div className="h-full flex flex-col items-center justify-center text-center p-12">
                        <div className={cn(
                          "w-16 h-16 rounded-2xl flex items-center justify-center mb-6 border",
                          theme === 'dark' ? "bg-zinc-800 border-zinc-700 shadow-none" : "bg-white border-zinc-100 shadow-xl shadow-zinc-100"
                        )}>
                          <MessagesSquare className={cn("w-8 h-8", theme === 'dark' ? "text-white" : "text-zinc-900")} />
                        </div>
                        <h1 className={cn("text-xl font-bold tracking-tight mb-2", theme === 'dark' ? "text-white" : "text-zinc-900")}>Chat evaluation lab</h1>
                        <p className="text-zinc-500 text-xs max-w-sm mb-10 font-medium">Quantitatively assess model performance through qualitative interaction. Verify reasoning and instruction-following consistency.</p>
                        
                        <div className="grid grid-cols-2 gap-4 w-full max-w-xl">
                           <Card className={cn("cursor-pointer transition-all text-left p-4", theme === 'dark' ? "hover:bg-zinc-800/50 border-zinc-800 bg-zinc-900 shadow-none" : "hover:border-zinc-300 bg-white/50 border-zinc-100")}>
                             <h4 className={cn("font-bold text-xs mb-1 capitalize tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>Multiturn reasoning</h4>
                             <p className="text-[10px] text-zinc-500 font-medium leading-relaxed">Evaluate the model's ability to maintain context across complex dialogue chains.</p>
                           </Card>
                           <Card className={cn("cursor-pointer transition-all text-left p-4", theme === 'dark' ? "hover:bg-zinc-800/50 border-zinc-800 bg-zinc-900 shadow-none" : "hover:border-zinc-300 bg-white/50 border-zinc-100")}>
                             <h4 className={cn("font-bold text-xs mb-1 capitalize tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>Synthetic creativity</h4>
                             <p className="text-[10px] text-zinc-500 font-medium leading-relaxed">Assess the ability to synthesize original concepts from high-entropy training data.</p>
                           </Card>
                        </div>
                     </div>
                   )}
                   {messages.map((m, i) => (
                     <div key={i} className={cn("flex gap-6 group", m.role === 'user' ? "flex-row-reverse" : "")}>
                        <div className={cn(
                          "w-9 h-9 rounded-xl shrink-0 flex items-center justify-center text-xs font-black border",
                          theme === 'dark' 
                            ? (m.role === 'user' ? "bg-zinc-800 text-zinc-100 border-zinc-700 shadow-none" : "bg-white text-black border-white shadow-none")
                            : (m.role === 'user' ? "bg-zinc-50 text-zinc-600 border-zinc-200 shadow-sm" : "bg-black text-white border-black shadow-sm")
                        )}>
                          {m.role === 'user' ? 'U' : <Bot className="w-5 h-5" />}
                        </div>
                        <div className={cn("flex-1 space-y-3", m.role === 'user' ? "text-right" : "")}>
                           <div className={cn(
                             "text-sm leading-relaxed whitespace-pre-wrap p-5 rounded-xl border",
                             m.role === 'user' 
                               ? (theme === 'dark' ? "bg-zinc-800 border-zinc-700 text-zinc-100 ml-12 shadow-none" : "bg-white border-zinc-200 text-zinc-700 ml-12 shadow-sm") 
                               : (theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-300 mr-12 shadow-none" : "bg-zinc-50 border-zinc-100 text-zinc-900 mr-12 shadow-sm")
                           )}>
                             {m.content}
                           </div>
                           {m.role === 'bot' && m.meta && (
                             <div className="flex items-center gap-2 px-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                <button title="Add as chosen response; paste a rejected alternative" onClick={() => addChatPreferencePair(i, 'chosen')} className="p-2 rounded-lg bg-white border border-zinc-100 text-zinc-400 hover:text-emerald-600 hover:border-emerald-200 transition-all shadow-sm">
                                  <ThumbsUp className="w-3.5 h-3.5" />
                                </button>
                                <button title="Add as rejected response; paste a preferred replacement" onClick={() => addChatPreferencePair(i, 'rejected')} className="p-2 rounded-lg bg-white border border-zinc-100 text-zinc-400 hover:text-rose-600 hover:border-rose-200 transition-all shadow-sm">
                                  <ThumbsDown className="w-3.5 h-3.5" />
                                </button>
                                <button title="Regenerate this response" onClick={() => retryBotMessage(i)} className="p-2 rounded-lg bg-white border border-zinc-100 text-zinc-400 hover:text-zinc-600 hover:border-zinc-300 transition-all shadow-sm">
                                  <RotateCcw className="w-3.5 h-3.5" />
                                </button>
                             </div>
                           )}
                        </div>
                     </div>
                   ))}
                   <div ref={chatEndRef} />
                </div>

                 <div className="sticky bottom-0 pb-8 bg-[var(--background)]/90 backdrop-blur-md pt-4">
                   <div className="relative group">
                      <textarea 
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendMessage())}
                        placeholder="Type a research prompt..."
                        className={cn(
                          "w-full border rounded-[1.5rem] p-5 pr-14 text-sm outline-none transition-all min-h-[64px] max-h-48 resize-none",
                          theme === 'dark' 
                            ? "bg-zinc-900 border-zinc-800 text-zinc-200 focus:border-zinc-700 focus:ring-zinc-700/5 shadow-none" 
                            : "bg-white border-zinc-200 text-zinc-800 focus:border-zinc-900 focus:ring-4 focus:ring-zinc-900/5 shadow-xl shadow-zinc-200/40"
                        )}
                      />
                      <button 
                        onClick={sendMessage}
                        disabled={isGenerating || !inputMessage.trim()}
                        className={cn(
                          "absolute right-4 bottom-4 p-2.5 disabled:opacity-30 rounded-xl transition-all",
                          theme === 'dark'
                            ? "bg-white text-black hover:bg-zinc-200 shadow-none"
                            : "bg-zinc-900 hover:bg-black text-white shadow-lg shadow-zinc-400"
                        )}
                      >
                        {isGenerating ? <div className={cn("w-4 h-4 border-2 rounded-full animate-spin", theme === 'dark' ? "border-black/30 border-t-black" : "border-white/30 border-t-white")} /> : <Send className="w-4 h-4" />}
                      </button>
                   </div>
                   <div className="flex items-center justify-between px-3 pt-3">
                      <div className="flex gap-8 items-center">
                        <div className="flex items-center gap-3 group relative">
                          <span className="text-[10px] text-zinc-400 font-bold uppercase tracking-widest leading-none">Temp</span>
                          {showTooltips && (
                            <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                               <strong>Sampling Temperature:</strong> Lowering results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive.
                            </div>
                          )}
                          <input 
                            type="range" 
                            min="0.1" 
                            max="2.0" 
                            step="0.1" 
                            value={temperature} 
                            onChange={(e) => setTemperature(parseFloat(e.target.value))}
                            className="w-20 accent-zinc-900 h-1"
                          />
                          <span className={cn("text-[10px] font-mono font-bold w-6", theme === 'dark' ? "text-zinc-200" : "text-zinc-900")}>{temperature.toFixed(1)}</span>
                        </div>
                        <div className="flex items-center gap-3 group relative">
                          <span className="text-[10px] text-zinc-400 font-bold uppercase tracking-widest leading-none">Penalty</span>
                          {showTooltips && (
                            <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                               <strong>Presence Penalty:</strong> Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
                            </div>
                          )}
                          <input 
                            type="range" 
                            min="0.0" 
                            max="2.0" 
                            step="0.1" 
                            value={presencePenalty} 
                            onChange={(e) => setPresencePenalty(parseFloat(e.target.value))}
                            className="w-20 accent-zinc-900 h-1"
                          />
                          <span className={cn("text-[10px] font-mono font-bold w-6", theme === 'dark' ? "text-zinc-200" : "text-zinc-900")}>{presencePenalty.toFixed(1)}</span>
                        </div>
                        <span className={cn("text-[10px] font-bold uppercase tracking-widest border-l pl-4", theme === 'dark' ? "text-zinc-500 border-zinc-800" : "text-zinc-400 border-zinc-200")}>Context: 1024</span>
                      </div>
                      <span className={cn("text-[10px] font-bold uppercase tracking-widest", theme === 'dark' ? "text-zinc-500" : "text-zinc-400")}>Status: IDLE</span>
                   </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'pretrain' && (
              <motion.div key="pretrain" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8 max-w-6xl mx-auto">
                 <TutorialBox 
                    title="Phase 1: Knowledge Acquisition (Parameters Enabled)" 
                    description="This stage involves unsupervised learning on large text corpora. Point the directory to a folder containing multiple .txt files. The engine will automatically mix and stream them into the model buffer."
                    icon={Database}
                 />

                 <div className="grid grid-cols-3 gap-8">
                    <Card title="Pretraining Protocol" subtitle="Configure core training parameters" className="col-span-2">
                       <div className="space-y-6">
                          <div className="space-y-2 group relative">
                             <div className="flex justify-between items-end">
                                <label className="text-[10px] font-bold text-zinc-400 capitalize tracking-widest">Corpus directory (Project local)</label>
                                <span className={cn("text-[10px] font-bold capitalize flex items-center gap-1", corpusDir ? "text-emerald-500" : "text-rose-500")}>
                                   {corpusDir ? <><ThumbsUp className="w-2.5 h-2.5" /> Validated</> : <><X className="text-xs" /> Required</>}
                                </span>
                             </div>
                             {showTooltips && (
                               <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-full p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                  <strong>Corpus Target:</strong> Point this to a folder within your project. The engine will recursively ingest all <code>.txt</code> files as raw training data.
                               </div>
                             )}
                             <TextField
                                type="text"
                                value={corpusDir}
                                onChange={(e) => setCorpusDir(e.target.value)}
                                placeholder="e.g. datasets/my_corpus"
                             />
                             <p className="text-[10px] font-medium text-zinc-400 italic">The engine will scan all subdirectories for .txt files recursively.</p>
                          </div>

                          <div className="grid grid-cols-3 gap-6">
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Training Epochs</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Epochs:</strong> The number of full passes through the training data. The system will automatically calculate total steps based on your corpus size.
                                </div>
                                <NumberField
                                   value={pretrainEpochs}
                                   onChange={setPretrainEpochs}
                                   min={1}
                                   integer
                                   ariaLabel="Training epochs"
                                />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Base LR</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Learning Rate:</strong> The step size of the optimizer. 3e-4 is the industry standard for stable pretraining of smaller models.
                                </div>
                                <NumberField
                                  value={pretrainLR}
                                  onChange={setPretrainLR}
                                  min={0}
                                  step={0.0001}
                                  ariaLabel="Base learning rate"
                                />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Val Split %</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Val Split:</strong> Percentage of data reserved for evaluation. Higher splits provide better metrics but reduce training data.
                                </div>
                                <NumberField
                                  value={pretrainValSplit}
                                  onChange={setPretrainValSplit}
                                  min={0}
                                  max={50}
                                  ariaLabel="Validation split percent"
                                />
                             </div>
                          </div>

                          <div className="grid grid-cols-3 gap-6">
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Micro Batch</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Micro Batch:</strong> Sequences processed per GPU step.
                                </div>
                                <NumberField
                                  value={pretrainBatchSize}
                                  onChange={setPretrainBatchSize}
                                  min={1}
                                  integer
                                  ariaLabel="Micro batch size"
                                />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Grad Accum</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Gradient Accumulation:</strong> Simulates larger batch sizes by summing gradients over multiple micro-batches.
                                </div>
                                <NumberField
                                  value={pretrainGradAccum}
                                  onChange={setPretrainGradAccum}
                                  min={1}
                                  integer
                                  ariaLabel="Gradient accumulation steps"
                                />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Warmup Steps</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Warmup:</strong> Number of steps to linearly increase LR from zero to base value. Prevents divergent gradients at start.
                                </div>
                                <NumberField
                                  value={pretrainWarmupSteps}
                                  onChange={setPretrainWarmupSteps}
                                  min={0}
                                  integer
                                  ariaLabel="Warmup steps"
                                />
                             </div>
                          </div>

                          <div className={cn(
                            "p-4 border rounded-xl flex items-center justify-between",
                            theme === 'dark' ? "bg-zinc-900 border-zinc-800" : "bg-zinc-900 text-white border-zinc-900"
                          )}>
                             <div className="flex items-center gap-3">
                                <Zap className="w-4 h-4 text-emerald-400 shrink-0" />
                                <span className="text-[10px] font-bold uppercase tracking-widest">Effective Global Batch Size</span>
                             </div>
                             <span className="text-sm font-black font-mono text-emerald-400">
                                {pretrainBatchSize * pretrainGradAccum}
                             </span>
                          </div>


                          <div className={cn(
                            "p-4 border rounded-xl flex gap-3",
                            theme === 'dark' ? "bg-zinc-900 border-zinc-800" : "bg-zinc-50 border-zinc-200"
                          )}>
                             <Terminal className="w-4 h-4 text-zinc-400 shrink-0 mt-0.5" />
                             <p className="text-[10px] text-zinc-500 leading-relaxed font-bold uppercase tracking-widest">
                               Engine will initialize with {settings?.vocab_size || 0} vocabulary size using BPETokenizer.
                             </p>
                          </div>

                          {metrics.meta?.step < metrics.meta?.max_steps && isTraining ? (
                             <button 
                                onClick={stopTraining}
                                className="w-full py-4 bg-rose-600 hover:bg-rose-500 text-white font-bold rounded-xl transition-all flex items-center justify-center gap-2"
                             >
                                <RotateCcw className="w-4 h-4" /> Terminate Active Run
                             </button>
                          ) : (
                             <RunButton onClick={startPretraining}>
                                Initialize Pretraining Run
                             </RunButton>
                          )}
                       </div>
                    </Card>
                    
                    <div className="space-y-8">
                       <Card title="Live Convergence" subtitle="Real-time loss tracking">
                          <div className="h-44 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart data={metrics.steps.slice(-20).map((s, i) => ({ step: s, val: metrics.train_loss[i] }))}>
                                <Line type="monotone" dataKey="val" stroke="#18181b" strokeWidth={2} dot={false} />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                          <div className={cn("mt-4 flex justify-between items-end pt-3 border-t", theme === 'dark' ? "border-zinc-800" : "border-zinc-50")}>
                             <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Perplexity</span>
                             <span className={cn("text-sm font-black font-mono", theme === 'dark' ? "text-white" : "text-zinc-900")}>
                                {metrics.meta?.last_train_perplexity ? metrics.meta.last_train_perplexity.toFixed(2) : "N/A"}
                             </span>
                          </div>
                       </Card>

                       <Card title="Runtime Statistics">
                          <div className="space-y-3">
                             {[
                                { label: "Progress", val: `Epoch ${metrics.meta?.epoch || 0} / ${metrics.meta?.total_epochs || pretrainEpochs}`, status: "ok" },
                                { label: "Step Count", val: `${metrics.meta?.step || 0} / ${metrics.meta?.max_steps || '...'}`, status: "ok" },
                                { label: "GPU Device", val: settings?.device || 'CPU', status: "ok" },
                                { label: "Engine State", val: metrics.meta?.step ? "TRAINING" : "IDLE", status: metrics.meta?.step ? "high" : "ok" }
                             ].map((stat, i) => (
                                <div key={i} className="flex justify-between items-center text-[11px] font-bold">
                                   <span className="text-zinc-400 uppercase tracking-widest">{stat.label}</span>
                                   <span className={cn("font-mono", stat.status === 'high' ? "text-emerald-500" : "text-zinc-800")}>{stat.val}</span>
                                </div>
                             ))}
                          </div>
                       </Card>
                    </div>
                 </div>

                 <div className="grid grid-cols-3 gap-8">
                    <Card title="Tokenizer Laboratory" subtitle="Build or restore the active BPE tokenizer" className="col-span-2">
                       <div className="space-y-6">
                           <div className="space-y-3">
                              <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest leading-none">Tokenizer Source Type</label>
                              <div className="grid grid-cols-2 gap-3">
                                 {[
                                   { id: 'file' as const, label: 'Single Text File', desc: 'Use one .txt file as the vocab basis.' },
                                   { id: 'corpus' as const, label: 'Corpus Folder', desc: 'Scan a folder recursively for .txt files.' }
                                 ].map((option) => (
                                   <button
                                     key={option.id}
                                     type="button"
                                     onClick={() => setTokSourceType(option.id)}
                                     className={cn(
                                       "rounded-xl border px-4 py-3 text-left transition-all",
                                       tokSourceType === option.id
                                         ? (theme === 'dark' ? "bg-zinc-100 text-black border-zinc-100" : "bg-zinc-900 text-white border-zinc-900")
                                         : (theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-400 hover:border-zinc-700" : "bg-zinc-50 border-zinc-200 text-zinc-600 hover:border-zinc-300")
                                     )}
                                   >
                                      <div className="text-xs font-black uppercase tracking-widest">{option.label}</div>
                                      <div className={cn("mt-1 text-[10px] font-medium", tokSourceType === option.id ? "opacity-70" : "text-zinc-500")}>{option.desc}</div>
                                   </button>
                                 ))}
                              </div>
                           </div>

                           <div className="space-y-2 group relative">
                              <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest leading-none">
                                {tokSourceType === 'file' ? 'Text File Path' : 'Corpus Folder Path'}
                              </label>
                              <div className="flex gap-3">
                                <TextField
                                  type="text"
                                  placeholder={tokSourceType === 'file' ? 'E:\\AGENT 3\\training\\FineFactualNews.txt' : 'corpus'}
                                  value={tokPath}
                                  onChange={(e) => setTokPath(e.target.value)}
                                  className="flex-1"
                                />
                                <button
                                  type="button"
                                  onClick={browseTokenizerSource}
                                  className={cn(
                                    "px-4 rounded-xl border text-xs font-black uppercase tracking-widest transition-all",
                                    theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-300 hover:bg-zinc-800" : "bg-white border-zinc-200 text-zinc-700 hover:bg-zinc-50"
                                  )}
                                >
                                  Browse
                                </button>
                              </div>
                              <p className="text-[10px] text-zinc-500 font-medium">
                                This field is required. Relative paths are resolved from the h2 repo root.
                              </p>
                           </div>

                           <div className="grid grid-cols-3 gap-6">
                              <div className="space-y-2 group relative">
                                 <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest leading-none">Vocabulary Size</label>
                                 <NumberField
                                   value={tokVocabSize}
                                   onChange={setTokVocabSize}
                                   min={256}
                                   integer
                                   ariaLabel="Tokenizer vocabulary size"
                                 />
                              </div>
                              <div className="space-y-2 group relative">
                                 <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest leading-none">Min Frequency</label>
                                 <NumberField
                                   value={tokMinFreq}
                                   onChange={setTokMinFreq}
                                   min={1}
                                   integer
                                   ariaLabel="Tokenizer minimum frequency"
                                 />
                              </div>
                              <div className="space-y-2 group relative">
                                 <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest leading-none">RAM Guard MB</label>
                                 <NumberField
                                   value={tokMaxInputMb}
                                   onChange={setTokMaxInputMb}
                                   min={0}
                                   integer
                                   ariaLabel="Tokenizer RAM guard megabytes"
                                 />
                                 <p className="text-[10px] text-zinc-500 font-medium">0 = auto-safe based on detected RAM.</p>
                              </div>
                           </div>

                           <div className={cn("rounded-xl border p-4 text-[10px] font-mono leading-6", theme === 'dark' ? "bg-zinc-950 border-zinc-800 text-zinc-400" : "bg-zinc-50 border-zinc-200 text-zinc-600")}>
                              <div><strong>Primary output:</strong> data/tokenizer.json</div>
                              <div><strong>Project copy:</strong> {settings?.project_tokenizer_path || 'projects/haiku_studio/tokenizer.json'}</div>
                              <div><strong>Bundled prebuilt:</strong> {settings?.prebuilt_tokenizer_path || 'studio/prebuilt/default_tokenizer.json'}</div>
                           </div>

                           <div className="grid grid-cols-2 gap-3">
                              <button 
                                onClick={startTokenizerTraining}
                                disabled={isTrainingTok || isTraining}
                                className={cn(
                                  "py-4 font-bold rounded-xl transition-all flex items-center justify-center gap-2 shadow-lg hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:hover:translate-y-0",
                                  theme === 'dark' ? "bg-zinc-200 text-black" : "bg-zinc-900 text-white"
                                )}
                              >
                                 {isTrainingTok ? <RotateCcw className="w-4 h-4 animate-spin" /> : <Database className="w-4 h-4" />}
                                 {isTrainingTok ? 'Tokenizer Running' : 'Build Tokenizer'}
                              </button>
                              <button
                                type="button"
                                onClick={restorePrebuiltTokenizer}
                                disabled={isTrainingTok || isTraining}
                                className={cn(
                                  "py-4 font-bold rounded-xl transition-all flex items-center justify-center gap-2 border disabled:opacity-50",
                                  theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-200 hover:bg-zinc-800" : "bg-white border-zinc-200 text-zinc-900 hover:bg-zinc-50"
                                )}
                              >
                                <RotateCcw className="w-4 h-4" />
                                Restore Prebuilt
                              </button>
                           </div>
                       </div>
                    </Card>
                    <div className="space-y-8">
                       <Card title="Tokenizer Insights">
                          <p className="text-[10px] text-zinc-500 leading-relaxed font-medium">
                            Tokenizer training now streams source progress into the System Kernel Output panel. The RAM guard samples large corpora instead of loading everything blindly, then saves both the runtime data tokenizer and the active project tokenizer copy.
                          </p>
                       </Card>
                    </div>
                 </div>

                 <Card title="Full Training Audit" subtitle="Global gradient and loss trajectory">
                    <div className="h-80 w-full mt-4">
                       <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={metrics.steps.map((s, i) => ({ step: s, train: metrics.train_loss[i], val: metrics.val_loss[i] }))}>
                             <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" />
                             <XAxis dataKey="step" stroke="#a1a1aa" fontSize={10} tickLine={false} axisLine={false} />
                             <YAxis stroke="#a1a1aa" fontSize={10} tickLine={false} axisLine={false} />
                             <RechartsTooltip contentStyle={{ background: theme === 'dark' ? '#18181b' : '#fff', border: theme === 'dark' ? '1px solid #27272a' : '1px solid #e4e4e7', borderRadius: '12px', fontSize: '10px', boxShadow: theme === 'dark' ? 'none' : '0 10px 15px -3px rgba(0,0,0,0.1)' }} />
                             <Line type="monotone" dataKey="train" stroke="#e4e4e7" strokeWidth={2} dot={false} />
                             <Line type="monotone" dataKey="val" stroke="#18181b" strokeWidth={2} dot={false} />
                          </LineChart>
                       </ResponsiveContainer>
                    </div>
                 </Card>
              </motion.div>
            )}

            {activeTab === 'sft' && (
              <motion.div key="sft" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8 max-w-6xl mx-auto">
                 <TutorialBox 
                    title="Phase 2: Instruction Tuning" 
                    description="Supervised Fine-Tuning (SFT) transforms a base model into a conversational assistant. This stage requires high-quality multi-turn dialogues. Gradient masking ensures the model only learns from assistant responses."
                    icon={Brain}
                 />

                 <div className="grid grid-cols-1 gap-8">
                    <Card title="SFT Configuration" subtitle="Refine dialogue parameters">
                       <div className="space-y-6">
                          <div className="space-y-2 group relative">
                             <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">SFT Training Data</label>
                             <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-full p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                <strong>SFT Dataset:</strong> Provide a <code>.jsonl</code> file containing multi-turn dialogues formatted with <code>user:</code> and <code>bot:</code> segments.
                             </div>
                             <TextField type="text" value={sftDataPath} onChange={(e) => setSftDataPath(e.target.value)} placeholder={settings?.project_sft_dir || "projects/haiku_studio/datasets/sft"} />
                          </div>

                          <div className="grid grid-cols-3 gap-6">
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Epochs</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Epochs:</strong> Number of times the optimizer passes through the entire SFT dataset. Usually 1-3 is sufficient.
                                </div>
                                <NumberField value={sftEpochs} onChange={setSftEpochs} min={1} integer ariaLabel="SFT epochs" />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Batch Size</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Batch Size:</strong> Number of conversation pairs processed simultaneously during one update step.
                                </div>
                                <NumberField value={sftBatchSize} onChange={setSftBatchSize} min={1} integer ariaLabel="SFT batch size" />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Learning Rate</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Fine-Tuning LR:</strong> Often much lower than pretraining LR (e.g., 5e-5) to prevent catastrophic forgetting of base knowledge.
                                </div>
                                <NumberField value={sftLR} onChange={setSftLR} min={0} step={0.000001} ariaLabel="SFT learning rate" />
                             </div>
                          </div>

                          <div className={cn(
                            "p-4 rounded-xl flex gap-3",
                            theme === 'dark' ? "bg-blue-900/10 border border-blue-900/20 text-blue-300" : "bg-blue-50 border border-blue-100 text-blue-800"
                          )}>
                             <MessagesSquare className="w-4 h-4 shrink-0 mt-0.5" />
                             <p className="text-xs leading-relaxed font-semibold">
                               Standard chat template detected: <code>user:/bot:</code> roles. System will auto-mask non-bot tokens.
                             </p>
                          </div>

                          <RunButton onClick={startSFT} disabled={isTraining}>
                             Start SFT Training
                          </RunButton>
                       </div>
                    </Card>
                 </div>
              </motion.div>
            )}

            {activeTab === 'dpo' && (
              <motion.div key="dpo" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8 max-w-6xl mx-auto">
                 <TutorialBox 
                    title="Phase 3: Preference Alignment" 
                    description="DPO uses prompt, chosen response, and rejected response pairs to nudge an SFT model toward the answers you prefer."
                    icon={Zap}
                    colorClass="bg-zinc-900 text-white border-zinc-900"
                 />

                 <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
                    <Card title="DPO Training" subtitle="Preference pairs, reference model, and alignment checkpoint" className="xl:col-span-2">
                       <div className="space-y-6">
                          {!settings?.dpo_ready ? (
                            <div className={cn("text-center py-12 space-y-4 border rounded-xl", theme === 'dark' ? "bg-rose-950/30 border-rose-900/50" : "bg-rose-50/60 border-rose-100")}>
                               <h4 className="font-bold text-rose-600 dark:text-rose-300 text-sm uppercase tracking-tight">dpo.py Missing</h4>
                               <p className="text-[11px] text-rose-600 dark:text-rose-200 max-w-[360px] mx-auto leading-relaxed">The UI expects a root-level dpo.py trainer. Restore dpo.py before launching alignment.</p>
                            </div>
                          ) : (
                            <div className="space-y-8">
                               <div className={cn("rounded-2xl border p-5 space-y-3", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-zinc-50 border-zinc-200")}>
                                  <div className="flex items-center gap-2">
                                    <Info className="w-4 h-4 text-blue-500" />
                                    <h4 className={cn("text-[11px] font-black uppercase tracking-widest", theme === 'dark' ? "text-zinc-100" : "text-zinc-900")}>Where DPO prompts come from</h4>
                                  </div>
                                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400 leading-relaxed font-medium">
                                    DPO reads preference pairs from the dataset path below. Each pair needs the original prompt, a preferred response, and a weaker rejected response. Chat Lab thumbs-up/down feedback writes pairs into the active project's DPO dataset automatically, and you can also place your own <code>.jsonl</code> files in that folder.
                                  </p>
                                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-[10px] font-semibold">
                                    <div className={cn("rounded-xl border p-3", theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-300" : "bg-white border-zinc-200 text-zinc-700")}><strong>1.</strong> Test replies in Chat Lab.</div>
                                    <div className={cn("rounded-xl border p-3", theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-300" : "bg-white border-zinc-200 text-zinc-700")}><strong>2.</strong> Save chosen/rejected pairs.</div>
                                    <div className={cn("rounded-xl border p-3", theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-300" : "bg-white border-zinc-200 text-zinc-700")}><strong>3.</strong> Start DPO training here.</div>
                                  </div>
                               </div>

                               <div className="grid grid-cols-2 gap-6">
                                  <div className="space-y-3 text-left col-span-2">
                                     <label className="text-[10px] font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider ml-1">Preference Dataset Path</label>
                                     <TextField value={dpoDataPath} onChange={(e) => setDpoDataPath(e.target.value)} placeholder={settings?.project_dpo_dir || "projects/haiku_studio/datasets/dpo"} />
                                     <p className="text-[9px] text-zinc-500 dark:text-zinc-400 font-medium px-1">Use a project DPO folder or a single JSONL/text file. JSONL rows should include prompt, chosen, and rejected fields.</p>
                                  </div>
                                  <div className="space-y-3 text-left">
                                     <label className="text-[10px] font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider ml-1">Beta</label>
                                     <NumberField value={dpoBeta} onChange={setDpoBeta} min={0} step={0.01} ariaLabel="DPO beta" />
                                     <p className="text-[9px] text-zinc-500 dark:text-zinc-400 font-medium px-1">Preference strength against the frozen reference model. Lower is stronger; higher is more conservative.</p>
                                  </div>
                                  <div className="space-y-3 text-left">
                                     <label className="text-[10px] font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider ml-1">Alignment LR</label>
                                     <NumberField value={dpoLR} onChange={setDpoLR} min={0} step={0.000001} ariaLabel="DPO learning rate" />
                                     <p className="text-[9px] text-zinc-500 dark:text-zinc-400 font-medium px-1">Use a small LR. DPO is an alignment pass, not a new pretraining run.</p>
                                  </div>
                                  <div className="space-y-3 text-left">
                                     <label className="text-[10px] font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider ml-1">Epochs</label>
                                     <NumberField value={dpoEpochs} onChange={setDpoEpochs} min={1} integer ariaLabel="DPO epochs" />
                                  </div>
                                  <div className="space-y-3 text-left">
                                     <label className="text-[10px] font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider ml-1">Batch Size</label>
                                     <NumberField value={dpoBatchSize} onChange={setDpoBatchSize} min={1} integer ariaLabel="DPO batch size" />
                                  </div>
                               </div>
                               <RunButton onClick={startDPO} disabled={isTraining || !settings?.dpo_ready}>
                                  Start DPO Training
                               </RunButton>
                               <p className="text-[10px] text-zinc-500 dark:text-zinc-400 leading-relaxed">
                                  If no reference checkpoint exists, dpo.py creates one from the configured SFT policy checkpoint and freezes it. Output saves to {settings?.dpo_checkpoint || 'projects/haiku_studio/checkpoints/model.dpo.pt'}.
                               </p>
                            </div>
                          )}
                       </div>
                    </Card>

                    <div className="space-y-8">
                       <Card title="DPO Stats" subtitle="Live from active project files">
                          <div className="space-y-6">
                             <div className="grid grid-cols-2 gap-3">
                               {[
                                 { label: 'Preference Pairs', value: dpoStats?.preference_pairs ?? settings?.dpo_buffer ?? 0, suffix: 'pairs' },
                                 { label: 'Dataset Files', value: dpoStats?.file_count ?? 0, suffix: 'files' },
                                 { label: 'Chat Lab Buffer', value: dpoStats?.feedback_pairs ?? 0, suffix: 'pairs' },
                                 { label: 'Latest Step', value: dpoStats?.latest_step ?? settings?.dpo_global_step ?? 0, suffix: 'step' }
                               ].map((stat) => (
                                 <div key={stat.label} className={cn("rounded-xl border p-4", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-zinc-50 border-zinc-200")}>
                                   <span className="text-[9px] font-black text-zinc-500 dark:text-zinc-400 uppercase tracking-widest">{stat.label}</span>
                                   <div className={cn("mt-2 text-2xl font-black font-mono tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>{stat.value}</div>
                                   <span className="text-[9px] text-zinc-400 uppercase tracking-wider">{stat.suffix}</span>
                                 </div>
                               ))}
                             </div>

                             <div className={cn("rounded-xl border p-4 space-y-2", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-zinc-50 border-zinc-200")}>
                               <p className="text-[9px] font-black text-zinc-500 dark:text-zinc-400 uppercase tracking-widest">Dataset Source</p>
                               <p className={cn("text-[10px] font-mono break-all leading-relaxed", theme === 'dark' ? "text-zinc-300" : "text-zinc-700")}>{dpoStats?.dataset_path || settings?.dpo_dataset || settings?.project_dpo_dir || 'projects/haiku_studio/datasets/dpo'}</p>
                               <p className="text-[9px] text-zinc-500 dark:text-zinc-400">Metrics log: {dpoStats?.log_path || 'projects/haiku_studio/logs/dpo_loss.jsonl'}</p>
                             </div>

                             <div className="space-y-2">
                               <p className="text-[9px] font-black text-zinc-500 dark:text-zinc-400 uppercase tracking-widest">Checkpoint Status</p>
                               {Object.entries(dpoStats?.checkpoints || {}).map(([key, ckptRaw]) => {
                                 const ckpt = ckptRaw as { path: string; exists: boolean; size_bytes?: number; modified_at?: string | null };
                                 return (
                                   <div key={key} className={cn("rounded-lg border p-3 flex items-start justify-between gap-3", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-white border-zinc-200")}>
                                     <div className="min-w-0">
                                       <p className={cn("text-[10px] font-black uppercase tracking-wider", ckpt.exists ? "text-emerald-500" : "text-zinc-400")}>{key} {ckpt.exists ? 'ready' : 'missing'}</p>
                                       <p className={cn("text-[9px] font-mono break-all", theme === 'dark' ? "text-zinc-500" : "text-zinc-500")}>{ckpt.path}</p>
                                     </div>
                                     <div className="text-right shrink-0">
                                       <p className="text-[9px] font-bold text-zinc-400">{formatBytes(ckpt.size_bytes)}</p>
                                       <p className="text-[8px] text-zinc-500">{ckpt.exists ? formatShortDate(ckpt.modified_at) : 'not found'}</p>
                                     </div>
                                   </div>
                                 );
                               })}
                             </div>

                             {dpoStats?.latest_metrics && (
                               <div className={cn("rounded-xl border p-4", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-zinc-50 border-zinc-200")}>
                                  <p className="text-[9px] font-black text-zinc-500 dark:text-zinc-400 uppercase tracking-widest mb-2">Last Metrics Row</p>
                                  <pre className={cn("text-[9px] whitespace-pre-wrap break-words font-mono", theme === 'dark' ? "text-zinc-300" : "text-zinc-700")}>{JSON.stringify(dpoStats.latest_metrics, null, 2)}</pre>
                               </div>
                             )}

                             <div className="pt-2 flex gap-2">
                                <button onClick={flushDPOFeedback} className={cn("flex-1 py-2.5 border font-bold rounded-lg text-[9px] uppercase tracking-wider transition-all", theme === 'dark' ? "bg-zinc-900 hover:bg-zinc-800 border-zinc-800 text-zinc-200" : "bg-zinc-50 hover:bg-zinc-100 border-zinc-200 text-zinc-700")}>Flush Feedback</button>
                                <button onClick={() => { fetchSettings(); fetchDpoStats(); }} className={cn("flex-1 py-2.5 border font-bold rounded-lg text-[9px] uppercase tracking-wider transition-all", theme === 'dark' ? "bg-white hover:bg-zinc-100 border-white text-black" : "bg-zinc-900 hover:bg-black border-zinc-900 text-white")}>Refresh</button>
                             </div>
                          </div>
                       </Card>
                    </div>
                 </div>
              </motion.div>
            )}

            {activeTab === 'hub' && (
              <motion.div key="hub" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                 <HuggingFaceHub theme={theme} />
              </motion.div>
            )}


            {activeTab === 'help' && (
              <motion.div key="help" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-10 max-w-5xl mx-auto pb-32">
                <div className="space-y-4 pt-8 text-center sm:text-left">
                  <div className="flex flex-wrap items-center justify-center sm:justify-start gap-3">
                    <span className={cn("px-2.5 py-1 text-[10px] font-black uppercase tracking-widest rounded-lg", theme === 'dark' ? "bg-white text-black" : "bg-black text-white")}>Field Manual</span>
                    <span className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em]">Beginner Workflow Guide</span>
                  </div>
                  <h1 className={cn("text-3xl font-black tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>Using Haiku Studio</h1>
                  <p className="text-sm text-zinc-500 dark:text-zinc-400 font-medium leading-relaxed max-w-3xl">
                    Haiku Studio is a local desktop workspace for building and testing small transformer language models. The UI manages projects, tokenizers, training runs, metrics, checkpoints, and chat testing while the h2 Python trainers do the actual model work behind the scenes.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {[
                    { label: '1. Create or load a project', desc: 'A project is the persistent folder for tokenizer, config, datasets, logs, and checkpoints.', icon: Database },
                    { label: '2. Build or restore tokenizer', desc: 'Use a text file or corpus folder to create data/tokenizer.json, or restore the bundled default tokenizer.', icon: FileText },
                    { label: '3. Train and evaluate', desc: 'Run pretraining, SFT, optional DPO, then test checkpoints in Chat Lab.', icon: Brain }
                  ].map((item) => {
                    const Icon = item.icon;
                    return (
                      <div key={item.label} className={cn("rounded-2xl border p-5 space-y-3", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-white border-zinc-200 shadow-sm")}>
                        <Icon className="w-5 h-5 text-emerald-500" />
                        <h3 className={cn("text-sm font-black", theme === 'dark' ? "text-white" : "text-zinc-900")}>{item.label}</h3>
                        <p className="text-[11px] text-zinc-500 dark:text-zinc-400 leading-relaxed font-medium">{item.desc}</p>
                      </div>
                    );
                  })}
                </div>

                <section className="space-y-5">
                  <h2 className={cn("text-xl font-black tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>Project system</h2>
                  <div className={cn("rounded-2xl border p-6 space-y-4", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-zinc-50 border-zinc-200")}>
                    <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed font-medium">
                      Projects are the source of truth. The <code>data/</code> folder is still used as the runtime staging area because the h2 trainers expect it, but Haiku Studio syncs the active project into <code>data/</code> before jobs run and saves outputs back into the project folder.
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-[11px] font-mono">
                      {[
                        'projects/<name>/tokenizer.json',
                        'projects/<name>/config/gpt_config.json',
                        'projects/<name>/datasets/corpus',
                        'projects/<name>/datasets/sft',
                        'projects/<name>/datasets/dpo',
                        'projects/<name>/checkpoints',
                        'projects/<name>/logs',
                        'projects/<name>/cache'
                      ].map((pathLabel) => (
                        <div key={pathLabel} className={cn("rounded-xl border px-4 py-3 break-all", theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-300" : "bg-white border-zinc-200 text-zinc-700")}>{pathLabel}</div>
                      ))}
                    </div>
                    <p className="text-[11px] text-zinc-500 dark:text-zinc-400 leading-relaxed">
                      Use <strong>Load Project Into Data</strong> when you want the selected project copied into the runtime folder. Use <strong>Save Runtime To Project</strong> when you want current runtime files copied back into the active project.
                    </p>
                  </div>
                </section>

                <section className="space-y-5">
                  <h2 className={cn("text-xl font-black tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>Tokenizer workflow</h2>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                    <div className={cn("rounded-2xl border p-6 space-y-4", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-white border-zinc-200 shadow-sm")}>
                      <h3 className={cn("text-sm font-black", theme === 'dark' ? "text-white" : "text-zinc-900")}>When to use the bundled tokenizer</h3>
                      <p className="text-[12px] text-zinc-500 dark:text-zinc-400 leading-relaxed font-medium">
                        Use the prebuilt tokenizer when you want to start quickly or keep compatibility with existing checkpoints. The restore button copies the bundled tokenizer into <code>data/tokenizer.json</code> and into the active project.
                      </p>
                    </div>
                    <div className={cn("rounded-2xl border p-6 space-y-4", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-white border-zinc-200 shadow-sm")}>
                      <h3 className={cn("text-sm font-black", theme === 'dark' ? "text-white" : "text-zinc-900")}>When to build a custom tokenizer</h3>
                      <p className="text-[12px] text-zinc-500 dark:text-zinc-400 leading-relaxed font-medium">
                        Build a tokenizer from either one representative text file or a corpus folder. A domain-specific tokenizer can help if your training data has unusual words, code, game names, usernames, or formatting that the default tokenizer does not handle well.
                      </p>
                    </div>
                  </div>
                  <div className={cn("rounded-2xl border p-5", theme === 'dark' ? "bg-amber-950/20 border-amber-900/40" : "bg-amber-50 border-amber-200")}>
                    <p className="text-[12px] text-amber-800 dark:text-amber-200 leading-relaxed font-semibold">
                      Tokenizer training can use a lot of RAM on very large corpora. The RAM guard samples large inputs instead of loading everything at once. If no file or folder is selected, the app should stop and show an error rather than guessing.
                    </p>
                  </div>
                </section>

                <section className="space-y-5">
                  <h2 className={cn("text-xl font-black tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>Training phases</h2>
                  <div className="space-y-4">
                    {[
                      { title: 'Pretraining', icon: Cpu, body: 'Teaches the base model general next-token prediction from raw text. Use the corpus dataset folder. This is the longest and most hardware-intensive phase.' },
                      { title: 'SFT', icon: MessagesSquare, body: 'Teaches assistant behavior using user:/bot: dialogue samples. The trainer masks user turns and learns from bot replies.' },
                      { title: 'DPO', icon: Zap, body: 'Aligns the SFT model with preference pairs. Each pair has a prompt, a chosen answer, and a rejected answer. DPO uses a frozen reference model so alignment does not drift too far from the policy model.' }
                    ].map((item) => {
                      const Icon = item.icon;
                      return (
                        <div key={item.title} className={cn("rounded-2xl border p-5 flex gap-4", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-white border-zinc-200 shadow-sm")}>
                          <div className={cn("w-11 h-11 rounded-xl flex items-center justify-center shrink-0", theme === 'dark' ? "bg-zinc-900 text-zinc-200" : "bg-zinc-100 text-zinc-900")}><Icon className="w-5 h-5" /></div>
                          <div>
                            <h3 className={cn("text-sm font-black", theme === 'dark' ? "text-white" : "text-zinc-900")}>{item.title}</h3>
                            <p className="text-[12px] text-zinc-500 dark:text-zinc-400 leading-relaxed font-medium mt-1">{item.body}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>

                <section className="space-y-5">
                  <h2 className={cn("text-xl font-black tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>DPO data format</h2>
                  <div className={cn("rounded-2xl border p-6 space-y-4", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-zinc-50 border-zinc-200")}>
                    <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed font-medium">
                      The DPO tab reads from the active project DPO dataset path. Chat Lab can create feedback pairs automatically, and you can also add JSONL files manually.
                    </p>
                    <pre className={cn("rounded-xl p-4 text-[11px] overflow-x-auto font-mono", theme === 'dark' ? "bg-zinc-900 text-zinc-300" : "bg-white text-zinc-700 border border-zinc-200")}>{`{"prompt":"user: Explain X\nbot:","chosen":"Clear preferred answer.","rejected":"Weaker answer."}`}</pre>
                    <p className="text-[11px] text-zinc-500 dark:text-zinc-400 leading-relaxed">
                      Good DPO pairs should compare two answers to the same prompt. Do not use unrelated chosen/rejected text. Keep the preferred answer genuinely better, not just longer.
                    </p>
                  </div>
                </section>

                <section className="space-y-5">
                  <h2 className={cn("text-xl font-black tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>Kernel output and errors</h2>
                  <div className={cn("rounded-2xl border p-6 space-y-4", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-white border-zinc-200 shadow-sm")}>
                    <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed font-medium">
                      The System Kernel Output panel shows backend logs from tokenizer training, pretraining, SFT, DPO, project sync, and app startup. Serious failures open the kernel automatically and are highlighted red. Warnings and non-fatal issues are highlighted yellow.
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-[11px] font-semibold">
                      <div className="rounded-xl border border-rose-300 bg-rose-50 text-rose-900 dark:bg-rose-950/40 dark:text-rose-100 dark:border-rose-800 p-3"><AlertTriangle className="w-4 h-4 mb-2" />Red means action required.</div>
                      <div className="rounded-xl border border-amber-300 bg-amber-50 text-amber-900 dark:bg-amber-950/40 dark:text-amber-100 dark:border-amber-800 p-3"><AlertTriangle className="w-4 h-4 mb-2" />Yellow means warning or fallback.</div>
                      <div className={cn("rounded-xl border p-3", theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-300" : "bg-zinc-50 border-zinc-200 text-zinc-700")}><Terminal className="w-4 h-4 mb-2" />Clear Buffer resets only the visible log panel.</div>
                    </div>
                  </div>
                </section>

                <section className="space-y-5">
                  <h2 className={cn("text-xl font-black tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>Common beginner mistakes</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {[
                      ['Changing tokenizer mid-project', 'Changing tokenizers after training makes old checkpoints incompatible unless the vocab is identical.'],
                      ['Training DPO before SFT', 'DPO expects a policy model that already knows how to answer. Run SFT first.'],
                      ['Using too much learning rate', 'If loss spikes, repeats, or collapses, lower the learning rate and resume from a better checkpoint.'],
                      ['Forgetting active project', 'Check the top bar before training. Outputs save under the active project folder.']
                    ].map(([title, body]) => (
                      <div key={title} className={cn("rounded-2xl border p-5", theme === 'dark' ? "bg-zinc-950/50 border-zinc-800" : "bg-white border-zinc-200 shadow-sm")}>
                        <h3 className={cn("text-sm font-black", theme === 'dark' ? "text-white" : "text-zinc-900")}>{title}</h3>
                        <p className="text-[12px] text-zinc-500 dark:text-zinc-400 leading-relaxed font-medium mt-2">{body}</p>
                      </div>
                    ))}
                  </div>
                </section>

                <footer className="pt-10 border-t border-[var(--border)] text-center">
                  <p className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.35em]">ROOTCOMPUTER · Haiku Studio</p>
                </footer>
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        {/* Platform Monitor / Console */}
        <div className={cn(
          "transition-all flex flex-col z-20 border-t",
          isTerminalOpen ? "h-72" : "h-12",
          theme === 'dark' ? "bg-zinc-900 border-zinc-800 shadow-none" : "bg-white border-zinc-200 shadow-2xl"
        )}>
           <div 
             onClick={() => setIsTerminalOpen(!isTerminalOpen)}
             className={cn("h-12 px-6 flex items-center justify-between cursor-pointer border-b", theme === 'dark' ? "border-zinc-800" : "border-zinc-50")}
           >
              <div className="flex items-center gap-3 min-w-0">
                <Terminal className={cn(
                  "w-4 h-4 shrink-0",
                  kernelAttention === 'error' ? "text-rose-500" : kernelAttention === 'warning' ? "text-amber-500" : "text-zinc-400"
                )} />
                <span className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">System Kernel Output</span>
                {!isTerminalOpen && kernelAttention !== 'normal' && (
                  <span className={cn(
                    "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[9px] font-black uppercase tracking-widest border",
                    kernelAttention === 'error'
                      ? (theme === 'dark' ? "bg-rose-950/70 border-rose-500/50 text-rose-100" : "bg-rose-50 border-rose-200 text-rose-800")
                      : (theme === 'dark' ? "bg-amber-950/70 border-amber-500/50 text-amber-100" : "bg-amber-50 border-amber-200 text-amber-800")
                  )}>
                    <AlertTriangle className="w-3 h-3" />
                    {kernelAttention === 'error' ? 'Action Required' : 'Warning'}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-4">
                 {isTerminalOpen && (
                   <button onClick={(e) => { e.stopPropagation(); setLogs([]); setKernelAttention('normal'); }} className="text-[10px] font-bold text-zinc-400 hover:text-rose-500 transition-colors uppercase tracking-widest flex items-center gap-1.5 px-3 py-1 hover:bg-rose-50 dark:hover:bg-rose-950/30 rounded-lg">
                      <Trash2 className="w-3 h-3" /> Clear Buffer
                   </button>
                 )}
                 <div className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-full border transition-colors",
                    kernelAttention === 'error'
                      ? (theme === 'dark' ? "bg-rose-950/60 border-rose-500/40" : "bg-rose-50 border-rose-200")
                      : kernelAttention === 'warning'
                        ? (theme === 'dark' ? "bg-amber-950/60 border-amber-500/40" : "bg-amber-50 border-amber-200")
                        : (theme === 'dark' ? "bg-zinc-800 border-zinc-700" : "bg-zinc-50 border-zinc-100")
                 )}>
                    <span className={cn(
                      "w-1.5 h-1.5 rounded-full",
                      kernelAttention === 'error' ? "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.7)]" :
                      kernelAttention === 'warning' ? "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.7)]" :
                      "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]"
                    )} />
                    <span className={cn(
                      "text-[10px] font-black uppercase tracking-widest",
                      kernelAttention === 'error'
                        ? (theme === 'dark' ? "text-rose-100" : "text-rose-800")
                        : kernelAttention === 'warning'
                          ? (theme === 'dark' ? "text-amber-100" : "text-amber-800")
                          : (theme === 'dark' ? "text-zinc-400" : "text-zinc-600")
                    )}>
                      {kernelAttention === 'error' ? 'Kernel Alert' : kernelAttention === 'warning' ? 'Kernel Warning' : 'Kernel Ready'}
                    </span>
                 </div>
              </div>
           </div>
           {isTerminalOpen && (
             <div 
               ref={logBoxRef}
               className={cn(
                 "flex-1 p-6 overflow-y-auto font-mono text-[11px] leading-7 select-text custom-scrollbar transition-colors duration-300",
                 theme === 'dark' ? "bg-zinc-950 text-zinc-400" : "bg-white text-zinc-500"
               )}
             >
                {logs.length === 0 && <span className="text-zinc-300 italic font-medium">Listening for engine cycles... system operational.</span>}
                {logs.map((entry, i) => {
                  const severity = entry.severity;
                  return (
                    <div key={entry.id} className={cn(
                      "flex gap-4 py-1.5 px-2 rounded-md transition-colors",
                      kernelSeverityClass(severity, theme)
                    )}>
                      <span className="text-zinc-400 dark:text-zinc-600 shrink-0 font-bold tabular-nums">{(i+1).toString().padStart(4, '0')}</span>
                      <span className="text-zinc-400 shrink-0 font-bold text-[9px] uppercase tracking-tighter mt-0.5 opacity-70">{entry.time}</span>
                      <span className={cn("shrink-0 px-2 py-0.5 rounded-md text-[8px] font-black uppercase tracking-widest h-fit mt-0.5", kernelSeverityLabelClass(severity, theme))}>
                        {severity === 'error' ? 'ERR' : severity === 'warning' ? 'WARN' : 'LOG'}
                      </span>
                      <span className={cn(
                        "break-all pr-4 font-medium",
                        severity === 'error' ? (theme === 'dark' ? "text-rose-100 font-bold" : "text-rose-950 font-bold") :
                        severity === 'warning' ? (theme === 'dark' ? "text-amber-100 font-semibold" : "text-amber-950 font-semibold") :
                        entry.line.includes('loss') ? (theme === 'dark' ? "text-zinc-200 font-bold" : "text-zinc-900 font-bold") :
                        (theme === 'dark' ? "text-zinc-400" : "text-zinc-500")
                      )}>
                        {entry.line}
                      </span>
                    </div>
                  );
                })}
             </div>
           )}
        </div>

      </div>

      {/* New Project Modal */}
      <AnimatePresence>
        {isNewProjectOpen && (
          <div className="fixed inset-0 z-[110] flex items-center justify-center p-6">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => !isCreatingProject && setIsNewProjectOpen(false)}
              className="absolute inset-0 bg-zinc-900/70 backdrop-blur-md"
            />
            <motion.div
              initial={{ scale: 0.98, opacity: 0, y: 10 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.98, opacity: 0, y: 10 }}
              className={cn(
                "relative w-full max-w-lg rounded-2xl shadow-2xl overflow-hidden border",
                theme === 'dark' ? "bg-zinc-950 border-zinc-800" : "bg-white border-zinc-200"
              )}
            >
              <div className="px-7 py-6 border-b border-[var(--border)] flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={cn("w-11 h-11 rounded-xl flex items-center justify-center", theme === 'dark' ? "bg-white text-black" : "bg-zinc-900 text-white")}>
                    <Plus className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-[var(--foreground)] tracking-tight">Create New Project</h3>
                    <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mt-1">Creates project folders and stages runtime files</p>
                  </div>
                </div>
                <button
                  onClick={() => setIsNewProjectOpen(false)}
                  disabled={isCreatingProject}
                  className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-xl transition-all text-zinc-400 disabled:opacity-40"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="p-7 space-y-6">
                <div className="space-y-2">
                  <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest ml-1">Project name</label>
                  <TextField
                    autoFocus
                    value={newProjectName}
                    onChange={(e) => setNewProjectName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') createProject();
                      if (e.key === 'Escape') setIsNewProjectOpen(false);
                    }}
                    placeholder="haiku_experiment"
                    className="font-bold"
                  />
                  <p className="text-[10px] text-zinc-500 leading-relaxed">
                    Names are converted to safe folder names. The project will be created under <span className="font-mono">projects/</span>.
                  </p>
                </div>

                <label className={cn("flex items-start gap-3 p-4 rounded-xl border cursor-pointer", theme === 'dark' ? "bg-zinc-900/50 border-zinc-800" : "bg-zinc-50 border-zinc-100")}>
                  <input
                    type="checkbox"
                    checked={newProjectSeedCurrent}
                    onChange={(e) => setNewProjectSeedCurrent(e.target.checked)}
                    className="mt-0.5"
                  />
                  <div>
                    <p className="text-xs font-bold text-[var(--foreground)]">Seed from current runtime</p>
                    <p className="text-[10px] text-zinc-500 leading-relaxed mt-1">
                      Copies the current tokenizer/config into the new project so it can be loaded immediately.
                    </p>
                  </div>
                </label>

                <div className={cn("rounded-xl border p-4 text-[10px] leading-relaxed", theme === 'dark' ? "bg-zinc-900/40 border-zinc-800 text-zinc-400" : "bg-zinc-50 border-zinc-100 text-zinc-500")}>
                  The app will create <span className="font-mono">checkpoints</span>, <span className="font-mono">logs</span>, <span className="font-mono">cache</span>, and dataset folders for corpus, SFT, and DPO.
                </div>
              </div>

              <div className="px-7 py-5 border-t border-[var(--border)] flex justify-end gap-3">
                <button
                  onClick={() => setIsNewProjectOpen(false)}
                  disabled={isCreatingProject}
                  className="px-5 py-2.5 rounded-xl text-[10px] font-bold text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-all uppercase tracking-wider disabled:opacity-40"
                >
                  Cancel
                </button>
                <button
                  onClick={createProject}
                  disabled={isCreatingProject || isTraining || !newProjectName.trim()}
                  className={cn(
                    "px-6 py-2.5 rounded-xl text-[10px] font-bold uppercase tracking-wider transition-all disabled:opacity-40 flex items-center gap-2",
                    theme === 'dark' ? "bg-white text-black hover:bg-zinc-100" : "bg-zinc-900 text-white hover:bg-zinc-800"
                  )}
                >
                  {isCreatingProject ? <RotateCcw className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
                  Create Project
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Settings Modal */}
      <AnimatePresence>
        {isSettingsOpen && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              exit={{ opacity: 0 }}
              onClick={() => setIsSettingsOpen(false)}
              className="absolute inset-0 bg-zinc-900/60 backdrop-blur-md"
            />
            <motion.div 
              initial={{ scale: 0.98, opacity: 0, y: 10 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.98, opacity: 0, y: 10 }}
              className={cn(
                "relative w-full max-w-5xl rounded-xl shadow-2xl overflow-hidden border transition-all duration-500",
                "bg-[var(--card)] border-[var(--border)]"
              )}
            >
              <div className="px-8 py-6 border-b border-[var(--border)] flex items-center justify-between bg-[var(--background)]/50 backdrop-blur-xl">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-xl bg-zinc-900 dark:bg-white flex items-center justify-center text-white dark:text-black shadow-lg">
                    <Settings className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-[var(--foreground)] tracking-tight">Settings</h3>
                    <div className="flex items-center gap-3 mt-0.5">
                       <span className={cn("px-2 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider", theme === 'dark' ? "bg-zinc-800 text-zinc-400" : "bg-zinc-100 text-zinc-500")}>Build v2.4.0</span>
                       <span className="text-[10px] font-semibold text-zinc-400 capitalize tracking-wide">Environment & hyperparameter tuning</span>
                    </div>
                  </div>
                </div>
                <button onClick={() => setIsSettingsOpen(false)} className="p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 rounded-xl transition-all text-zinc-400">
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="flex h-[70vh]">
                 {/* Modal Sidebar */}
                  <div className={cn("w-64 border-r border-[var(--border)] p-6 space-y-8", theme === 'dark' ? "bg-zinc-900/50" : "bg-zinc-50/30")}>
                    <div className="space-y-4">
                       <h4 className="text-[10px] font-bold text-zinc-400 dark:text-zinc-500 uppercase tracking-widest px-1 opacity-50">App Settings</h4>
                       <div className="space-y-2">
                           <button onClick={toggleTheme} className={cn(
                             "group w-full flex items-center justify-between p-3 rounded-xl transition-all text-[10px] font-bold tracking-wider border shadow-sm",
                             theme === 'dark' ? "bg-zinc-900/50 border-zinc-800 hover:bg-zinc-800 shadow-none" : "bg-white border-zinc-100 hover:bg-zinc-50"
                           )}>
                               <span className="text-zinc-500 flex items-center gap-3 capitalize">
                                  {theme === 'light' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
                                  {theme === 'light' ? 'Light' : 'Dark'} mode
                               </span>
                           </button>
                           <button onClick={() => setShowTooltips(!showTooltips)} className={cn(
                             "w-full flex items-center justify-between p-3 rounded-xl transition-all text-[10px] font-bold tracking-wider border shadow-sm",
                             theme === 'dark' ? "bg-zinc-900/50 border-zinc-800 hover:bg-zinc-800 shadow-none" : "bg-white border-zinc-100 hover:bg-zinc-50"
                           )}>
                               <span className="text-zinc-500 flex items-center gap-3 capitalize">
                                  <HelpCircle className="w-3.5 h-3.5" />
                                  UX hints
                               </span>
                               <div className={cn("w-8 h-4 rounded-full relative transition-all", showTooltips ? "bg-emerald-500" : "bg-zinc-200 dark:bg-zinc-800")}>
                                  <div className={cn("absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all shadow-sm", showTooltips ? "right-0.5" : "left-0.5")} />
                               </div>
                           </button>
                       </div>
                    </div>

                    <div className="space-y-4 pt-8 border-t border-[var(--border)]">
                       <h4 className="text-[10px] font-bold text-zinc-400 dark:text-zinc-500 uppercase tracking-widest px-1 opacity-50">Project Settings</h4>
                       <button 
                         onClick={detectHardware}
                         disabled={autoConfigPending}
                         className={cn(
                           "w-full py-4 rounded-xl text-[10px] font-bold uppercase tracking-widest transition-all flex items-center justify-center gap-2",
                           theme === 'dark' ? "bg-zinc-800 text-zinc-200 hover:bg-zinc-700 shadow-none" : "bg-zinc-900 text-white hover:bg-zinc-800 shadow-lg shadow-zinc-200/50"
                         )}
                       >
                          {autoConfigPending ? <RotateCcw className="w-3.5 h-3.5 animate-spin" /> : <Cpu className="w-3.5 h-3.5" />}
                          Re-scan Device
                       </button>
                    </div>
                 </div>

                 {/* Modal Scroll Area */}
                 <div className="flex-1 p-8 overflow-y-auto custom-scrollbar">
                    <div className="space-y-12 pb-20">
                       <section className="space-y-6">
                            <h4 className="text-xs font-bold text-[var(--foreground)] capitalize tracking-wider flex items-center gap-3">
                               <div className="w-1 h-4 bg-emerald-500 rounded-full" />
                               Global environment
                            </h4>
                            <div className={cn(
                              "p-6 rounded-xl space-y-6 border",
                              theme === 'dark' ? "bg-zinc-900/50 border-zinc-800" : "bg-zinc-50 border-zinc-100"
                            )}>
                               <div className="grid grid-cols-2 gap-8">
                                  <div className="space-y-1">
                                     <span className="text-[10px] font-bold text-zinc-400 capitalize tracking-wide">Inference device</span>
                                     <p className="text-xs font-semibold tabular-nums uppercase truncate">{settings?.device || 'Scanning...'}</p>
                                  </div>
                                  <div className="space-y-1">
                                     <span className="text-[10px] font-bold text-zinc-400 capitalize tracking-wide">Kernel version</span>
                                     <p className="text-xs font-semibold tabular-nums">2.4.0-stable</p>
                                  </div>
                               </div>
                            </div>
                       </section>

                       {hardwareInfo && (
                         <section className="space-y-6">
                            <h4 className="text-xs font-bold text-zinc-900 dark:text-white capitalize tracking-wider flex items-center gap-3">
                               <div className="w-1 h-4 bg-emerald-500 rounded-full" />
                               Hardware analysis result
                            </h4>
                            <div className="p-6 bg-zinc-900 rounded-xl text-white space-y-6">
                               <div className="grid grid-cols-2 gap-6">
                                  <div className="space-y-1">
                                     <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wide text-emerald-400">Accelerator</span>
                                     <p className="text-xs font-semibold">{hardwareInfo.hardware.name}</p>
                                  </div>
                                  <div className="space-y-1">
                                     <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wide text-emerald-400">Total VRAM</span>
                                     <p className="text-xs font-semibold">{hardwareInfo.hardware.vram_total.toFixed(1)} GB</p>
                                  </div>
                                  <div className="space-y-1">
                                     <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wide text-emerald-400">Compute Cap</span>
                                     <p className="text-xs font-semibold">{hardwareInfo.hardware.compute_capability}</p>
                                  </div>
                                  <div className="space-y-1">
                                     <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wide text-emerald-400">FP16 Support</span>
                                     <p className="text-xs font-semibold">{hardwareInfo.hardware.supports_bf16 ? 'Full Native ✓' : 'Emulated ⚠️'}</p>
                                  </div>
                               </div>
                               <div className="pt-6 border-t border-zinc-800 space-y-4">
                                  <div className="flex items-center justify-between">
                                     <div>
                                        <h5 className="text-[10px] font-bold uppercase tracking-wider text-zinc-400">Tier Recommendation</h5>
                                        <p className="text-base font-bold text-emerald-400 tracking-tight">{hardwareInfo.recommendation.tier}</p>
                                     </div>
                                     <button 
                                       onClick={applyAutoRecommendation}
                                       className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-white text-[10px] font-bold uppercase tracking-wider rounded-lg transition-all"
                                     >
                                        Apply Specs
                                     </button>
                                  </div>
                                 </div>
                              </div>
                           </section>
                        )}

                       <section className="space-y-6">
                          <h4 className="text-xs font-bold text-[var(--foreground)] capitalize tracking-wider flex items-center gap-3">
                             <div className="w-1 h-4 bg-zinc-900 dark:bg-white rounded-full" />
                             Project directories
                          </h4>
                          <div className="space-y-3">
                             <div className="grid grid-cols-1 gap-2">
                                {projects.map((p, i) => (
                                  <div key={i} className={cn(
                                    "p-4 rounded-xl border flex items-center justify-between group transition-all duration-200 gap-4",
                                    activeProject === p 
                                      ? (theme === 'dark' ? "bg-emerald-500/10 border-emerald-500/40 text-white" : "bg-emerald-50 border-emerald-200 text-zinc-900 shadow-sm") 
                                      : (theme === 'dark' ? "bg-zinc-950/40 border-zinc-800 text-zinc-200 hover:border-zinc-700" : "bg-white border-zinc-200 text-zinc-900 hover:border-zinc-300")
                                  )}>
                                     <div className="flex items-start gap-4 min-w-0">
                                        <div className={cn("w-2 h-2 rounded-full mt-1.5 shrink-0", activeProject === p ? "bg-emerald-500" : "bg-zinc-300 dark:bg-zinc-700")} />
                                        <div className="flex flex-col min-w-0">
                                           <span className="text-sm font-bold tracking-tight truncate">{p}</span>
                                           <span className={cn("text-[10px] font-mono break-all leading-relaxed", activeProject === p ? "text-emerald-700 dark:text-emerald-300" : "text-zinc-500 dark:text-zinc-400")}>projects/{p}</span>
                                        </div>
                                     </div>
                                     {activeProject !== p ? (
                                       <button onClick={() => loadProject(p)} className={cn("px-3 py-1.5 text-[9px] font-bold uppercase tracking-wider rounded-lg border transition-all shrink-0", theme === 'dark' ? "bg-zinc-900 border-zinc-700 text-zinc-200 hover:bg-zinc-800" : "bg-zinc-900 border-zinc-900 text-white hover:bg-black")}>
                                          MOUNT
                                       </button>
                                     ) : (
                                       <span className="px-3 py-1.5 text-[9px] font-black uppercase tracking-wider rounded-lg bg-emerald-500 text-white shrink-0">Active</span>
                                     )}
                                  </div>
                                ))}
                             </div>
                             <button
                               type="button"
                               onClick={() => openNewProjectDialog()}
                               disabled={isTraining}
                               className={cn(
                                 "w-full mt-4 px-4 py-4 rounded-xl text-[10px] font-bold uppercase tracking-widest outline-none transition-all flex items-center justify-center gap-3 border disabled:opacity-40",
                                 theme === 'dark' 
                                   ? "bg-zinc-900/40 border-zinc-800 hover:border-white text-white" 
                                   : "bg-white border-[#F1F1F1] hover:border-zinc-900 text-zinc-500"
                               )}
                             >
                               <Plus className="w-4 h-4" />
                               Initialize New Project
                             </button>
                          </div>
                       </section>

                       <section className="space-y-6">
                          <div className="flex items-center justify-between">
                             <h4 className="text-xs font-bold text-[var(--foreground)] capitalize tracking-wider flex items-center gap-3 text-left">
                                <div className="w-1 h-4 bg-rose-500 rounded-full" />
                                Model architecture
                             </h4>
                             <button 
                               onClick={() => setIsArchitectureLocked(!isArchitectureLocked)}
                               className={cn(
                                 "flex items-center gap-2 px-3 py-1.5 rounded-lg text-[9px] font-bold uppercase tracking-wider transition-all border",
                                 isArchitectureLocked 
                                  ? "bg-zinc-50 dark:bg-zinc-800 border-zinc-100 dark:border-zinc-800 text-zinc-400" 
                                  : "bg-rose-50 dark:bg-rose-950/30 border-rose-200 dark:border-rose-900 text-rose-600 dark:text-rose-400"
                               )}
                             >
                               {isArchitectureLocked ? <><Lock className="w-3 h-3" /> Locked</> : <><Unlock className="w-3 h-3" /> Editable</>}
                             </button>
                          </div>
                          <div className="grid grid-cols-2 gap-8 text-left">
                             <div className="space-y-4">
                                {[
                                   { label: "Transformer layers", val: settings?.model_layers || 0, key: 'model_layers' },
                                   { label: "Embedding dimension", val: settings?.model_dim || 0, key: 'model_dim' },
                                   { label: "Attention heads", val: settings?.model_heads || 0, key: 'model_heads' },
                                   { label: "Block size (Context)", val: settings?.block_size || 0, key: 'block_size' },
                                   { label: "Vocab size", val: settings?.vocab_size || 0, key: 'vocab_size' }
                                ].map((arch, i) => (
                                  <div key={i} className="space-y-1.5">
                                     <label className="text-[10px] font-bold text-zinc-400 dark:text-zinc-500 capitalize tracking-wider ml-1 opacity-70">{arch.label}</label>
                                     <NumberField
                                       value={arch.val}
                                       disabled={isArchitectureLocked}
                                       integer
                                       min={0}
                                       onChange={(value) => setSettings(prev => prev ? ({ ...prev, [arch.key as keyof AppSettings]: value }) : null)}
                                       className="py-2.5 text-xs font-bold"
                                       ariaLabel={arch.label}
                                     />
                                  </div>
                                ))}
                             </div>
                             <div className={cn(
                               "p-8 rounded-xl border space-y-6 flex flex-col justify-center",
                               theme === 'dark' 
                                 ? "bg-zinc-900/40 border-zinc-800" 
                                 : "bg-white border-[#F1F1F1]"
                             )}>
                                <div className="space-y-1">
                                   <span className="text-[10px] font-bold text-zinc-400 dark:text-zinc-500 capitalize tracking-widest text-left block">Parameter count</span>
                                   <p className="text-3xl font-bold text-[#9f9fa9] tracking-tight text-left">{(settings?.model_params || 0).toLocaleString()}</p>
                                </div>
                                <div className="flex items-center gap-2">
                                   <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                                   <span className="text-[9px] font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Inference Profile Loaded</span>
                                </div>
                                <p className="text-[10px] text-zinc-400 leading-relaxed font-medium text-left opacity-70">
                                   Modifying architecture requires weights initialization. 
                                   Structural changes reset learned gradients.
                                </p>
                             </div>
                          </div>
                       </section>
                    </div>
                 </div>
              </div>

               <div className="px-8 py-6 border-t border-[var(--border)] flex justify-end gap-4 bg-[var(--background)]/50 backdrop-blur-xl">
                  <button 
                    onClick={() => {
                      fetchSettings();
                      setIsSettingsOpen(false);
                    }}
                    className="px-6 py-2.5 rounded-xl text-[10px] font-bold text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-all uppercase tracking-wider"
                  >
                    Cancel
                  </button>
                  <button 
                    onClick={async () => {
                      try {
                        const payload = {
                          ...settings,
                          theme,
                          show_tooltips: showTooltips,
                          active_project: activeProject,
                          projects
                        };
                        await axios.post('/api/settings', payload);
                        setSettings(payload as AppSettings);
                        setIsSettingsOpen(false);
                      } catch (e) {}
                    }}
                    className="px-8 py-2.5 bg-zinc-900 dark:bg-white text-white dark:text-black rounded-xl text-[10px] font-bold uppercase tracking-wider shadow-lg shadow-zinc-200/50 dark:shadow-none hover:bg-zinc-800 dark:hover:bg-zinc-100 transition-all font-bold"
                  >
                    Apply Config
                  </button>
               </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
      </div>
    </ThemeContext.Provider>
  );
}
