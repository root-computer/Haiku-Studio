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
  Info
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

// --- Context ---
const ThemeContext = React.createContext<{ theme: 'light' | 'dark', showTooltips: boolean }>({ theme: 'light', showTooltips: true });
const useTheme = () => React.useContext(ThemeContext);

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

export default function App() {
  const [activeTab, setActiveTab] = useState<'home' | 'chat' | 'pretrain' | 'sft' | 'dpo' | 'help' | 'hub'>('home');
  const [deployMenuOpen, setDeployMenuOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [logCursor, setLogCursor] = useState(0);
  const [metrics, setMetrics] = useState<Metrics>({ steps: [], train_loss: [], val_loss: [], meta: {} });
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [hardwareInfo, setHardwareInfo] = useState<any>(null);
  const [autoConfigPending, setAutoConfigPending] = useState(false);
  const [projects, setProjects] = useState<string[]>(['haiku_studio']);
  const [activeProject, setActiveProject] = useState('haiku_studio');

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

  // Poll logs and metrics
  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const res = await axios.get(`/api/logs?since=${logCursor}`);
        if (res.data.lines?.length > 0) {
          setLogs(prev => [...prev, ...res.data.lines.map((l: any) => l[1])].slice(-1000));
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
    }, 2000);

    return () => clearInterval(interval);
  }, [logCursor]);

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

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    // Ideally sync with backend too
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (logBoxRef.current) {
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
    }
  }, [logs]);

  const createProject = async () => {
    if (isTraining) return;
    const raw = window.prompt('New project name:', 'haiku_experiment');
    const name = (raw || '').trim();
    if (!name) return;
    try {
      const res = await axios.post('/api/projects/create', { name, seed_current: true });
      setProjects(res.data.projects || []);
      setLogs(prev => [...prev, `[project] ${res.data.status}: ${res.data.project}`].slice(-1000));
      await loadProject(name);
    } catch (e: any) {
      alert(e?.response?.data?.error || 'Failed to create project.');
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
      setLogs(prev => [...prev, `[project] Loaded ${name}. Project tokenizer/config are now staged into data/.`].slice(-1000));
      fetchSettings();
    } catch (e: any) {
      alert(e?.response?.data?.error || 'Failed to load project.');
    }
  };

  const syncProjectToData = async () => {
    try {
      setIsTerminalOpen(true);
      const res = await axios.post('/api/projects/sync-to-data');
      setLogs(prev => [...prev, `[project] ${res.data.status}`].slice(-1000));
      fetchSettings();
    } catch (e: any) {
      alert(e?.response?.data?.error || 'Failed to sync project into data/.');
    }
  };

  const saveRuntimeToProject = async () => {
    try {
      setIsTerminalOpen(true);
      const res = await axios.post('/api/projects/save-runtime');
      setLogs(prev => [...prev, `[project] ${res.data.status}`].slice(-1000));
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
      setLogs(prev => [...prev, `[tokenizer] ${res.data.status}. Saving to data/tokenizer.json and project tokenizer copy.`].slice(-1000));
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
      setLogs(prev => [...prev, `[tokenizer] ${res.data.status}: data/tokenizer.json restored from bundled prebuilt tokenizer.`].slice(-1000));
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
            <span className="text-[10px] font-semibold text-zinc-400 capitalize tracking-wider mt-1 opacity-70">h2 engine · optional UI</span>
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
                 onClick={createProject}
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
                          setLogs(prev => [...prev, `[SYSTEM] h2 trainers save checkpoints and metrics into the active project folder.`, `[SYSTEM] Use Hugging Face Push to export saved artifacts.`]);
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
                      <p className="text-[10px] text-zinc-500 mt-2 truncate">{settings?.project_dir}</p>
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
                    <button onClick={createProject} disabled={isTraining} className={cn("px-4 py-2 rounded-xl text-[10px] font-bold uppercase tracking-wider border disabled:opacity-40", theme === 'dark' ? "bg-white border-white text-black hover:bg-zinc-200" : "bg-zinc-900 border-zinc-900 text-white hover:bg-black")}>Create Project</button>
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
                             <input 
                                type="text" 
                                value={corpusDir}
                                onChange={(e) => setCorpusDir(e.target.value)}
                                placeholder="e.g. datasets/my_corpus" 
                                className={cn(
                                  "w-full border rounded-xl px-4 py-3 text-sm font-mono outline-none transition-all",
                                  theme === 'dark' 
                                    ? "bg-zinc-900 border-zinc-800 text-zinc-300 focus:border-zinc-700 focus:bg-zinc-900/80 shadow-none" 
                                    : "bg-zinc-50 border-zinc-200 text-zinc-600 focus:border-zinc-900 focus:bg-white"
                                )} 
                             />
                             <p className="text-[10px] font-medium text-zinc-400 italic">The engine will scan all subdirectories for .txt files recursively.</p>
                          </div>

                          <div className="grid grid-cols-3 gap-6">
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Training Epochs</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Epochs:</strong> The number of full passes through the training data. The system will automatically calculate total steps based on your corpus size.
                                </div>
                                <input 
                                   type="number" 
                                   value={pretrainEpochs}
                                   onChange={(e) => setPretrainEpochs(Number(e.target.value))}
                                   className={cn(
                                     "w-full rounded-xl px-4 py-3 text-sm font-mono border transition-all outline-none",
                                     theme === 'dark' 
                                       ? "bg-zinc-900 border-zinc-800 text-zinc-200 focus:border-zinc-700" 
                                       : "bg-zinc-50 border-zinc-200 text-zinc-900 focus:bg-white focus:border-zinc-900"
                                   )} 
                                />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Base LR</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Learning Rate:</strong> The step size of the optimizer. 3e-4 is the industry standard for stable pretraining of smaller models.
                                </div>
                                <input 
                                  type="number" 
                                  step="0.0001"
                                  value={pretrainLR} 
                                  onChange={(e) => setPretrainLR(Number(e.target.value))} 
                                  className={cn(
                                   "w-full rounded-xl px-4 py-3 text-sm font-mono border transition-all outline-none",
                                   theme === 'dark' 
                                      ? "bg-zinc-900 border-zinc-800 text-zinc-200 focus:border-zinc-700" 
                                      : "bg-zinc-50 border-zinc-200 text-zinc-900 focus:bg-white focus:border-zinc-900"
                                 )} />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Val Split %</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Val Split:</strong> Percentage of data reserved for evaluation. Higher splits provide better metrics but reduce training data.
                                </div>
                                <input 
                                  type="number" 
                                  value={pretrainValSplit} 
                                  onChange={(e) => setPretrainValSplit(Number(e.target.value))} 
                                  className={cn(
                                   "w-full rounded-xl px-4 py-3 text-sm font-mono border transition-all outline-none",
                                   theme === 'dark' 
                                      ? "bg-zinc-900 border-zinc-800 text-zinc-200 focus:border-zinc-700" 
                                      : "bg-zinc-50 border-zinc-200 text-zinc-900 focus:bg-white focus:border-zinc-900"
                                 )} />
                             </div>
                          </div>

                          <div className="grid grid-cols-3 gap-6">
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Micro Batch</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Micro Batch:</strong> Sequences processed per GPU step.
                                </div>
                                <input 
                                  type="number" 
                                  value={pretrainBatchSize} 
                                  onChange={(e) => setPretrainBatchSize(Number(e.target.value))} 
                                  className={cn(
                                   "w-full rounded-xl px-4 py-3 text-sm font-mono border transition-all outline-none",
                                   theme === 'dark' 
                                      ? "bg-zinc-900 border-zinc-800 text-zinc-200 focus:border-zinc-700" 
                                      : "bg-zinc-50 border-zinc-200 text-zinc-900 focus:bg-white focus:border-zinc-900"
                                 )} />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Grad Accum</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Gradient Accumulation:</strong> Simulates larger batch sizes by summing gradients over multiple micro-batches.
                                </div>
                                <input 
                                  type="number" 
                                  value={pretrainGradAccum} 
                                  onChange={(e) => setPretrainGradAccum(Number(e.target.value))} 
                                  className={cn(
                                   "w-full rounded-xl px-4 py-3 text-sm font-mono border transition-all outline-none",
                                   theme === 'dark' 
                                      ? "bg-zinc-900 border-zinc-800 text-zinc-200 focus:border-zinc-700" 
                                      : "bg-zinc-50 border-zinc-200 text-zinc-900 focus:bg-white focus:border-zinc-900"
                                 )} />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Warmup Steps</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Warmup:</strong> Number of steps to linearly increase LR from zero to base value. Prevents divergent gradients at start.
                                </div>
                                <input 
                                  type="number" 
                                  value={pretrainWarmupSteps} 
                                  onChange={(e) => setPretrainWarmupSteps(Number(e.target.value))} 
                                  className={cn(
                                   "w-full rounded-xl px-4 py-3 text-sm font-mono border transition-all outline-none",
                                   theme === 'dark' 
                                      ? "bg-zinc-900 border-zinc-800 text-zinc-200 focus:border-zinc-700" 
                                      : "bg-zinc-50 border-zinc-200 text-zinc-900 focus:bg-white focus:border-zinc-900"
                                 )} />
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
                             <button 
                                onClick={startPretraining}
                                className={cn(
                                   "w-full py-4 font-bold rounded-xl transition-all flex items-center justify-center gap-2",
                                   theme === 'dark' ? "bg-white text-black hover:bg-zinc-100" : "bg-black text-white hover:bg-zinc-800 shadow-xl shadow-zinc-300"
                                 )}
                             >
                                <Zap className="w-4 h-4" /> Initialize Pretraining Run
                             </button>
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
                                <input 
                                  type="text"
                                  placeholder={tokSourceType === 'file' ? 'E:\\AGENT 3\\training\\FineFactualNews.txt' : 'corpus'}
                                  value={tokPath} 
                                  onChange={(e) => setTokPath(e.target.value)} 
                                  className={cn(
                                    "flex-1 border rounded-xl px-4 py-3 text-sm font-mono outline-none transition-all",
                                    theme === 'dark' ? "bg-zinc-900 border-zinc-800" : "bg-zinc-50 border-zinc-200"
                                  )} 
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
                                 <input 
                                   type="number" 
                                   min={256}
                                   value={tokVocabSize} 
                                   onChange={(e) => setTokVocabSize(Number(e.target.value))} 
                                   className={cn(
                                     "w-full border rounded-xl px-4 py-3 text-sm font-mono outline-none transition-all",
                                     theme === 'dark' ? "bg-zinc-900 border-zinc-800" : "bg-zinc-50 border-zinc-200"
                                   )} 
                                 />
                              </div>
                              <div className="space-y-2 group relative">
                                 <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest leading-none">Min Frequency</label>
                                 <input 
                                   type="number"
                                   min={1}
                                   value={tokMinFreq} 
                                   onChange={(e) => setTokMinFreq(Number(e.target.value))} 
                                   className={cn(
                                     "w-full border rounded-xl px-4 py-3 text-sm font-mono outline-none transition-all",
                                     theme === 'dark' ? "bg-zinc-900 border-zinc-800" : "bg-zinc-50 border-zinc-200"
                                   )} 
                                 />
                              </div>
                              <div className="space-y-2 group relative">
                                 <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest leading-none">RAM Guard MB</label>
                                 <input 
                                   type="number"
                                   min={0}
                                   value={tokMaxInputMb} 
                                   onChange={(e) => setTokMaxInputMb(Number(e.target.value))} 
                                   className={cn(
                                     "w-full border rounded-xl px-4 py-3 text-sm font-mono outline-none transition-all",
                                     theme === 'dark' ? "bg-zinc-900 border-zinc-800" : "bg-zinc-50 border-zinc-200"
                                   )} 
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

                 <div className="grid grid-cols-3 gap-8">
                    <Card title="SFT Configuration" subtitle="Refine dialogue parameters" className="col-span-2">
                       <div className="space-y-6">
                          <div className="space-y-2 group relative">
                             <label className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">SFT Training Data</label>
                             <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-full p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                <strong>SFT Dataset:</strong> Provide a <code>.jsonl</code> file containing multi-turn dialogues formatted with <code>user:</code> and <code>bot:</code> segments.
                             </div>
                             <input type="text" value={sftDataPath} onChange={(e) => setSftDataPath(e.target.value)} placeholder={settings?.project_sft_dir || "projects/haiku_studio/datasets/sft"} className={cn("w-full border rounded-xl px-4 py-3 text-sm font-mono outline-none", theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-300" : "bg-zinc-50 border-zinc-200 text-zinc-600")} />
                          </div>

                          <div className="grid grid-cols-3 gap-6">
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Epochs</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Epochs:</strong> Number of times the optimizer passes through the entire SFT dataset. Usually 1-3 is sufficient.
                                </div>
                                <input type="number" value={sftEpochs} onChange={(e) => setSftEpochs(Number(e.target.value))} className={cn("w-full border rounded-xl px-4 py-3 text-sm font-mono", theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-300" : "bg-zinc-50 border-zinc-200 text-zinc-900")} />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Batch Size</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Batch Size:</strong> Number of conversation pairs processed simultaneously during one update step.
                                </div>
                                <input type="number" value={sftBatchSize} onChange={(e) => setSftBatchSize(Number(e.target.value))} className={cn("w-full border rounded-xl px-4 py-3 text-sm font-mono", theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-300" : "bg-zinc-50 border-zinc-200 text-zinc-900")} />
                             </div>
                             <div className="space-y-2 group relative">
                                <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Learning Rate</label>
                                <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-50 w-48 p-3 bg-zinc-900 text-white text-[10px] rounded-xl shadow-2xl border border-zinc-800">
                                   <strong>Fine-Tuning LR:</strong> Often much lower than pretraining LR (e.g., 5e-5) to prevent catastrophic forgetting of base knowledge.
                                </div>
                                <input type="number" step="0.000001" value={sftLR} onChange={(e) => setSftLR(Number(e.target.value))} className={cn("w-full border rounded-xl px-4 py-3 text-sm font-mono", theme === 'dark' ? "bg-zinc-900 border-zinc-800 text-zinc-300" : "bg-zinc-50 border-zinc-200 text-zinc-900")} />
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

                          <button onClick={startSFT} disabled={isTraining} className="w-full py-4 bg-black hover:bg-zinc-800 disabled:bg-zinc-300 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-all shadow-xl shadow-zinc-300">
                             Start SFT Training
                          </button>
                       </div>
                    </Card>
                    <div className="space-y-6">
                       <Card title="SFT Insights">
                          <div className="space-y-4">
                             <div className="flex justify-between items-center">
                                <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">Samples</span>
                                <span className="text-xs font-bold font-mono">1,240</span>
                             </div>
                             <div className="flex justify-between items-center">
                                <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">Avg Seq Len</span>
                                <span className="text-xs font-bold font-mono">512</span>
                             </div>
                             <div className="w-full bg-zinc-100 rounded-full h-1.5 overflow-hidden">
                                <div className="bg-emerald-500 h-full w-[65%]" />
                             </div>
                             <p className="text-[10px] text-zinc-500 font-medium">Dataset split: 80/20 train/val</p>
                          </div>
                       </Card>
                    </div>
                 </div>
              </motion.div>
            )}

            {activeTab === 'dpo' && (
              <motion.div key="dpo" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8 max-w-6xl mx-auto">
                 <TutorialBox 
                    title="Phase 3: Preference Alignment" 
                    description="Direct Preference Optimization trains from prompt/chosen/rejected pairs using a frozen reference checkpoint from the SFT model."
                    icon={Zap}
                    colorClass="bg-zinc-900 text-white border-zinc-900"
                 />

                 <div className="grid grid-cols-3 gap-8">
                    <Card title="DPO Training" subtitle="Preference pairs, reference model, and alignment checkpoint" className="col-span-2">
                       <div className="space-y-6">
                          {!settings?.dpo_ready ? (
                            <div className="text-center py-12 space-y-4 border border-rose-100 rounded-xl bg-rose-50/60">
                               <h4 className="font-bold text-rose-900 text-sm uppercase tracking-tight">dpo.py Missing</h4>
                               <p className="text-[11px] text-rose-600 max-w-[360px] mx-auto leading-relaxed">The UI expects a root-level dpo.py trainer. Restore dpo.py before launching alignment.</p>
                            </div>
                          ) : (
                            <div className="space-y-8">
                               <div className="grid grid-cols-2 gap-6">
                                  <div className="space-y-3 text-left col-span-2">
                                     <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider ml-1">Preference Dataset Path</label>
                                     <input value={dpoDataPath} onChange={(e) => setDpoDataPath(e.target.value)} className="w-full bg-zinc-50 border border-zinc-200 rounded-xl px-4 py-3 text-sm font-mono focus:border-zinc-900 focus:bg-white transition-all outline-none" />
                                     <p className="text-[9px] text-zinc-400 font-medium px-1">Folder or file containing JSONL/text prompt/chosen/rejected pairs. Chat Lab feedback writes into the active project's DPO dataset folder by default.</p>
                                  </div>
                                  <div className="space-y-3 text-left">
                                     <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider ml-1">Beta</label>
                                     <input type="number" value={dpoBeta} onChange={(e) => setDpoBeta(Number(e.target.value))} step={0.01} className="w-full bg-zinc-50 border border-zinc-200 rounded-xl px-4 py-3 text-sm font-mono focus:border-zinc-900 focus:bg-white transition-all outline-none" />
                                     <p className="text-[9px] text-zinc-400 font-medium px-1">DPO preference strength against the frozen reference model.</p>
                                  </div>
                                  <div className="space-y-3 text-left">
                                     <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider ml-1">Alignment LR</label>
                                     <input type="number" value={dpoLR} onChange={(e) => setDpoLR(Number(e.target.value))} step={0.000001} className="w-full bg-zinc-50 border border-zinc-200 rounded-xl px-4 py-3 text-sm font-mono focus:border-zinc-900 focus:bg-white transition-all outline-none" />
                                     <p className="text-[9px] text-zinc-400 font-medium px-1">Small learning rate for preference optimization.</p>
                                  </div>
                                  <div className="space-y-3 text-left">
                                     <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider ml-1">Epochs</label>
                                     <input type="number" value={dpoEpochs} onChange={(e) => setDpoEpochs(Number(e.target.value))} min={1} className="w-full bg-zinc-50 border border-zinc-200 rounded-xl px-4 py-3 text-sm font-mono focus:border-zinc-900 focus:bg-white transition-all outline-none" />
                                  </div>
                                  <div className="space-y-3 text-left">
                                     <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider ml-1">Batch Size</label>
                                     <input type="number" value={dpoBatchSize} onChange={(e) => setDpoBatchSize(Number(e.target.value))} min={1} className="w-full bg-zinc-50 border border-zinc-200 rounded-xl px-4 py-3 text-sm font-mono focus:border-zinc-900 focus:bg-white transition-all outline-none" />
                                  </div>
                               </div>
                               <button onClick={startDPO} disabled={isTraining || !settings?.dpo_ready} className="w-full py-4 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-300 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-all shadow-xl shadow-emerald-100 flex items-center justify-center gap-2">
                                  <Zap className="w-4 h-4" /> Start DPO Training
                               </button>
                               <p className="text-[10px] text-zinc-500 leading-relaxed">
                                  If no reference checkpoint exists, dpo.py creates one from the configured SFT policy checkpoint and freezes it. Output saves to {settings?.dpo_checkpoint || 'projects/haiku_studio/checkpoints/model.dpo.pt'}.
                               </p>
                            </div>
                          )}
                       </div>
                    </Card>

                    <div className="space-y-8">
                       <Card title="DPO Status">
                          <div className="space-y-8">
                             <div className="flex flex-col gap-1 text-left">
                                <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Preference Pairs</span>
                                <div className="text-3xl font-bold text-zinc-900 font-mono tracking-tight">{settings?.dpo_buffer ?? 0} <span className="text-xs text-zinc-300 font-medium tracking-normal">Pairs</span></div>
                             </div>
                             <div className="flex flex-col gap-1 text-left">
                                <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider">DPO Eval Step</span>
                                <div className="text-3xl font-bold text-zinc-900 font-mono tracking-tight">{settings?.dpo_global_step ?? 0}</div>
                             </div>
                             <div className="pt-6 border-t border-zinc-100 flex gap-2">
                                <button onClick={flushDPOFeedback} className="flex-1 py-2.5 bg-zinc-50 hover:bg-zinc-100 border border-zinc-200 text-zinc-600 font-bold rounded-lg text-[9px] uppercase tracking-wider transition-all">Flush Feedback</button>
                                <button onClick={fetchSettings} className="flex-1 py-2.5 bg-zinc-900 hover:bg-black text-white font-bold rounded-lg text-[9px] uppercase tracking-wider transition-all shadow-lg shadow-zinc-200">Refresh</button>
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
              <motion.div key="help" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-12 max-w-4xl mx-auto pb-32">
                    <div className="space-y-3 mb-16 pt-8 text-center sm:text-left">
                       <h1 className={cn("text-2xl font-black tracking-tighter", theme === 'dark' ? "text-white" : "text-zinc-900")}>Technical Field Manual</h1>
                       <div className="flex items-center justify-center sm:justify-start gap-3">
                          <span className={cn("px-2.5 py-1 text-[10px] font-black uppercase tracking-widest rounded-lg shadow-sm", theme === 'dark' ? "bg-white text-black" : "bg-black text-white")}>v2.4.0 — STABLE</span>
                          <p className="text-zinc-400 font-bold text-xs uppercase tracking-widest opacity-80">Official Engineering Reference & Documentation</p>
                       </div>
                    </div>

                <div className="space-y-24">
                   {/* Section 1: Philosophy */}
                   <section className="space-y-8">
                      <div className="flex items-center gap-5">
                         <div className={cn("w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg border", theme === 'dark' ? "bg-zinc-800 border-zinc-700 text-white" : "bg-white border-zinc-200 text-zinc-900")}>
                            <Zap className="w-7 h-7" />
                         </div>
                         <div>
                            <h3 className="text-lg font-black text-zinc-900 tracking-tight">1. Neural Philosophy & Core Architecture</h3>
                            <p className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em] mt-0.5 opacity-70">Knowledge Pressure & Transformer Synthesis</p>
                         </div>
                      </div>
                      <div className="prose prose-zinc prose-sm max-w-none text-zinc-600 leading-relaxed space-y-6">
                         <p className="text-sm font-medium leading-loose italic border-l-4 border-zinc-200 pl-6 py-2 bg-zinc-50/50 rounded-r-xl">
                            "Haiku Studio is not a consumer chat wrapper. It is a high-performance Neural Weight Synthesis Environment designed for the complete lifecycle of LLM creation—from raw knowledge acquisition to human alignment."
                         </p>
                         <div className="grid grid-cols-1 sm:grid-cols-2 gap-8 mt-8 text-zinc-500">
                           <div className="space-y-3">
                              <h5 className="text-[11px] font-black uppercase tracking-widest text-zinc-900">Static Logic vs. Statistical Pressure</h5>
                              <p className="text-[12px] font-medium leading-relaxed">
                                Unlike traditional software, neural networks are not "coded"; they are <strong>pressured</strong> into specific configurations through iterative optimization. Haiku Studio treats the transformer as a blank statistical slate.
                              </p>
                           </div>
                           <div className="space-y-3">
                              <h5 className="text-[11px] font-black uppercase tracking-widest text-zinc-900">Architecture: RoPE-GQA Core</h5>
                              <ul className="text-[12px] font-medium space-y-2 list-none">
                                 <li className="flex gap-2">
                                    <span className="text-black font-black">RoPE:</span> 
                                    <span>Rotary Positional Embeddings for infinite context extrapolation by rotating token vectors.</span>
                                </li>
                                 <li className="flex gap-2">
                                    <span className="text-black font-black">GQA:</span> 
                                    <span>Grouped-Query Attention for high-speed inference of MQA with the understanding of MHA.</span>
                                </li>
                              </ul>
                           </div>
                         </div>
                      </div>
                   </section>

                   {/* Section 2: Topology */}
                   <section className="space-y-10">
                      <div className="flex items-center gap-5">
                         <div className={cn("w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg border", theme === 'dark' ? "bg-zinc-800 border-zinc-700 text-white" : "bg-white border-zinc-200 text-zinc-900")}>
                            <Globe className="w-7 h-7" />
                         </div>
                         <div>
                            <h3 className="text-lg font-black text-zinc-900 tracking-tight">2. Interface Topology & Navigation</h3>
                            <p className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em] mt-0.5 opacity-70">Navigation Protocol & Engine Monitoring</p>
                         </div>
                      </div>
                      
                      <div className="space-y-12">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                           <div className="space-y-5">
                              <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-zinc-400 border-b border-zinc-100 pb-2">Primary Sidebar Operations</h4>
                              <div className="space-y-3">
                                 {[
                                    { label: "Home Studio", desc: "Mission control. Central dashboard for project lifecycle management." },
                                    { label: "Chat Lab", desc: "Interactive verification. Manually assess model reasoning and instruction follow." },
                                    { label: "Pretraining", desc: "Knowledge acquisition. Foundational training from raw text corpora." },
                                    { label: "Instruct SFT", desc: "Behavioral shaping. Fine-tuning the base model into an assistant role." },
                                    { label: "Alignment DPO", desc: "Human alignment. Preference-based optimization using comparative pairs." }
                                 ].map((item, id) => (
                                    <div key={id} className="flex gap-4 items-start p-4 bg-zinc-50 border border-zinc-100 rounded-2xl hover:bg-white transition-all hover:shadow-md hover:-translate-y-0.5 pointer-events-none">
                                       <div className="w-2 h-2 rounded-full bg-zinc-300 mt-1.5" />
                                       <div className="space-y-1">
                                          <p className="text-[11px] font-black text-zinc-900 uppercase tracking-widest">{item.label}</p>
                                          <p className="text-[11px] text-zinc-500 font-medium italic">{item.desc}</p>
                                       </div>
                                    </div>
                                 ))}
                              </div>
                           </div>
                           <div className="space-y-5">
                              <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-zinc-400 border-b border-zinc-100 pb-2">Global & Terminal Controls</h4>
                              <div className="space-y-6">
                                 <div className="p-6 bg-zinc-900 rounded-3xl text-white space-y-4 shadow-xl">
                                    <div className="flex justify-between items-center border-b border-zinc-800 pb-3">
                                       <span className="text-[11px] font-black uppercase tracking-widest text-zinc-400">System Kernel</span>
                                       <div className="flex gap-2">
                                          <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                                          <span className="w-2 h-2 rounded-full bg-amber-500" />
                                          <span className="w-2 h-2 rounded-full bg-emerald-500" />
                                       </div>
                                    </div>
                                    <p className="text-[11px] text-zinc-400 font-medium leading-relaxed">
                                       The terminal provides real-time telemetry. In <strong>Diagnostic Mode</strong>, monitor raw loss values, gradient norms, and hardware status. Check here for OOM (Out of Memory) or NaN Gradient warnings.
                                    </p>
                                    <div className="pt-4 grid grid-cols-2 gap-4">
                                       <div className="space-y-1">
                                          <span className="text-[10px] font-black text-zinc-500 uppercase">Input Selector</span>
                                          <p className="text-[10px] font-bold text-zinc-300">Switch project containers.</p>
                                       </div>
                                       <div className="space-y-1">
                                          <span className="text-[10px] font-black text-zinc-500 uppercase">Deploy Pipeline</span>
                                          <p className="text-[10px] font-bold text-zinc-300">Serialize weight tensors.</p>
                                       </div>
                                    </div>
                                 </div>
                              </div>
                           </div>
                        </div>
                      </div>
                   </section>

                   {/* Section 3: Chat Lab */}
                   <section className="space-y-10">
                      <div className="flex items-center gap-5">
                         <div className={cn("w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg border", theme === 'dark' ? "bg-zinc-800 border-zinc-700 text-white" : "bg-white border-zinc-200 text-zinc-900")}>
                            <MessagesSquare className="w-7 h-7" />
                         </div>
                         <div>
                            <h3 className="text-lg font-black text-zinc-900 tracking-tight">3. Qualitative Assessment: Chat Lab</h3>
                            <p className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em] mt-0.5 opacity-70">Dialog Verification & Preference collection</p>
                         </div>
                      </div>
                      <div className="space-y-8">
                         <p className="text-[13px] text-zinc-500 font-medium leading-relaxed max-w-3xl">
                            Loss curves are important, but dialogue is the ultimate test. The Chat Lab allows you to verify reasoning fluidity and protocol adherence in real-time.
                         </p>
                         <div className="p-10 bg-zinc-950 rounded-[2.5rem] text-white shadow-2xl relative overflow-hidden group">
                           <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity">
                              <Terminal className="w-40 h-40 rotate-12" />
                           </div>
                           <h4 className="text-[12px] font-black uppercase tracking-[0.3em] mb-8 text-zinc-500 border-b border-zinc-900 pb-4">Sampling Knob Engineering</h4>
                           <div className="grid grid-cols-1 md:grid-cols-2 gap-12 relative z-10">
                              <div className="space-y-4">
                                 <span className="text-[11px] font-black text-emerald-500 uppercase tracking-widest flex items-center gap-2">
                                    Temperature <div className="h-0.5 flex-1 bg-zinc-900" />
                                 </span>
                                 <p className="text-[12px] text-zinc-400 leading-relaxed font-medium">
                                    Adjusts token probability entropy. 
                                    <span className="block mt-4 text-[10px] font-black text-zinc-600 italic">Hint: 0.7 for logic, 1.2+ for creative brainstorming.</span>
                                 </p>
                              </div>
                              <div className="space-y-4">
                                 <span className="text-[11px] font-black text-blue-500 uppercase tracking-widest flex items-center gap-2">
                                    Presence Penalty <div className="h-0.5 flex-1 bg-zinc-900" />
                                 </span>
                                 <p className="text-[12px] text-zinc-400 leading-relaxed font-medium">
                                    Prevents linguistic loops by penalizing tokens that have already appeared in the current context windown.
                                 </p>
                              </div>
                           </div>
                           <div className="mt-12 pt-8 border-t border-zinc-900 flex flex-col sm:flex-row items-center gap-6">
                              <div className="flex gap-2">
                                 <div className="w-10 h-10 rounded-full bg-zinc-900 p-2.5 text-emerald-500 flex items-center justify-center border border-zinc-800"><ThumbsUp className="w-5 h-5" /></div>
                                 <div className="w-10 h-10 rounded-full bg-zinc-900 p-2.5 text-rose-500 flex items-center justify-center border border-zinc-800"><ThumbsDown className="w-5 h-5" /></div>
                              </div>
                              <div>
                                 <h5 className="text-[11px] font-black uppercase tracking-widest text-zinc-200">The Preference Loop</h5>
                                 <p className="text-[11px] text-zinc-500 font-medium">Feedback flagged here directly populates your <strong>Phase 3 DPO Buffer</strong> for reinforcement training.</p>
                              </div>
                           </div>
                         </div>
                      </div>
                   </section>

                   {/* Section 4: Pretraining */}
                   <section className="space-y-12">
                      <div className="flex items-center gap-5">
                         <div className={cn("w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg border", theme === 'dark' ? "bg-zinc-800 border-zinc-700 text-white" : "bg-white border-zinc-200 text-zinc-900")}>
                            <Database className="w-7 h-7" />
                         </div>
                         <div>
                            <h3 className="text-lg font-black text-zinc-900 tracking-tight">4. Phase 1: Foundational Pretraining</h3>
                            <p className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em] mt-0.5 opacity-70">Infinite Data Ingestion & knowledge Acquisition</p>
                         </div>
                      </div>

                       <div className="space-y-10">
                          <div className="grid grid-cols-1 md:grid-cols-5 gap-10">
                             <div className="md:col-span-3 space-y-6">
                                <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-zinc-900 border-l-4 border-black pl-4">Recursive Multi-File Streamer</h4>
                                <p className="text-xs text-zinc-500 font-medium leading-relaxed">
                                   Point the engine to any local directory. The pretraining kernel scans for raw <code>.txt</code> data, creates an <strong>Inter-Doc Buffer</strong>, and packs tokens into full context windows (e.g., 2048) to maximize hardware FLOPs utilization.
                                </p>
                                <div className="space-y-2 pt-4">
                                   <div className="flex items-center gap-3">
                                      <div className="w-1.5 h-1.5 rounded-full bg-zinc-900" />
                                      <p className="text-[11px] font-bold text-zinc-600 uppercase tracking-widest">LR Strategy: Linear Warmup + Cosine Decay</p>
                                   </div>
                                   <p className="text-[11px] text-zinc-400 font-medium pl-4">Allows weights to stabilize during initial steps before aggressive optimization triggers.</p>
                                </div>
                             </div>
                             <div className="md:col-span-2 space-y-6">
                                <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-zinc-400">Success Indicators</h4>
                                <div className="p-6 bg-zinc-50 border border-zinc-100 rounded-3xl space-y-4">
                                   <div className="space-y-1">
                                      <p className="text-[10px] font-black text-black uppercase">Training Loss</p>
                                      <p className="text-[10px] text-zinc-500 font-medium">Logarithmic descent target. Spikes usually indicate LR is too high.</p>
                                   </div>
                                   <div className="space-y-1 border-t border-zinc-200 pt-4">
                                      <p className="text-[10px] font-black text-black uppercase">Perplexity (PPL)</p>
                                      <p className="text-[10px] text-zinc-500 font-medium">Measurement of next-token surprise. Targets for base stability are typically &lt; 15.0.</p>
                                   </div>
                                </div>
                             </div>
                          </div>
                       </div>
                   </section>

                   {/* Section 5: SFT */}
                   <section className="space-y-10">
                      <div className="flex items-center gap-5">
                         <div className={cn("w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg border", theme === 'dark' ? "bg-zinc-800 border-zinc-700 text-white" : "bg-white border-zinc-200 text-zinc-900")}>
                            <Brain className="w-7 h-7" />
                         </div>
                         <div>
                            <h3 className="text-lg font-black text-zinc-900 tracking-tight">5. Phase 2: Instruct Tuning (SFT)</h3>
                            <p className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em] mt-0.5 opacity-70">Supervised Fine-Tuning & Protocol Formatting</p>
                         </div>
                      </div>
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                         <div className="space-y-6">
                            <p className="text-sm font-medium text-zinc-500 leading-relaxed">
                               SFT transforms a neutral knowledge base into a specific assistant persona. Protocol requires structured <strong>JSONL</strong> multi-turn dialogue data.
                            </p>
                            <div className="p-6 bg-amber-50 border border-amber-100 rounded-2xl space-y-3">
                               <h5 className="text-[10px] font-black uppercase tracking-widest text-amber-900 flex items-center gap-2"><Lock className="w-3 h-3" /> Gradient Masking</h5>
                               <p className="text-[11px] text-amber-800 font-medium leading-relaxed">
                                  Haiku Studio identifies <code>user:</code> and <code>bot:</code> tokens. It applies 0.0 weight to user prompts, ensuring the model only learns the assistant's predictive behavior.
                               </p>
                            </div>
                         </div>
                         <div className="space-y-6">
                            <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-zinc-400 border-b border-zinc-100 pb-2">The Golden Data Format</h4>
                            <div className="bg-zinc-900 rounded-2xl p-5 font-mono text-[10px] text-zinc-400 border border-zinc-800 shadow-inner overflow-x-auto whitespace-pre leading-loose">
                               <span className="text-zinc-600">// JSONL Entry format</span>{"\n"}
                               {"{"}"text": "<span className="text-emerald-500">user:</span> How do I synthesize salts?\n<span className="text-blue-500">bot:</span> Salt synthesis involves..."{"}"}
                            </div>
                            <p className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em] text-center pt-2 italic">Optimal Pass: 1-3 Epochs on High-Quality Corpora.</p>
                         </div>
                      </div>
                   </section>

                   {/* Section 6: DPO */}
                   <section className="space-y-12">
                      <div className="flex items-center gap-4">
                         <div className={cn("w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg border", theme === 'dark' ? "bg-zinc-800 border-zinc-700 text-white" : "bg-white border-zinc-200 text-zinc-900")}>
                            <Zap className="w-7 h-7" />
                         </div>
                         <div>
                            <h3 className="text-lg font-black text-zinc-900 tracking-tight">6. Phase 3: Alignment (DPO)</h3>
                            <p className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em] mt-0.5 opacity-70">Direct Preference Optimization & Human Alignment</p>
                         </div>
                      </div>
                      <div className="p-12 border-2 border-dashed border-zinc-200 rounded-[3rem] space-y-10 bg-zinc-50/30">
                         <div className="space-y-4 max-w-2xl">
                            <h4 className="text-[11px] font-black uppercase tracking-[0.2em] text-zinc-900">The Reference Model Invariant</h4>
                            <p className="text-xs text-zinc-500 leading-relaxed font-medium">
                               DPO requires a persistent <strong>Reference Model</strong>—a static snapshot of your model BEFORE DPO starts. The math compares log-probabilities between the two to ensure preference without losing general reasoning intelligence.
                            </p>
                         </div>
                         <div className="grid grid-cols-1 sm:grid-cols-2 gap-16 border-t border-zinc-200 pt-10">
                            <div className="space-y-4">
                               <span className="text-[11px] font-black text-rose-500 uppercase tracking-widest flex items-center gap-2">Beta (KL Constraint) <Info className="w-3 h-3" /></span>
                               <p className="text-[12px] text-zinc-400 leading-relaxed font-medium italic">
                                  <strong>High Beta (0.5+):</strong> Stable, subtle alignment.{"\n"}
                                  <strong>Low Beta (0.1-0.3):</strong> Drastic shifts, higher risk of "Model Collapse."
                               </p>
                            </div>
                            <div className="space-y-4">
                               <span className="text-[11px] font-black text-zinc-900 uppercase tracking-widest">The Comparative Gate</span>
                               <p className="text-[12px] text-zinc-400 leading-relaxed font-medium italic">
                                  Updates only trigger when the Preference Buffer reaches your batch ceiling. Ensure your Chat Lab evaluations are diverse.
                               </p>
                            </div>
                         </div>
                      </div>
                   </section>

                   {/* Section 7: HF */}
                   <section className="space-y-10">
                      <div className="flex items-center gap-5">
                         <div className={cn("w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg border", theme === 'dark' ? "bg-zinc-800 border-zinc-700 text-white" : "bg-white border-zinc-200 text-zinc-900")}>
                            <HuggingFaceIcon className="w-8 h-8" />
                         </div>
                         <div>
                            <h3 className="text-lg font-black text-zinc-900 tracking-tight">7. External Integration: Hugging Face Ecosystem</h3>
                            <p className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em] mt-0.5 opacity-70">Direct Hub Ingestion & Weight Publishing</p>
                         </div>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                        <div className="p-8 bg-zinc-900 rounded-[2rem] text-white shadow-2xl space-y-6">
                           <h4 className="text-[11px] font-black uppercase tracking-widest text-emerald-400">A. Hub Sync (Datasets)</h4>
                           <p className="text-[11px] text-zinc-400 font-medium leading-relaxed">
                              Search trillions of tokens directly from the dataset hub. This bypasses manual sharding. Haiku Studio streams relevant shards into your project containers automatically.
                           </p>
                           <div className="flex gap-2">
                              <span className="px-2 py-0.5 bg-zinc-800 text-[9px] font-bold text-zinc-500 rounded uppercase">Streaming Enabled</span>
                              <span className="px-2 py-0.5 bg-zinc-800 text-[9px] font-bold text-zinc-500 rounded uppercase">Auto-Auth</span>
                           </div>
                        </div>
                        <div className="p-8 bg-zinc-900 rounded-[2rem] text-white shadow-2xl space-y-6">
                           <h4 className="text-[11px] font-black uppercase tracking-widest text-blue-400">B. Global Push (Weights)</h4>
                           <p className="text-[11px] text-zinc-400 font-medium leading-relaxed">
                              Export your finalized weights via Secure Multipart Upload. System auto-generates <code>config.json</code> and weights in <code>.safetensors</code> format for instant cloud deployment.
                           </p>
                           <div className="flex items-center gap-2 text-[9px] font-black text-zinc-500 uppercase tracking-widest">
                              <Lock className="w-3 h-3" /> Requires Write Token (hf_...)
                           </div>
                        </div>
                      </div>
                   </section>

                   {/* Section 8: Diagnostics */}
                   <section className="space-y-10">
                      <div className="flex items-center gap-5">
                         <div className={cn("w-14 h-14 rounded-2xl flex items-center justify-center shadow-lg border", theme === 'dark' ? "bg-zinc-800 border-zinc-700 text-white" : "bg-white border-zinc-200 text-zinc-900")}>
                            <HelpCircle className="w-7 h-7" />
                         </div>
                         <div>
                            <h3 className="text-lg font-black text-zinc-900 tracking-tight">8. System Diagnostics & Troubleshooting</h3>
                            <p className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em] mt-0.5 opacity-70">Engineering Resolution Protocols</p>
                         </div>
                      </div>
                      <div className="space-y-6">
                         {[
                            { q: "Case 701: The Kernel Stutter", s: "Terminal/UI freezes for 5-15 seconds.", r: "Checkpoint Serialization. Writing weights (~2GB+) to disk is a blocking I/O operation. Increase interval if on HDD." },
                            { q: "Case 804: Attention Collapse", s: "Model repeats same phrase or exits prematurely.", r: "Gradient Explosion. Decrease Learning Rate immediately. This occurs when 1 layer's weight magnitude drowns the rest." },
                            { q: "Case 902: VRAM Fragmentation", s: "OOM errors despite sufficient device memory.", r: "Use the 'Re-scan Device' toggle in Settings. This triggers garbage collection and defragments the VRAM pool." }
                         ].map((item, i) => (
                            <div key={i} className="p-10 bg-white border border-zinc-100 rounded-[2rem] space-y-6 shadow-sm border-b-4 border-b-zinc-200">
                               <div className="flex justify-between items-center group">
                                 <h4 className="text-sm font-black text-zinc-900 uppercase tracking-widest">{item.q}</h4>
                                 <div className="w-10 h-10 rounded-full bg-zinc-50 border border-zinc-100 flex items-center justify-center group-hover:bg-black group-hover:text-white transition-all text-zinc-400"><BookOpen className="w-4 h-4" /></div>
                               </div>
                               <div className="grid grid-cols-1 sm:grid-cols-2 gap-10">
                                  <div className="space-y-2">
                                     <span className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em]">Observation Symptoms</span>
                                     <p className="text-[12px] text-zinc-500 font-bold italic leading-relaxed">{item.s}</p>
                                  </div>
                                  <div className="space-y-2 border-l border-zinc-100 pl-8">
                                     <span className="text-[10px] font-black text-zinc-400 uppercase tracking-[0.2em]">Engineering Resolution</span>
                                     <p className="text-[12px] text-zinc-600 font-medium leading-relaxed italic">{item.r}</p>
                                  </div>
                               </div>
                            </div>
                         ))}
                      </div>
                   </section>

                   <footer className="pt-24 border-t border-zinc-100 text-center space-y-4">
                      <div className="flex items-center justify-center gap-6 opacity-30">
                         <div className="h-px w-20 bg-zinc-300" />
                         <Globe className="w-5 h-5 text-zinc-900" />
                         <div className="h-px w-20 bg-zinc-300" />
                      </div>
                      <p className="text-[11px] font-black text-zinc-300 uppercase tracking-[0.5em]">
                         © 2026 ROOTCOMPUTER DEVELOPMENT — PRODUCTION REFERENCE
                      </p>
                   </footer>
                </div>
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
              <div className="flex items-center gap-3">
                <Terminal className="w-4 h-4 text-zinc-400" />
                <span className="text-[10px] font-black uppercase tracking-[0.2em] text-zinc-500">System Kernel Output</span>
              </div>
              <div className="flex items-center gap-4">
                 {isTerminalOpen && (
                   <button onClick={(e) => { e.stopPropagation(); setLogs([]); }} className="text-[10px] font-bold text-zinc-400 hover:text-rose-500 transition-colors uppercase tracking-widest flex items-center gap-1.5 px-3 py-1 hover:bg-rose-50 rounded-lg">
                      <Trash2 className="w-3 h-3" /> Clear Buffer
                   </button>
                 )}
                 <div className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-full border",
                    theme === 'dark' ? "bg-zinc-800 border-zinc-700" : "bg-zinc-50 border-zinc-100"
                 )}>
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
                    <span className={cn("text-[10px] font-black uppercase tracking-widest", theme === 'dark' ? "text-zinc-400" : "text-zinc-600")}>Kernel Ready</span>
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
                {logs.map((log, i) => (
                  <div key={i} className={cn(
                    "flex gap-4 py-1 last:border-0 hover:bg-zinc-50/50 dark:hover:bg-zinc-800/30 transition-colors",
                    theme === 'dark' ? "border-zinc-800" : "border-zinc-50 border-b"
                  )}>
                    <span className="text-zinc-400 dark:text-zinc-600 shrink-0 font-bold tabular-nums">{(i+1).toString().padStart(4, '0')}</span>
                    <span className="text-zinc-400 shrink-0 font-bold text-[9px] uppercase tracking-tighter mt-0.5 opacity-60">{new Date().toLocaleTimeString()}</span>
                    <span className={cn(
                      "break-all pr-4 font-medium",
                      log.includes('ERROR') ? "text-rose-500 font-bold" : 
                      log.includes('loss') ? (theme === 'dark' ? "text-zinc-200 font-bold" : "text-zinc-900 font-bold") : 
                      (theme === 'dark' ? "text-zinc-400" : "text-zinc-500")
                    )}>
                      {log}
                    </span>
                  </div>
                ))}
             </div>
           )}
        </div>

      </div>

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
                                    "p-4 rounded-xl border flex items-center justify-between group transition-all duration-200",
                                    activeProject === p 
                                      ? (theme === 'dark' ? "bg-white border-white text-black" : "bg-zinc-900 border-zinc-900 text-white shadow-xl shadow-zinc-200/50") 
                                      : "bg-white dark:bg-zinc-900/40 border-zinc-100 dark:border-zinc-800 text-zinc-900 dark:text-white hover:border-zinc-300"
                                  )}>
                                     <div className="flex items-center gap-4">
                                        <div className={cn("w-2 h-2 rounded-full", activeProject === p ? "bg-emerald-500" : "bg-zinc-200 dark:bg-zinc-700")} />
                                        <div className="flex flex-col">
                                           <span className="text-sm font-bold tracking-tight">{p}</span>
                                           <span className={cn("text-[9px] font-semibold opacity-50 font-mono", activeProject === p ? "text-zinc-400" : "text-zinc-500")}>/PROJECTS/{p}</span>
                                        </div>
                                     </div>
                                     {activeProject !== p && (
                                       <button onClick={() => setActiveProject(p)} className="px-3 py-1.5 bg-zinc-900 dark:bg-white text-white dark:text-black text-[9px] font-bold uppercase tracking-wider rounded-lg opacity-0 group-hover:opacity-100 transition-all">
                                          MOUNT
                                       </button>
                                     )}
                                  </div>
                                ))}
                             </div>
                             <div className="relative group mt-4">
                                <Plus className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-400" />
                                <input 
                                  type="text" 
                                  placeholder="INITIALIZE NEW PROJECT..." 
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter') {
                                      const val = (e.target as any).value;
                                      if (val) {
                                        setProjects([...projects, val]);
                                        (e.target as any).value = '';
                                      }
                                    }
                                  }}
                                  className={cn(
                                    "w-full pl-12 pr-4 py-4 rounded-xl text-[10px] font-bold uppercase tracking-widest outline-none transition-all",
                                    theme === 'dark' 
                                      ? "bg-zinc-900/40 border-zinc-800 focus:border-white text-white" 
                                      : "bg-white border-[#F1F1F1] focus:border-zinc-900 text-zinc-400 placeholder:text-zinc-300"
                                  )}
                                />
                             </div>
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
                                     <input 
                                       type="number" 
                                       value={arch.val}
                                       disabled={isArchitectureLocked}
                                       onChange={(e) => setSettings(prev => prev ? ({ ...prev, [arch.key as keyof AppSettings]: Number(e.target.value) }) : null)}
                                       className={cn(
                                         "w-full rounded-xl px-4 py-2.5 text-xs font-mono font-bold border transition-all outline-none",
                                         isArchitectureLocked 
                                           ? "bg-zinc-50 dark:bg-zinc-900/30 border-transparent text-zinc-400 opacity-60 cursor-not-allowed" 
                                           : "bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 focus:border-zinc-900 dark:focus:border-white text-zinc-900 dark:text-white"
                                       )} 
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
