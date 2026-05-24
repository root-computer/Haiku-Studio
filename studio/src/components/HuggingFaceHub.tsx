import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Download, 
  ExternalLink, 
  FileText, 
  Loader2,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { motion } from '../lib/motion-shim';
import { cn } from '../lib/utils';
import axios from 'axios';
import { HuggingFaceIcon } from './HuggingFaceIcon';

interface Dataset {
  id: string;
  name?: string;
  author?: string;
  lastModified?: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
}

export default function HuggingFaceHub({ theme }: { theme: 'light' | 'dark' }) {
  const [activeSubTab, setActiveSubTab] = useState<'explorer' | 'publisher'>('explorer');
  const [search, setSearch] = useState('text');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [downloadStatus, setDownloadStatus] = useState<{id: string, status: string} | null>(null);

  // Publisher state
  const [repoName, setRepoName] = useState('');
  const [hfToken, setHfToken] = useState('');
  const [isPublishing, setIsPublishing] = useState(false);
  const [publishStatus, setPublishStatus] = useState<string | null>(null);

  const fetchDatasets = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await axios.get(`/api/hub/datasets?search=${search}`);
      setDatasets(res.data);
    } catch (err) {
      setDatasets([]);
      setError("Failed to fetch datasets from the hub.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    setError(null);
    setDownloadStatus(null);
    setPublishStatus(null);
  }, [activeSubTab]);

  useEffect(() => {
    if (activeSubTab === 'explorer') {
      const timer = setTimeout(() => {
        fetchDatasets();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [search, activeSubTab]);

  const handleDownload = async (datasetId: string) => {
    setDownloadingId(datasetId);
    setDownloadStatus(null);
    try {
      const res = await axios.post('/api/hub/download-dataset', {
        datasetId,
        filename: 'train.txt' 
      });
      setDownloadStatus({ id: datasetId, status: 'Success! Saved to corpus.' });
    } catch (err) {
      setError(`Could not find a standard 'train.txt' in ${datasetId}.`);
    } finally {
      setDownloadingId(null);
    }
  };

  const handlePublish = async () => {
    if (!repoName || !hfToken) {
      setError("Repository name and Hugging Face Token are required.");
      return;
    }
    setIsPublishing(true);
    setError(null);
    setPublishStatus(null);
    try {
      const res = await axios.post('/api/hub/export-model', {
        repoName,
        hfToken
      });
      setPublishStatus("Model successfully pushed to Hugging Face Hub!");
    } catch (err: any) {
      setError(err.response?.data?.error || "Failed to export model to the hub.");
    } finally {
      setIsPublishing(false);
    }
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-20">
      <div className="flex items-center justify-between">
        <div className="flex flex-col gap-2">
          <h1 className={cn("text-2xl font-bold tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>
            Hugging Face Sync
          </h1>
          <p className="text-sm text-zinc-500 font-medium leading-relaxed">
            Interface directly with the world's largest repository of models and datasets.
          </p>
        </div>
        
        <div className={cn(
          "flex p-1 rounded-xl border shrink-0",
          theme === 'dark' ? "bg-zinc-900 border-zinc-800" : "bg-white border-zinc-200 shadow-sm"
        )}>
          <button 
            onClick={() => setActiveSubTab('explorer')}
            className={cn(
              "px-4 py-2 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all",
              activeSubTab === 'explorer' 
                ? (theme === 'dark' ? "bg-white text-black" : "bg-black text-white") 
                : "text-zinc-400 hover:text-zinc-600"
            )}
          >
            Dataset Explorer
          </button>
          <button 
            onClick={() => setActiveSubTab('publisher')}
            className={cn(
              "px-4 py-2 text-[10px] font-black uppercase tracking-widest rounded-lg transition-all",
              activeSubTab === 'publisher' 
                ? (theme === 'dark' ? "bg-white text-black" : "bg-black text-white") 
                : "text-zinc-400 hover:text-zinc-600"
            )}
          >
            Model Publisher
          </button>
        </div>
      </div>

      {activeSubTab === 'explorer' ? (
        <>
          <div className="relative group">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-zinc-400 group-focus-within:text-zinc-600 transition-colors" />
            <input 
              type="text" 
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search for datasets (e.g. 'shakespeare', 'code', 'philosophy')..."
              className={cn(
                "w-full pl-12 pr-6 py-4 rounded-2xl border text-sm font-medium transition-all outline-none",
                theme === 'dark' 
                  ? "bg-zinc-900 border-zinc-800 text-white focus:border-zinc-600" 
                  : "bg-white border-zinc-200 text-zinc-900 focus:border-zinc-400 focus:shadow-lg focus:shadow-zinc-100"
              )}
            />
          </div>

          {error && (
            <div className="p-4 bg-rose-50 border border-rose-100 rounded-xl flex items-center gap-3 text-rose-600 text-xs font-bold">
              <AlertCircle className="w-4 h-4" />
              {error}
            </div>
          )}

          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-20 gap-4">
              <Loader2 className="w-8 h-8 text-zinc-400 animate-spin" />
              <span className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Scanning the hub...</span>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {datasets.map((ds) => (
                <motion.div 
                  key={ds.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn(
                    "p-6 rounded-2xl border flex flex-col gap-4 group transition-all hover:shadow-xl",
                    theme === 'dark' 
                      ? "bg-zinc-900 border-zinc-800 hover:border-zinc-700" 
                      : "bg-white border-zinc-100 hover:border-zinc-300 shadow-sm shadow-zinc-100"
                  )}
                >
                  <div className="flex items-start justify-between">
                    <div className="w-10 h-10 rounded-xl bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center text-zinc-900 dark:text-white shrink-0 shadow-inner">
                      <FileText className="w-5 h-5" />
                    </div>
                    <button 
                      onClick={() => window.open(`https://huggingface.co/datasets/${ds.id}`, '_blank')}
                      className="p-2 text-zinc-400 hover:text-zinc-600 dark:hover:text-white transition-colors"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </button>
                  </div>

                  <div className="flex flex-col gap-1 min-w-0">
                    <h3 className={cn("text-sm font-bold truncate", theme === 'dark' ? "text-white" : "text-zinc-900")}>
                      {ds.id.split('/').pop()}
                    </h3>
                    <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-wider truncate">
                      {ds.id.split('/')[0]}
                    </span>
                  </div>

                  <div className="flex items-center gap-4 mt-2">
                    <div className="flex items-center gap-1.5 capitalize text-[10px] font-bold text-zinc-500">
                      <Download className="w-3 h-3" />
                      {ds.downloads?.toLocaleString() || 0}
                    </div>
                    <div className="flex items-center gap-1.5 capitalize text-[10px] font-bold text-zinc-500">
                      <HuggingFaceIcon className="w-3 h-3 text-emerald-500" />
                      {ds.id.includes('/') ? 'Community' : 'Official'}
                    </div>
                  </div>

                  <div className="mt-auto pt-4 flex flex-col gap-2">
                    {downloadStatus?.id === ds.id ? (
                      <div className="flex items-center gap-2 text-emerald-500 text-[10px] font-bold uppercase tracking-wider py-3 justify-center">
                        <CheckCircle2 className="w-4 h-4" />
                        {downloadStatus.status}
                      </div>
                    ) : (
                      <button 
                        disabled={downloadingId !== null}
                        onClick={() => handleDownload(ds.id)}
                        className={cn(
                          "w-full py-3 rounded-xl text-[10px] font-black uppercase tracking-widest flex items-center justify-center gap-2 transition-all",
                          theme === 'dark' 
                            ? "bg-white text-black hover:bg-zinc-100 disabled:opacity-50" 
                            : "bg-black text-white hover:bg-zinc-800 shadow-md shadow-zinc-200 disabled:bg-zinc-300"
                        )}
                      >
                        {downloadingId === ds.id ? (
                          <>
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                            Injesting...
                          </>
                        ) : (
                          <>
                            <Download className="w-3.5 h-3.5" />
                            Sync to Corpus
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          )}

          {datasets.length === 0 && !isLoading && (
            <div className="text-center py-20 border-2 border-dashed border-zinc-100 rounded-3xl">
              <HuggingFaceIcon className="w-12 h-12 text-zinc-200 mx-auto mb-4" />
              <p className="text-zinc-400 font-bold text-sm">No datasets found matching your criteria.</p>
            </div>
          )}
        </>
      ) : (
        <div className="max-w-2xl mx-auto py-12">
           <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className={cn(
              "p-10 rounded-3xl border shadow-2xl shadow-zinc-200/50 space-y-10",
              theme === 'dark' ? "bg-zinc-900 border-zinc-800 shadow-none" : "bg-white border-zinc-100"
            )}
           >
              <div className="text-center space-y-3">
                 <div className="w-16 h-16 rounded-2xl bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center text-zinc-900 dark:text-white mx-auto shadow-inner">
                    <HuggingFaceIcon className="w-8 h-8" />
                 </div>
                 <h2 className={cn("text-xl font-bold tracking-tight", theme === 'dark' ? "text-white" : "text-zinc-900")}>Publish Model Weights</h2>
                 <p className="text-xs text-zinc-500 font-medium px-4">Export your current project weights directly to a new or existing Hugging Face repository.</p>
              </div>

              <div className="space-y-6">
                 <div className="space-y-2">
                    <label className="text-[10px] font-black text-zinc-400 uppercase tracking-widest ml-1">Target Repository</label>
                    <input 
                      type="text" 
                      value={repoName}
                      onChange={(e) => setRepoName(e.target.value)}
                      placeholder="username/my-haiku-model"
                      className={cn(
                        "w-full px-4 py-4 rounded-xl border text-sm transition-all focus:ring-2",
                        theme === 'dark' ? "bg-zinc-800 border-zinc-700 text-white focus:ring-zinc-700" : "bg-zinc-50 border-zinc-200 text-zinc-900 focus:ring-zinc-200"
                      )}
                    />
                 </div>

                 <div className="space-y-2">
                    <label className="text-[10px] font-black text-zinc-400 uppercase tracking-widest ml-1">Hugging Face API Token</label>
                    <input 
                      type="password" 
                      value={hfToken}
                      onChange={(e) => setHfToken(e.target.value)}
                      placeholder="hf_..."
                      className={cn(
                        "w-full px-4 py-4 rounded-xl border text-sm transition-all focus:ring-2",
                        theme === 'dark' ? "bg-zinc-800 border-zinc-700 text-white focus:ring-zinc-700" : "bg-zinc-50 border-zinc-200 text-zinc-900 focus:ring-zinc-200"
                      )}
                    />
                    <p className="text-[9px] text-zinc-400 font-medium px-1 italic">We never persist your token. It is used only for this session's upload stream.</p>
                 </div>

                 {error && (
                    <div className="p-4 bg-rose-50 border border-rose-100 rounded-xl flex items-center gap-3 text-rose-600 text-[10px] font-bold uppercase transition-all animate-pulse">
                      <AlertCircle className="w-4 h-4" />
                      {error}
                    </div>
                 )}

                 {publishStatus && (
                    <div className="p-4 bg-emerald-50 border border-emerald-100 rounded-xl flex items-center gap-3 text-emerald-600 text-[10px] font-bold uppercase transition-all">
                      <CheckCircle2 className="w-4 h-4" />
                      {publishStatus}
                    </div>
                 )}

                 <button 
                  disabled={isPublishing}
                  onClick={handlePublish}
                  className={cn(
                    "w-full py-5 rounded-2xl text-xs font-black uppercase tracking-widest flex items-center justify-center gap-3 transition-all",
                    theme === 'dark' 
                      ? "bg-white text-black hover:bg-zinc-100 disabled:opacity-50" 
                      : "bg-black text-white hover:bg-zinc-800 shadow-xl shadow-zinc-200 disabled:bg-zinc-300"
                  )}
                 >
                    {isPublishing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Pushing Weights...
                      </>
                    ) : (
                      <>
                        <HuggingFaceIcon className="w-4 h-4" />
                        Initiate Global Push
                      </>
                    )}
                 </button>
              </div>
           </motion.div>
        </div>
      )}
    </div>
  );
}
