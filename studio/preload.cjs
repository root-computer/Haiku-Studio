const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('haikuStudio', {
  pickTokenizerFile: () => ipcRenderer.invoke('haiku:pick-tokenizer-file'),
  pickCorpusFolder: () => ipcRenderer.invoke('haiku:pick-corpus-folder'),
});
