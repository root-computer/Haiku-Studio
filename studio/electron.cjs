const { app, BrowserWindow, dialog, Menu, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');
const fs = require('fs');

const studioDir = __dirname;
const port = Number(process.env.HAIKU_STUDIO_PORT || 3000);
const isWin = process.platform === 'win32';
const externalBackend = process.env.HAIKU_STUDIO_EXTERNAL_BACKEND === '1' || process.env.HAIKU_STUDIO_EXTERNAL_BACKEND === 'true';
const logDir = path.join(studioDir, 'logs');
const mainLogPath = path.join(logDir, 'electron-main.log');
const backendLogPath = path.join(logDir, 'backend.log');
const rendererLogPath = path.join(logDir, 'renderer.log');
const distIndexPath = path.join(studioDir, 'dist', 'index.html');
const preloadPath = path.join(studioDir, 'preload.cjs');
const forceDev = process.env.HAIKU_STUDIO_DEV === '1' || process.env.HAIKU_STUDIO_DEV === 'true';
const backendMode = forceDev || !fs.existsSync(distIndexPath) ? 'development' : 'production';

let serverProcess = null;
let mainWindow = null;
let shuttingDown = false;
let shutdownPosted = false;

function ensureLogDir() {
  try { fs.mkdirSync(logDir, { recursive: true }); } catch {}
}

function appendLog(filePath, line) {
  try {
    ensureLogDir();
    fs.appendFileSync(filePath, line, 'utf8');
  } catch (err) {
    try { console.error(`[Haiku Studio] log write failed for ${filePath}:`, err); } catch {}
  }
}

function appendMain(line) { appendLog(mainLogPath, line); }
function appendBackend(line) { appendLog(backendLogPath, line); }
function appendRenderer(line) { appendLog(rendererLogPath, line); }

ensureLogDir();
try {
  fs.writeFileSync(
    mainLogPath,
    `[Haiku Studio] Electron main launch ${new Date().toISOString()}\n` +
      `[Haiku Studio] studioDir=${studioDir}\n` +
      `[Haiku Studio] port=${port}\n` +
      `[Haiku Studio] externalBackend=${externalBackend}\n` +
      `[Haiku Studio] electron=${process.versions.electron || 'unknown'} node=${process.versions.node}\n`,
    'utf8',
  );
} catch {}

process.on('uncaughtException', (err) => {
  appendMain(`[uncaughtException] ${err && err.stack ? err.stack : String(err)}\n`);
  tryShowFatalError('Haiku Studio crashed', err && err.message ? err.message : String(err));
});

process.on('unhandledRejection', (reason) => {
  appendMain(`[unhandledRejection] ${reason && reason.stack ? reason.stack : String(reason)}\n`);
});

try {
  app.disableHardwareAcceleration();
  app.commandLine.appendSwitch('disable-gpu');
  app.commandLine.appendSwitch('disable-software-rasterizer');
} catch (err) {
  appendMain(`[gpu-disable-warning] ${err && err.stack ? err.stack : String(err)}\n`);
}

function tryShowFatalError(title, message) {
  const fullMessage = `${message}\n\nLogs:\n${mainLogPath}\n${backendLogPath}\n${rendererLogPath}`;
  appendMain(`[fatal] ${title}: ${message}\n`);
  try {
    if (app && app.isReady()) dialog.showErrorBox(title, fullMessage);
    else console.error(fullMessage);
  } catch {
    try { console.error(fullMessage); } catch {}
  }
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function diagnosticHtml(title, detail) {
  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(title)}</title>
  <style>
    body { margin: 0; background: #09090b; color: #e4e4e7; font-family: Segoe UI, Arial, sans-serif; }
    .wrap { max-width: 900px; margin: 12vh auto; padding: 32px; border: 1px solid #27272a; border-radius: 18px; background: #111114; }
    h1 { margin: 0 0 12px; font-size: 24px; }
    p { color: #a1a1aa; line-height: 1.55; }
    code, pre { background: #18181b; border: 1px solid #27272a; border-radius: 10px; padding: 10px; display: block; white-space: pre-wrap; color: #f4f4f5; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>${escapeHtml(title)}</h1>
    <p>${escapeHtml(detail)}</p>
    <p>Check these logs:</p>
    <code>${escapeHtml(mainLogPath)}\n${escapeHtml(backendLogPath)}\n${escapeHtml(rendererLogPath)}</code>
  </div>
</body>
</html>`;
}

function waitForServer(url, attempts = 160) {
  appendMain(`[waitForServer] ${url}\n`);
  return new Promise((resolve, reject) => {
    let count = 0;
    const tick = () => {
      count += 1;
      const req = http.get(url, (res) => {
        res.resume();
        if (res.statusCode && res.statusCode >= 200 && res.statusCode < 500) {
          appendMain(`[waitForServer] ready ${url} status=${res.statusCode} attempt=${count}\n`);
          resolve();
        } else if (count >= attempts) {
          reject(new Error(`Haiku Studio server returned HTTP ${res.statusCode} for ${url}.`));
        } else {
          setTimeout(tick, 500);
        }
      });
      req.on('error', (err) => {
        if (count === 1 || count % 20 === 0) appendMain(`[waitForServer] not ready ${url} attempt=${count}: ${err.message}\n`);
        if (count >= attempts) reject(new Error(`Haiku Studio server did not start at ${url}.`));
        else setTimeout(tick, 500);
      });
      req.setTimeout(1000, () => req.destroy(new Error('timeout')));
    };
    tick();
  });
}

function getBackendCommand() {
  if (isWin) {
    const comspec = process.env.ComSpec || 'C:\\Windows\\System32\\cmd.exe';
    return { command: comspec, args: ['/d', '/c', 'npm', 'run', 'server'] };
  }
  return { command: 'npm', args: ['run', 'server'] };
}

function startBackendIfNeeded() {
  if (externalBackend) {
    appendMain('[backend] using externally launched backend from launch_haiku_studio.bat\n');
    return;
  }

  fs.writeFileSync(
    backendLogPath,
    `[Haiku Studio] Backend launch ${new Date().toISOString()}\n` +
      `[Haiku Studio] studioDir=${studioDir}\n` +
      `[Haiku Studio] port=${port}\n` +
      `[Haiku Studio] mode=${backendMode}\n` +
      `[Haiku Studio] dist=${fs.existsSync(distIndexPath) ? distIndexPath : 'missing; using Vite dev middleware'}\n`,
    'utf8',
  );
  fs.writeFileSync(rendererLogPath, `[Haiku Studio] Renderer launch ${new Date().toISOString()}\n`, 'utf8');

  const { command, args } = getBackendCommand();
  appendBackend(`[Haiku Studio] command=${command}\n[Haiku Studio] args=${args.join(' ')}\n`);

  serverProcess = spawn(command, args, {
    cwd: studioDir,
    env: {
      ...process.env,
      HAIKU_STUDIO_PORT: String(port),
      HAIKU_STUDIO_ALLOW_SHUTDOWN: '1',
      NODE_ENV: backendMode,
      FORCE_COLOR: '0',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
    shell: false,
  });

  serverProcess.stdout.on('data', (chunk) => appendBackend(chunk.toString()));
  serverProcess.stderr.on('data', (chunk) => appendBackend(chunk.toString()));
  serverProcess.on('error', (err) => appendBackend(`[Haiku Studio] backend process error: ${err && err.stack ? err.stack : String(err)}\n`));
  serverProcess.on('exit', (code, signal) => {
    appendBackend(`[Haiku Studio] backend exited code=${code} signal=${signal}\n`);
    serverProcess = null;
    if (!shuttingDown && mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(diagnosticHtml('Haiku Studio backend stopped', `The backend exited with code ${code}.`))}`).catch(() => {});
    }
  });
}


function registerIpcHandlers() {
  ipcMain.handle('haiku:pick-tokenizer-file', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Select tokenizer training text file',
      properties: ['openFile'],
      filters: [
        { name: 'Text data', extensions: ['txt', 'text', 'jsonl', 'md'] },
        { name: 'All files', extensions: ['*'] },
      ],
    });
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) return '';
    return result.filePaths[0];
  });

  ipcMain.handle('haiku:pick-corpus-folder', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Select tokenizer corpus folder',
      properties: ['openDirectory'],
    });
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) return '';
    return result.filePaths[0];
  });
}

async function createWindow() {
  appendMain('[createWindow] starting\n');
  try {
    Menu.setApplicationMenu(null);
    startBackendIfNeeded();
    await waitForServer(`http://127.0.0.1:${port}/api/health`);
    await waitForServer(`http://127.0.0.1:${port}/`);
  } catch (err) {
    appendMain(`[createWindow] backend failed: ${err && err.stack ? err.stack : String(err)}\n`);
    createDiagnosticWindow('Haiku Studio backend failed to start', err && err.message ? err.message : String(err));
    return;
  }

  mainWindow = new BrowserWindow({
    width: 1500,
    height: 950,
    minWidth: 1180,
    minHeight: 760,
    backgroundColor: '#09090b',
    title: 'Haiku Studio',
    autoHideMenuBar: true,
    menuBarVisible: false,
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      preload: preloadPath,
      devTools: true,
    },
  });

  mainWindow.removeMenu();

  mainWindow.webContents.on('console-message', (_event, level, message, line, sourceId) => appendRenderer(`[console:${level}] ${message} (${sourceId}:${line})\n`));
  mainWindow.webContents.on('did-fail-load', (_event, errorCode, errorDescription, validatedURL) => appendRenderer(`[did-fail-load] ${errorCode} ${errorDescription} ${validatedURL}\n`));
  mainWindow.webContents.on('render-process-gone', (_event, details) => appendRenderer(`[render-process-gone] ${JSON.stringify(details)}\n`));
  mainWindow.webContents.on('crashed', () => appendRenderer('[crashed] renderer crashed\n'));

  mainWindow.webContents.on('did-finish-load', async () => {
    appendRenderer(`[did-finish-load] ${mainWindow.webContents.getURL()}\n`);
    try {
      const mounted = await mainWindow.webContents.executeJavaScript(
        `new Promise(resolve => setTimeout(() => { const root = document.getElementById('root'); resolve(Boolean(root && root.children.length)); }, 1500))`,
      );
      appendRenderer(`[mount-check] root_has_children=${mounted}\n`);
      if (!mounted) {
        await mainWindow.webContents.executeJavaScript(
          `document.body.innerHTML = ${JSON.stringify(diagnosticHtml('Haiku Studio frontend did not mount', 'The backend is running, but React did not render anything into #root. Check studio/logs/renderer.log for the actual frontend error.'))}`,
        );
      }
    } catch (err) {
      appendRenderer(`[mount-check-error] ${err && err.stack ? err.stack : String(err)}\n`);
    }
  });

  mainWindow.once('ready-to-show', () => {
    appendMain('[window] ready-to-show\n');
    mainWindow.show();
  });

  try {
    appendMain(`[loadURL] http://127.0.0.1:${port}\n`);
    await mainWindow.loadURL(`http://127.0.0.1:${port}`);
  } catch (err) {
    appendRenderer(`[loadURL failed] ${err && err.stack ? err.stack : String(err)}\n`);
    await mainWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(diagnosticHtml('Haiku Studio failed to load', err && err.message ? err.message : String(err)))}`);
    mainWindow.show();
  }
}

function createDiagnosticWindow(title, detail) {
  appendMain(`[diagnosticWindow] ${title}: ${detail}\n`);
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 720,
    backgroundColor: '#09090b',
    title,
    autoHideMenuBar: true,
    menuBarVisible: false,
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: false },
  });
  mainWindow.removeMenu();
  mainWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(diagnosticHtml(title, detail))}`).catch((err) => appendMain(`[diagnostic load failed] ${err && err.stack ? err.stack : String(err)}\n`));
}

function requestBackendShutdown() {
  if (shutdownPosted) return;
  shutdownPosted = true;
  const req = http.request({ hostname: '127.0.0.1', port, path: '/api/shutdown', method: 'POST', timeout: 750 }, (res) => {
    res.resume();
    appendMain(`[shutdown] backend responded ${res.statusCode}\n`);
  });
  req.on('error', (err) => appendMain(`[shutdown] backend request failed: ${err.message}\n`));
  req.on('timeout', () => req.destroy(new Error('shutdown timeout')));
  req.end();
}

function stopBackend() {
  shuttingDown = true;
  appendMain('[stopBackend] requested\n');
  if (externalBackend) {
    requestBackendShutdown();
    return;
  }
  if (!serverProcess || !serverProcess.pid) return;
  if (isWin) {
    spawn(process.env.ComSpec || 'C:\\Windows\\System32\\cmd.exe', ['/d', '/s', '/c', `taskkill /pid ${serverProcess.pid} /T /F`], { stdio: 'ignore', windowsHide: true });
  } else {
    serverProcess.kill('SIGINT');
  }
}

app.whenReady().then(() => {
  registerIpcHandlers();
  return createWindow();
}).catch((err) => {
  appendMain(`[whenReady.catch] ${err && err.stack ? err.stack : String(err)}\n`);
  tryShowFatalError('Haiku Studio failed to launch', err && err.message ? err.message : String(err));
  app.quit();
});

app.on('before-quit', stopBackend);
app.on('window-all-closed', () => {
  stopBackend();
  app.quit();
});
