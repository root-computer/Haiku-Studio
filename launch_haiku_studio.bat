@echo off
setlocal EnableExtensions
cd /d "%~dp0studio"

if not exist logs mkdir logs
> logs\launcher.log echo [Haiku Studio] Launcher started %DATE% %TIME%
>> logs\launcher.log echo [Haiku Studio] studio=%CD%

where npm >nul 2>nul
if errorlevel 1 (
  echo [Haiku Studio] npm was not found. Install Node.js LTS, then run this launcher again.
  >> logs\launcher.log echo [Haiku Studio] ERROR: npm not found
  pause
  exit /b 1
)

if not exist .npmrc (
  > .npmrc echo registry=https://registry.npmjs.org/
  >> .npmrc echo fund=false
  >> .npmrc echo audit=false
)

set NEED_INSTALL=0
if not exist node_modules set NEED_INSTALL=1
if not exist node_modules\.bin\electron.cmd set NEED_INSTALL=1
if not exist node_modules\.bin\tsx.cmd set NEED_INSTALL=1
if not exist node_modules\.bin\vite.cmd set NEED_INSTALL=1

if "%NEED_INSTALL%"=="1" (
  echo [Haiku Studio] Installing or repairing UI dependencies from npmjs.org...
  >> logs\launcher.log echo [Haiku Studio] npm install --registry=https://registry.npmjs.org/ --no-audit --no-fund
  call npm install --registry=https://registry.npmjs.org/ --no-audit --no-fund >> logs\launcher.log 2>&1
  if errorlevel 1 (
    echo [Haiku Studio] npm install failed. See studio\logs\launcher.log
    echo [Haiku Studio] If this folder was reused from a failed install, close Haiku Studio and run repair_studio_deps.bat once.
    pause
    exit /b 1
  )
) else (
  >> logs\launcher.log echo [Haiku Studio] node_modules already present; skipping npm install
)

if not exist dist\index.html (
  echo [Haiku Studio] Building desktop UI bundle...
  >> logs\launcher.log echo [Haiku Studio] npm run build
  call npm run build >> logs\launcher.log 2>&1
  if errorlevel 1 (
    echo [Haiku Studio] UI build failed. See studio\logs\launcher.log
    pause
    exit /b 1
  )
)

set HAIKU_STUDIO_PORT=39177
set HAIKU_STUDIO_EXTERNAL_BACKEND=1
set HAIKU_STUDIO_ALLOW_SHUTDOWN=1
set NODE_ENV=production
set FORCE_COLOR=0

rem Clear stale backend log and launch backend outside Electron.
echo [Haiku Studio] Starting backend in hidden window...
>> logs\launcher.log echo [Haiku Studio] Starting backend: npm run server
> logs\backend.log echo [Haiku Studio] External backend launch %DATE% %TIME%
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -WindowStyle Hidden -FilePath 'cmd.exe' -WorkingDirectory '%CD%' -ArgumentList '/d','/s','/c','npm run server ^>^> logs\backend.log 2^>^&1'" >> logs\launcher.log 2>&1
if errorlevel 1 (
  echo [Haiku Studio] Backend process failed to launch. See studio\logs\launcher.log
  pause
  exit /b 1
)

echo [Haiku Studio] Opening desktop app...
>> logs\launcher.log echo [Haiku Studio] Starting Electron
call npm run studio -- --enable-logging --log-file="%CD%\logs\electron-chromium.log" >> logs\electron-cli.log 2>&1
set EXITCODE=%ERRORLEVEL%
>> logs\launcher.log echo [Haiku Studio] Electron exited with code %EXITCODE%

if not "%EXITCODE%"=="0" (
  echo [Haiku Studio] Electron exited with code %EXITCODE%.
  echo Logs are in studio\logs\launcher.log, electron-main.log, electron-cli.log, backend.log, and renderer.log
  pause
)
exit /b %EXITCODE%
