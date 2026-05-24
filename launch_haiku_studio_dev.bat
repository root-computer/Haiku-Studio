@echo off
setlocal EnableExtensions
cd /d "%~dp0studio"
if not exist logs mkdir logs
> logs\launcher-dev.log echo [Haiku Studio] Dev launcher started %DATE% %TIME%

where npm >nul 2>nul
if errorlevel 1 (
  echo [Haiku Studio] npm was not found. Install Node.js LTS, then run this launcher again.
  >> logs\launcher-dev.log echo [Haiku Studio] ERROR: npm not found
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
  call npm install --registry=https://registry.npmjs.org/ --no-audit --no-fund >> logs\launcher-dev.log 2>&1
  if errorlevel 1 (
    echo [Haiku Studio] npm install failed. See studio\logs\launcher-dev.log
    echo [Haiku Studio] If this folder was reused from a failed install, close Haiku Studio and run repair_studio_deps.bat once.
    pause
    exit /b 1
  )
)

set HAIKU_STUDIO_PORT=39177
set HAIKU_STUDIO_DEV=1
set HAIKU_STUDIO_EXTERNAL_BACKEND=1
set HAIKU_STUDIO_ALLOW_SHUTDOWN=1
set NODE_ENV=development
set FORCE_COLOR=0

> logs\backend.log echo [Haiku Studio] External DEV backend launch %DATE% %TIME%
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -WindowStyle Hidden -FilePath 'cmd.exe' -WorkingDirectory '%CD%' -ArgumentList '/d','/s','/c','npm run server ^>^> logs\backend.log 2^>^&1'" >> logs\launcher-dev.log 2>&1

call npm run studio -- --enable-logging --log-file="%CD%\logs\electron-chromium.log" >> logs\electron-cli.log 2>&1
set EXITCODE=%ERRORLEVEL%
>> logs\launcher-dev.log echo [Haiku Studio] Electron exited with code %EXITCODE%
if not "%EXITCODE%"=="0" pause
exit /b %EXITCODE%
