@echo off
setlocal EnableExtensions
cd /d "%~dp0studio"
if not exist logs mkdir logs
> logs\repair-deps.log echo [Haiku Studio] Dependency repair started %DATE% %TIME%

echo [Haiku Studio] This repairs a failed/partial npm install for the Studio UI.
echo [Haiku Studio] Close all Haiku Studio/Electron/Node windows before continuing.
echo.
choice /C YN /M "Continue with dependency repair"
if errorlevel 2 exit /b 1

where npm >nul 2>nul
if errorlevel 1 (
  echo [Haiku Studio] npm was not found. Install Node.js LTS first.
  >> logs\repair-deps.log echo [Haiku Studio] ERROR: npm not found
  pause
  exit /b 1
)

> .npmrc echo registry=https://registry.npmjs.org/
>> .npmrc echo fund=false
>> .npmrc echo audit=false

rem Remove only Studio dependency/build folders. This does not touch h2 checkpoints, corpora, configs, or Python files.
if exist node_modules (
  echo [Haiku Studio] Removing studio\node_modules...
  >> logs\repair-deps.log echo [Haiku Studio] rmdir /s /q node_modules
  rmdir /s /q node_modules >> logs\repair-deps.log 2>&1
)
if exist dist (
  echo [Haiku Studio] Removing studio\dist...
  rmdir /s /q dist >> logs\repair-deps.log 2>&1
)

echo [Haiku Studio] Installing dependencies from npmjs.org...
>> logs\repair-deps.log echo [Haiku Studio] npm install --registry=https://registry.npmjs.org/ --no-audit --no-fund
call npm install --registry=https://registry.npmjs.org/ --no-audit --no-fund >> logs\repair-deps.log 2>&1
if errorlevel 1 (
  echo [Haiku Studio] npm install failed. See studio\logs\repair-deps.log
  echo [Haiku Studio] If removal failed with EPERM, reboot Windows or close stuck node/electron processes, then rerun this script.
  pause
  exit /b 1
)

echo [Haiku Studio] Building UI bundle...
call npm run build >> logs\repair-deps.log 2>&1
if errorlevel 1 (
  echo [Haiku Studio] UI build failed. See studio\logs\repair-deps.log
  pause
  exit /b 1
)

echo [Haiku Studio] Repair complete. Run launch_haiku_studio.bat from the repo root.
pause
exit /b 0
