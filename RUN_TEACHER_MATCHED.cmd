@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Run this launcher from the OTSE repository root with .venv installed.
  exit /b 1
)
".venv\Scripts\python.exe" -u run_teacher_matched.py --output "training_outputs\otse-auto-source-all-v1" %*
exit /b %errorlevel%
