@echo off
rem Launches (from the project folder) the two follow-up sweeps described in experiments\grammar_pressures\README.md.
rem Safe to re-run: finished runs are skipped.
cd /d "%~dp0.."
set PY=python
if exist "..\.venv-event\Scripts\python.exe" set PY=..\.venv-event\Scripts\python.exe
%PY% -c "import torch; print('torch', torch.__version__)" || (echo PyTorch not found in this Python. & pause & exit /b 1)
%PY% -m experiments.grammar_pressures.sweep --output results\grammar_pressures\lead --seeds 0 1 2 3 4 5 6 7 8 9 --updates 20000 --workers 6 --conditions lead_h10 lead_h20 lead_h30
%PY% -m experiments.grammar_pressures.sweep --output results\grammar_pressures\mixed --seeds 0 1 2 3 4 --updates 12000 --workers 6 --conditions mixed_plain mixed_graded mixed_noise mixed_graded_noise
echo.
echo Both sweeps finished. Summaries are in results\grammar_pressures\lead\summary.md and results\grammar_pressures\mixed\summary.md
pause
