@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ====================================================
echo Starting Traffic Signal Optimization Web Dashboard...
echo ====================================================
call .venv\Scripts\activate.bat
streamlit run dashboard\app.py
pause
