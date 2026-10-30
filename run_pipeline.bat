@echo off
echo ====================================================
echo Starting AI Traffic Video Processing Pipeline...
echo ====================================================
call .venv\Scripts\activate.bat
python main.py --video assets\videos\sample_traffic.mp4
pause
