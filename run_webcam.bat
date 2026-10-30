@echo off
echo ====================================================
echo Starting Live Webcam AI Traffic Detection...
echo (Press 'q' inside video window to exit)
echo ====================================================
call .venv\Scripts\activate.bat
python main.py --webcam
pause
