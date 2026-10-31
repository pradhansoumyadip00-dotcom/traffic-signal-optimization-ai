@echo off
title AI Traffic Management - Live Public Tunnel
cd /d "%~dp0"

.venv\Scripts\python start_tunnel.py
pause
