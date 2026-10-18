@echo off
rem Double-click to launch NanoSlicer (Windows). Place next to nanoslicer.py or keep in launchers\.
cd /d "%~dp0.."
where pythonw >nul 2>nul && (start "" pythonw nanoslicer.py & exit /b)
where pyw >nul 2>nul && (start "" pyw nanoslicer.py & exit /b)
python nanoslicer.py
pause
