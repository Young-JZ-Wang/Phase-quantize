@echo off
rem Double-click to launch the Phase Mask Quantizer GUI (Windows). Place next to phase_mask_quantizer.py or keep in launchers\.
cd /d "%~dp0.."
where pythonw >nul 2>nul && (start "" pythonw phase_mask_quantizer.py & exit /b)
where pyw >nul 2>nul && (start "" pyw phase_mask_quantizer.py & exit /b)
python phase_mask_quantizer.py
pause
