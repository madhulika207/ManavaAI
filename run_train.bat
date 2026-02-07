@echo off
echo Starting Mistral Fine-Tuning...
echo Using virtual environment from: C:\Users\hp\.gemini\antigravity\scratch\mistral_finetune\venv

"C:\Users\hp\.gemini\antigravity\scratch\mistral_finetune\venv\Scripts\python.exe" train.py

if %errorlevel% neq 0 (
    echo.
    echo Error occurred!
    echo.
    if %errorlevel% == 1 (
         echo Check if you have NVIDIA Drivers installed.
    )
    pause
) else (
    echo Training complete!
    pause
)
