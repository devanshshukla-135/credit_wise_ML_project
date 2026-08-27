@echo off
cd /d "%~dp0"
call C:\ProgramData\anaconda3\Scripts\activate.bat
streamlit run app.py
pause