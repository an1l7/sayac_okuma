@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [HATA] Python sanal ortami yok. Once README'deki kurulum adimlarini uygulayin.
  pause
  exit /b 1
)
set "MODEL_LOCAL_ONLY=0"
echo AquaLens Studio - Internet/ilk model indirme izinli
 echo Tarayici: http://127.0.0.1:8001
".venv\Scripts\python.exe" -u -m uvicorn main:app --host 127.0.0.1 --port 8001 --log-level info
pause
