@echo off
rem Double-click to open the ShanHaiJue desktop interface.
rem 双击本文件即可打开山海决图形界面。
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python 3 was not found. Install it from https://www.python.org/downloads/
  echo 未找到 Python 3，请从 https://www.python.org/downloads/ 安装后重试。
  pause
  exit /b 1
)

python -c "import importlib.util,sys; m=[x for x in ('numpy','scipy','sklearn','joblib','pandas','openpyxl') if importlib.util.find_spec(x) is None]; print('Missing / 缺少: '+', '.join(m)) if m else None; sys.exit(1 if m else 0)"
if errorlevel 1 (
  echo Installing dependencies / 正在安装依赖 ...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Install failed. Run manually: pip install -r requirements.txt
    echo 安装失败，请手动执行：pip install -r requirements.txt
    pause
    exit /b 1
  )
)

python gui\app.py
if errorlevel 1 pause
