#!/bin/bash
# Double-click to open the ShanHaiJue desktop interface.
# 双击本文件即可打开山海决图形界面。
cd "$(dirname "$0")" || exit 1

PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3 was not found. Install it from https://www.python.org/downloads/ and try again."
  echo "未找到 Python 3。请从 https://www.python.org/downloads/ 安装后重试。"
  read -r -p "Press Enter to close / 按回车关闭 " _
  exit 1
fi

"$PY" - <<'CHECK' || {
import importlib.util, sys
missing = [m for m in ("numpy", "scipy", "sklearn", "joblib", "pandas", "openpyxl")
           if importlib.util.find_spec(m) is None]
if missing:
    print("Missing packages / 缺少依赖: " + ", ".join(missing))
    sys.exit(1)
CHECK
  echo
  echo "Installing dependencies / 正在安装依赖 ..."
  "$PY" -m pip install -r requirements.txt || {
    echo "Install failed. Run manually: pip install -r requirements.txt"
    echo "安装失败，请手动执行：pip install -r requirements.txt"
    read -r -p "Press Enter to close / 按回车关闭 " _
    exit 1
  }
}

exec "$PY" gui/app.py
