#!/usr/bin/env bash
# learning AI at www.haotianblog.com
#
# Build a macOS .app around the CPU distribution's desktop interface.
#
#   ./script/package_macos_app.sh
#   open "dist/EIS-PEM ShanHaiJue.app"
#
# The bundle carries the identification and certification pipeline, the frozen validated
# system, the reference CSVs, and a Python runtime, so the app launches without the user
# installing anything. It is a LOCAL bundle: the embedded runtime is a virtual environment
# built from this machine's Python, and moving the app to another Mac still needs a frozen
# runtime plus Developer ID signing and notarization.
set -euo pipefail

APP_NAME="EIS-PEM ShanHaiJue"
BUNDLE_ID="com.eispem.shanhaijue"
EXECUTABLE_NAME="eis-pem-shanhaijue"
VERSION="1.0.0"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
APP_DIR="$RESOURCES_DIR/app"
VENV_DIR="$RESOURCES_DIR/venv"
ICONSET_DIR="$RESOURCES_DIR/AppIcon.iconset"
ICON_FILE="$RESOURCES_DIR/AppIcon.icns"

# ----------------------------------------------------------------------------------
# base interpreter: needs tkinter, which several Python builds ship without
# ----------------------------------------------------------------------------------
BASE_PYTHON="${BASE_PYTHON:-}"
if [[ -z "$BASE_PYTHON" ]]; then
  for cand in /opt/anaconda3/bin/python3 /opt/homebrew/bin/python3 python3 /usr/bin/python3; do
    path="$(command -v "$cand" 2>/dev/null || true)"
    [[ -n "$path" ]] || continue
    if "$path" -c 'import tkinter, venv' >/dev/null 2>&1; then BASE_PYTHON="$path"; break; fi
  done
fi
if [[ -z "$BASE_PYTHON" ]]; then
  echo "No Python 3 with tkinter and venv was found." >&2
  echo "Install one (python.org builds include tkinter) or set BASE_PYTHON=/path/to/python3." >&2
  exit 1
fi
echo "Base interpreter: $BASE_PYTHON"

# ----------------------------------------------------------------------------------
# the distribution must be complete before anything is packaged
# ----------------------------------------------------------------------------------
required_paths=(
  "shanhaijue/__init__.py"
  "shanhaijue/panel.py"
  "shanhaijue/arms.py"
  "shanhaijue/features.py"
  "shanhaijue/router.py"
  "shanhaijue/certifier.py"
  "shanhaijue/pipeline.py"
  "shanhaijue/vendor/eispem/seis_model.py"
  "shanhaijue/vendor/eispem/seis_pipeline.py"
  "gui/app.py"
  "gui/csv_io.py"
  "gui/i18n.py"
  "gui/params.py"
  "bin/certify.py"
  "bin/selftest.py"
  "artifacts/validated_system/frozen_system.joblib"
  "artifacts/validated_system/VALIDATED_PERFORMANCE.json"
  "examples/calibration_grid_25x120.csv"
  "requirements.txt"
  "README.md"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$ROOT_DIR/$path" ]]; then
    echo "Missing required distribution path: $path" >&2
    exit 1
  fi
done

if grep -q "^import jax" "$ROOT_DIR/shanhaijue/arms.py"; then
  echo "This is the GPU distribution. The app is built from the CPU distribution," >&2
  echo "whose estimator ensemble runs on scipy and needs no CUDA device." >&2
  exit 1
fi

rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$APP_DIR"

# ----------------------------------------------------------------------------------
# payload: the program, the frozen system, the reference CSVs
# ----------------------------------------------------------------------------------
for tree in shanhaijue gui bin examples; do
  rsync -a --exclude "__pycache__" --exclude "*.pyc" "$ROOT_DIR/$tree" "$APP_DIR/"
done
mkdir -p "$APP_DIR/artifacts"
rsync -a "$ROOT_DIR/artifacts/validated_system" "$APP_DIR/artifacts/"
cp "$ROOT_DIR/requirements.txt" "$ROOT_DIR/README.md" "$APP_DIR/"
[[ -f "$ROOT_DIR/LICENSE" ]] && cp "$ROOT_DIR/LICENSE" "$APP_DIR/"

# ----------------------------------------------------------------------------------
# embedded runtime
# ----------------------------------------------------------------------------------
echo "Building the embedded runtime (this downloads numpy/scipy/scikit-learn once) ..."
"$BASE_PYTHON" -m venv --copies "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
"$VENV_DIR/bin/python" -m pip install --quiet -r "$ROOT_DIR/requirements.txt"

# requirements.txt states the window in which the frozen system loads AND scores
# identically. The bundle is not a range, it is one build, so it gets the exact version the
# system was frozen under — same scores, and no InconsistentVersionWarning in the log.
FROZEN_SKLEARN="$("$VENV_DIR/bin/python" - "$ROOT_DIR/artifacts/validated_system/ENVIRONMENT.json" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1]))["frozen_under"]["scikit-learn"])
except Exception:
    print("")
PY
)"
if [[ -n "$FROZEN_SKLEARN" ]]; then
  echo "Pinning the embedded scikit-learn to $FROZEN_SKLEARN, the frozen system's own version ..."
  "$VENV_DIR/bin/python" -m pip install --quiet "scikit-learn==$FROZEN_SKLEARN"
fi

# A venv resolves its own prefix from the location of its interpreter, so the bundle
# survives being moved into /Applications. It still reads `home` out of pyvenv.cfg, which
# is why this is a local-machine app rather than a redistributable one.
if [[ -L "$VENV_DIR/bin/python" ]]; then
  python_target="$(readlink "$VENV_DIR/bin/python")"
  if [[ "$python_target" = /* && -x "$python_target" ]]; then
    rm "$VENV_DIR/bin/python"
    cp "$python_target" "$VENV_DIR/bin/python"
    chmod +x "$VENV_DIR/bin/python"
  fi
fi

# ----------------------------------------------------------------------------------
# executables
# ----------------------------------------------------------------------------------
cat > "$MACOS_DIR/$EXECUTABLE_NAME" <<'SH'
#!/usr/bin/env bash
# learning AI at www.haotianblog.com
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RESOURCES_DIR="$BUNDLE_DIR/Contents/Resources"
APP_DIR="$RESOURCES_DIR/app"
LOG_DIR="$HOME/Library/Logs/EIS-PEM ShanHaiJue"
LOG_FILE="$LOG_DIR/app.log"
mkdir -p "$LOG_DIR"

PYTHON="$RESOURCES_DIR/venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON:-}" || ! -x "$PYTHON" ]]; then
  /usr/bin/osascript -e 'display dialog "EIS-PEM ShanHaiJue could not find a usable Python runtime." buttons {"OK"} default button "OK" with icon caution' || true
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1
export EIS_PEM_APP_BUNDLE="$BUNDLE_DIR"
cd "$APP_DIR"

{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] launching the desktop interface"
  echo "python: $PYTHON"
  echo "app:    $APP_DIR"
} >> "$LOG_FILE"

exec "$PYTHON" "$APP_DIR/gui/app.py" >> "$LOG_FILE" 2>&1
SH

# One command-line entry per bin/ script, so the bundle is usable without the window.
for entry in certify:certify selftest:selftest run_panel:run-panel \
             train_router:train-router freeze_system:freeze-system \
             validate_system:validate-system import_frozen:import-frozen; do
  src="${entry%%:*}"; name="${entry##*:}"
  cat > "$MACOS_DIR/shanhaijue-$name" <<SH
#!/usr/bin/env bash
# learning AI at www.haotianblog.com
set -euo pipefail
BUNDLE_DIR="\$(cd "\$(dirname "\$0")/../.." && pwd)"
APP_DIR="\$BUNDLE_DIR/Contents/Resources/app"
PYTHON="\$BUNDLE_DIR/Contents/Resources/venv/bin/python"
[[ -x "\$PYTHON" ]] || PYTHON="\$(command -v python3 || true)"
if [[ -z "\${PYTHON:-}" || ! -x "\$PYTHON" ]]; then
  echo "No usable Python runtime in the bundle." >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
cd "\$APP_DIR"
exec "\$PYTHON" "\$APP_DIR/bin/$src.py" "\$@"
SH
  chmod +x "$MACOS_DIR/shanhaijue-$name"
done

chmod +x "$MACOS_DIR/$EXECUTABLE_NAME"

# ----------------------------------------------------------------------------------
# Info.plist
# ----------------------------------------------------------------------------------
cat > "$CONTENTS_DIR/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>$APP_NAME</string>
  <key>CFBundleExecutable</key>
  <string>$EXECUTABLE_NAME</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>CFBundleIdentifier</key>
  <string>$BUNDLE_ID</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>$VERSION</string>
  <key>CFBundleVersion</key>
  <string>$VERSION</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

# ----------------------------------------------------------------------------------
# icon: a Nyquist arc over an instrument-panel grid, generated rather than shipped
# ----------------------------------------------------------------------------------
mkdir -p "$ICONSET_DIR"
"$VENV_DIR/bin/python" - "$ICONSET_DIR" <<'PY'
# learning AI at www.haotianblog.com
from __future__ import annotations

import math
import struct
import sys
import zlib
from pathlib import Path


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def _write_png(path: Path, size: int) -> None:
    rows = []
    arc = max(1.0, size * 0.016)
    axis = max(1.0, size * 0.007)
    for y in range(size):
        row = bytearray()
        yn = y / max(1, size - 1)
        for x in range(size):
            xn = x / max(1, size - 1)
            r = int(12 + 20 * xn + 10 * (1 - yn))
            g = int(24 + 44 * xn + 12 * (1 - yn))
            b = int(58 + 96 * (1 - yn) + 38 * xn)

            if (abs((xn * 5) % 1.0) < 0.012 and 0.18 < yn < 0.82) or \
               (abs((yn * 5) % 1.0) < 0.012 and 0.14 < xn < 0.86):
                r, g, b = min(255, r + 10), min(255, g + 16), min(255, b + 28)

            # the depressed semicircle of a charge-transfer arc
            t = min(1.0, max(0.0, (xn - 0.18) / 0.68))
            curve = 0.64 - 0.24 * math.sin(math.pi * t)
            if 0.18 < xn < 0.86 and abs(y - curve * size) <= arc:
                r, g, b = 84, 226, 255

            # the certified subset: three claimed coordinates on the arc
            for cx in (0.36, 0.52, 0.70):
                ct = min(1.0, max(0.0, (cx - 0.18) / 0.68))
                cy = 0.64 - 0.24 * math.sin(math.pi * ct)
                if math.hypot((xn - cx) * size, (yn - cy) * size) < size * 0.035:
                    r, g, b = 255, 206, 84

            if (abs(xn - 0.16) * size <= axis and 0.20 < yn < 0.82) or \
               (abs(yn - 0.78) * size <= axis and 0.14 < xn < 0.88):
                r, g, b = 210, 224, 255

            row.extend((r, g, b, 255))
        rows.append(b"\x00" + bytes(row))

    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
           + _chunk(b"IEND", b""))
    path.write_bytes(png)


target = Path(sys.argv[1])
for filename, size in {
    "icon_16x16.png": 16, "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32, "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128, "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256, "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512, "icon_512x512@2x.png": 1024,
}.items():
    _write_png(target / filename, size)
PY

if command -v iconutil >/dev/null 2>&1; then
  iconutil -c icns "$ICONSET_DIR" -o "$ICON_FILE"
  rm -rf "$ICONSET_DIR"
fi

# ----------------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------------
cat > "$RESOURCES_DIR/PipelineManifest.txt" <<EOF
Application: $APP_NAME
Bundle ID: $BUNDLE_ID
Version: $VERSION
Packaged at: $(date -u '+%Y-%m-%dT%H:%M:%SZ')
Source root: $ROOT_DIR
Base interpreter: $BASE_PYTHON
Embedded runtime: $("$VENV_DIR/bin/python" -c 'import sys; print(sys.version.split()[0])')

Git branch: $(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
Git commit: $(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
Git status:
$(git -C "$ROOT_DIR" status --short -- "$ROOT_DIR" 2>/dev/null || echo unavailable)

Frozen system:
$(cat "$ROOT_DIR/artifacts/validated_system/import_record.json")

Packaged:
- desktop interface: gui/
- pipeline package: shanhaijue/ including the vendored physics layer
- command-line entries: bin/
- frozen validated system: artifacts/validated_system/
- reference CSVs: examples/
- docs: README.md, LICENSE when present
- excluded: nothing else exists in the distribution; it ships programs only

Installed packages:
$("$VENV_DIR/bin/python" -m pip list --format=freeze 2>/dev/null)
EOF

# ----------------------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------------------
for path in "$APP_DIR/gui/app.py" "$APP_DIR/shanhaijue/pipeline.py" \
            "$APP_DIR/artifacts/validated_system/frozen_system.joblib" \
            "$VENV_DIR/bin/python" "$MACOS_DIR/$EXECUTABLE_NAME"; do
  if [[ ! -e "$path" ]]; then
    echo "Packaged app is missing required file: $path" >&2
    exit 1
  fi
done

plutil -lint "$CONTENTS_DIR/Info.plist" >/dev/null

PYTHONDONTWRITEBYTECODE=1 "$VENV_DIR/bin/python" - "$APP_DIR" <<'PY'
# learning AI at www.haotianblog.com
import sys
sys.path.insert(0, sys.argv[1])

import tkinter  # noqa: F401
import numpy  # noqa: F401
import scipy  # noqa: F401
import sklearn  # noqa: F401
import joblib  # noqa: F401
import openpyxl  # noqa: F401

from shanhaijue import panel, arms, features, router, certifier, pipeline  # noqa: F401

frozen = pipeline.load_frozen(sys.argv[1] + "/artifacts/validated_system/frozen_system.joblib")
conditions, frequencies = pipeline.frozen_grid(frozen)
assert frozen["certifier"].thresholds, "frozen system carries no thresholds"
print(f"    smoke test: {len(conditions)} conditions x {frequencies.size} frequencies, "
      f"budgets {sorted(frozen['certifier'].thresholds)}")
PY

find "$APP_BUNDLE" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$APP_BUNDLE" -type f -name "*.pyc" -delete

if command -v codesign >/dev/null 2>&1; then
  if codesign --force --deep --sign - "$APP_BUNDLE" >/dev/null 2>&1; then
    codesign --verify --deep --strict "$APP_BUNDLE" 2>/dev/null || \
      echo "Warning: ad-hoc codesign verification failed; the bundle was still built." >&2
  else
    echo "Warning: ad-hoc codesign failed; the bundle was still built." >&2
  fi
fi

echo "Built macOS app: $APP_BUNDLE"
echo "Size: $(du -sh "$APP_BUNDLE" | cut -f1)"
echo "Manifest: $RESOURCES_DIR/PipelineManifest.txt"
