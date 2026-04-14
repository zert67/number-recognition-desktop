#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# 사전 점검
if ! command -v pyinstaller &> /dev/null; then
  echo "Error: pyinstaller not found. Run: pip install -r requirements_desktop.txt" >&2
  exit 1
fi

if [ ! -f "model/model.h5" ]; then
  echo "Error: model/model.h5 not found. Run: python model/train.py" >&2
  exit 1
fi

echo "NumberRecognition.app 빌드 시작..."

# 이전 빌드 정리
rm -rf build dist NumberRecognition.spec

pyinstaller \
  --onedir \
  --windowed \
  --name "NumberRecognition" \
  --add-data "model/model.h5:model" \
  --add-data "image_utils.py:." \
  desktop_app.py

echo ""
echo "빌드 완료: dist/NumberRecognition.app"
echo "실행: open dist/NumberRecognition.app"
