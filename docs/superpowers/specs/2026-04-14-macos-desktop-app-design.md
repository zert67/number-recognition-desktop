# macOS Desktop App — Design Spec

**Date:** 2026-04-14
**Goal:** 기존 숫자 인식 프로젝트를 더블클릭으로 실행 가능한 macOS `.app` 번들로 패키징
**Approach:** Tkinter UI + PyInstaller (Flask 제거, 순수 데스크탑 앱)

---

## 1. 전체 구조

기존 Flask 웹 버전은 그대로 유지하고, 데스크탑 앱을 별도 파일로 추가한다.

```
number-recognition/
├── desktop_app.py           # Tkinter 앱 (신규)
├── build_app.sh             # PyInstaller 빌드 스크립트 (신규)
├── requirements_desktop.txt # 데스크탑 전용 의존성 (신규)
│
├── model/
│   ├── train.py             # 기존 유지
│   └── model.h5             # 기존 유지 (번들에 포함)
│
│   ── 기존 파일 (변경 없음) ──
├── app.py
├── templates/
├── static/
├── tests/
└── requirements.txt
```

### 데이터 흐름

1. 앱 시작 시 `model/model.h5` 1회 로드
2. 사용자가 Tkinter Canvas에 마우스로 숫자를 그림
3. Predict 버튼 클릭 → Canvas를 PIL Image로 캡처
4. 28×28 그레이스케일 리사이즈 + 정규화 (0~1)
5. 모델 예측 → 숫자(0~9) + 신뢰도(%) 결과 표시
6. Clear 버튼 → Canvas 초기화, 결과 리셋

Flask 서버 없이 Python이 모델을 직접 호출한다.

---

## 2. Tkinter UI

### 창 레이아웃

```
┌─────────────────────────────┐
│      손글씨 숫자 인식         │  타이틀바
│  ┌─────────────────────┐    │
│  │                     │    │
│  │   Canvas 280×280    │    │  검정 배경, 흰 펜 (lineWidth 18)
│  │                     │    │
│  └─────────────────────┘    │
│  [ Predict ]  [ Clear ]      │
│                              │
│      결과: 7                 │  font size 48
│      신뢰도: 98.3%           │  font size 18
└─────────────────────────────┘
```

### 동작

| 요소 | 동작 |
|------|------|
| Canvas | 마우스 드래그로 흰색 선 그리기 (검정 배경) |
| Predict 버튼 | Canvas → PIL Image → 모델 예측 → 결과 표시 |
| Clear 버튼 | Canvas 검정으로 초기화, 결과 `-` 로 리셋 |
| 빈 Canvas 예측 | "먼저 숫자를 그려주세요" 메시지 표시 |

### 색상

- 배경: `#1a1a2e` (다크)
- Canvas 배경: `#000000`
- 펜: `#ffffff`
- Predict 버튼: `#e94560`
- 결과 텍스트: `#ffffff`

---

## 3. PyInstaller 패키징

### `requirements_desktop.txt`

```
tensorflow>=2.12,<3.0
pillow>=9.0
numpy>=1.23,<2.0
pyinstaller
```

### `build_app.sh`

```bash
#!/bin/bash
set -e

pyinstaller \
  --onedir \
  --windowed \
  --name "NumberRecognition" \
  --add-data "model/model.h5:model" \
  desktop_app.py

echo "빌드 완료: dist/NumberRecognition.app"
```

### 주요 PyInstaller 옵션

| 옵션 | 설명 |
|------|------|
| `--onedir` | 단일 디렉토리 번들 (TensorFlow 크기로 인해 `--onefile`보다 안정적) |
| `--windowed` | 터미널 창 없이 실행 |
| `--add-data "model/model.h5:model"` | 학습된 모델을 번들 내 `model/` 폴더에 포함 |

### 모델 경로 처리

번들 실행 시 `sys._MEIPASS`로 리소스 경로가 변경되므로 `desktop_app.py`에서 경로를 동적으로 처리한다:

```python
import sys, pathlib

def get_model_path():
    if getattr(sys, 'frozen', False):
        base = pathlib.Path(sys._MEIPASS)
    else:
        base = pathlib.Path(__file__).parent
    return base / 'model' / 'model.h5'
```

---

## 4. 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements_desktop.txt

# 2. 모델이 없으면 학습
python model/train.py

# 3. 개발 중 직접 실행
python desktop_app.py

# 4. .app 빌드
bash build_app.sh

# 5. 앱 실행
open dist/NumberRecognition.app
```

---

## 5. 기존 Flask 웹 버전과의 관계

| | Flask 웹 버전 | Tkinter 데스크탑 버전 |
|--|--|--|
| 실행 | `python app.py` | `python desktop_app.py` 또는 `.app` |
| UI | 브라우저 Canvas | Tkinter Canvas |
| 모델 호출 | HTTP API | 직접 호출 |
| 배포 | 서버 필요 | `.app` 번들 |

두 버전은 동일한 `model/model.h5`를 공유한다.
