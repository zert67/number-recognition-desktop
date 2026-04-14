# macOS Desktop App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 숫자 인식 프로젝트를 더블클릭으로 실행 가능한 macOS `.app` 번들로 만든다.

**Architecture:** Tkinter Canvas UI가 모델을 직접 호출한다 (Flask 없음). 앱 시작 시 `model/model.h5`를 로드하고, 마우스 드래그로 숫자를 그리면 PIL로 캡처해 28×28로 변환 후 예측한다. PyInstaller로 Python + 모델을 단일 `.app` 번들로 패키징한다.

**Tech Stack:** Python 3, TensorFlow/Keras, Tkinter, Pillow, NumPy, PyInstaller

---

## File Map

| 파일 | 역할 |
|------|------|
| `desktop_app.py` | Tkinter 앱 — UI, 그리기, 예측, 결과 표시 |
| `requirements_desktop.txt` | 데스크탑 전용 의존성 |
| `build_app.sh` | PyInstaller 빌드 스크립트 |
| `tests/test_desktop_app.py` | desktop_app 유닛 테스트 |

---

## Task 1: 데스크탑 의존성 설정

**Files:**
- Create: `requirements_desktop.txt`

- [ ] **Step 1: `requirements_desktop.txt` 작성**

```
tensorflow>=2.12,<3.0
pillow>=9.0
numpy>=1.23,<2.0
pyinstaller
```

- [ ] **Step 2: pyinstaller 설치**

```bash
pip install pyinstaller
```

Expected: `Successfully installed pyinstaller-...`

- [ ] **Step 3: 설치 확인**

```bash
pyinstaller --version
```

Expected: `6.x.x` 또는 유사한 버전 출력

- [ ] **Step 4: Commit**

```bash
git add requirements_desktop.txt
git commit -m "chore: add desktop app dependencies"
```

---

## Task 2: Tkinter 데스크탑 앱

**Files:**
- Create: `desktop_app.py`
- Create: `tests/test_desktop_app.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_desktop_app.py`:
```python
import io
import numpy as np
import pytest
from PIL import Image


def make_white_stroke_image():
    """흰색 획이 있는 280×280 PIL 이미지 (RGBA) 반환"""
    img = Image.new('RGBA', (280, 280), (0, 0, 0, 255))
    pixels = img.load()
    for x in range(100, 180):
        pixels[x, 140] = (255, 255, 255, 255)
    return img


def make_blank_image():
    """완전히 검정인 280×280 PIL 이미지 반환"""
    return Image.new('RGBA', (280, 280), (0, 0, 0, 255))


def test_preprocess_returns_correct_shape():
    from desktop_app import preprocess_canvas_image
    img = make_white_stroke_image()
    arr = preprocess_canvas_image(img)
    assert arr.shape == (1, 28, 28, 1)


def test_preprocess_normalizes_values():
    from desktop_app import preprocess_canvas_image
    img = make_white_stroke_image()
    arr = preprocess_canvas_image(img)
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_is_blank_canvas_true_for_black_image():
    from desktop_app import is_blank_canvas
    img = make_blank_image()
    assert is_blank_canvas(img) is True


def test_is_blank_canvas_false_for_drawn_image():
    from desktop_app import is_blank_canvas
    img = make_white_stroke_image()
    assert is_blank_canvas(img) is False


def test_get_model_path_returns_path_object():
    from desktop_app import get_model_path
    import pathlib
    path = get_model_path()
    assert isinstance(path, pathlib.Path)
    assert path.name == 'model.h5'
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
pytest tests/test_desktop_app.py -v
```

Expected: `ImportError` (desktop_app.py 없음)

- [ ] **Step 3: `desktop_app.py` 작성**

```python
import sys
import pathlib
import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from PIL import Image, ImageGrab
import tensorflow as tf


# ── 모델 경로 (개발 환경 / PyInstaller 번들 양쪽 지원) ──────────────────────

def get_model_path() -> pathlib.Path:
    """개발 환경과 PyInstaller 번들 양쪽에서 model.h5 경로를 반환한다."""
    if getattr(sys, 'frozen', False):
        base = pathlib.Path(sys._MEIPASS)
    else:
        base = pathlib.Path(__file__).parent
    return base / 'model' / 'model.h5'


# ── 이미지 전처리 ─────────────────────────────────────────────────────────────

def preprocess_canvas_image(img: Image.Image) -> np.ndarray:
    """PIL RGBA 이미지 → (1, 28, 28, 1) float32 배열 (0~1 정규화)."""
    gray = img.convert('L')
    resized = gray.resize((28, 28), Image.LANCZOS)
    arr = np.array(resized, dtype='float32') / 255.0
    return arr.reshape(1, 28, 28, 1)


def is_blank_canvas(img: Image.Image) -> bool:
    """이미지가 완전히 검정(빈 Canvas)이면 True를 반환한다."""
    arr = np.array(img.convert('L'))
    return arr.max() <= 10


# ── Tkinter 앱 ───────────────────────────────────────────────────────────────

class NumberRecognitionApp(tk.Tk):
    CANVAS_SIZE = 280
    PEN_WIDTH = 18
    BG_COLOR = '#1a1a2e'
    CANVAS_BG = '#000000'
    PEN_COLOR = '#ffffff'
    BTN_PREDICT_BG = '#e94560'
    BTN_CLEAR_BG = '#444444'

    def __init__(self):
        super().__init__()
        self.title('손글씨 숫자 인식')
        self.configure(bg=self.BG_COLOR)
        self.resizable(False, False)

        self._model = None
        self._last_x = 0
        self._last_y = 0

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        pad = {'padx': 20, 'pady': 10}

        # 캔버스
        self.canvas = tk.Canvas(
            self,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg=self.CANVAS_BG,
            highlightthickness=2,
            highlightbackground='#e94560',
        )
        self.canvas.pack(**pad)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)

        # 버튼
        btn_frame = tk.Frame(self, bg=self.BG_COLOR)
        btn_frame.pack(pady=(0, 10))

        tk.Button(
            btn_frame,
            text='Predict',
            bg=self.BTN_PREDICT_BG,
            fg='white',
            font=('Helvetica', 14, 'bold'),
            relief='flat',
            padx=20, pady=6,
            command=self._on_predict,
        ).pack(side='left', padx=8)

        tk.Button(
            btn_frame,
            text='Clear',
            bg=self.BTN_CLEAR_BG,
            fg='white',
            font=('Helvetica', 14),
            relief='flat',
            padx=20, pady=6,
            command=self._on_clear,
        ).pack(side='left', padx=8)

        # 결과 표시
        self.result_var = tk.StringVar(value='-')
        self.confidence_var = tk.StringVar(value='')

        tk.Label(
            self,
            textvariable=self.result_var,
            font=('Helvetica', 48, 'bold'),
            bg=self.BG_COLOR,
            fg='white',
        ).pack()

        tk.Label(
            self,
            textvariable=self.confidence_var,
            font=('Helvetica', 18),
            bg=self.BG_COLOR,
            fg='#aaaaaa',
        ).pack(pady=(0, 20))

    def _load_model(self):
        model_path = get_model_path()
        if not model_path.exists():
            self.result_var.set('오류')
            self.confidence_var.set(f'모델 없음: python model/train.py 실행')
            return
        self._model = tf.keras.models.load_model(str(model_path))

    def _on_press(self, event):
        self._last_x, self._last_y = event.x, event.y

    def _on_drag(self, event):
        x, y = event.x, event.y
        self.canvas.create_line(
            self._last_x, self._last_y, x, y,
            fill=self.PEN_COLOR,
            width=self.PEN_WIDTH,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
            smooth=True,
        )
        self._last_x, self._last_y = x, y

    def _capture_canvas(self) -> Image.Image:
        """Canvas 내용을 PIL RGBA 이미지로 반환한다."""
        self.update_idletasks()
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        return ImageGrab.grab(bbox=(x, y, x + w, y + h)).convert('RGBA')

    def _on_predict(self):
        if self._model is None:
            return
        img = self._capture_canvas()
        if is_blank_canvas(img):
            self.result_var.set('-')
            self.confidence_var.set('먼저 숫자를 그려주세요')
            return
        arr = preprocess_canvas_image(img)
        predictions = self._model.predict(arr, verbose=0)[0]
        digit = int(np.argmax(predictions))
        confidence = float(predictions[digit])
        self.result_var.set(str(digit))
        self.confidence_var.set(f'신뢰도: {confidence * 100:.1f}%')

    def _on_clear(self):
        self.canvas.delete('all')
        self.result_var.set('-')
        self.confidence_var.set('')


if __name__ == '__main__':
    app = NumberRecognitionApp()
    app.mainloop()
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
pytest tests/test_desktop_app.py -v
```

Expected:
```
PASSED tests/test_desktop_app.py::test_preprocess_returns_correct_shape
PASSED tests/test_desktop_app.py::test_preprocess_normalizes_values
PASSED tests/test_desktop_app.py::test_is_blank_canvas_true_for_black_image
PASSED tests/test_desktop_app.py::test_is_blank_canvas_false_for_drawn_image
PASSED tests/test_desktop_app.py::test_get_model_path_returns_path_object
```

- [ ] **Step 5: 앱 직접 실행 확인**

```bash
python desktop_app.py
```

Expected: 창이 열리고, Canvas에 그림을 그린 뒤 Predict 버튼을 누르면 숫자와 신뢰도가 표시됨. Ctrl+C로 종료.

- [ ] **Step 6: Commit**

```bash
git add desktop_app.py tests/test_desktop_app.py
git commit -m "feat: add Tkinter desktop app"
```

---

## Task 3: PyInstaller 빌드 스크립트

**Files:**
- Create: `build_app.sh`

- [ ] **Step 1: `build_app.sh` 작성**

```bash
#!/bin/bash
set -e

echo "NumberRecognition.app 빌드 시작..."

# 이전 빌드 정리
rm -rf build dist NumberRecognition.spec

pyinstaller \
  --onedir \
  --windowed \
  --name "NumberRecognition" \
  --add-data "model/model.h5:model" \
  desktop_app.py

echo ""
echo "빌드 완료: dist/NumberRecognition.app"
echo "실행: open dist/NumberRecognition.app"
```

- [ ] **Step 2: 실행 권한 부여**

```bash
chmod +x build_app.sh
```

- [ ] **Step 3: .gitignore에 빌드 아티팩트 추가**

`.gitignore` 파일 끝에 다음을 추가:
```
# PyInstaller build artifacts
build/
dist/
*.spec
```

```bash
# 편집 후 확인
cat .gitignore
```

- [ ] **Step 4: Commit**

```bash
git add build_app.sh .gitignore
git commit -m "feat: add PyInstaller build script"
```

---

## Task 4: .app 번들 빌드 및 검증

**Files:**
- No new files (build output goes to `dist/`, which is gitignored)

- [ ] **Step 1: 빌드 실행**

```bash
bash build_app.sh
```

Expected: `빌드 완료: dist/NumberRecognition.app` 출력

TensorFlow를 포함하므로 시간이 걸릴 수 있음 (2~10분).

- [ ] **Step 2: 번들 구조 확인**

```bash
ls dist/NumberRecognition.app/Contents/MacOS/
ls dist/NumberRecognition.app/Contents/Resources/ | head -10
```

Expected: `NumberRecognition` 실행 파일과 리소스 파일들 존재

- [ ] **Step 3: 모델 파일 포함 확인**

```bash
find dist/NumberRecognition.app -name "model.h5"
```

Expected: `dist/NumberRecognition.app/.../model/model.h5` 경로 출력

- [ ] **Step 4: 앱 실행 확인**

```bash
open dist/NumberRecognition.app
```

Expected: 앱 창이 열리고, Canvas에 그림을 그려 Predict가 정상 동작함을 확인

- [ ] **Step 5: 전체 테스트 실행**

```bash
pytest tests/ -v
```

Expected: 모든 테스트 PASSED (기존 4개 + 신규 5개 = 9개)

- [ ] **Step 6: 최종 Commit**

```bash
git add .
git commit -m "chore: verify .app build and all tests passing"
```
