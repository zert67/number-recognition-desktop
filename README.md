# Number Recognition — macOS Desktop App

MNIST 데이터셋으로 학습한 CNN 모델을 Tkinter GUI로 감싼 macOS 데스크탑 앱입니다.  
Python 없이 더블클릭으로 실행할 수 있는 `.app` 번들로 패키징됩니다.

## 데모

앱 창에서 Canvas에 숫자(0~9)를 그리고 **Predict** 버튼을 누르면 인식 결과와 신뢰도를 표시합니다.

## 기술 스택

- **모델**: TensorFlow/Keras CNN (MNIST, 99%+ 정확도)
- **GUI**: Tkinter (Python 기본 내장)
- **이미지 처리**: Pillow (PIL) — in-memory 버퍼, Screen Recording 권한 불필요
- **패키징**: PyInstaller (`.app` 번들)
- **테스트**: pytest

## 프로젝트 구조

```
number-recognition-desktop/
├── desktop_app.py          # Tkinter 앱 메인
├── image_utils.py          # 이미지 전처리 유틸 (Tkinter 의존 없음)
├── build_app.sh            # PyInstaller 빌드 스크립트
├── model/
│   ├── train.py            # CNN 학습 스크립트
│   └── model.h5            # 학습된 모델 (gitignore)
├── tests/
│   └── test_desktop_app.py # 유닛 테스트
└── requirements_desktop.txt
```

## 시작하기

### 1. 의존성 설치

```bash
pip install -r requirements_desktop.txt
```

### 2. 모델 학습

```bash
python model/train.py
```

학습 완료 후 `model/model.h5` 파일이 생성됩니다. (약 2~3분 소요)

### 3. 앱 직접 실행 (개발 모드)

```bash
python desktop_app.py
```

### 4. macOS `.app` 번들 빌드

```bash
bash build_app.sh
```

빌드 완료 후 `dist/NumberRecognition.app`이 생성됩니다. (TensorFlow 포함으로 2~10분 소요)

```bash
open dist/NumberRecognition.app
```

## 테스트

```bash
pytest tests/ -v
```

## 관련 레포

Flask 기반 웹 앱 버전은 아래 레포를 참고하세요.  
👉 [number-recognition](https://github.com/zert67/number-recognition)
