import sys
import pathlib
import numpy as np
from PIL import Image, ImageDraw


def get_model_path() -> pathlib.Path:
    """개발 환경과 PyInstaller 번들 양쪽에서 model.h5 경로를 반환한다."""
    if getattr(sys, 'frozen', False):
        base = pathlib.Path(sys._MEIPASS)
    else:
        base = pathlib.Path(__file__).parent
    return base / 'model' / 'model.h5'


def preprocess_canvas_image(img: Image.Image) -> np.ndarray:
    """PIL 이미지 → (1, 28, 28, 1) float32 배열 (0~1 정규화)."""
    gray = img.convert('L')
    resized = gray.resize((28, 28), Image.LANCZOS)
    arr = np.array(resized, dtype='float32') / 255.0
    return arr.reshape(1, 28, 28, 1)


def is_blank_canvas(img: Image.Image) -> bool:
    """이미지가 완전히 검정(빈 Canvas)이면 True를 반환한다."""
    arr = np.array(img.convert('L'))
    return bool(arr.max() <= 10)
