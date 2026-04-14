import numpy as np
import pytest
from PIL import Image


def make_white_stroke_image():
    """흰색 획이 있는 280×280 PIL 이미지 반환"""
    img = Image.new('RGB', (280, 280), 'black')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.line([(100, 140), (180, 140)], fill='white', width=18)
    return img


def make_blank_image():
    """완전히 검정인 280×280 PIL 이미지 반환"""
    return Image.new('RGB', (280, 280), 'black')


def test_preprocess_returns_correct_shape():
    from image_utils import preprocess_canvas_image
    img = make_white_stroke_image()
    arr = preprocess_canvas_image(img)
    assert arr.shape == (1, 28, 28, 1)


def test_preprocess_normalizes_values():
    from image_utils import preprocess_canvas_image
    img = make_white_stroke_image()
    arr = preprocess_canvas_image(img)
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_is_blank_canvas_true_for_black_image():
    from image_utils import is_blank_canvas
    img = make_blank_image()
    assert is_blank_canvas(img) is True


def test_is_blank_canvas_false_for_drawn_image():
    from image_utils import is_blank_canvas
    img = make_white_stroke_image()
    assert is_blank_canvas(img) is False


def test_get_model_path_returns_path_object():
    from image_utils import get_model_path
    import pathlib
    path = get_model_path()
    assert isinstance(path, pathlib.Path)
    assert path.name == 'model.h5'
