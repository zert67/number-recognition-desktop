import sys
import pathlib
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from image_utils import get_model_path, preprocess_canvas_image, is_blank_canvas


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

        # in-memory draw buffer (Screen Recording 권한 불필요)
        self._pil_image = Image.new('RGB', (self.CANVAS_SIZE, self.CANVAS_SIZE), 'black')
        self._pil_draw = ImageDraw.Draw(self._pil_image)

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        pad = {'padx': 20, 'pady': 10}

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
            self.confidence_var.set('모델 없음: python model/train.py 실행')
            return
        try:
            self._model = tf.keras.models.load_model(str(model_path))
        except Exception as e:
            self.result_var.set('오류')
            self.confidence_var.set(f'모델 로드 실패: {e}')

    def _on_press(self, event):
        self._last_x, self._last_y = event.x, event.y

    def _on_drag(self, event):
        x, y = event.x, event.y
        # Tkinter canvas에 그리기
        self.canvas.create_line(
            self._last_x, self._last_y, x, y,
            fill=self.PEN_COLOR,
            width=self.PEN_WIDTH,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
            smooth=True,
        )
        # PIL in-memory 버퍼에도 동일하게 그리기
        self._pil_draw.line(
            [self._last_x, self._last_y, x, y],
            fill='white',
            width=self.PEN_WIDTH,
        )
        self._last_x, self._last_y = x, y

    def _on_predict(self):
        if self._model is None:
            self.result_var.set('오류')
            self.confidence_var.set('모델 없음: python model/train.py 실행')
            return
        if is_blank_canvas(self._pil_image):
            self.result_var.set('-')
            self.confidence_var.set('먼저 숫자를 그려주세요')
            return
        arr = preprocess_canvas_image(self._pil_image)
        predictions = self._model.predict(arr, verbose=0)[0]
        digit = int(np.argmax(predictions))
        confidence = float(predictions[digit])
        self.result_var.set(str(digit))
        self.confidence_var.set(f'신뢰도: {confidence * 100:.1f}%')

    def _on_clear(self):
        self.canvas.delete('all')
        self._pil_image = Image.new('RGB', (self.CANVAS_SIZE, self.CANVAS_SIZE), 'black')
        self._pil_draw = ImageDraw.Draw(self._pil_image)
        self.result_var.set('-')
        self.confidence_var.set('')


if __name__ == '__main__':
    app = NumberRecognitionApp()
    app.mainloop()
