from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Line, Rectangle
from kivy.core.window import Window
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import numpy as np
from PIL import Image, ImageOps
import threading
import os

# ML backends (prefer TFLite for Android)
_tflite_available = False
try:
    from tflite_runtime.interpreter import Interpreter  # lightweight runtime
    _tflite_available = True
except Exception:
    try:
        # fallback to full TF if available (desktop/dev)
        from tensorflow.lite import Interpreter  # type: ignore
        _tflite_available = True
    except Exception:
        _tflite_available = False

# Keras fallback (desktop)
_keras_available = False
try:
    from tensorflow.keras import models
    _keras_available = True
except Exception:
    _keras_available = False

class DrawingWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lines = []
        with self.canvas:
            # white background so exports aren't transparent on Android
            Color(1, 1, 1, 1)
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg_rect, size=self._update_bg_rect)

    def _update_bg_rect(self, *args):
        if hasattr(self, '_bg_rect'):
            self._bg_rect.pos = self.pos
            self._bg_rect.size = self.size

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            with self.canvas:
                Color(0, 0, 0, 1)
                touch.ud['line'] = Line(points=[touch.x, touch.y], width=12)
        
    def on_touch_move(self, touch):
        if touch.ud.get('line'):
            touch.ud['line'].points += [touch.x, touch.y]
            
    def clear_canvas(self):
        self.canvas.clear()
        # re-draw white background after clearing
        with self.canvas:
            Color(1, 1, 1, 1)
            self._bg_rect = Rectangle(pos=self.pos, size=self.size)

class HandwritingApp(App):
    def build(self):
        # 리소스에서 모델 경로 탐색 (APK에 동봉)
        # 우선순위: TFLite -> Keras H5
        tflite_path = resource_find('mnist_cnn.tflite')
        h5_path = resource_find('mnist_cnn.h5')
        self.model_backend = None
        self.model_path = None
        if _tflite_available and tflite_path:
            self.model_backend = 'tflite'
            self.model_path = tflite_path
        elif _keras_available and h5_path:
            self.model_backend = 'keras'
            self.model_path = h5_path

        self.layout = BoxLayout(orientation='vertical')
        self.drawing_widget = DrawingWidget(size_hint=(1, 0.7))
        self.layout.add_widget(self.drawing_widget)

        # 버튼들을 담을 수평 레이아웃
        button_layout = BoxLayout(size_hint=(1, 0.1), spacing=10, padding=10)
        self.clear_button = Button(text='지우기')
        self.predict_button = Button(text='예측')
        self.visualize_button = Button(text='신경망 시각화')
        self.clear_button.bind(on_press=self.clear_canvas)
        self.predict_button.bind(on_press=self.predict)
        self.visualize_button.bind(on_press=self.visualize_network)
        
        button_layout.add_widget(self.clear_button)
        button_layout.add_widget(self.predict_button)
        button_layout.add_widget(self.visualize_button)
        self.layout.add_widget(button_layout)

        # 결과 레이블
        self.result_label = Label(text='', size_hint=(1, 0.2))
        self.layout.add_widget(self.result_label)

        # 모델 로딩
        threading.Thread(target=self.load_model, daemon=True).start()
        
        return self.layout

    def load_model(self):
        def _set_msg(msg):
            Clock.schedule_once(lambda dt: setattr(self.result_label, 'text', msg))

        if not self.model_path or not self.model_backend:
            _set_msg('모델 파일을 찾을 수 없습니다. (mnist_cnn.tflite 또는 mnist_cnn.h5)')
            return

        try:
            if self.model_backend == 'tflite':
                interpreter = Interpreter(model_path=self.model_path)
                interpreter.allocate_tensors()
                self._tflite_interpreter = interpreter
                self._tflite_input = interpreter.get_input_details()
                self._tflite_output = interpreter.get_output_details()
                _set_msg('TFLite 모델 로딩 완료')
            elif self.model_backend == 'keras':
                self.model = models.load_model(self.model_path)
                _set_msg('Keras 모델 로딩 완료')
        except Exception as e:
            _set_msg(f'모델 로딩 실패: {e}')

    def clear_canvas(self, instance):   
        self.drawing_widget.clear_canvas()
        self.result_label.text = ""

    def predict(self, instance):
        # 모델 준비 확인
        backend_ready = (self.model_backend == 'tflite' and hasattr(self, '_tflite_interpreter')) or (
            self.model_backend == 'keras' and hasattr(self, 'model')
        )
        if not backend_ready:
            self.result_label.text = '모델 로딩 중...'
            return

        # 캔버스를 이미지로 변환 (RGBA -> L, invert, resize 28x28)
        core_image = self.drawing_widget.export_as_image()
        texture = core_image.texture
        pil = Image.frombytes('RGBA', texture.size, texture.pixels)
        pil = pil.convert('L')
        pil = ImageOps.invert(pil)
        pil = pil.resize((28, 28), Image.LANCZOS)

        img_array = np.array(pil, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, 28, 28, 1)

        if self.model_backend == 'tflite':
            input_detail = self._tflite_input[0]
            output_detail = self._tflite_output[0]
            # adjust dtype if needed
            input_tensor = img_array.astype(input_detail['dtype'])
            self._tflite_interpreter.set_tensor(input_detail['index'], input_tensor)
            self._tflite_interpreter.invoke()
            predictions = self._tflite_interpreter.get_tensor(output_detail['index'])
        else:
            predictions = self.model.predict(img_array)

        # 결과 표시
        top3 = np.argsort(predictions[0])[::-1][:3]
        result_text = f"예측: {top3[0]}\n"
        result_text += "\n".join([
            f"{i+1}위: {n} ({predictions[0][n]*100:.1f}%)" for i, n in enumerate(top3)
        ])
        self.result_label.text = result_text

    def visualize_network(self, instance):
        """신경망 구조를 시각화합니다."""
        try:
            # 필요 시에만 matplotlib 임포트 (앱 시작 시 의존성 강제 방지)
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import matplotlib.font_manager as fm
            
            # 한글 폰트 설정 - 더 강력한 방법
            import platform
            system = platform.system()
            
            if system == 'Windows':
                # Windows에서 사용 가능한 한글 폰트들
                font_candidates = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
            elif system == 'Darwin':  # macOS
                font_candidates = ['AppleGothic', 'Arial Unicode MS']
            else:  # Linux
                font_candidates = ['DejaVu Sans', 'Liberation Sans']
            
            # 사용 가능한 폰트 찾기
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            selected_font = None
            
            for font in font_candidates:
                if font in available_fonts:
                    selected_font = font
                    break
            
            if selected_font:
                plt.rcParams['font.family'] = selected_font
            else:
                # 폰트를 찾지 못한 경우 기본 설정
                plt.rcParams['font.family'] = 'DejaVu Sans'
            
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
            
            plt.switch_backend('Agg')  # GUI 없이 렌더링

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 8)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # 레이어 정의 (CNN 구조)
            layers = [
                {'name': 'Input\n28x28', 'x': 1, 'y': 4, 'width': 1, 'height': 2, 'color': 'lightblue'},
                {'name': 'Conv2D\n32 filters\n3x3', 'x': 2.5, 'y': 3.5, 'width': 1.5, 'height': 1, 'color': 'lightgreen'},
                {'name': 'MaxPool\n2x2', 'x': 2.5, 'y': 2, 'width': 1.5, 'height': 0.8, 'color': 'lightcoral'},
                {'name': 'Conv2D\n64 filters\n3x3', 'x': 4.5, 'y': 3.5, 'width': 1.5, 'height': 1, 'color': 'lightgreen'},
                {'name': 'MaxPool\n2x2', 'x': 4.5, 'y': 2, 'width': 1.5, 'height': 0.8, 'color': 'lightcoral'},
                {'name': 'Flatten', 'x': 6.5, 'y': 3, 'width': 1, 'height': 1, 'color': 'lightyellow'},
                {'name': 'Dense\n128', 'x': 7.5, 'y': 4, 'width': 1, 'height': 1.5, 'color': 'lightpink'},
                {'name': 'Dense\n10', 'x': 7.5, 'y': 2, 'width': 1, 'height': 1, 'color': 'lightpink'}
            ]
            
            # 레이어 그리기
            for layer in layers:
                rect = patches.Rectangle(
                    (layer['x'], layer['y']), 
                    layer['width'], 
                    layer['height'],
                    linewidth=2, 
                    edgecolor='black', 
                    facecolor=layer['color'],
                    alpha=0.7
                )
                ax.add_patch(rect)
                
                # 레이어 이름
                ax.text(
                    layer['x'] + layer['width']/2, 
                    layer['y'] + layer['height']/2,
                    layer['name'], 
                    ha='center', 
                    va='center', 
                    fontsize=8, 
                    weight='bold',
                    fontfamily='monospace'  # 고정폭 폰트 사용
                )
            
            # 연결선 그리기
            connections = [
                (1.5, 4, 2.5, 4),  # Input -> Conv1
                (3.25, 3.5, 3.25, 2.4),  # Conv1 -> MaxPool1
                (3.25, 2, 4.5, 2),  # MaxPool1 -> Conv2
                (5.25, 3.5, 5.25, 2.4),  # Conv2 -> MaxPool2
                (5.25, 2, 6.5, 3),  # MaxPool2 -> Flatten
                (7, 3, 7.5, 4.75),  # Flatten -> Dense1
                (8, 4.75, 8, 2.5),  # Dense1 -> Dense2
            ]
            
            for x1, y1, x2, y2 in connections:
                ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.1, head_length=0.1, 
                        fc='black', ec='black', alpha=0.6)
            
            # 제목과 설명
            ax.text(5, 7.5, 'MNIST CNN Neural Network Structure', ha='center', fontsize=16, weight='bold', fontfamily='sans-serif')
            ax.text(5, 6.8, 'Convolutional Neural Network for Handwritten Digit Recognition', ha='center', fontsize=12, fontfamily='sans-serif')
            
            # 범례
            legend_elements = [
                patches.Patch(color='lightblue', label='Input Layer'),
                patches.Patch(color='lightgreen', label='Convolutional Layer'),
                patches.Patch(color='lightcoral', label='Pooling Layer'),
                patches.Patch(color='lightyellow', label='Flatten'),
                patches.Patch(color='lightpink', label='Dense Layer')
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
            
            # 저장 및 표시
            plt.tight_layout()
            plt.savefig('neural_network_visualization.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            self.result_label.text = "Neural Network Visualization Complete!\nCheck neural_network_visualization.png file."
            
        except Exception as e:
            self.result_label.text = f"Visualization Error: {e}"

if __name__ == '__main__':
    HandwritingApp().run()