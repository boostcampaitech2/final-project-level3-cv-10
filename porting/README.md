# Pytorch to Android Porting
### - Pytorch로 학습한 model(.pt/.pth)을 ONNX(.onnx), Tensorflow(.pb), TFLite(.tflite) 파일로 변환
### - FP16과 Optimization option을 활성화
---

## 실행 방법
- porting.ipynb 실행

---
###  Reference
- https://github.com/sithu31296/PyTorch-ONNX-TFLite#onnx-model-inference
- https://github.com/sithu31296/PyTorch-ONNX-TFLite
- https://www.tensorflow.org/lite/performance/post_training_quantization?hl=ko

---
### 업데이트 내역
21.12.08
- Pytorch to TFLite 변환 코드 구현
 
21.12.09
- baseline/models/deeplabv3.py 의 deeplabv3-mobileNetv3-large 모델을 활용하도록 변경
- **deeplabv3.py의 import경로를 각자 환경에 맞게 고쳐줘야 합니다.**
---
### 주의할 점
- porting.ipynb 파일을 따라 진행하는 과정에서, 여러 library를 install 해야합니다. 모든 과정은 https://github.com/sithu31296/PyTorch-ONNX-TFLite#onnx-model-inference 를 통해 진행하였기 때문에, 해당 링크에서 install해야 하는 library에 대해 알 수 있습니다.
- onnx-tf의 경우 위 링크를 따라가도 적용이 잘 안될 수 있음. 해결방법은 https://github.com/onnx/onnx-tensorflow/issues/550 참고
