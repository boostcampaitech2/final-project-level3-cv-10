# Porting(모델 경량화 및 최적화, TFLite로 변환)

모델 경량화 및 최적화와, Pytorch 모델을 ONNX, TFLite 모델로 변환하는 기능을 담은 폴더입니다.

변환 과정은 https://github.com/sithu31296/PyTorch-ONNX-TFLite#onnx-model-inference 를 참고했습니다.

---
model : model weight를 load하기 위한 .pth 파일 저장(활용할 때 경로지정 필요)

---
### 업데이트 내역
21.12.08
- Pytorch to TFLite 변환 코드 구현

---
### 주의할 점
- porting.ipynb 파일을 따라 진행하는 과정에서, 여러 library를 install 해야합니다. 모든 과정은 https://github.com/sithu31296/PyTorch-ONNX-TFLite#onnx-model-inference 를 통해 진행하였기 때문에, 해당 링크에서 install해야 하는 library에 대해 알 수 있습니다.
