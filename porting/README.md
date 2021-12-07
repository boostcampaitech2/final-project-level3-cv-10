# Porting(모델 경량화 및 최적화, TFLite로 변환)

모델 경량화 및 최적화와, Pytorch 모델을 ONNX, TFLite 모델로 변환하는 기능을 담은 폴더입니다.

변환 과정은 https://github.com/sithu31296/PyTorch-ONNX-TFLite#onnx-model-inference 를 참고했습니다.

---
model : model weight를 load하기 위한 .pth 파일 저장(활용할 때 경로지정 필요)

---
21.12.08
- Pytorch to TFLite 변환 코드 구현
