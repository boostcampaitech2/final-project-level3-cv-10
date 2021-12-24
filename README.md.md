# README.md

# **Recognition of a walking danger zone for the visually impaired using segmentation**

---

# **권예환_T2012, 권용범_T2013, 김경민_T2018, 심재빈_T2124, 정현종_T2208**

## **Overview**

---

![ezgif.com-gif-maker (1).gif](README%20md%20f4dd3370b16740868aa4959bb7c01681/ezgif.com-gif-maker_(1).gif)

시각장애인 분들은 여전히 길에서 보행 시에 많은 위험에 노출됩니다. 시각의 혜택을 받지 못하기 때문에 진행 방향을 잡는 것부터가 어렵습니다. 이를 위해 인도 곳곳에는 점자블록이 설치되어 길 안내 용도로 사용되는데, 이 또한 직접 점자블록 위를 걸을 때 의미가 있는 것이고, 점자블록 밖에 서 있는 경우는 블록이 어디 있는지 찾기 쉽지 않습니다.

이러한 이유로 저희는 시각장애인의 눈을 대신하여, Semantic Segmentation을 통해 인도의 지면 정보를 파악하고, 시각장애인에게 유용한 정보를 제공하는 앱을 구현하고자 하였습니다.

## Application

---

## Structure

---

```
final-project-level3-cv-10
├─baseline  # Deeplabpv3
├─android   # android codes
└─porting   # torch → onnx → Tensorflow → TFlite
```

## Descrption

---

- Segmentation
- On-Demand ML
- Realtime
- TTS

## Install APK

---

Android (only)

- Install APK on Device

*ios not supported

## Running on Android Studio

---

Settings

- JDK : 1.8.0
- Android AVD : Pixel4a, API Level 30

## Model

---

Deeplab v3

- backbone : MobileNet V3-Large
- Optimizaton : float32 → FP16

## Dataset

---

인도(人道) 보행 영상 : [https://aihub.or.kr/aidata/136](https://aihub.or.kr/aidata/136) 

- SurfaceMasking Dataset

## License

---

Dataset : CC-BY-SA

Application Icon : CC-BY

Pytorch : Facebook Copyright

Onnx : Apache License 2.0

Tensorflow : Apache License 2.0

Tensorflow Lite : Apache License 2.0

Android Studio : 

- Binaries : Freeware
- Source code : Apache License 2.0