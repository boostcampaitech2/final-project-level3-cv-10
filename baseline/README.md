수정 중에 있습니다.

# MobileNetV3-DeepLabV3
Our model is based on [Pytorch's DeepLabV3](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py).

## Introduction
본 팀은 인도 보행 안전을 위해 보행 표면의 종류를 파악하여 사용자에게 위험 정보를 전달 하려고 하였습니다. 따라서 보행 표면 정보를 인식하기 위해 segmentation을 활용하여 전달하고자 하였습니다. 모델 선정 기준은 아래와 같습니다.

모바일 기기에서 작동할 수 있으며, 자료가 있는 모델을 우선 선정하였습니다. Semantic segmentation task에서 잘 알려진 DeepLabV3를 활용하였습니다. 본 팀은 모바일 변환과 구동을 중점적으로 하여 진행하였습니다.

## Result
| Backbone          | Seg. Head | Pretrain | Reduced_tail* | Crop Size | mIoU** | #params(M) | MACs(G) |
| :---------------- | :-------: | :------: | :-----------: | :-------: | :----: | :--------: | :-----: |
| MobileNetV3-Small | DeepLabV3 | ImageNet |     False     |  320x240  |   -    |    6.13    |  1.69   |
| MobileNetV3-Small | DeepLabV3 | ImageNet |     True      |  320x240  |   -    |    3.51    |   1.0   |
| MobileNetV3-Large | DeepLabV3 | ImageNet |     False     |  320x240  |   -    |   11.03    |  2.91   |
| MobileNetV3-Large | DeepLabV3 | ImageNet |     True      |  320x240  |   -    |    6.11    |  1.68   |

*: MobileNetV3의 Last Stage의 channel을 half-down 하는 것을 말합니다.

**: -

## Data
[AI Hub](https://aihub.or.kr/)의 [인도(人道) 보행 영상](https://aihub.or.kr/aidata/136)

## Model
```
.
├── datasets.py
├── models
│   ├── deeplabv3.py
│   └── mobilenetv3.py
├── run.py
├── train.py
└── utils.py
```
### Backbone
모바일에서 원활한 구동이 가능해야 함으로 MobileNetV3 선택하였습니다. 또한 학습 효율을 위해 pytorch에서 제공하는 ImageNet pretrained 모델을 사용하여 학습을 진행했습니다.

- Small
- Large

Reduced_tail 옵션을 사용한 경우,기존 pretrained weight에서 last stage의 channel 크기가 상이하여 해당 부분을 제거하고 load 하였습니다.

### Segmentation Head

학습시 auxiliary loss를 활성화 하여 진행했습니다.


## Usage
```
python run.py
```
