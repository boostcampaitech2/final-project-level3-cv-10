# MobileNetV3-DeepLabV3
Our model is based on [Pytorch's DeepLabV3](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py).

## Introduction
본 팀은 인도 보행 안전을 위해 보행 노면의 종류를 파악하여 사용자에게 위험 정보를 전달 하려고 하였습니다. 따라서 보행 노면 정보를 인식하기 위해 segmentation을 활용하여 전달하고자 하였습니다. 모델 선정 기준은 아래와 같습니다.

모바일 기기에서 작동할 수 있으며, 자료가 있는 모델을 우선 선정하였습니다. Semantic segmentation task에서 잘 알려진 DeepLabV3를 활용하였습니다. 본 팀은 모바일 변환과 구동을 중점적으로 하여 진행하였습니다.

## Result
| Backbone          | Seg. Head | Pretrain | Reduced_tail* | Crop Size | #params(M) | MACs(G) |
| :---------------- | :-------: | :------: | :-----------: | :-------: | :--------: | :-----: |
| MobileNetV3-Small | DeepLabV3 | ImageNet |     False     |  320x240  |    6.13    |  1.69   |
| MobileNetV3-Small | DeepLabV3 | ImageNet |     True      |  320x240  |    3.51    |   1.0   |
| MobileNetV3-Large | DeepLabV3 | ImageNet |     False     |  320x240  |   11.03    |  2.91   |
| MobileNetV3-Large | DeepLabV3 | ImageNet |     True      |  320x240  |    6.11    |  1.68   |

*: MobileNetV3의 Last Stage의 channel 크기를 반으로 줄입니다.

mIoU는 전 모델 모두 약 0.46 정도의 성능을 가지고 있습니다. 큰 모델부터 시작하여 점진적으로 모델의 크기를 줄여가며 가능성을 확인하였습니다.

## Data
[AI Hub](https://aihub.or.kr/)의 [인도(人道) 보행 영상](https://aihub.or.kr/aidata/136)을 이용하여 진행하였습니다.

Semantic segmentation 데이터이며 총 21개의 CLASS로 구성이 되어 있습니다. (학습 시에는 배경을 포함한 22개의 CLASS를 이용했습니다.)

| name                         | #anns |
| :--------------------------- | ----: |
| alley_crosswalk              |  1419 |
| alley_damaged                |  1558 |
| alley_normal                 | 28473 |
| alley_speed_bump             |  1863 |
| bike_lane                    |  3585 |
| braille_guide_blocks_damaged |   208 |
| braille_guide_blocks_normal  | 11172 |
| caution_zone_grating         | 13840 |
| caution_zone_manhole         | 20946 |
| caution_zone_repair_zone     |   206 |
| caution_zone_stairs          |  1483 |
| caution_zone_tree_zone       | 10867 |
| roadway_crosswalk            |  3459 |
| roadway_normal               | 25758 |
| sidewalk_asphalt             |  2301 |
| sidewalk_blocks              | 40003 |
| sidewalk_cement              |  7809 |
| sidewalk_damaged             |  3042 |
| sidewalk_other               |   941 |
| sidewalk_soil_stone          |   834 |
| sidewalk_urethane            |  1618 |


학습을 위해 COCO format으로 변경하였으며, Stratified group k-fold를 적용하여 train, validation 데이터를 나누었습니다.

원본 이미지 크기는 1080p(16:9)이지만, 학습시 240p(4:3)로 조정하여 진행하였습니다.

## Model
- components
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

### Model config
```python
model = deeplabv3_mobilenet_v3(
    pretrained_backbone=True,
    aux_loss=True,
    small=True,
    reduced_tail=True,
    grid_mode=False,
)
```
- `pretrained_backbone`, (bool): MobileNetV3의 ImageNet pre-trained weight를 불러옵니다.
- `aux_loss`, (bool): Train 시 auxiliary loss를 사용하여 학습합니다.
- `small`, (bool): MobileNetV3-Small을 사용하여 진행합니다.
- `reduced_tail`, (bool): MobileNetV3의 last stage의 channel 크기를 반으로 감소합니다.
- `grid_mode`, (bool): DeepLabV3 Head 에서 나오는 output을 resize하지 않고 원본 크기를 그대로 사용합니다.

### Backbone

모바일에서 원활한 구동이 가능해야 함으로 MobileNetV3 선택하였습니다. 또한 학습 효율을 위해 pytorch에서 제공하는 ImageNet pretrained 모델을 사용하여 학습을 진행했습니다.

Reduced_tail 옵션을 사용한 경우,기존 pretrained weight에서 last stage의 channel 크기가 상이하여 해당 부분을 제거하고 load 하였습니다.

### Segmentation Head

학습시 auxiliary loss를 활성화 하여 진행했습니다.

grid output 사용시 정보 손실이 존재하지 않습니다. 사유는 DeepLabV3 head에서 output (15, 20) 자체의 resolution이 작고, 이것을 `F.interpolate`를 거쳐 원래 이미지 크기로 만들어 주기 때문에 mask 크기가 커져도 원본 output이 작기 때문에 정보량이 증가되지 않습니다. 따라서 output을 그대로 시용해서 mobile의 mask flatten time에서 장점을 얻게 됩니다.

## Usage
```
python run.py
```
학습 진행은 `run.py`를 통해 진행이 됩니다.

## References

1. https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py

2. https://aihub.or.kr/aidata/136
