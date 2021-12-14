## Model

```
.
├── datasets.py
├── models
│   ├── deeplabv3.py
│   ├── mobilenetv3.py
│   └── quantization_mobilenetv3.py
├── run.py
├── train.py
└── utils.py
```

### Usage
```
python3 run.py
```

1. run.py 91번 line에서 optimizer 변경하시면 됩니다.

2. pretrained_backbone=True로 하면 ImageNet으로 pretrained된 모델을 불러옵니다.
```
model = deeplabv3_mobilenet_v3_large(pretrained=False,
                                     pretrained_backbone=True,
                                     aux_loss=False)
```



### Image transfrom
```
# dataset.py; L31
def get_train_transform():
    return A.Compose([
        # A.Resize(360, 640),
        # A.RandomCrop(360, 480),
        A.Resize(240, 427),
        A.RandomCrop(240, 320),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])
 ```