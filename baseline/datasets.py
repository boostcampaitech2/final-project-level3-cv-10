import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pycocotools.coco import COCO

from utils import timer
# class_colormap = pd.read_csv(
#     "/opt/ml/segmentation/baseline_code/class_dict.csv")
# palette = class_colormap.iloc[:, 1:].values.astype(np.uint8)

CLASSES = [
    'background', 'alley_crosswalk', 'alley_damaged', 'alley_normal',
    'alley_speed_bump', 'bike_lane', 'braille_guide_blocks_damaged',
    'braille_guide_blocks_normal', 'caution_zone_grating',
    'caution_zone_manhole', 'caution_zone_repair_zone', 'caution_zone_stairs',
    'caution_zone_tree_zone', 'roadway_crosswalk', 'roadway_normal',
    'sidewalk_asphalt', 'sidewalk_blocks', 'sidewalk_cement',
    'sidewalk_damaged', 'sidewalk_other', 'sidewalk_soil_stone',
    'sidewalk_urethane'
]


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


def get_valid_transform():
    return A.Compose([
        A.Resize(240, 427),
        A.RandomCrop(240, 320),
        A.Normalize(),
        ToTensorV2()
    ])


@timer
class CustomDataset(Dataset):
    """COCO format"""
    def __init__(
        self,
        data_json,
        mode='train',
        transforms=None,
        image_root_path: str = '/opt/ml/data/final-project/images',
        # image_root_path: str = '/opt/ml/data/final-project/train_images',
    ):
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.coco = COCO(data_json)
        self.cocoImgIds = self.coco.getImgIds()
        self.image_root_path = image_root_path

    def __getitem__(
        self,
        index: int,
    ):
        # dataset이 index되어 list처럼 동작
        image_infos = self.coco.loadImgs(self.cocoImgIds[index])[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(
            os.path.join(self.image_root_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        # images /= 255.0

        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            masks = np.zeros((image_infos["height"], image_infos["width"]),
                             np.uint8)
            # General trash = 1, ... , Cigarette = 10
            # anns = sorted(anns,
            #               key=lambda idx: len(idx['segmentation'][0]),
            #               reverse=True)
            # pdb.set_trace()

            for ann in anns:
                annCatId = ann['category_id']

                masks[self.coco.annToMask(ann) == 1] = annCatId

            # transform -> albumentations 라이브러리 활용
            if self.transforms is not None:
                transformed = self.transforms(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos

        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transforms is not None:
                transformed = self.transforms(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.cocoImgIds)