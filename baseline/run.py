## built-in
import pdb
import time
import logging
from datetime import datetime

## 3rd
# torch
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim import lr_scheduler

from torch.utils.data import DataLoader

# Custom
from models.deeplabv3 import deeplabv3_mobilenet_v3_large
from train import train_model
from datasets import CustomDataset
from datasets import get_train_transform

now = datetime.now()
cur_time_str = now.strftime("%d%m%Y_%H%M")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)-8s %(levelname)-6s %(message)s',
    datefmt='%m-%d %H:%M',
    filename=f'{cur_time_str}.log',
    filemode='w')

logger = logging.getLogger("train")


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    config = dict(epochs=25, batch_size=24, lr=1e-4)

    # NUM_CLASS = 10
    # CLASSES = ("Background", "General trash", "Paper", "Paper pack", "Metal",
    #            "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery",
    #            "Clothing")
    TRAIN_PROCESS = ['train', 'val']
    TRAIN_JSON = {
        "train":
        f"/opt/ml/Git/final-project-level3-cv-10/data/sample.train.json",
        "val":
        f"/opt/ml/Git/final-project-level3-cv-10/data/sample.valid.json",
    }

    defined_transforms = {
        'train': get_train_transform(),
        'val': get_train_transform(),
    }

    _time = time.perf_counter()
    recycle_dataset = {
        x: CustomDataset(
            data_json=TRAIN_JSON[x],
            transforms=defined_transforms[x],
        )
        for x in TRAIN_PROCESS
    }
    logger.info(f"Dataset progress. {time.perf_counter() - _time:.4f}s")

    _time = time.perf_counter()
    dataloaders = {
        x: DataLoader(
            recycle_dataset[x],
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        for x in TRAIN_PROCESS
    }
    logger.info(f"Dataloader progress. {time.perf_counter() - _time:.4f}s")
    # pdb.set_trace()

    _time = time.perf_counter()
    model = deeplabv3_mobilenet_v3_large()
    model.to(device)
    logger.info(f"Model progress. {time.perf_counter() - _time:.4f}s")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
    )

    loss_func = nn.CrossEntropyLoss()

    # training
    train_model(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        device=device,
        criterion=loss_func,
        # scheduler=scheduler,
        num_epochs=config['epochs'],
        train_process=TRAIN_PROCESS,
    )
    torch.save(model.state_dict(), 'model_weights.pth')


if __name__ == "__main__":

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main()

    # import wandb

    # api = wandb.Api()

    # run is specified by <entity>/<project>/<run_id>
    # run = api.run("booduck4ai/effv2s_pretrained/")

    # sweep_config = {
    #     'method': 'grid',  #grid, random
    #     'metric': {
    #         'name': 'person_loss',
    #         # 'name': 'label_loss',
    #         'goal': 'minimize'
    #     },
    #     'parameters': {
    #         'epochs': {
    #             'values': [20]
    #         },
    #         'batch_size': {
    #             'values': [244]
    #         },
    #     }
    # }
    # sweep_id = wandb.sweep(sweep_config, project='test')

    # wandb.agent(sweep_id, main)
