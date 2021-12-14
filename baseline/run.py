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

# import nni
import wandb
from torch.utils.data import DataLoader

# Custom
from models.deeplabv3 import deeplabv3_mobilenet_v3_large
from train import train
from datasets import CustomDataset
from datasets import get_train_transform, get_valid_transform

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


def main(args):

    TRAIN_PROCESS = ['train', 'val']
    TRAIN_JSON = {
        "train":
        # "/opt/ml/Git/final-project-level3-cv-10/data/sample.train.json",
        '/opt/ml/data/final-project/images/train.json',
        "val":
        # "/opt/ml/Git/final-project-level3-cv-10/data/sample.valid.json",
        '/opt/ml/data/final-project/images/val.json',
    }

    wandb.config = {**args}

    defined_transforms = {
        'train': get_train_transform(),
        'val': get_valid_transform(),
    }

    recycle_dataset = {
        x: CustomDataset(
            data_json=TRAIN_JSON[x],
            transforms=defined_transforms[x],
        )
        for x in TRAIN_PROCESS
    }

    _time = time.perf_counter()
    dataloaders = {
        x: DataLoader(
            recycle_dataset[x],
            batch_size=args['batch_size'],
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
    model = deeplabv3_mobilenet_v3_large(pretrained=False,
                                         pretrained_backbone=False,
                                         aux_loss=False)
    model.load_state_dict(torch.load('model_weights.pth'))
    model.to(device)
    logger.info(f"Model progress. {time.perf_counter() - _time:.4f}s")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args['lr'],
    )

    loss_func = nn.CrossEntropyLoss()

    # training
    N_EPOCH = args['epochs']
    valid_mIoU = 0
    for epoch in range(N_EPOCH):
        logger.info(f"Epoch {epoch + 1:>2}/{N_EPOCH} ----------")
        metric = train(
            epoch=epoch,
            model=model,
            dataloaders=dataloaders,
            optimizer=optimizer,
            device=device,
            criterion=loss_func,
            # scheduler=scheduler,
            train_process=TRAIN_PROCESS,
            autocast_enabled=args['fp16'],
        )
        valid_mIoU = metric['val']['mIoU']
        valid_IoU = metric['val']['IoU']
        valid_loss = metric['val']['loss']

        train_mIoU = metric['train']['mIoU']
        train_IoU = metric['train']['IoU']
        train_loss = metric['train']['loss']

        wandb.log({
            "train/loss": train_loss,
            "train/mIoU": train_mIoU,
            "val/loss": valid_loss,
            "val/mIoU": valid_mIoU,
            **train_IoU,
            **valid_IoU,
        })
    #     nni.report_intermediate_result(valid_mIoU)
    # nni.report_final_result(valid_mIoU)

    torch.save(model.state_dict(), 'model_weights.after15.pth')

    return valid_mIoU / N_EPOCH


if __name__ == "__main__":

    wandb.init(project="final-prj", entity="booduck4ai")

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # params = nni.get_next_parameter()
    params = dict(epochs=30, lr=5e-5, batch_size=32, fp16=True)
    main(params)

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
