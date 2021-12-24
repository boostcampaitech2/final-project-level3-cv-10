# built-in
import time
import logging

# 3rd
import numpy as np
from tqdm import tqdm

# torch
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

from utils import label_accuracy_score, add_hist
from datasets import CLASSES

logger = logging.getLogger('train')


def train(
    epoch: int,
    model: torch.nn.Module,
    dataloaders,
    optimizer,
    criterion,
    device,
    scheduler=None,
    train_process: list = ['train', 'val'],
    autocast_enabled: bool = False,
    aux: bool = False,
):

    if autocast_enabled:
        scaler = GradScaler()

    metric = {}

    for phase in train_process:

        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        n_iter = 0
        cost_time = 0
        eval_time = 0
        hist_time = 0
        total_loss = 0
        total_aux_loss = 0
        total_out_loss = 0

        hist = np.zeros((22, 22))
        for images, masks, image_infos in tqdm(dataloaders[phase]):
            n_iter += 1
            optimizer.zero_grad()

            images = torch.stack(images).to(device)
            masks = torch.stack(masks).long().to(device)

            _time = time.perf_counter()
            with torch.set_grad_enabled(phase == 'train'), autocast(
                    enabled=autocast_enabled):
                outputs = model.forward(images)
                if aux:
                    aux_outputs = outputs['aux']
                    outputs = outputs['out']

                    aux_loss = criterion(aux_outputs, masks)
                    out_loss = criterion(outputs, masks)
                    loss = out_loss + aux_loss
                    total_aux_loss += aux_loss.cpu().item()
                    total_out_loss += out_loss.cpu().item()
                else:
                    loss = criterion(outputs, masks)

                if phase == 'train':
                    if not autocast_enabled:
                        loss.backward()
                        optimizer.step()
                    else:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

            total_loss += loss.cpu().item()

            cost_time += time.perf_counter() - _time

            _time = time.perf_counter()
            masks = masks.cpu().numpy()
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            hist = add_hist(hist, masks, outputs, n_class=22)
            hist_time += time.perf_counter() - _time

        #
        total_loss /= n_iter
        _time = time.perf_counter()
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        logger.info(
            f"{phase.upper():5}: Epoch [{epoch+1}], Loss: {round(total_loss, 4)}, mIoU: {round(mIoU, 4)}"
        )
        print_metric(IoU)
        eval_time += time.perf_counter() - _time
        logger.info(
            f"{phase.upper():5} TIME: Cost - {cost_time:.2f}s, Hist - {hist_time:.2f}s, Eval - {eval_time:.2f}s"
        )

        if aux:
            total_aux_loss /= n_iter
            total_out_loss /= n_iter
            metric[phase] = dict(loss=total_loss,
                                 out_loss=total_out_loss,
                                 aux_loss=total_aux_loss,
                                 mIoU=mIoU,
                                 IoU={
                                     f"{phase}/{class_name}": round(IoU, 4)
                                     for class_name, IoU in zip(CLASSES, IoU)
                                 })
        else:
            metric[phase] = dict(loss=total_loss,
                                 mIoU=mIoU,
                                 IoU={
                                     f"{phase}/{class_name}": round(IoU, 4)
                                     for class_name, IoU in zip(CLASSES, IoU)
                                 })

    return metric


def test(model,
         dataloaders,
         optimizer,
         criterion,
         device,
         autocast_enabled: bool = False,
         phase: str = 'test'):

    model.eval()  # Set model to evaluate mode

    n_iter = 0
    cost_time = 0
    eval_time = 0
    hist_time = 0
    epoch_loss = 0

    hist = np.zeros((22, 22))
    for images, masks, image_infos in tqdm(dataloaders[phase]):
        n_iter += 1
        optimizer.zero_grad()

        images = torch.stack(images).to(device)
        masks = torch.stack(masks).long().to(device)

        _time = time.perf_counter()
        outputs = model.forward(images)['out']
        cost_time += time.perf_counter() - _time

        _time = time.perf_counter()
        masks = masks.cpu().numpy()
        outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        hist = add_hist(hist, masks, outputs, n_class=22)
        hist_time += time.perf_counter() - _time

    _time = time.perf_counter()
    acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
    print_metric(IoU)
    eval_time += time.perf_counter() - _time
    logger.info(
        f"{phase.upper():5} TIME: Cost - {cost_time:.2f}s, Hist - {hist_time:.2f}s, Eval - {eval_time:.2f}s"
    )


def print_metric(values):
    _metric_by_class = {
        class_name: round(IoU, 4)
        for class_name, IoU in zip(CLASSES, values)
    }
    logger.info(f'IoU by class :')
    max_row = 4
    n_row = 0
    template = ''
    for key, _values in _metric_by_class.items():
        template += f"| {key:28} | {_values:.4f} |"
        n_row += 1
        if max_row == n_row:
            logger.info(template)
            template = ''
            n_row = 0
    logger.info(template)