# built-in
import pdb
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

logger = logging.getLogger('train')

class_names = [
    'background', 'alley_crosswalk', 'alley_damaged', 'alley_normal',
    'alley_speed_bump', 'bike_lane', 'braille_guide_blocks_damaged',
    'braille_guide_blocks_normal', 'caution_zone_grating',
    'caution_zone_manhole', 'caution_zone_repair_zone', 'caution_zone_stairs',
    'caution_zone_tree_zone', 'roadway_crosswalk', 'roadway_normal',
    'sidewalk_asphalt', 'sidewalk_blocks', 'sidewalk_cement',
    'sidewalk_damaged', 'sidewalk_other', 'sidewalk_soil_stone',
    'sidewalk_urethane'
]


def train_model(
    model,
    dataloaders,
    optimizer,
    criterion,
    device,
    num_epochs=5,
    scheduler=None,
    epoch_div=5,
    train_process=['train', 'val'],
    with_amp: bool = False,
):

    autocast_enabled = False
    if with_amp:
        scaler = GradScaler()
        autocast_enabled = True

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1:>2}/{num_epochs} ----------")

        for phase in train_process:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
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
                # masks = torch.stack(masks).to(images)
                # pdb.set_trace()

                _time = time.perf_counter()
                with torch.set_grad_enabled(phase == 'train'), autocast(
                        enabled=autocast_enabled):
                    outputs = model.forward(images)['out']

                    # pdb.set_trace()
                    loss = criterion(outputs, masks)
                    epoch_loss += loss.cpu().item()

                    if phase == 'train':
                        if not with_amp:
                            loss.backward()
                            optimizer.step()
                        else:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                cost_time += time.perf_counter() - _time

                _time = time.perf_counter()
                masks = masks.cpu().numpy()
                # outputs = outputs.detach().cpu().numpy()
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()

                hist = add_hist(hist, masks, outputs, n_class=22)
                hist_time += time.perf_counter() - _time

            _time = time.perf_counter()
            epoch_loss /= n_iter
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            IoU_by_class = {
                class_name: round(IoU, 4)
                for class_name, IoU in zip(class_names, IoU)
            }
            logger.info(
                f"{phase.upper():5}: Epoch [{epoch+1}/{num_epochs}], Loss: {round(epoch_loss, 4)}, mIoU: {round(mIoU, 4)}"
            )
            logger.info(f'IoU by class :')
            max_row = 4
            n_row = 0
            template = ''
            for key, values in IoU_by_class.items():
                template += f"| {key:28} | {values:.4f} |"
                n_row += 1
                if max_row == n_row:
                    logger.info(template)
                    template = ''
                    n_row = 0
            logger.info(template)

            eval_time += time.perf_counter() - _time
            logger.info(
                f"{phase.upper():5} TIME: Cost - {cost_time:.2f}s, Hist - {hist_time:.2f}s, Eval - {eval_time:.2f}s"
            )
