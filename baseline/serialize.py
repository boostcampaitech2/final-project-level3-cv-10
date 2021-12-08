import torch
# import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

from models.deeplabv3 import deeplabv3_mobilenet_v3_large

model = deeplabv3_mobilenet_v3_large(pretrained=False, pretrained_backbone=False)
model.state_dict(torch.load('model_weights.pth'))
model.eval()
example = torch.rand(1, 3, 640, 480)
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("deeplabv3_scripted_final_prj.ptl")
