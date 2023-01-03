import moco.builder
from torchvision import models
import torch

# parameters for ACO
moco_dim = 128
moco_k = 20480
moco_m = 0.99
moco_t = 0.7
seq_length = 10
mlp = True
ckpt_dir = None

# set target resnet
model = models.resnet34()
model.train()

b = models.resnet34()

# load checkpoint
# ckpt = torch.load('/home/ywang3/workplace/ACO/code/seq10/checkpoint_0116.pth.tar') 
# print(f"The checkpoint is on arch : {ckpt['arch']}")
# ckpt = ckpt['state_dict']
ckpt = b.state_dict()
from collections import OrderedDict
d = OrderedDict()

'''
for name, _ in model.named_parameters():
    if f'encoder_q.{name}' in ckpt.keys():
        d[name] = ckpt[f'encoder_q.{name}']
    else:
        print(f'Missing From Ckpt : {name}')
'''

model.load_state_dict(ckpt)
