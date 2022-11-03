# author:c19h
# datetime:2022/11/2 16:16
import torch

from data_generation import Event_DataModule
from trainer import EvNetModel
import json
import pytorch_lightning as pl
from torchsummary import summary
import numpy as np
# %%
train_params = json.load(open('./pretrained_models/DVS128_10_24ms_dwn/all_params.json', 'r'))
# %%
data_params = train_params['data_params']
backbone_params = train_params['backbone_params']
clf_params = train_params['clf_params']
optim_params = train_params['optim_params']
data = Event_DataModule(**data_params)
# %%
features = next(iter(data.train_dataloader()))
#%%
model = EvNetModel(backbone_params=backbone_params,
                   clf_params=clf_params,
                   optim_params=optim_params,
                   loss_weights=None)
summary(model,input_size=[(20,128,109,144), (20,128,109,2)])