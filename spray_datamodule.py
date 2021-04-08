import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn.model_selection import train_test_split

class spraydata(torch.utils.data.Dataset):
  def __init__(self,data):
    self.data=data
  def __getitem__(self,index):
    data = self.data[index,:7]
    return data
  def __len__(self):
    return self.data.shape[0]

class spray_dm(LightningDataModule):
  
  def __init__(self,hparams):
    super().__init__()
    self.hparams = hparams

  def prepare_data(self):
    return

  def setup(self,stage=None):
    self.train_data = spraydata(torch.load(self.hparams.data_path+'train_data.pt'))
    self.val_data = spraydata(torch.load(self.hparams.data_path+'test_data.pt'))

  def train_dataloader(self):
    return DataLoader(self.train_data, batch_size=self.hparams.batch_size, num_workers=8, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.hparams.batch_size, num_workers=8)

  def test_dataloader(self):
    return DataLoader(self.val_data, batch_size=self.hparams.batch_size, num_workers=8)