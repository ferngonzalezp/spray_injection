import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch.nn.functional as F
import matplotlib.pyplot as plt

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data,0,mode='fan_out')
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data,0,mode='fan_out')

def calc_gradient_penalty(netD, real_data, generated_data,l = 10):
    # GP strengt
    LAMBDA = l

    b_size = real_data.size()[0]

    # Calculate interpolation
    bs = [b_size]
    for i in range(len(real_data[0].shape)):
        bs.append(1)
    alpha = torch.rand(bs).to(real_data.device)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.type_as(real_data)

    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()

class generator(nn.Module):
  def __init__(self,hparams):
    super().__init__()
    self.generator = nn.Sequential(*mlp(hparams.n_layers,
                                            hparams.n_neurons,
                                            hparams.latent_dim,
                                            hparams.output_dim))
  
  def forward(self,x):
      return self.generator(x)

class discriminator(nn.Module):
  def __init__(self,hparams):
    super().__init__()
    self.discriminator = nn.Sequential(*mlp(hparams.n_layers,
                                            hparams.n_neurons,
                                            hparams.output_dim,
                                            1))
  
  def forward(self,x):
      return self.discriminator(x)
  
def mlp(n_layers,n_neurons,input_dim,output_dim):
    layers = []
    for i in range(n_layers):
      if i > n_layers-2:
        n_neurons = output_dim
        layers.append(nn.Linear(input_dim,n_neurons))
        input_dim = n_neurons
        layers.append(nn.BatchNorm1d(n_neurons))
        layers.append(nn.GELU())
      else:
        layers.append(nn.Linear(input_dim,n_neurons))
        input_dim = n_neurons
        layers.append(nn.BatchNorm1d(n_neurons))
        layers.append(nn.GELU())
    return layers

class spray_injection(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams
    self.save_hyperparameters()
    self.generator = generator(self.hparams)
    self.generator.apply(weights_init)
    self.discriminator = discriminator(self.hparams)
    self.discriminator.apply(weights_init)
    self.pz = torch.distributions.log_normal.LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))

  @staticmethod
  def add_model_specific_args(parent_parser):
      parser = ArgumentParser(parents=[parent_parser], add_help=False)
      parser.add_argument('--batch_size', type=int, default=64)
      parser.add_argument('--lr', type=float, default=1e-3)
      parser.add_argument('--latent_dim', type=int, default=100)
      parser.add_argument('--output_dim', type=int, default=7)
      parser.add_argument('--n_layers', type=int, default=4)
      parser.add_argument('--n_neurons', type=int, default=64)
      parser.add_argument('--ckpt_path', default = './checkpt.ckpt', type = str)
      parser.add_argument('--data_path', default = './sprayData/', type = str)
      return parser
  
  def loss(self,x,y):
    return -torch.mean((y)) + torch.mean((x)) 

  def forward(self,x):
    return self.generator(x)

  def training_step(self,batch,batch_idx,optimizer_idx):
    y = batch
    z = self.pz.sample(torch.Size([y.shape[0],self.hparams.latent_dim])).squeeze(-1).type_as(y)
    x = self(z)
    if optimizer_idx==0:
      grad_penalty = calc_gradient_penalty(self.discriminator,y,x)
      loss = self.loss(self.discriminator(x),self.discriminator(y)) + grad_penalty
      self.log('disc_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
      return loss
    if optimizer_idx==1:
      loss = -torch.mean(self.discriminator(x))
      self.log('gen_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
      return loss

  def validation_step(self,batch,batch_idx):
    y = batch
    z = torch.randn((y.shape[0],self.hparams.latent_dim)).type_as(y)
    x = self(z)
    loss = torch.abs(self.loss(self.discriminator(x),self.discriminator(y)))
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss 
  
  def configure_optimizers(self):
      opt_G =  torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0,0.9))
      opt_D =  torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0,0.9))
      return [opt_D, opt_G]
  
  def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
    optimizer.zero_grad()

  # Alternating schedule for optimizer steps (ie: GANs)
  def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
    if optimizer_idx == 0:
        if batch_nb % 1 == 0 :
           optimizer.step(closure=closure)

    if optimizer_idx == 1:
        if batch_nb % 1 == 0 :
           optimizer.step(closure=closure)