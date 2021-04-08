from argparse import ArgumentParser
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from spray_injection import spray_injection
from spray_datamodule import spray_dm

def main(hparams):
    checkpoint_callback = ModelCheckpoint(
    dirpath=os.getcwd(),
    save_top_k=True,
    verbose=True,
    monitor='val_loss',
    mode='min',
    save_last=True,
    prefix='')
    model = spray_injection(hparams)
    dm = spray_dm(hparams)
    trainer = Trainer.from_argparse_args(hparams,checkpoint_callback=checkpoint_callback,
                         auto_select_gpus = True,
                         progress_bar_refresh_rate=1)
    trainer.fit(model, dm)                           
if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser = spray_injection.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)