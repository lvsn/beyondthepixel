## python3 train.py --dataroot dataset/calibrated64 --name RescaleGammaHDRRescaleNoSolidAngles --rescale_HDR True --no-use_solid_angles_map

import tensorflow #Import tensorflow otherwise the code crashes
                    #https://github.com/pytorch/pytorch/issues/81140#issuecomment-1230524321
import numpy as np
import torch
from options.options import Options
from dataManagement.dataset import Dataset, find_size
from LightningModule.LitUnet import LitFixupUnet, buildUnet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
from glob import glob
from natsort import natsorted

import matplotlib.pyplot as plt

if __name__ ==  '__main__':
    parser = Options()
    opt = parser.parse()
    opt.phase = 'train'

    default_root_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(default_root_dir):
        os.makedirs(default_root_dir)

    train_dataset = Dataset(opt, phase='train')
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=1, shuffle=True)
    val_dataset = Dataset(opt, phase='val')
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=1, shuffle=False)

    sizey, sizex = find_size(train_data_loader)
    opt.size_y = sizey
    opt.size_x = sizex

    Unet_network = buildUnet(opt, )

    lightning_module = LitFixupUnet(Unet_network, opt)

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if opt.version == 'None':
        checkpoints = glob(save_dir + '/**/last.ckpt', recursive=True)
        if not checkpoints:
            raise ValueError("No checkpoint found at path: "+ save_dir)
        checkpoint = natsorted(checkpoints)[-1]
    else:
        checkpoints = glob(save_dir + "/lightning_logs/version_" + opt.version + "/**/last.ckpt", recursive=True)
        if not checkpoints:
            checkpoints = glob(save_dir + "/lightning_logs/version_" + opt.version + "/**/*.ckpt", recursive=True)
            if not checkpoints:
                raise ValueError("No checkpoint found at path: "+ save_dir + "/lightning_logs/version_" + opt.version)
        checkpoint = natsorted(checkpoints)[-1]

    #save every epoch's model
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, save_last=False)

    logger = pl.loggers.TensorBoardLogger(save_dir=default_root_dir, version=int(opt.version), name="lightning_logs")

    if opt.early_stop:
        trainer = pl.Trainer(max_epochs=opt.n_epoch, 
                            val_check_interval=1.0, 
                            accelerator="gpu", devices=1,
                            logger=logger,
                            default_root_dir=default_root_dir,
                            callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=500, mode="min")])
    else:
        trainer = pl.Trainer(max_epochs=opt.n_epoch, 
                            val_check_interval=1.0, 
                            logger=logger,
                            accelerator="gpu", devices=1, 
                            default_root_dir=default_root_dir,
                            callbacks=[checkpoint_callback])
    
    #save config file:
    #setattr(opt,'version',str(trainer.logger.version))
    #print(opt.version)
    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    filename = trainer.logger.log_dir + "/config.txt"
    parser.save(filename, opt)

    #train
    trainer.fit(model=lightning_module, ckpt_path=checkpoint, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
