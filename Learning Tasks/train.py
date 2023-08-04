
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

import matplotlib.pyplot as plt

if __name__ ==  '__main__':
    parser = Options()
    opt = parser.parse()
    opt.phase = 'train'

    default_root_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(default_root_dir):
        os.makedirs(default_root_dir)

    train_dataset = Dataset(opt, phase='train')
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=10, shuffle=True)
    val_dataset = Dataset(opt, phase='val')
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=10, shuffle=False)

    sizey, sizex = find_size(train_data_loader)
    opt.size_y = sizey
    opt.size_x = sizex

    Unet_network = buildUnet(opt, )

    lightning_module = LitFixupUnet(Unet_network, opt)

    #save every epoch's model
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, save_last=True)

    if opt.early_stop:
        trainer = pl.Trainer(max_epochs=opt.n_epoch, 
                            val_check_interval=1.0, 
                            accelerator="gpu", devices=1, 
                            default_root_dir=default_root_dir,
                            callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=500, mode="min")])
    else:
        trainer = pl.Trainer(max_epochs=opt.n_epoch, 
                            val_check_interval=1.0, 
                            accelerator="gpu", devices=1, 
                            default_root_dir=default_root_dir,
                            callbacks=[checkpoint_callback])
    
    #save config file:
    setattr(opt,'version',str(trainer.logger.version))
    print(opt.version)
    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    filename = trainer.logger.log_dir + "/config.txt"
    parser.save(filename, opt)

    #train
    trainer.fit(model=lightning_module, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)