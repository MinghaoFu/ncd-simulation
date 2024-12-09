import torch
import random
import argparse
import numpy as np
import ipdb as pdb
import os, pwd, yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

import sys
sys.path.append('..')
from LiLY.modules.nonparam import ModularShifts
from LiLY.tools.utils import load_yaml, setup_seed
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import warnings
warnings.filterwarnings('ignore')
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset
from General.gen_data import nonparametric_ts

class SyntheticDataset(Dataset):
    
    def __init__(self, directory='./data', dataset="nonparametric"):
        super().__init__()
        self.path = os.path.join(directory, dataset, "data.npz")
        self.npz = np.load(self.path)
        self.data = {}
        for key in ["zt", "xt", "ct", "ht", "st"]:
            self.data[key] = self.npz[key]

        self.B = np.load(os.path.join(directory, dataset, "B.npy"))
        self.Bs = np.load(os.path.join(directory, dataset, "Bs.npy"))
        self.B_mask = np.load(os.path.join(directory, dataset, "B_mask.npy"))

    def __len__(self):
        return len(self.data["xt"])

    def __getitem__(self, idx):
        zt = torch.from_numpy(self.data["zt"][idx].astype('float32'))
        xt = torch.from_numpy(self.data["xt"][idx].astype('float32'))
        ct = torch.from_numpy(self.data["ct"][idx].astype('float32'))
        ht = torch.from_numpy(self.data["ht"][idx].astype('float32'))
        st = torch.from_numpy(self.data["st"][idx].astype('float32'))   
        sample = {"zt": zt, "xt": xt, "ct": ct, "ht": ht, "st": st} 
        return sample
    

def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    abs_file_path = './nonparam.yaml'
    cfg = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")

    pl.seed_everything(args.seed)
    name = cfg['DATASET'] + str(cfg['VAE']['INPUT_DIM']) + "mask"
    log_dir = os.path.join('.', cfg['PROJ_NAME'], name)
    wandb_logger = WandbLogger(project=cfg['PROJ_NAME'], name=name, save_dir=log_dir)

    #nonparametric_ts(observed_size=cfg['VAE']['INPUT_DIM'], dyn_latent_size=cfg['VAE']['DYN_DIM'], obs_latent_size=cfg['SPLINE']['OBS_DIM'])
    
    data = SyntheticDataset(dataset=cfg['DATASET'])

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']
    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])

    train_loader = DataLoader(train_data, 
                              batch_size=cfg['VAE']['TRAIN_BS'], 
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=False,
                              shuffle=True)

    val_loader = DataLoader(val_data, 
                            batch_size=cfg['VAE']['VAL_BS'], 
                            pin_memory=cfg['VAE']['PIN'],
                            num_workers=cfg['VAE']['CPU'],
                            shuffle=False)

    if cfg['LOAD_CHECKPOINT']:
        model = ModularShifts.load_from_checkpoint(checkpoint_path=cfg['CHECKPOINT'], # if save hyperparameter
                            #strict=False,                       
                            input_dim=cfg['VAE']['INPUT_DIM'],
                            length=cfg['VAE']['LENGTH'],
                            obs_dim=cfg['SPLINE']['OBS_DIM'],
                            dyn_dim=cfg['VAE']['DYN_DIM'],
                            lag=cfg['VAE']['LAG'],
                            nclass=cfg['VAE']['NCLASS'],
                            hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                            dyn_embedding_dim=cfg['VAE']['DYN_EMBED_DIM'],
                            obs_embedding_dim=cfg['SPLINE']['OBS_EMBED_DIM'],
                            trans_prior=cfg['VAE']['TRANS_PRIOR'],
                            lr=cfg['VAE']['LR'],
                            infer_mode=cfg['VAE']['INFER_MODE'],
                            bound=cfg['SPLINE']['BOUND'],
                            count_bins=cfg['SPLINE']['BINS'],
                            order=cfg['SPLINE']['ORDER'],
                            beta=cfg['VAE']['BETA'],
                            gamma=cfg['VAE']['GAMMA'],
                            sigma=cfg['VAE']['SIMGA'],
                            decoder_dist=cfg['VAE']['DEC']['DIST'],
                            correlation=cfg['MCC']['CORR'])
    else:
        model = ModularShifts(input_dim=cfg['VAE']['INPUT_DIM'],
                            length=cfg['VAE']['LENGTH'],
                            obs_dim=cfg['SPLINE']['OBS_DIM'],
                            dyn_dim=cfg['VAE']['DYN_DIM'],
                            lag=cfg['VAE']['LAG'],
                            nclass=cfg['VAE']['NCLASS'],
                            hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                            dyn_embedding_dim=cfg['VAE']['DYN_EMBED_DIM'],
                            obs_embedding_dim=cfg['SPLINE']['OBS_EMBED_DIM'],
                            trans_prior=cfg['VAE']['TRANS_PRIOR'],
                            lr=cfg['VAE']['LR'],
                            infer_mode=cfg['VAE']['INFER_MODE'],
                            bound=cfg['SPLINE']['BOUND'],
                            count_bins=cfg['SPLINE']['BINS'],
                            order=cfg['SPLINE']['ORDER'],
                            beta=cfg['VAE']['BETA'],
                            gamma=cfg['VAE']['GAMMA'],
                            sigma=cfg['VAE']['SIMGA'],
                            decoder_dist=cfg['VAE']['DEC']['DIST'],
                            correlation=cfg['MCC']['CORR'],
                            masks=None, #data.B_mask,
                            B=data.B,
                            instantaneous=True)
        
    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    checkpoint_callback = ModelCheckpoint(monitor='val_mcc', 
                                          save_top_k=1, 
                                          mode='max')

    early_stop_callback = EarlyStopping(monitor="val_mcc", 
                                        min_delta=0.00, 
                                        patience=50, 
                                        verbose=False, 
                                        mode="max")

    trainer = pl.Trainer(default_root_dir=log_dir,
                         #gpus=cfg['VAE']['GPU'], 
                         #accelerator="auto", 
                         logger=wandb_logger,
                         val_check_interval = cfg['MCC']['FREQ'],
                         max_epochs=cfg['VAE']['EPOCHS'],
                         callbacks=[checkpoint_callback], 
                         devices=1,
                         strategy='ddp_find_unused_parameters_true' 
                         ) #strategy='ddp_find_unused_parameters_true'

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        default='modular_4',
        type=str
    )

    argparser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=770
    )

    args = argparser.parse_args()
    main(args)
