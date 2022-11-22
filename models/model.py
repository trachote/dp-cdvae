from typing import Any, Dict

# import hydra
import numpy as np
import omegaconf
import torch
# import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm

from common.datamodule import CrystDataModule
import copy
import json
import os
from omegaconf import DictConfig, OmegaConf
import glob

class MODEL(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.hparams = cfg.model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device = "cpu"
        self.hparams.data = cfg.data
        self.current_epoch = 0
        self.logs = {'train':[], 'val':[], 'test':[]}
        self.train_checkpoint_path = None
        self.val_checkpoint_path = None
        self.min_val_loss = float('inf')
        self.min_val_epoch = 0
        self.train_log = None
        self.val_log = None
        self.model_name = "model"

    def init(self):
        self.init_datamodule()
        self.init_optimizer()
        self.init_scheduler()

        with open(self.cfg.output_dir + "/hparams.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg=self.cfg))

        self.to(self.device)

        self.load_checkpoint()
        
        self.init_dataloader()

    def init_optimizer(self):
        print(f"Instantiating Optimizer <{self.cfg.optim.optimizer._target_}>")
        # if self.cfg.optim.optimizer._target_=='Adam':
        self.optimizer = torch.optim.Adam(self.parameters(), **{k:v for k, v in self.cfg.optim.optimizer.items() if k!='_target_'})        

    def init_scheduler(self):
        print(f"Instantiating LR Scheduler <{self.cfg.optim.lr_scheduler._target_}>")
        # if self.cfg.optim.optimizer.optimizer._target_=='ReduceLROnPlateau':
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **{k:v for k, v in self.cfg.optim.lr_scheduler.items() if k!='_target_'})

    def init_datamodule(self):
        if self.cfg.data.datamodule._target_=='CrystDataModule':
            self.datamodule = CrystDataModule(**{k:v for k, v in self.cfg.data.datamodule.items() if k!='_target_'})
            self.lattice_scaler = self.datamodule.lattice_scaler.copy()
            self.scaler = self.datamodule.scaler.copy()
            torch.save(self.datamodule.lattice_scaler, self.cfg.output_dir  + '/lattice_scaler.pt')
            torch.save(self.datamodule.scaler, self.cfg.output_dir  + '/prop_scaler.pt')

    def init_dataloader(self):
        self.datamodule.setup("fit")
        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()[0]

    def train_start(self):
        print(">>>TRAINING START<<<")
        pass

    def train_end(start):
        print(">>>TRAINING END<<<")
        pass

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # impelment training step and return loss
        return NotImplementedError

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # impelment validation step and return loss
        return NotImplementedError

    def clip_grad_value_(self):
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=0.5)

    def train_step_end(self, e):
        # for examination
        pass

    def val_step_end(self, e):
        # for examination
        pass

    def train_epoch_start(self, e):
        self.clear_log_dict()
        self.train_log = None
        self.val_log = None

    def val_epoch_start(self, e):
        pass    

    def train_epoch_end(self, e):
        log_dict = {'epoch':e}
        log_dict.update({k:np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k] for x in self.logs['train']]) for k in self.logs['train'][0].keys()})

        with open(self.cfg.output_dir + "/train_metrics.json", 'a') as f:
            f.write(json.dumps({k:v for k, v in log_dict.items()}))
            f.write('\r\n')

        self.train_log = log_dict

    def val_epoch_end(self, e):
        log_dict = {'epoch':e}
        log_dict.update({k:np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k] for x in self.logs['val']]) for k in self.logs['val'][0].keys()})

        with open(self.cfg.output_dir + "/val_metrics.json", 'a') as f:
            f.write(json.dumps({k:v for k, v in log_dict.items()}))
            f.write('\r\n')

        self.val_log = log_dict

        if self.val_log['val_loss'] < self.min_val_loss:
            self.min_val_loss = self.val_log['val_loss']
            self.min_val_epoch = e
            self.val_checkpoint_path = self.save_checkpoint(model_checkpoint_path=self.val_checkpoint_path, suffix="val")


    def train_val_epoch_end(self, e):

        if e % self.cfg.logging.log_freq_every_n_epoch == 0:
            self.logging(e)

        if e % self.cfg.checkpoint_freq_every_n_epoch == 0:
            self.train_checkpoint_path = self.save_checkpoint(model_checkpoint_path=self.train_checkpoint_path, suffix="train")

        self.current_epoch += 1

    def load_checkpoint(self):
        # ckpts = list(self.cfg.output_dir.glob(f'*={self.model_name}=*.ckpt'))
        ckpts = list(glob.glob(f'{self.cfg.output_dir}/*={self.model_name}=*.ckpt'))        
        if len(ckpts) > 0:
            ckpt_epochs = np.array([int(ckpt.parts[-1].split('.')[0].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
            print(f">>>>> Load model from checkpoint {ckpt}")
            ckpt = torch.load(ckpt)

            
            self.current_epoch = ckpt['epoch'] + 1

            self.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            if self.cfg.optim.use_lr_scheduler:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

            self.train_checkpoint_path = f"{self.cfg.output_dir}/epoch={ckpt['epoch']}={self.model_name}=train.ckpt"

            ckpts = list(glob.glob(f'{self.cfg.output_dir}/*={self.model_name}=val.ckpt'))
            # list(self.cfg.output_dir.glob(f'*={self.model_name}=val.ckpt'))

            if len(ckpts) > 0:

                ckpt_epochs = np.array([int(ckpt.parts[-1].split('.')[0].split('=')[1]) for ckpt in ckpts])
                ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
                print(f">>>>> Load val model from checkpoint {ckpt}")

                ckpt = torch.load(ckpt)

                self.min_val_epoch = ckpt['epoch']
                self.min_val_loss = torch.tensor(ckpt['val_loss'])

                print("min val epoch: ", self.min_val_epoch)
                print("min val loss: ", self.min_val_loss.item())            

                self.val_checkpoint_path = f"{self.cfg.output_dir}/epoch={ckpt['epoch']}={self.model_name}=val.ckpt"
        else:
            print(f">>>>> New Training")        

    def save_checkpoint(self, model_checkpoint_path, suffix="val", logs={}):
        model_checkpoint = {
            'model_state_dict':copy.deepcopy(self.state_dict()), 
            'optimizer_state_dict':copy.deepcopy(self.optimizer.state_dict()), 
            'scheduler_state_dict':copy.deepcopy(self.scheduler.state_dict()),
            'epoch':self.current_epoch, 
            'train_loss':self.train_log['train_loss'], 
            'val_loss':self.val_log['val_loss'] if self.val_log else None 
        }

        model_checkpoint.update(logs)

        new_model_checkpoint_path = f"{self.cfg.output_dir}/epoch={self.current_epoch}={self.model_name}={suffix}.ckpt"

        if new_model_checkpoint_path != model_checkpoint_path:
            if model_checkpoint_path:
                os.remove(model_checkpoint_path)
            print("Save model checkpoint: ", new_model_checkpoint_path)
            print("\tmodel checkpoint train loss: ", model_checkpoint['train_loss'])
            print("\tmodel checkpoint val loss: ", model_checkpoint['val_loss'])
            torch.save(model_checkpoint, new_model_checkpoint_path)

        return new_model_checkpoint_path

    def early_stopping(self, e):
        if e - self.min_val_epoch > self.cfg.data.early_stopping_patience_epoch:
            print("Early stopping")
            return True

        return False

    def logging(self, e):
        print(f"Epoch {e:5d}:")
        print(f"\tTrain Loss:{self.train_log['train_loss']}")
        if self.val_log:
            print(f"\tVal Loss:{self.val_log['val_loss']}")

    def log_dict(self, log_dict, prefix):
        self.logs[prefix].append(log_dict)

    def clear_log_dict(self):
        for x in self.logs:
            self.logs[x] = []
