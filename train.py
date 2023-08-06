import time
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import os
import json
import copy
import argparse
import random
from common.model_utils import get_model

def main(cfg):

    if cfg.train.deterministic:
        torch.manual_seed(cfg.train.random_seed)
        torch.cuda.manual_seed(cfg.train.random_seed)
        torch.cuda.manual_seed_all(cfg.train.random_seed)
        np.random.seed(cfg.train.random_seed)
        random.seed(cfg.train.random_seed)    
    
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = get_model(cfg)

    model.init()
    
    model.train_start()

    for e in range(model.current_epoch, cfg.train.pl_trainer.max_epochs):
        tick = time.time()
        model.train()

        model.train_epoch_start(e)
        for batch_idx, batch in enumerate(model.train_dataloader):
            loss = model.training_step(batch.to(model.device), batch_idx)
            model.optimizer.zero_grad()
            loss.backward()
            model.clip_grad_value_()
            model.optimizer.step()
            model.train_step_end(e)

        model.train_epoch_end(e)

        if e % cfg.logging.check_val_every_n_epoch == 0:

            model.eval()

            model.val_epoch_start(e)

            with torch.no_grad():
                outs = []
                for val_batch_idx, val_batch in enumerate(model.val_dataloader):
                    val_out = model.validation_step(val_batch.to(model.device), val_batch_idx)
                    outs.append(val_out.detach())
                    model.val_step_end(e)

            model.val_epoch_end(e)

            if cfg.optim.use_lr_scheduler:
                model.scheduler.step(torch.mean(torch.stack([x for x in outs])))

        model.train_val_epoch_end(e)
        print(f"\tTraining time: {time.time() - tick} s")
        if e % 10 == 0:
            pass#model.noise_model.gamma.show_schedule()
        
        if model.early_stopping(e):
            break
    
    model.train_end(e)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--predict_property', default=False)
    parser.add_argument('--predict_property_class', default=False)
    parser.add_argument('--early_stop', type=int, default=300)

    args = parser.parse_args()

    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("now", lambda x: time.strftime(x))

    cfg = OmegaConf.load(args.config_path)
    cfg.output_dir = args.output_path
    
    if args.predict_property is not None:
        cfg.model.predict_property = args.predict_property

    if args.predict_property_class is not None:
        cfg.model.predict_property_class = args.predict_property_class      
    
    cfg.data = OmegaConf.load("./conf/data/"+cfg.data+".yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml(cfg)), resolve=True))
    cfg.data.early_stopping_patience_epoch = args.early_stop

    main(cfg)