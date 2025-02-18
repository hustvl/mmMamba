"""
Finetuning functions to do post-distillation
"""
import os
from os.path import join
from omegaconf import OmegaConf

import torch
from torch.nn import Module

from src.utils.setup import update_config_from_args
from src.dataloaders import load_data
from src.trainer import get_trainer, get_optimizer, get_scheduler
from accelerate import Accelerator

def prepare_stage3_configs(args, model_config: dict,
                             distill_stage3_config_name: str = None,
                             distill_stage3_checkpoint_name: str = None,
                             config_dir='./configs/experiment'):
    """
    Prepare finetuning configs
    """
    # Load finetuning config
    distill_stage3_config = (distill_stage3_config_name if distill_stage3_config_name is not None else 
                       distill_stage3_checkpoint_name.split('-f=')[-1].split('-')[0])
    distill_stage3_config_path = join(config_dir, f'{distill_stage3_config}.yaml')
    distill_stage3_config = OmegaConf.load(distill_stage3_config_path)
    distill_stage3_config = update_config_from_args(distill_stage3_config, args,
                                              ignore_args=['lr', 'weight_decay'])
    # Update data tokenizer to match model
    if getattr(distill_stage3_config.dataset, 'pretrained_model_config', None) is not None:
        for k in ['pretrained_model_name_or_path', 'cache_dir']:
            distill_stage3_config.dataset.pretrained_model_config[k] = model_config['model'][k]
    # Set finetuning args
    for arg, argv in distill_stage3_config.trainer.items():
        if arg != 'name':
            setattr(args, arg, argv)
    for _config in ['dataloader', 'optimizer', 'lr_scheduler']:
        setattr(args, _config, OmegaConf.to_container(getattr(distill_stage3_config, _config)))
    return distill_stage3_config, args


def get_trainer_stage3(model, tokenizer, checkpoint_suffix, distill_stage3_config: dict, device: torch.device, 
                  args: any, wandb: any, initial_eval: bool = False, teacher_model=None, logger=None):
    """
    Initialize finetuning trainer
    """
    #model.to(device)  # if using a fused optimizer
    accelerator = Accelerator(gradient_accumulation_steps=distill_stage3_config.trainer.gradient_accumulation_steps, log_with="wandb")
    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model=model, **distill_stage3_config.optimizer)
    for key, value in dict(distill_stage3_config.lr_scheduler).items():
        if 'step' in key and isinstance(value, (int, float)):
            distill_stage3_config.lr_scheduler[key] = value * accelerator.num_processes
    scheduler = get_scheduler(optimizer=optimizer, **distill_stage3_config.lr_scheduler)

    dataloaders  = load_data(model, tokenizer, distill_stage3_config.dataset, distill_stage3_config.dataloader) 
    train_loader = dataloaders[distill_stage3_config.trainer.train_split]
    eval_loader  = dataloaders[distill_stage3_config.trainer.val_split]

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    model.train()

    OurTrainer = get_trainer(distill_stage3_config.trainer.name)

    trainer = OurTrainer(accelerator=accelerator,
                         model=model,
                         args=args,
                         train_loader=train_loader,
                         eval_loader=eval_loader,
                         optimizer_and_scheduler=(optimizer, scheduler),
                         device=device,
                         wandb=wandb,
                         checkpoint_suffix=checkpoint_suffix,
                         max_length=distill_stage3_config.dataset.dataset_config.max_length,
                         teacher_model=teacher_model,
                         logger=logger,
                         **distill_stage3_config.trainer)
    return trainer