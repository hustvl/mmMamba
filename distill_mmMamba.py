"""
Script to distill pretrained Transformers into linear attention variants
"""
import sys
import os
from os.path import join

import argparse
import torch
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer
import copy

sys.path.append('./src')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.utils.setup import (
    init_wandb, seed_everything, flatten_config, get_run_name_from_args,
    update_config_from_args, update_model_config_from_args,
)
from src.utils.logging import print_config, print_header
from src.utils.logging_v2 import get_logger, LogCallback

from src.dataloaders import load_data
from src.trainer import get_trainer, get_optimizer, get_scheduler
from src.stage3 import prepare_stage3_configs, get_trainer_stage3

from src.model.pretrained import get_pretrained_loader
from src.model.load_model import load_and_convert_attns, load_and_convert_attns_shadow, load_and_convert_distill_stage3
from src.model.convert_model import toggle_attention, remove_base_attention
from src.model.utils import count_parameters
from accelerate import Accelerator
from hovle.modeling_internvl_chat import InternVLChatModel
# from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='mmMamba')
    parser.add_argument("--run_name", type=str, default='1_distill')
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--distill_stage1_config", type=str, default=None)
    parser.add_argument("--distill_stage2_config", type=str, default=None)
    parser.add_argument("--distill_stage3_config", type=str, default=None)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--load_distill_checkpoint", type=str, default=None)
    
    # Override default configs
    # Feature map / model
    parser.add_argument("--attention_type", type=str, default=None)
    parser.add_argument("--tie_qk_kernels", action='store_true', default=None)
    parser.add_argument("--train_qk", action='store_true', default=None)
    parser.add_argument("--state_chunk_len", type=int, default=None)
    
    # Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_distill_stage3_steps", type=int, default=None)

    parser.add_argument("--no_peft_grad_ckpt", action='store_true', default=None)
    
    # Dataloading
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    # Evaluation
    parser.add_argument("--no_init_eval", action='store_true', default=False)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)

    # Miscellaneous
    parser.add_argument("--huggingface_token", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default='./checkpoints')
    parser.add_argument("--results_dir", type=str, default='./results')
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action='store_true', default=None)
    parser.add_argument("--no_cuda", action='store_true', default=None)
    parser.add_argument("--no_wandb", action='store_true', default=None)
    parser.add_argument("--wandb_entity", type=str, default='TaoHongyuan')
    parser.add_argument("--debug", action='store_true', default=None)
    parser.add_argument("--no_attention_mask", action='store_true', default=None)

    parser.add_argument("--train_stage1", action='store_true', default=False)
    parser.add_argument("--train_stage2", action='store_true', default=False)
    parser.add_argument("--train_stage3", action='store_true', default=False)

    args = parser.parse_args()
    return args

def main():
    # ------
    # SET UP
    # ------
    args = get_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = get_logger(f"train_{args.run_name}_log", log_dir=args.checkpoint_dir)
    args.results_dir = join(args.results_dir, args.model_config)
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    seed_everything(args.seed)
    args.device = torch.device('cuda')
    logger.info(args)
    # Load model configs

    model_config_path = join('./configs/model', f'{args.model_config}.yaml')
    model_config = OmegaConf.load(model_config_path)
    model_config = update_model_config_from_args(model_config, args)
    
    # WandB logging
    wandb = init_wandb(args)

    # Get pretrained model
    if 'HoVLE' in model_config.model.pretrained_model_name_or_path:
        model = InternVLChatModel.from_pretrained(
            model_config.model.pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_config.model.pretrained_model_name_or_path, trust_remote_code=True, use_fast=False)
    else:
        model_loader = get_pretrained_loader(**model_config.model,
                                            huggingface_token=args.huggingface_token)
        tokenizer = model_loader.load_tokenizer()
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        model = model_loader.load(model_type=args.attention_type)
        model.state_chunk_len = model_config['attention']['state_chunk_len']

    # Convert model
    try:
        args.attention_type = model_config['attention']['attention_type']
    except AttributeError:
        args.attention_type = 'mamba2'

    if args.verbose:
        print_header('*** Initial Model ***', logger=logger)
        logger.info(f"Model: {model}")

    checkpoint_path = None
    distill_done = False

    # --------
    # TRAINING
    # --------

    if args.train_stage1: 
        distill_stage1_config_path = join('./configs/experiment', f'{args.distill_stage1_config}.yaml')
        distill_stage1_config = OmegaConf.load(distill_stage1_config_path)
        distill_stage1_config = update_config_from_args(distill_stage1_config, args)

        args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks

        # Update data tokenizer to match model
        for k in ['pretrained_model_name_or_path', 'cache_dir']:
            distill_stage1_config.dataset.pretrained_model_config[k] = model_config.model[k]

        # Update optimizer if specified
        if 'optimizer' in model_config:
            for k, v in model_config.optimizer.items():
                distill_stage1_config.optimizer[k] = v

        accelerator = Accelerator(gradient_accumulation_steps=distill_stage1_config.trainer.gradient_accumulation_steps, log_with="wandb")

        if wandb is not None:
            distill_stage1_config['model'] = model_config  # Combine for logging
            _flattened = {'model': model_config,
                        'model_config': args.model_config,  # config file names
                        'distill_stage1_config': args.distill_stage1_config,
                        'distill_checkpoint': args.load_distill_checkpoint,
                        'replicate': args.replicate}
            flatten_config(OmegaConf.to_container(distill_stage1_config), _flattened, '')
            wandb.config.update(_flattened)

        if accelerator.is_main_process:
            print_header('Distillation Config', logger=logger)
            print_config(distill_stage1_config, logger=logger)
            print_header('Model Config', logger=logger)
            print_config(model_config, logger=logger)


        checkpoint_path = args.load_distill_checkpoint
        model = load_and_convert_attns(model, model_config,
                                        accelerator=accelerator,
                                        logger=logger,
                                        attention_type=args.attention_type, 
                                        checkpoint_path=checkpoint_path, 
                                        print_model=False,
                                        peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                        train_attention=True)

        if distill_stage1_config.trainer.name is not None:  # Get data for distilling
            dataloaders  = load_data(model, tokenizer, distill_stage1_config.dataset, distill_stage1_config.dataloader)
            train_loader = dataloaders[distill_stage1_config.trainer.train_split]
            eval_loader  = dataloaders[distill_stage1_config.trainer.val_split]
                    
            if args.verbose and accelerator.is_main_process:
                print_header('*** Dataset preview ***', logger=logger)
                for ix, data in enumerate(train_loader):
                    print_info = f"-> Train data input_ids.shape: {data['input_ids'].shape}"
                    print(print_info) if logger is None else logger.info(print_info)
                    break
                for ix, data in enumerate(eval_loader):
                    print_info = f"-> Eval  data input_ids.shape: {data['input_ids'].shape}"
                    print(print_info) if logger is None else logger.info(print_info)
                    break
                
                for ix, data in enumerate(dataloaders[distill_stage1_config.trainer.val_split]):
                    print_info = f"-> Prompt: {tokenizer.batch_decode(data['input_ids'])[0]}"
                    print(print_info) if logger is None else logger.info(print_info)
                    if 'position_ids' in data:
                        print_info = f"-> Position IDs: shape: {data['position_ids'].shape}"
                        print(print_info) if logger is None else logger.info(print_info)
                        print_info = f"Position IDs: {data['position_ids']}"
                        print(print_info) if logger is None else logger.info(print_info)
                    break

            if accelerator.is_main_process:  # Look at model
                print_header('*** Distill Trainable Parameters ***', logger=logger)
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        print(f'├── {n} (dtype = {p.dtype})') if logger is None else logger.info(f'├── {n} (dtype = {p.dtype})')

            # Log some stats
            distill_stage1_config.model_train_params = count_parameters(model, requires_grad=True)
            distill_stage1_config.model_total_params = count_parameters(model, requires_grad=False)
            pct_trainable = distill_stage1_config.model_train_params / distill_stage1_config.model_total_params
        
            if accelerator.is_main_process:
                print_header('*** Distillation Parameter Counts ***', logger=logger)
                print_info = f"├── Number training to distill:  {distill_stage1_config.model_train_params}"
                print(print_info) if logger is None else logger.info(print_info)
                print_info = f"├── Number of total parameters:  {distill_stage1_config.model_total_params}"
                print(print_info) if logger is None else logger.info(print_info)
                print_info = f"├── Percent training to distill: {pct_trainable * 100:.3f}%"
                print(print_info) if logger is None else logger.info(print_info)
        
            # Get optimizer and scheduler
            optimizer = get_optimizer(model=model, **distill_stage1_config.optimizer)
            for key, value in dict(distill_stage1_config.lr_scheduler).items():
                if 'step' in key and isinstance(value, (int, float)):
                    distill_stage1_config.lr_scheduler[key] = value * accelerator.num_processes
            scheduler = get_scheduler(optimizer=optimizer, **distill_stage1_config.lr_scheduler)
            model = toggle_attention(model, train=True)

            model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
            
            # Load trainer 
            for arg, argv in distill_stage1_config.trainer.items():
                if arg != 'name':
                    setattr(args, arg, argv)
            for _config in ['dataloader', 'optimizer', 'lr_scheduler']:
                setattr(args, _config, OmegaConf.to_container(getattr(distill_stage1_config, _config)))
        
            OurTrainer = get_trainer(distill_stage1_config.trainer.name)
            trainer = OurTrainer(accelerator=accelerator,
                                model=model, 
                                args=args,
                                train_loader=train_loader,
                                eval_loader=eval_loader,
                                optimizer_and_scheduler=(optimizer, scheduler),
                                device=args.device,
                                wandb=wandb,
                                checkpoint_suffix='_distill_all_stage1',
                                save_results=True,
                                max_length=distill_stage1_config.dataset.dataset_config.max_length,
                                logger=logger,
                                **distill_stage1_config.trainer)

            # Train / distill model
            if accelerator.is_main_process: 
                print_header('*** Distilling Attentions ***', logger=logger)
                print_info = f"├── Experiment name: {args.run_name}"
                print(print_info) if logger is None else logger.info(print_info)
                print_info = f"├── Device: {args.device}"
                print(print_info) if logger is None else logger.info(print_info)
                print_info = f"├── Seed: {args.seed}"
                print(print_info) if logger is None else logger.info(print_info)
            
            model = trainer.train()
            model = toggle_attention(model, train=False)
            if not args.train_stage2:
                model = remove_base_attention(model)

            args.load_distill_checkpoint = trainer.last_val_checkpoint_path
            print_info = "Done distilling all stage 1"
            print(print_info) if logger is None else logger.info(print_info)
            distill_done = True
            os._exit(0)
            sys.exit(0)
    
    if args.train_stage2: 
        distill_stage2_config_path = join('./configs/experiment', f'{args.distill_stage2_config}.yaml')
        distill_stage2_config = OmegaConf.load(distill_stage2_config_path)
        distill_stage2_config = update_config_from_args(distill_stage2_config, args)

        args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks

        # Update data tokenizer to match model
        for k in ['pretrained_model_name_or_path', 'cache_dir']:
            distill_stage2_config.dataset.pretrained_model_config[k] = model_config.model[k]

        # Update optimizer if specified
        if 'optimizer' in model_config:
            for k, v in model_config.optimizer.items():
                distill_stage2_config.optimizer[k] = v

        accelerator = Accelerator(gradient_accumulation_steps=distill_stage2_config.trainer.gradient_accumulation_steps, log_with="wandb")

        if wandb is not None:
            distill_stage2_config['model'] = model_config  # Combine for logging
            _flattened = {'model': model_config,
                        'model_config': args.model_config,  # config file names
                        'distill_stage2_config': args.distill_stage2_config,
                        'distill_checkpoint': args.load_distill_checkpoint,
                        'replicate': args.replicate}
            flatten_config(OmegaConf.to_container(distill_stage2_config), _flattened, '')
            wandb.config.update(_flattened)

        if accelerator.is_main_process:
            print_header('Distillation Config', logger=logger)
            print_config(distill_stage2_config, logger=logger)
            print_header('Model Config', logger=logger)
            print_config(model_config, logger=logger)

        checkpoint_path = args.load_distill_checkpoint
        model = load_and_convert_attns_shadow(model, model_config,
                                                accelerator=accelerator,
                                                attention_type=args.attention_type, 
                                                checkpoint_path=checkpoint_path,
                                                print_model=False, 
                                                train_converted=False,
                                                peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                                train_attention=False,
                                                logger=logger)

        model = toggle_attention(model, train=True)

        dataloaders  = load_data(model, tokenizer, distill_stage2_config.dataset, distill_stage2_config.dataloader)
        train_loader = dataloaders[distill_stage2_config.trainer.train_split]
        eval_loader  = dataloaders[distill_stage2_config.trainer.val_split]

        optimizer = get_optimizer(model=model, **distill_stage2_config.optimizer)
        for key, value in dict(distill_stage2_config.lr_scheduler).items():
            if 'step' in key and isinstance(value, (int, float)):
                distill_stage2_config.lr_scheduler[key] = value * accelerator.num_processes
        scheduler = get_scheduler(optimizer=optimizer, **distill_stage2_config.lr_scheduler)

        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
        
        # Load trainer 
        for arg, argv in distill_stage2_config.trainer.items():
            if arg != 'name':
                setattr(args, arg, argv)
        for _config in ['dataloader', 'optimizer', 'lr_scheduler']:
            setattr(args, _config, OmegaConf.to_container(getattr(distill_stage2_config, _config)))
    
        OurTrainer = get_trainer(distill_stage2_config.trainer.name)
        trainer = OurTrainer(accelerator=accelerator,
                                model=model, 
                                args=args,
                                train_loader=train_loader,
                                eval_loader=eval_loader,
                                optimizer_and_scheduler=(optimizer, scheduler),
                                device=args.device,
                                wandb=wandb,
                                checkpoint_suffix='_distill_all_stage2',
                                save_results=True,
                                max_length=distill_stage2_config.dataset.dataset_config.max_length,
                                logger=logger,
                                **distill_stage2_config.trainer)
        if accelerator.is_main_process:  # Look at model
            print_header('*** Distill Trainable Parameters ***', logger=logger)
            for n, p in model.named_parameters():
                if p.requires_grad:
                    print(f'├── {n} (dtype = {p.dtype})') if logger is None else logger.info(f'├── {n} (dtype = {p.dtype})')
        # Log some stats
        distill_stage2_config.model_train_params = count_parameters(model, requires_grad=True)
        distill_stage2_config.model_total_params = count_parameters(model, requires_grad=False)
        pct_trainable = distill_stage2_config.model_train_params / distill_stage2_config.model_total_params
    
        if accelerator.is_main_process:
            print_header('*** Distillation Parameter Counts ***', logger=logger)
            print_info = f"├── Number training to distill:  {distill_stage2_config.model_train_params}"
            print(print_info) if logger is None else logger.info(print_info)
            print_info = f"├── Number of total parameters:  {distill_stage2_config.model_total_params}"
            print(print_info) if logger is None else logger.info(print_info)
            print_info = f"├── Percent training to distill: {pct_trainable * 100:.3f}%"
            print(print_info) if logger is None else logger.info(print_info)
        # Train / distill model
        if accelerator.is_main_process: 
            print_header('*** Distilling Stage 2 Attentions ***', logger=logger)
            print_info = f'├── Experiment name: {args.run_name}'
            print(print_info) if logger is None else logger.info(print_info)
            print_info = f'├── Device: {args.device}'
            print(print_info) if logger is None else logger.info(print_info)
            print_info = f'├── Seed: {args.seed}'
            print(print_info) if logger is None else logger.info(print_info)
            print_info = f'├── Begin Distillation Stage 2 by loading weights from {checkpoint_path}...'
            print(print_info) if logger is None else logger.info(print_info)
        model = trainer.train()

        # Prepare for downstream distill_stage3 / eval
        model = toggle_attention(model, train=False)
        model = remove_base_attention(model)

        args.load_distill_checkpoint = trainer.best_val_checkpoint_path
        print_info = "Done distilling stage 2"
        print(print_info) if logger is None else logger.info(print_info)
        distill_done = True
        os._exit(0)
        sys.exit(0)


    if args.train_stage3:
        distill_stage3_config_path = join('./configs/experiment', f'{args.distill_stage3_config}.yaml')
        distill_stage3_config = OmegaConf.load(distill_stage3_config_path)
        distill_stage3_config = update_config_from_args(distill_stage3_config, args)

        args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks

        # Update data tokenizer to match model
        for k in ['pretrained_model_name_or_path', 'cache_dir']:
            distill_stage3_config.dataset.pretrained_model_config[k] = model_config.model[k]

        # Update optimizer if specified
        if 'optimizer' in model_config:
            for k, v in model_config.optimizer.items():
                distill_stage3_config.optimizer[k] = v

        accelerator = Accelerator(gradient_accumulation_steps=distill_stage3_config.trainer.gradient_accumulation_steps, log_with="wandb")

        if wandb is not None:
            distill_stage3_config['model'] = model_config  # Combine for logging
            _flattened = {'model': model_config,
                        'model_config': args.model_config,  # config file names
                        'distill_stage3_config': args.distill_stage3_config,
                        'distill_checkpoint': args.load_distill_checkpoint,
                        'replicate': args.replicate}
            flatten_config(OmegaConf.to_container(distill_stage3_config), _flattened, '')
            wandb.config.update(_flattened)

        if accelerator.is_main_process:
            print_header('Distillation Config', logger=logger)
            print_config(distill_stage3_config, logger=logger)
            print_header('Model Config', logger=logger)
            print_config(model_config, logger=logger)

        distill_stage3_config, args = prepare_stage3_configs(args, model_config, args.distill_stage3_config)

        checkpoint_path = args.load_distill_checkpoint
        
        teacher_model = InternVLChatModel.from_pretrained(
            model_config.model.pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False).eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        if distill_done == False:
            model = load_and_convert_attns(model, model_config, 
                                            accelerator=accelerator,
                                            distill_stage3_config=distill_stage3_config,
                                            attention_type=args.attention_type, 
                                            checkpoint_path=checkpoint_path,
                                            print_model=False, 
                                            train_converted=False,
                                            peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                            logger=logger,
                                            train_attention=False)

            model = toggle_attention(model, train=False)
            model = remove_base_attention(model)
        
        if accelerator.is_main_process:
            print_info = f'-> Distilled checkpoint loaded from {args.load_distill_checkpoint}!'
            print(print_info) if logger is None else logger.info(print_info)
    

        if args.max_distill_stage3_steps is not None:
            args.max_steps = args.max_distill_stage3_steps

        checkpoint_path = None
        model = load_and_convert_distill_stage3(model, distill_stage3_config, 
                                            accelerator=accelerator,
                                            checkpoint_path=checkpoint_path,
                                            print_model=False,
                                            peft_gradient_checkpointing=not args.no_peft_grad_ckpt,
                                            logger=logger)

        if args.verbose and accelerator.is_main_process:
            print_header(f'*** Final Trainable distill_stage3 parameters ***', logger=logger)
            for n, p in model.named_parameters():
                if p.requires_grad:
                    print(f'├── {n} ({p.dtype})') if logger is None else logger.info(f'├── {n} ({p.dtype})')

        distill_stage3_trainer = get_trainer_stage3(model, tokenizer, "_ft_embedding", distill_stage3_config, args.device, args, wandb, teacher_model=teacher_model, logger=logger)
        if args.verbose and accelerator.is_main_process:
            print_header('distill_stage3 config', logger=logger)
            print_config(distill_stage3_config, logger=logger)
        # Log some stats

        if accelerator.is_main_process:  # Look at model
                print_header('*** Distill Trainable Parameters ***', logger=logger)
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        print(f'├── {n} (dtype = {p.dtype})') if logger is None else logger.info(f'├── {n} (dtype = {p.dtype})')
                        
        distill_stage3_config.model_train_params = count_parameters(model, requires_grad=True)
        distill_stage3_config.model_total_params = count_parameters(model, requires_grad=False)
        pct_trainable = distill_stage3_config.model_train_params / distill_stage3_config.model_total_params
    
        if accelerator.is_main_process:
            print_header('*** Distill_stage3 Parameter Counts ***', logger=logger)
            print_info = f'├── Number training to distill:  {distill_stage3_config.model_train_params}'
            print(print_info) if logger is None else logger.info(print_info)
            print_info = f'├── Number of total parameters:  {distill_stage3_config.model_total_params}'
            print(print_info) if logger is None else logger.info(print_info)
            print_info = f'├── Percent training to distill: {pct_trainable * 100:.3f}%'
            print(print_info) if logger is None else logger.info(print_info)
        if accelerator.is_main_process:
            print_header('*** Distill_stage3 all***', logger=logger)
            print_info = f'├── Experiment name: {args.run_name}'
            print(print_info) if logger is None else logger.info(print_info)
            print_info = f'├── Device: {args.device}'
            print(print_info) if logger is None else logger.info(print_info)
            print_info = f'├── Seed: {args.seed}'
            print(print_info) if logger is None else logger.info(print_info)
            print_info = f'├── Begin distill_stage3 by loading weights from {args.load_distill_checkpoint}...'
            print(print_info) if logger is None else logger.info(print_info)
        model = distill_stage3_trainer.train()
        args.load_distill_checkpoint = distill_stage3_trainer.best_val_checkpoint_path
        print_info = "Done distill_stage3"
        print(print_info) if logger is None else logger.info(print_info)
        os._exit(0)
        sys.exit(0)

if __name__ == '__main__':
    main()
