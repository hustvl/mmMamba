"""
Helpers to load checkpoints for learned feature maps (attentions) or other parameters
"""
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.utils.logging import print_header, _format_arg
from .convert_model import convert_attention, convert_attention_shadow
from .peft import create_peft_config
from accelerate import Accelerator

def load_and_convert_attns(model: nn.Module,
                           model_config: dict,
                           accelerator: Accelerator = None,
                           distill_stage3_config: dict = None,
                           logger=None,
                           attention_type: str = None,
                           checkpoint_path: str = None,
                           print_model: bool = False,
                           train_converted: bool = True,  # Should be false if loading distill checkpoint by default
                           peft_gradient_checkpointing: bool = None,
                           train_attention: bool = False,  # Should be true if converting attentions for first time,
                           freeze_weights: bool = True,
                           rank: int = 0,
                           remove_base_attn: bool = True,
                          ) -> nn.Module:
    """
    Load trained attention kernel parameter weights
    """
    if freeze_weights:
        for p in model.parameters():
            p.requires_grad = False

    if attention_type is not None:  # override default
        model_config['attention']['attention_type'] = attention_type
    model_config['attention']['rank'] = rank   # multi-gpu debugging

    model = convert_attention(model, model_config['attention'], 
                              train_attention, remove_base_attn, accelerator=accelerator, distill_stage3_config=distill_stage3_config)

    for name, param in model.named_parameters():
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            param.requires_grad = False
        if accelerator is not None and accelerator.is_main_process:
            print(name) if logger is None else logger.info(name)
        else:
            print(name)

    # Load any trained attentions
    if checkpoint_path is not None:
        print(f'Loading weights from {checkpoint_path}...') if logger is None else logger.info(f'Loading weights from {checkpoint_path}...')
        state_dict = torch.load(checkpoint_path)['model_state_dict']
        _keys = model.load_state_dict(state_dict, strict=False)
        try:
            assert len(_keys.unexpected_keys) == 0
            if accelerator is not None and accelerator.is_main_process:
                print_header('*** All expected keys matched successfully ***', logger=logger)
                if print_model:
                    for k in state_dict.keys():
                        print(k) if logger is None else logger.info(k)  
            else:
                print('All expected keys matched successfully')
                if print_model:
                    for k in state_dict.keys():
                        print(k) if logger is None else logger.info(k)  
        except Exception as e:
            if accelerator is not None and accelerator.is_main_process:
                print(e) if logger is None else logger.info(e)
                print_header('*** Error: unexpected keys in checkpoint ***', logger=logger)
                print('Unexpected keys:')
                for k in _keys.unexpected_keys:
                    print(k) if logger is None else logger.info(k) 
            else:
                print(e)
                print_header('*** Error: unexpected keys in checkpoint ***')
                print('Unexpected keys:')
                for k in _keys.unexpected_keys:
                    print(k)

    if print_model and accelerator.is_main_process:  # Look at model
        print_header('*** Distill Trainable Parameters ***', logger=logger)
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})') if logger is None else logger.info(f'├── {n} (dtype = {p.dtype})')

    return model

def load_and_convert_attns_shadow(model: nn.Module,
                           model_config: dict,
                           accelerator: Accelerator = None,
                           attention_type: str = None,
                           checkpoint_path: str = None,
                           print_model: bool = False,
                           train_converted: bool = True,  # Should be false if loading distill checkpoint by default
                           peft_gradient_checkpointing: bool = None,
                           train_attention: bool = False,  # Should be true if converting attentions for first time,
                           freeze_weights: bool = True,
                           rank: int = 0,
                           remove_base_attn: bool = True,
                           logger=None,
                          ) -> nn.Module:
    """
    Load trained attention kernel parameter weights
    """
    if freeze_weights:
        for p in model.parameters():
            p.requires_grad = False

    if attention_type is not None:  # override default
        model_config['attention']['attention_type'] = attention_type
    model_config['attention']['rank'] = rank   # multi-gpu debugging

    model = convert_attention_shadow(model, model_config['attention'], 
                              train_attention, remove_base_attn, accelerator=accelerator, logger=logger)

    # Load any trained attentions
    if checkpoint_path is not None:
        if accelerator.is_main_process:
            print(f'Loading weights from {checkpoint_path}...')
        state_dict = torch.load(checkpoint_path)['model_state_dict']
        _keys = model.load_state_dict(state_dict, strict=False)
        try:
            assert len(_keys.unexpected_keys) == 0
            if accelerator.is_main_process:
                print_header('*** All expected keys matched successfully ***', logger=logger)
                if print_model:
                    for k in state_dict.keys():
                        print(k) if logger is None else logger.info(k)  
        except Exception as e:
            if accelerator.is_main_process:
                print(e) if logger is None else logger.info(e)
                print_header('*** Error: unexpected keys in checkpoint ***', logger=logger)
                print('Unexpected keys:') if logger is None else logger.info('Unexpected keys:')
                for k in _keys.unexpected_keys:
                    print(k) if logger is None else logger.info(k)

    if print_model and accelerator.is_main_process:  # Look at model
        print_header('*** Trainable Parameters ***', logger=logger)
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})') if logger is None else logger.info(f'├── {n} (dtype = {p.dtype})')

    return model

def load_and_convert_distill_stage3(model: nn.Module,
                              distill_stage3_config: dict,
                              accelerator: Accelerator = None,
                              checkpoint_path: str = None,
                              print_model: bool = False,
                              peft_gradient_checkpointing: bool = None,
                              rank: int = 0,
                              logger=None,
                              **kwargs: any):
    """
    Load trained model weights
    """
    for p in model.parameters():
        p.requires_grad = False

    # Keep specified weights trainable
    if 'trainable_weights' in distill_stage3_config.distill_stage3:
        for name in distill_stage3_config.distill_stage3['trainable_weights']:
            for n, p in model.named_parameters():
                if name in n and "embedding_model" in n: 
                    if 'softmax_attention' in distill_stage3_config.distill_stage3:
                        layer = int(n.split('encoder.')[-1].split('.')[0])
                        if layer not in distill_stage3_config.distill_stage3['softmax_attention']:
                            p.requires_grad = True
                    else:
                        p.requires_grad = True
                if name in n and "language_model" in n: 
                    if 'softmax_attention' in distill_stage3_config.distill_stage3:
                        layer = int(n.split('layers.')[-1].split('.')[0]) + 8
                        if layer not in distill_stage3_config.distill_stage3['softmax_attention']:
                            p.requires_grad = True
                    else:
                        p.requires_grad = True
        
    # Load weights
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path)['model_state_dict']
        _keys = model.load_state_dict(state_dict, strict=False)
        try:
            assert len(_keys.unexpected_keys) == 0
            if accelerator is not None and accelerator.is_main_process:
                print_header('*** All expected keys matched successfully ***', logger=logger)
            else:
                print('All expected keys matched successfully')
        except Exception as e:
            if accelerator is not None and accelerator.is_main_process:
                print(e) if logger is None else logger.info(e)
                print_header('*** Error: unexpected keys in checkpoint ***', logger=logger)
                print('Unexpected keys:') if logger is None else logger.info('Unexpected keys:')
                for k in _keys.unexpected_keys:
                    print(k) if logger is None else logger.info(k)
            else:
                print(e)
                print_header('*** Error: unexpected keys in checkpoint ***')
                print('Unexpected keys:')
                for k in _keys.unexpected_keys:
                    print(k)

    if print_model and accelerator is not None and accelerator.is_main_process:  # Look at model
        print_header('*** Model ***', logger=logger)
        print(model) if logger is None else logger.info(model)
    else:
        print_header('*** Model ***')
        print(model)

    if print_model and accelerator.is_main_process:  # Look at model
        print_header('*** distill_stage3 Trainable Parameters ***', logger=logger)
        count = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n}.requires_grad: {p.requires_grad}') if logger is None else logger.info(f'├── {n}.requires_grad: {p.requires_grad}')
                count += 1
        if count == 0:
            print('(none)') if logger is None else logger.info('(none)')

    return model
