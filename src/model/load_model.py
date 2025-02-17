"""
Helpers to load checkpoints for learned feature maps (attentions) or other parameters
"""
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.utils.logging import print_header, _format_arg
from .convert_model import convert_attention, convert_attention_shadow, convert_embedding_attention, convert_embedding_attention_shadow
from .peft import create_peft_config
from accelerate import Accelerator

def load_and_convert_attns(model: nn.Module,
                           model_config: dict,
                           accelerator: Accelerator = None,
                           finetune_config: dict = None,
                           logger=None,
                           attention_type: str = None,
                           checkpoint_path: str = None,
                           print_model: bool = False,
                           merge_loras: bool = False,
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
                              train_attention, remove_base_attn, accelerator=accelerator, finetune_config=finetune_config)

    # Add low-rank adapters
    peft_key = 'peft'  # inconsistency across configs... why do this to myself
    if 'peft_config' in model_config['attention']:
        peft_key = 'peft_config'
    if peft_key in model_config['attention']:
        peft_config = model_config['attention'][peft_key]
        model, peft_config = create_peft_config(model, peft_config, 
                                                model_config['model']['torch_dtype'],
                                                preserve_requires_grad=train_converted,
                                                use_gradient_checkpointing=peft_gradient_checkpointing)
    else:
        peft_config = None

    for name, param in model.named_parameters():
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            param.requires_grad = False
        if accelerator is not None and accelerator.is_main_process:
            print(name) if logger is None else logger.info(name)
        else:
            print(name)

    """if print_model and rank == 0:  # Look at model
        print_header('*** Model before checkpoint load ***')
        print(model)"""

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

    """if print_model and rank == 0:  # Look at model
        print_header('*** Model ***')
        print(model)"""
    if merge_loras:
        model = model.merge_and_unload()
        if print_model and accelerator.is_main_process:
            print_header('*** Model (after merging adapters) ***', logger=logger)
            print(model) if logger is None else logger.info(model)
    if print_model and accelerator.is_main_process:  # Look at model
        print_header('*** Distill Trainable Parameters ***', logger=logger)
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})') if logger is None else logger.info(f'├── {n} (dtype = {p.dtype})')
    return model, peft_config

def load_and_convert_attns_shadow(model: nn.Module,
                           model_config: dict,
                           accelerator: Accelerator = None,
                           attention_type: str = None,
                           checkpoint_path: str = None,
                           print_model: bool = False,
                           merge_loras: bool = False,
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

    # Add low-rank adapters
    peft_key = 'peft'  # inconsistency across configs... why do this to myself
    if 'peft_config' in model_config['attention']:
        peft_key = 'peft_config'
    if peft_key in model_config['attention']:
        peft_config = model_config['attention'][peft_key]
        model, peft_config = create_peft_config(model, peft_config, 
                                                model_config['model']['torch_dtype'],
                                                preserve_requires_grad=train_converted,
                                                use_gradient_checkpointing=peft_gradient_checkpointing)
    else:
        peft_config = None

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

    if merge_loras:
        model = model.merge_and_unload()
        if print_model and accelerator.is_main_process:
            print_header('*** Model (after merging adapters) ***', logger=logger)
            print(model) if logger is None else logger.info(model)
    if print_model and accelerator.is_main_process:  # Look at model
        print_header('*** Trainable Parameters ***', logger=logger)
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})') if logger is None else logger.info(f'├── {n} (dtype = {p.dtype})')
    return model, peft_config

def load_and_convert_embedding_attns(model: nn.Module,
                           model_config: dict,
                           accelerator: Accelerator = None,
                           logger=None,
                           finetune_config: dict = None,
                           attention_type: str = None,
                           checkpoint_path: str = None,
                           print_model: bool = False,
                           merge_loras: bool = False,
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

    model = convert_embedding_attention(model, model_config['attention'], 
                              train_attention, remove_base_attn, accelerator=accelerator, finetune_config=finetune_config)

    # Add low-rank adapters
    peft_key = 'peft'  # inconsistency across configs... why do this to myself
    if 'peft_config' in model_config['attention']:
        peft_key = 'peft_config'
    if peft_key in model_config['attention']:
        peft_config = model_config['attention'][peft_key]
        model, peft_config = create_peft_config(model, peft_config, 
                                                model_config['model']['torch_dtype'],
                                                preserve_requires_grad=train_converted,
                                                use_gradient_checkpointing=peft_gradient_checkpointing)
    else:
        peft_config = None

    for name, param in model.named_parameters():
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            param.requires_grad = False
        #if accelerator.is_main_process:
        print(name) if logger is None else logger.info(name)

    """if print_model and rank == 0:  # Look at model
        print_header('*** Model before checkpoint load ***')
        print(model)"""

    # Load any trained attentions
    if checkpoint_path is not None:
        print(f'Loading weights from {checkpoint_path}...') if logger is None else logger.info(f'Loading weights from {checkpoint_path}...')
        state_dict = torch.load(checkpoint_path)['model_state_dict']
        _keys = model.load_state_dict(state_dict, strict=False)
        try:
            assert len(_keys.unexpected_keys) == 0
            #if accelerator.is_main_process:
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
            #if accelerator.is_main_process:
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
    """if print_model and rank == 0:  # Look at model
        print_header('*** Model ***')
        print(model)"""
    if merge_loras:
        model = model.merge_and_unload()
        if print_model and accelerator.is_main_process:
            print_header('*** Model (after merging adapters) ***', logger=logger)
            print(model) if logger is None else logger.info(model)
    if print_model and accelerator.is_main_process:  # Look at model
        print_header('*** Distill Trainable Parameters ***', logger=logger)
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})') if logger is None else logger.info(f'├── {n} (dtype = {p.dtype})')
    return model, peft_config

def load_and_convert_embedding_attns_shadow(model: nn.Module,
                           model_config: dict,
                           accelerator: Accelerator = None,
                           attention_type: str = None,
                           checkpoint_path: str = None,
                           print_model: bool = False,
                           merge_loras: bool = False,
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

    model = convert_embedding_attention_shadow(model, model_config['attention'], 
                              train_attention, remove_base_attn, accelerator=accelerator, logger=logger)

    # Add low-rank adapters
    peft_key = 'peft'  # inconsistency across configs... why do this to myself
    if 'peft_config' in model_config['attention']:
        peft_key = 'peft_config'
    if peft_key in model_config['attention']:
        peft_config = model_config['attention'][peft_key]
        model, peft_config = create_peft_config(model, peft_config, 
                                                model_config['model']['torch_dtype'],
                                                preserve_requires_grad=train_converted,
                                                use_gradient_checkpointing=peft_gradient_checkpointing)
    else:
        peft_config = None

    # Load any trained attentions
    if checkpoint_path is not None:
        if accelerator.is_main_process:
            print_info = f'Loading weights from {checkpoint_path}...'
            print(print_info) if logger is None else logger.info(print_info)
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

    if merge_loras:
        model = model.merge_and_unload()
        if print_model and accelerator.is_main_process:
            print_header('*** Model (after merging adapters) ***', logger=logger)
            print(model) if logger is None else logger.info(model)
    if print_model and accelerator.is_main_process:  # Look at model
        print_header('*** Trainable Parameters ***', logger=logger)
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n} (dtype = {p.dtype})') if logger is None else logger.info(f'├── {n} (dtype = {p.dtype})')
    return model, peft_config

def load_and_convert_finetune(model: nn.Module,
                              finetune_config: dict,
                              accelerator: Accelerator = None,
                              checkpoint_path: str = None,
                              print_model: bool = False,
                              merge_loras: bool = False,
                              peft_gradient_checkpointing: bool = None,
                              rank: int = 0,
                              logger=None,
                              **peft_kwargs: any):
    """
    Load trained adapter / model weights
    """
    # Add low-rank adapters
    peft_config = None
    if finetune_config.finetune.method == 'lora':
        if getattr(finetune_config.finetune, 'kwargs', None) is not None:#实验性注释，正常需要去掉---------------
            model, peft_config = create_peft_config(
                model, finetune_config.finetune,
                use_gradient_checkpointing=peft_gradient_checkpointing,
                **peft_kwargs,
            )
        for p in model.parameters():
            p.requires_grad = False
        # Keep specified weights trainable
        if 'trainable_weights' in finetune_config.finetune:
            for name in finetune_config.finetune['trainable_weights']:
                for n, p in model.named_parameters():
                    if name in n and "embedding_model" not in n:
                        p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        # Keep specified weights trainable
        if 'trainable_weights' in finetune_config.finetune:
            for name in finetune_config.finetune['trainable_weights']:
                for n, p in model.named_parameters():
                    if name in n and "embedding_model" not in n: 
                        if 'layers_to_ignore_language' in finetune_config.finetune:
                            layer = int(n.split('layers.')[-1].split('.')[0])
                            if layer not in finetune_config.finetune['layers_to_ignore']:
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

    if merge_loras:
        try:
            model = model.merge_and_unload()
            if print_model and accelerator.is_main_process:
                print_header('*** Model (after merging adapters) ***', logger=logger)
                print(model) if logger is None else logger.info(model)
        except Exception as e:
            print(e) if logger is None else logger.info(e)

    if print_model and accelerator.is_main_process:  # Look at model
        print_header('*** Finetune Trainable Parameters ***', logger=logger)
        count = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n}.requires_grad: {p.requires_grad}') if logger is None else logger.info(f'├── {n}.requires_grad: {p.requires_grad}')
                count += 1
        if count == 0:
            print('(none)') if logger is None else logger.info('(none)')

    return model, peft_config

def load_and_convert_embedding_finetune(model: nn.Module,
                              finetune_config: dict,
                              accelerator: Accelerator = None,
                              checkpoint_path: str = None,
                              print_model: bool = False,
                              merge_loras: bool = False,
                              peft_gradient_checkpointing: bool = None,
                              rank: int = 0,
                              logger=None,
                              **peft_kwargs: any):
    """
    Load trained adapter / model weights
    """
    # Add low-rank adapters
    peft_config = None
    if finetune_config.finetune.method == 'lora':
        if getattr(finetune_config.finetune, 'kwargs', None) is not None:#实验性注释，正常需要去掉---------------
            model, peft_config = create_peft_config(
                model, finetune_config.finetune,
                use_gradient_checkpointing=peft_gradient_checkpointing,
                **peft_kwargs,
            )
        for p in model.parameters():
            p.requires_grad = False
        # Keep specified weights trainable
        if 'trainable_weights' in finetune_config.finetune:
            for name in finetune_config.finetune['trainable_weights']:
                for n, p in model.named_parameters():
                    # if name in n and "language_model" not in n:
                    if name in n:
                        p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        # Keep specified weights trainable
        if 'trainable_weights' in finetune_config.finetune:
            for name in finetune_config.finetune['trainable_weights']:
                for n, p in model.named_parameters():
                    # if name in n and "language_model" not in n: 
                    if name in n: 
                        if 'layers_to_ignore_embedding' in finetune_config.finetune:
                            layer = int(n.split('layers.')[-1].split('.')[0])
                            if layer not in finetune_config.finetune['layers_to_ignore_embedding']:
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

    if accelerator is not None and accelerator.is_main_process:  
        print_header('*** Model ***', logger=logger)
        print(model) if logger is None else logger.info(model)
    else:
        print_header('*** Model ***')
        print(model)

    if merge_loras:
        try:
            model = model.merge_and_unload()
            if print_model and accelerator.is_main_process:
                print_header('*** Model (after merging adapters) ***', logger=logger)
                print(model) if logger is None else logger.info(model)
        except Exception as e:
            print(e) if logger is None else logger.info(e)

    if print_model and accelerator.is_main_process:  # Look at model
        print_header('*** Finetune Trainable Parameters ***', logger=logger)
        count = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f'├── {n}.requires_grad: {p.requires_grad}') if logger is None else logger.info(f'├── {n}.requires_grad: {p.requires_grad}')
                count += 1
        if count == 0:
            print('(none)') if logger is None else logger.info('(none)')

    return model, peft_config