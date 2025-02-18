"""
Attention conversion helpers
"""
from functools import partial
from tqdm import tqdm
import torch.nn as nn
from accelerate import Accelerator


def convert_attention(model: nn.Module, 
                      attention_config: dict, 
                      train_attention: bool = False,
                      remove_base_attn: bool = True,
                      accelerator: Accelerator = None,
                      distill_stage3_config: dict = None):
    """
    Call to convert all attention layers
    """

    if attention_config.attention_type != 'softmax':
        layers = traverse_layers(model)
        embedding_layers = traverse_embedding_layers(model)
        for layer_idx, layer in enumerate(tqdm(embedding_layers, disable=not accelerator.is_main_process if accelerator is not None else False)):
            if distill_stage3_config is None or layer_idx not in distill_stage3_config['distill_stage3']['softmax_attention']:
                if attention_config.attention_name == "self_attn":
                    layer.self_attn = convert_quadratic_to_linear(
                        layer, attention_config, embedding_layers, train_attention, remove_base_attn,
                    )
                    layer.self_attn.converted = True
                else:
                    layer.attention = convert_quadratic_to_linear(
                        layer, attention_config, embedding_layers, train_attention, remove_base_attn,
                    )
                    layer.attention.converted = True
            else:  # Freeze any preserved softmax attention layers
                for p in layer.parameters():
                    p.requires_grad = False

        for layer_idx, layer in enumerate(tqdm(layers, desc='Converting attentions...', disable=not accelerator.is_main_process if accelerator is not None else False)):
            layer_idx = layer_idx + 8
            if distill_stage3_config is None or layer_idx not in distill_stage3_config['distill_stage3']['softmax_attention']:
                if attention_config.attention_name == "self_attn":
                    layer.self_attn = convert_quadratic_to_linear(
                        layer, attention_config, layers, train_attention, remove_base_attn,
                    )
                    layer.self_attn.converted = True
                else:
                    layer.attention = convert_quadratic_to_linear(
                        layer, attention_config, layers, train_attention, remove_base_attn,
                    )
                    layer.attention.converted = True
            else:  # Freeze any preserved softmax attention layers
                for p in layer.parameters():
                    p.requires_grad = False
                    
    else:
        #if accelerator.is_main_process:
        print(f'-> attention_config.attention_type is {attention_config.attention_type}; not converting attentions')
    return model


def convert_attention_shadow(model: nn.Module, 
                      attention_config: dict, 
                      train_attention: bool = False,
                      remove_base_attn: bool = True,
                      accelerator: Accelerator = None,
                      logger=None):
    """
    Call to convert all attention layers
    """
    if attention_config.attention_type != 'softmax':
        layers = traverse_layers(model)
        embedding_layers = traverse_embedding_layers(model)
        logger.info(f'-> Converting shadow attentions...')

        for layer_idx, layer in enumerate(tqdm(embedding_layers, desc='Converting shadow attentions...', disable=not accelerator.is_main_process)):
            if attention_config.attention_name == "self_attn":
                layer.self_attn = convert_quadratic_to_linear(
                    layer, attention_config, embedding_layers, train_attention, remove_base_attn,
                )
                layer.self_attn.converted = True
                layer.self_attn.create_shadowweight()
            else:
                layer.attention = convert_quadratic_to_linear(
                    layer, attention_config, embedding_layers, train_attention, remove_base_attn,
                )
                layer.attention.converted = True
                layer.attention.create_shadowweight()


        for layer_idx, layer in enumerate(tqdm(layers, desc='Converting shadow attentions...', disable=not accelerator.is_main_process)):
            layer_idx = layer_idx + 8
            if attention_config.attention_name == "self_attn":
                layer.self_attn = convert_quadratic_to_linear(
                    layer, attention_config, layers, train_attention, remove_base_attn,
                )
                layer.self_attn.converted = True
                layer.self_attn.create_shadowweight()
            else:
                layer.attention = convert_quadratic_to_linear(
                    layer, attention_config, layers, train_attention, remove_base_attn,
                )
                layer.attention.converted = True
                layer.attention.create_shadowweight()

    else:
        if accelerator.is_main_process: 
            print_info = f'-> attention_config.attention_type is {attention_config.attention_type}; not converting attentions'
            print(print_info) if logger is None else logger.info(print_info)
    return model


def toggle_attention(model: nn.Module, train: bool = False):
    """
    Make attentions trainable if train is True
    -> Set train_attention = False when finetuning
    """
    for layer in traverse_layers(model):
        try:
            layer.self_attn.train_attention = train
        except:
            layer.attention.train_attention = train

    for layer in traverse_embedding_layers(model):
        try:
            layer.self_attn.train_attention = train
        except:
            layer.attention.train_attention = train

    return model

def remove_base_attention(model: nn.Module):
    """
    Remove teacher attention after distillation (if we keep it)
    """
    for layer in traverse_layers(model):
        try:
            if getattr(layer.self_attn, 'base_attn', False):
                del layer.self_attn.base_attn
        except:
            if getattr(layer.attention, 'base_attn', False):
                del layer.attention.base_attn

    for layer in traverse_embedding_layers(model):
        try:
            if getattr(layer.self_attn, 'base_attn', False):
                del layer.self_attn.base_attn
        except:
            if getattr(layer.attention, 'base_attn', False):
                del layer.attention.base_attn

    return model
        

def traverse_layers(model: nn.Module, verbose: bool = False):
    """
    Return list of model layers
    """
    try:
        layers = model.model.layers
        if verbose:
            print('-> Loading from model.model.layers')
    except AttributeError as e: # if base model
        if verbose:
            print(e)
        try:
            layers = model.layers
            if verbose:
                print('-> Loading from model.layers')
        except AttributeError as e1:  # If we make a PEFT model
            if verbose:
                print(e1)
            try:
                layers = model.base_model.model.model.layers
                if verbose:
                    print('-> Loading from model.base_model.model.model.layers')
            except AttributeError as e2:
                if verbose:
                    print(e2)
                layers = model.language_model.model.layers
                if verbose:
                    print('-> Loading from model.language_model.model.layers')

    return layers

def traverse_embedding_layers(model: nn.Module, verbose: bool = False):
    """
    Return list of model layers
    """
    layers = model.embedding_model.encoder

    return layers


def convert_quadratic_to_linear(layer: nn.Module,
                            attention_config: dict,
                            layers: list[nn.Module],  # list of layers
                            train_attention: bool = False,
                            remove_base_attn: bool = True):
    """
    Converts a single layer's attention layer as specified by attention_config
    """
    try:
        return get_attention(**attention_config)(
            base_attn=layer.self_attn,
            layer_idx=layer.self_attn.layer_idx,  # Transformers v4.36
            max_layer_idx=len(layers) - 1,
            train_attention=train_attention,
            remove_base_attn=remove_base_attn,
            use_D=attention_config.mamba2.use_D,
            use_qknorm=attention_config.mamba2.use_qknorm,
            use_conv=attention_config.mamba2.use_conv,
            use_gnorm=attention_config.mamba2.use_gnorm,
            use_A=attention_config.mamba2.use_A,
            inherit_qkv=attention_config.mamba2.inherit_qkv,
            mimic_init=attention_config.mamba2.mimic_init,
            stage1=attention_config.stage1,
            stage2=attention_config.stage2,
        )
    except:
        return get_attention(**attention_config)(
            base_attn=layer.attention,
            layer_idx=layer.attention.layer_idx,  # Transformers v4.36
            max_layer_idx=len(layers) - 1,
            train_attention=train_attention,
            remove_base_attn=remove_base_attn,
            use_D=attention_config.mamba2.use_D,
            use_qknorm=attention_config.mamba2.use_qknorm,
            use_conv=attention_config.mamba2.use_conv,
            use_gnorm=attention_config.mamba2.use_gnorm,
            use_A=attention_config.mamba2.use_A,
            inherit_qkv=attention_config.mamba2.inherit_qkv,
            mimic_init=attention_config.mamba2.mimic_init,
            stage1=attention_config.stage1,
            stage2=attention_config.stage2,
        )


def get_attention(attention_type: str, **kwargs: any):
    kwargs['attention_type'] = attention_type

    if attention_type == 'mamba2':
        from .linear_attention import Mamba2_Attention
        return partial(Mamba2_Attention, **kwargs)
    
    else:
        print(f'-> attention_type {attention_type} not handled... returning None')
        return None


def get_attention_cache(attention_type: str, past_key_values: any = None):
    """
    Determine how we store past keys and values when generating
    """
    if attention_type is None:
        return past_key_values
    
    elif 'softmax' in attention_type:
        return past_key_values
