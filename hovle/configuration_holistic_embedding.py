# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
from typing import Union
import json

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class HolisticEmbeddingConfig(PretrainedConfig):

    model_type = 'holistic_embedding'

    def __init__(
            self,
            num_hidden_layers=32,
            initializer_factor=1e-5,
            use_autoregressive_loss=False,
            # vision embedding
            num_channels=3,
            patch_size=14,
            image_size=224,
            # attention layer
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            attention_bias=False,
            attention_dropout=0.0,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            rope_scaling=None,
            # mlp layer
            intermediate_size=11008,
            mlp_bias=False,
            hidden_act='silu',
            # rms norm
            rms_norm_eps=1e-5,
            # pretraining
            pretraining_tp=1,
            use_ls=True,
            use_img_start_end_tokens=True,
            special_token_maps={},
            llm_vocab_size=92553,
            llm_hidden_size=2048,
            attn_implementation='flash_attention_2',
            downsample_ratio=0.5,
            img_context_token_id=92546,
            pixel_shuffle_loc="pre",
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.initializer_factor = initializer_factor
        self.use_autoregressive_loss = use_autoregressive_loss

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.hidden_act = hidden_act

        self.rms_norm_eps = rms_norm_eps

        self.pretraining_tp = pretraining_tp   
        self.use_ls = use_ls
        self.use_img_start_end_tokens = use_img_start_end_tokens
        
        self.special_token_maps = special_token_maps
        self.llm_vocab_size = llm_vocab_size
        self.llm_hidden_size = llm_hidden_size
        self.attn_implementation = attn_implementation
        self.downsample_ratio = downsample_ratio
        self.img_context_token_id = img_context_token_id
        self.pixel_shuffle_loc = pixel_shuffle_loc

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'vision_config' in config_dict:
            config_dict = config_dict['vision_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)
    
    @classmethod
    def from_dict_path(cls, config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)
