import sys
import os
import argparse
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import numpy as np
import json

sys.path.append('./src')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.model.load_model import load_and_convert_distill_stage3, load_and_convert_attns
from src.model.convert_model import remove_base_attention, toggle_attention
from hovle.modeling_internvl_chat import InternVLChatModel



B_INST, E_INST = "[INST]", "[/INST]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_path', type=str, required=True,
                        help='Path to teacher model')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to first checkpoint')
    parser.add_argument('--final_model_path', type=str, required=True,
                        help='Path to final model')
    parser.add_argument('--name', type=str, required=True,
                        help='Output name for saving model checkpoint')
    parser.add_argument('--model_config', type=str, required=True,
                        help='the path to model_config')
    parser.add_argument('--distill_stage3_config', type=str, required=True,
                        help='the path to model_config')
    args = parser.parse_args()

    model_config = OmegaConf.load(args.model_config)
    distill_stage3_config = OmegaConf.load(args.distill_stage3_config)

    model = InternVLChatModel.from_pretrained(
                args.teacher_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path, trust_remote_code=True, use_fast=False)

    model = load_and_convert_attns(model, model_config, 
                                    attention_type='mamba2', 
                                    checkpoint_path=args.ckpt_path, 
                                    print_model=None,
                                    train_attention=False)
    
    model = toggle_attention(model, train=False)
    model = remove_base_attention(model)

    model = load_and_convert_distill_stage3(model, distill_stage3_config, 
                                                            checkpoint_path=args.ckpt_path)
    
    for layer in model.embedding_model.encoder:
        if hasattr(layer.attention, 'change_weight'):
            layer.attention.change_weight()

    for layer in model.language_model.model.layers:
        if hasattr(layer.attention, 'change_weight'):
            layer.attention.change_weight()     
    
    model.eval()
    ckpt_path = f"{args.final_model_path}/{args.name}"

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)

    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path) 

    config_file = f"{ckpt_path}/config.json"
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["_name_or_path"] = ckpt_path
    config["architectures"] = "mmMambaChatModel"
    config["auto_map"]["AutoConfig"] = "configuration_mmMamba_chat.mmMambaChatConfig"
    config["auto_map"]["AutoModel"] = "modeling_mmMamba_chat.mmMambaChatModel"
    config["auto_map"]["AutoModelForCausalLM"] = "modeling_mmMamba_chat.mmMambaChatModel"
    config["embedding_config"]["model_type"] = "mmMamba_embedding"
    config["llm_config"]["architectures"] = "mmMambaForCausalLM"
    config["llm_config"]["auto_map"]["AutoConfig"] = "configuration_mmMamba.mmMambaConfig"
    config["llm_config"]["auto_map"]["AutoModel"] = "modeling_mmMamba.mmMambaForCausalLM"
    config["llm_config"]["auto_map"]["AutoModelForCausalLM"] = "modeling_mmMamba.mmMambaForCausalLM"
    config["llm_config"]["model_type"] = "mmMamba"
    config["model_type"] = "mmMamba_chat"

    config["embedding_config"]["layers_block_type"] = ["mamba2","mamba2","mamba2","mamba2","mamba2","mamba2","mamba2","mamba2"]
    config["llm_config"]["layers_block_type"] = ["mamba2","mamba2","mamba2","mamba2","mamba2","mamba2","mamba2","mamba2",
                                                 "mamba2","mamba2","mamba2","mamba2","mamba2","mamba2","mamba2","mamba2",
                                                 "mamba2","mamba2","mamba2","mamba2","mamba2","mamba2","mamba2","mamba2"]
    embedding_softmax_attention_list = [x for x in distill_stage3_config["distill_stage3"]["softmax_attention"] if x < 8]
    llm_softmax_attention_list = [x for x in distill_stage3_config["distill_stage3"]["softmax_attention"] if x >= 8]
    for index in embedding_softmax_attention_list:
        config["embedding_config"]["layers_block_type"][index] = "mha"
    for index in llm_softmax_attention_list:
        config["llm_config"]["layers_block_type"][index-8] = "mha"


    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print("done")
    return {}
    

main()


