#!/bin/bash


export PYTHONPATH="${PWD}:${PYTHONPATH}"  

accelerate launch \
  --config_file  "./configs/accelerate/default_config.yaml"  \
  --main_process_port  "11122"  \
    distill_mmMamba.py \
  --model_config distill_mmMamba \
  --distill_stage1_config distill_stage1_mmMamba \
  --checkpoint_dir ./checkpoints \
  --train_stage1  \
  --verbose \
  --seed 0 \
  --replicate 0   \
  --load_distill_checkpoint "path/to/ckpt(kept in stage1)"
