#!/bin/bash


export PYTHONPATH="${PWD}:${PYTHONPATH}"  

accelerate launch \
  --config_file  "./configs/accelerate/default_config.yaml"  \
  --main_process_port  "11122"  \
    distill_mmMamba.py \
  --model_config distill_mmMamba \
  --distill_stage3_config distill_stage3_mmMamba \
  --checkpoint_dir ./checkpoints \
  --train_stage3  \
  --verbose \
  --seed 0 \
  --replicate 0   \
  --load_distill_checkpoint "path/to/ckpt(inherit from stage2 or kept in stage3)"
