#!/bin/bash


export PYTHONPATH="${PWD}:${PYTHONPATH}"  

accelerate launch \
  --config_file  "./configs/accelerate/default_config.yaml"  \
  --num_processes   "4"  \
  --main_process_port  "11122"  \
    distill_mmMamba.py \
  --model_config ./configs/model/distill_hovle_7b_lk_smd_fd64 \
  --distill_stage1_config ./configs/experiment/distill_stage1_mmMamba \
  --distill_stage2_config ./configs/experiment/distill_stage2_mmMamba \
  --distill_stage3_config ./configs/experiment/distill_stage3_mmMamba \
  --checkpoint_dir ./checkpoints \
  --train_stage1  \
  --train_stage2  \
  --train_stage3  \
  --lk_zero_init \
  --verbose \
  --seed 0 \
  --replicate 0
