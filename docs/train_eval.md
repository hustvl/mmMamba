# mmMamba Training and Evaluation

## Training

Our code includes a three-stage training process. You can switch between different distillation stages by changing the command-line parameters.

### Main Command-Line Parameters

#### Training Configuration Files

- `--distill_stage1_config` : Configuration file for Stage 1.
- `--distill_stage2_config` : Configuration file for Stage 2.
- `--distill_stage3_config` : Configuration file for Stage 3.

#### Training Stage Switches

- `--train_stage1` : Whether to conduct training for Stage 1 (default: `false`).
- `--train_stage2` : Whether to conduct training for Stage 2 (default: `false`).
- `--train_stage3` : Whether to conduct training for Stage 3 (default: `false`).

#### Checkpoint load Path

- `--load_distill_checkpoint` : Path to load the distillation checkpoint.
  
#### Checkpoint store Path

- `--checkpoint_dir` : Path to store the distillation checkpoint.


Our code currently supports training in phases and does not support completing the entire training process in one go. You can specify which phase of training to execute by providing command-line arguments. Checkpoints will be periodically saved during each phase of training. Below are the script commands for executing the three phases of training:

```bash
bash scripts/distill_stage1.sh
```
```bash
bash scripts/distill_stage2.sh
```
```bash
bash scripts/distill_stage3.sh
```
#### Notes
- The --load_distill_checkpoint in the script is optional and is used to import an existing checkpoint. Please replace it with the corresponding ckpt path.
- In Stage 1 and Stage 2, all layers of the model are replaced with Mamba2 and aligned through distillation. 
- In Stage 3, you can determine which layers to retain as the original multi-head attention layers by customizing the `distill_stage3/softmax_attention` list in `configs/experiment/distill_stage3_mmMamba.yaml` to train the mmMamba-Hybrid model.

## Evaluation

We use VLMEvalKit and InternVL Evalkit to evaluate our model. Among the metrics, GQA and POPE are evaluated using InternVL Evalkit, while the other metrics are assessed with VLMEvalKit.  Below is the detailed evaluation process:
### VLMEvalkit
We provides a step-by-step guide to evaluate the mmMamba model using VLMEvalKit.

#### Step-by-Step Evaluation Process

1. **Download the mmMamba Model Weights**
   - Download the mmMamba model weights from Hugging Face.

2. **Clone the VLMEvalKit Repository**
   - Clone the VLMEvalKit repository from its [GitHub repository](https://github.com/open-compass/VLMEvalKit).

3. **Integrate mmMamba Model into VLMEvalKit**
   - Copy `mmMamba.py` from `./eval/mmMamba.py` to `VLMEvalKit/vlmeval/vlm/`.
   - Add the `mmMambaChat` class to `VLMEvalKit/vlmeval/vlm/__init__.py` with:
     ```python
     from .mmMamba import mmMambaChat
     ```

4. **Add mmMamba Model to Configuration**
   - Add the mmMamba model to `VLMEvalKit/vlmeval/config.py` with:
     ```python
     'mmMamba': partial(mmMambaChat, model_path='/path/to/mmMamba', version='V2.0')
     ```

5. **Run the Evaluation**
   - Run the evaluation with the following command:
     ```bash
     torchrun \
         --nproc-per-node=${GPUS} \
         --master_port=${MASTER_PORT} \
         run.py --data <DatasetName> --model mmMamba
     ```

#### Notes

- Replace `<DatasetName>` with the actual name of the dataset you want to use for evaluation as what is provided in VLMEvalKit.
- Ensure that you have the necessary dependencies installed and that your environment is correctly set up for running the evaluation.
- Adjust the paths and parameters as needed to fit your specific setup and requirements.

By following these steps, you should be able to successfully evaluate the mmMamba model using the provided tools and configurations.

### InternVL Evalkit
This README provides a step-by-step guide to evaluate the mmMamba model using InternVL2 for the GQA and POPE metrics.

#### Step-by-Step Evaluation Process

1. **Download the mmMamba Model Weights**
   - Download the mmMamba model weights from Hugging Face

2. **Clone the InternVL Repository**
   - Clone the [InternVL repository](https://github.com/internvl/internvl) and navigate to the `internvl_chat` directory.
  
3. **Modify the InternVL Model Initialization**
   - Change lines 46-48 in `internvl_chat/internvl/model/__init__.py` to:
     ```python
     model = AutoModel.from_pretrained(
         args.checkpoint,
         torch_dtype=torch.bfloat16,
         low_cpu_mem_usage=False,
         trust_remote_code=True).eval()
     ```

4. **Follow the evaluation process**
   - Follow the evaluation process outlined in the [InternVL2 Series Evaluation documentation](https://internvl.readthedocs.io/en/latest/internvl2.0/evaluation.html).

#### Notes

- Ensure that you have the necessary dependencies installed and that your environment is correctly set up for running the evaluation.
- Adjust the paths and parameters as needed to fit your specific setup and requirements.

By following these steps, you should be able to successfully evaluate the mmMamba model using InternVL2 for the specified metrics.

