# Post-training Weight Converting and Inference with Transformers
## Post-training Weight Converting

After the training is completed, you need to perform the final step to convert your saved checkpoint into Hugging Face model weight files. 

This guide provides instructions for converting model weights using the provided script `convert_weight.py`. This process is essential for preparing your model for deployment or further inference.

### Usage

To convert model weights, use the following command:

```python
python convert_weight.py \
      --teacher_path "path/to/HoVLE"  \
      --ckpt_path  "path/to/checkpoint"  \
      --final_model_path "/path/to/huggingface_model"  \
      --name "the name you wish to give to the converted model."  \
      --model_config  configs/model/distill_mmMamba.yaml  \
      --distill_stage3_config configs/experiment/distill_stage3_mmMamba.yaml
```

### Command-Line Arguments
- --teacher_path: Path to the teacher model weights.
- --ckpt_path: Path to the checkpoint file you want to convert.
- --final_model_path: Path where the converted model will be saved.
- --name: The name you wish to give to the converted model..
- --model_config: Path to the model configuration file.
- --distill_stage3_config: Path to the distillation stage 3 configuration file.

### Detailed Steps
#### 1.Prepare the Paths
- Ensure that you have the correct paths for the teacher model, checkpoint, and where you want to save the final model.
- Update the paths in the command according to your file system structure.

#### 2.Run the Conversion Script
- Execute the command in your terminal or command prompt.
- The script will perform the necessary conversions and save the model to the specified path.

#### 3.Verify the Conversion
- After the script completes, check the specified final_model_path to ensure the model has been saved correctly.

#### 4.Add model execution code:
Include these Python files from [the Hugging Face model repository](https://huggingface.co/hustvl/mmMamba-linear) into your model weight file:
- configuration_mmMamba.py
- configuration_mmMamba_chat.py
- configuration_mmMamba_embedding.py
- conversation.py
- fused_norm_gate.py
- modeling_mmMamba.py
- modeling_mmMamba_chat.py
- modeling_mmMamba_embedding.py

After this, you can follow the steps in the "Quick Start" section below to perform inference tasks with your own trained model.


## Quick Start Guide for mmMamba Inference

We provide example code to run mmMamba inference using the Transformers library.


### Main Dependencies for Model Inference

Below are the primary dependencies required for model inference:
- torch==2.1.0
- torchvision==0.16.0
- torchaudio==2.1.0
- transformers==4.37.2
- peft==0.10.0
- triton==3.2.0
- [mamba_ssm](https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
- [causal_conv1d](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8%2Bcu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
- [flash_attn](https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.0/flash_attn-2.6.0%2Bcu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)
(Please note that you need to select and download the corresponding .whl file based on your environment.)
- peft
- omegaconf
- rich
- accelerate
- sentencepiece
- decord
- seaborn


### Inference with Transformers

```python
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


path = 'hustvl/mmMamba-linear'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
pixel_values = load_image('/path/to/image', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)

# pure-text conversation (纯文本对话)
question = 'Hello, who are you?'
response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')


# single-image single-round conversation (图文对话)
question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')
```

### Note: 
The model path can point to either the model we uploaded on Hugging Face or the model converted by yourself in "Post-training Weight Converting" section.
