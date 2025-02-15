# Getting Started

## Preparation of dataset
We perform the distillation recipe on [VLM-SFT dataset](https://huggingface.co/datasets/YangyiYY/VLM-SFT) curated by [SOLO](https://github.com/Yangyi-Chen/SOLO/blob/main/SFT_GUIDE.md).

```bash
huggingface-cli download --repo-type dataset yangyiy/VLM-SFT --local-dir ./data/vlm_sft
```

Then you should unzip all the chunk.tar.gz files into the 'images' folder.

## Preparation of teacher model HoVLE
We use the pre-trained [HoVLE](https://huggingface.co/OpenGVLab/HoVLE) model as the teacher model.

```bash
huggingface-cli download  OpenGVLab/HoVLE --local-dir ./weights/HoVLE
```

> [!NOTE] 
> If you do not have the access to huggingface, you can use the mirror site by setting the environment variable `export HF_ENDPOINT=https://hf-mirror.com`.






