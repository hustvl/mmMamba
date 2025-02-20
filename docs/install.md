# Installation for mmMamba
This repository was built using Python 3.10, you can create a new conda environment and install the dependencies by the following commands.
```bash
conda create -n mmMamba python=3.10
conda activate mmMamba
pip install -r requirements.txt
```

## Note: 

- If you encounter errors during the execution of `pip install -r requirements.txt`, please manually install the libraries mentioned in the requirements file one by one using the `pip install` command.
- If you encounter errors while installing the libraries `mamba-ssm`, `causal_conv1d`, and `flash_attn` using pip, please download the corresponding `.whl` files that match your environment from the website([mamba-ssm](https://github.com/state-spaces/mamba/releases),[causal_conv1d](https://github.com/Dao-AILab/causal-conv1d/releases),[flash_attn](https://github.com/Dao-AILab/flash-attention/releases)), and then install them using `pip install`.
