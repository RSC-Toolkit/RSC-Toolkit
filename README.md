# Respiratory Sound Classification Tool Kit

## 🚀 Quick Start

### 1\. Environment Setup

It's recommended to use a virtual environment. This project requires **Python \>= 3.9**.

```bash
# Create and activate a virtual environment (conda or venv)
conda create -n rsc_toolkit python=3.9
conda activate rsc_toolkit

# Install the toolkit from github
pip install git+https://github.com/RSC-Toolkit/RSC-Toolkit

# or install from source if you need to edit the code
git clone git+https://github.com/RSC-Toolkit/RSC-Toolkit
cd rsc_tool_kit
pip install -e .
```

### 2\. Dataset Setup

You can automatically download and extract the official ICBHI dataset with:

```bash
rsc-toolkit datasets download icbhi --datasets_download_path ./datasets
```
This will create ./datasets/ICBHI_final_database/ by default
```
.datasets/ICBHI_final_database/
  └── wavs/
      ├── 101_1b1_Al_sc_Meditron.wav
      └── ...
```

Next, update your `config.yaml` to point to this directory:

```yaml
# In config.yaml
datasets:
  trainer_dataset:
    data_folder: "path/to/ICBHI_final_database" # ⬅️ Update this path
```

### 3\. Run Training

Launch your first training run with the default configuration (EfficientNet-B0 with PatchMix-CL).

```bash
rsc-toolkit train
# or override with the config you want to use.
rsc-toolkit train configs/icbhi/intra/cnn6_ce.yaml
```

Outputs, logs, and model checkpoints will be saved to the directory specified by `trainer.output_dir` in the config.

-----

## 🐳 Docker

Build and run inside Docker:

```bash
# build base image
docker build -f docker/Dockerfile -t rsc-toolkit .
# build dev image (editable install)
docker build -f docker/Dockerfile.dev -t rsc-toolkit-dev .

# run interactively with all GPUs and mounted source
docker run --gpus all -v .:/src --rm -it rsc-toolkit-dev bash
# may need --shm-size=8g

# prepare dataset
rsc-toolkit dataset download icbhi
rsc-toolkit dataset download hflungv1
rsc-toolkit dataset download sprlung

# run training
rsc-toolkit train --config configs/icbhi/intra/bts_ce.yaml
rsc-toolkit train --config configs/hflungv1/intra/bts_ce.yaml
rsc-toolkit train --config configs/sprsound/intra/bts_ce.yaml
```

## Development

