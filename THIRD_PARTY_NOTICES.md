# Third-party notices

RSC-Toolkit is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE): use for research, education and other
noncommercial purposes is permitted; commercial use requires separate permission from the licensor.

Some files contain code ported or adapted from third-party projects. **That code remains under its original license**,
listed below; the full license texts are in [`LICENSES/`](LICENSES). Each such file starts with a comment that names its
source. The PolyForm Noncommercial terms apply to the rest of the repository.

## Code

| Upstream project | Files in this repository | License | Copyright |
|---|---|---|---|
| [patch-mix_contrastive_learning](https://github.com/raymin0223/patch-mix_contrastive_learning) | `rsc_toolkit/models/ast_patchmix.py`, `rsc_toolkit/methods/patchmix.py`, `rsc_toolkit/models/projector.py`, `rsc_toolkit/augmentations/specaugment.py`, `rsc_toolkit/trainer/utils/moving_average.py`, `rsc_toolkit/trainer/callbacks/ma_update.py`, `rsc_toolkit/trainer/callbacks/paper_cosine_lr.py`, `rsc_toolkit/trainer/callbacks/icbhi_best_checkpoint.py` | MIT ([text](LICENSES/MIT-patch-mix_contrastive_learning.txt)) | Copyright (c) 2025 Sangmin Bae |
| [AST](https://github.com/YuanGongND/ast) | `rsc_toolkit/models/backbone/ast.py`, `rsc_toolkit/models/ast.py`, `rsc_toolkit/models/ast_patchmix.py` | BSD-3-Clause ([text](LICENSES/BSD-3-Clause-ast.txt)) | Copyright (c) 2021, Yuan Gong |
| [AMS-D](https://github.com/CcnNnn/AMS-D) | `rsc_toolkit/models/ast_amsd.py`, `rsc_toolkit/methods/amsd.py`, `rsc_toolkit/metrics/icbhi/binary_score.py`, `rsc_toolkit/datasets/amsd_data.py`, `rsc_toolkit/trainer/callbacks/amsd_best_checkpoint.py`, `rsc_toolkit/trainer/callbacks/amsd_epoch.py` | MIT ([text](LICENSES/MIT-AMS-D.txt)) | Copyright (c) 2026 Nuo Chen |
| [RespireNet](https://github.com/microsoft/RespireNet) | `rsc_toolkit/models/respirenet.py`, `rsc_toolkit/processors/respirenet_image.py`, `rsc_toolkit/datasets/respirenet_chunks.py`, `rsc_toolkit/trainer/callbacks/respirenet_vote_eval.py`, `rsc_toolkit/augmentations/respirenet_raw.py` | MIT ([text](LICENSES/MIT-RespireNet.txt)) | Copyright (c) Microsoft Corporation |
| [nlpaug](https://github.com/makcedward/nlpaug) 0.0.14 | `rsc_toolkit/augmentations/respirenet_raw.py` | MIT ([text](LICENSES/MIT-nlpaug.txt)) | Copyright (c) 2019 Edward Ma |
| [PrototypeSound](https://github.com/L3S/PrototypeSound) | `rsc_toolkit/models/protonet.py`, `rsc_toolkit/datasets/protonet_data.py`, `rsc_toolkit/trainer/callbacks/protonet_class_weight.py` | MIT ([text](LICENSES/MIT-PrototypeSound.txt)) | Copyright (c) 2021 L3S Research Center |
| [PANNs (audioset_tagging_cnn)](https://github.com/qiuqiangkong/audioset_tagging_cnn) | `rsc_toolkit/models/cnn14_gapaug.py`, `rsc_toolkit/models/backbone/cnn.py` | MIT ([text](LICENSES/MIT-audioset_tagging_cnn.txt)) | Copyright (c) 2018-2020 Qiuqiang Kong |
| [OPERA](https://github.com/evelyn0414/OPERA), based on [HTS-AT](https://github.com/RetroCirce/HTS-Audio-Transformer) and [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | `rsc_toolkit/models/opera/htsat.py` | MIT ([OPERA](LICENSES/MIT-OPERA.txt), [HTS-AT](LICENSES/MIT-HTS-Audio-Transformer.txt), [Swin](LICENSES/MIT-Swin-Transformer.txt)) | Copyright (c) 2024 OPERA; Copyright (c) 2022 Knut(Ke) Chen; Copyright (c) Microsoft Corporation |
| [BEATs (unilm)](https://github.com/microsoft/unilm/tree/master/beats) | `rsc_toolkit/models/beats_upstream/` | MIT ([text](LICENSES/MIT-unilm.txt)) | Copyright (c) Microsoft Corporation |
| [VMamba](https://github.com/MzeroMiko/VMamba) | `rsc_toolkit/models/backbone/vmamba/vssm.py` | MIT ([text](LICENSES/MIT-VMamba.txt)) | Copyright (c) 2024 MzeroMiko |
| [DASS](https://github.com/Saurabhbhati/DASS) | `rsc_toolkit/models/backbone/vmamba/vssm.py`, `rsc_toolkit/models/backbone/vmamba/cross_scan.py`, `rsc_toolkit/models/backbone/vmamba/selective_scan.py` | BSD-3-Clause ([text](LICENSES/BSD-3-Clause-DASS.txt)) | Copyright (c) 2024, Saurabhchand Bhati |
| [Mamba](https://github.com/state-spaces/mamba) (reference selective scan) | `rsc_toolkit/models/backbone/vmamba/selective_scan.py` | Apache-2.0 ([text](LICENSES/Apache-2.0.txt)) | Copyright 2023 Tri Dao, Albert Gu |
| [BTS-CARD](https://github.com/RSC-Toolkit/BTS-CARD) | `rsc_toolkit/models/bts_card.py`, `rsc_toolkit/datasets/btscard_sprs.py`, `rsc_toolkit/datasets/sprsound_ood.py`, `rsc_toolkit/trainer/callbacks/btscard.py` | Apache-2.0 ([text](LICENSES/Apache-2.0.txt)) | Copyright the BTS-CARD authors |
| [SupCon-DAT](https://github.com/RSC-Toolkit/SupCon-DAT) | `rsc_toolkit/datasets/sprsound/data/generate_annotation.py` | MIT ([text](LICENSES/MIT-SupCon-DAT.txt)) | Copyright (c) 2025 Respiratory Sound Classification Toolkit |
| [IBN-Net](https://github.com/XingangPan/IBN-Net) | `rsc_toolkit/models/resnest_ibn_domaintransfer.py` (`IBN` block) | MIT ([text](LICENSES/MIT-IBN-Net.txt)) | Copyright (c) 2017 Wei Yang |
| [AudioMAE](https://github.com/facebookresearch/AudioMAE), including a patched window attention from [timm](https://github.com/huggingface/pytorch-image-models) | `rsc_toolkit/models/audiomae_dde.py` | CC BY-NC 4.0 ([text](LICENSES/CC-BY-NC-4.0.txt)); timm parts Apache-2.0 ([text](LICENSES/Apache-2.0.txt)) | Copyright (c) Meta Platforms, Inc. and affiliates; Copyright 2019 Ross Wightman |
| [AFNO-transformer](https://github.com/NVlabs/AFNO-transformer) | `rsc_toolkit/models/addrsc/denoise.py` (`AFNO1D`) | CC BY-NC-SA 4.0 ([text](LICENSES/CC-BY-NC-SA-4.0.txt)) | Copyright (c) NVIDIA Corporation |
| [EVAR (eval-audio-repr)](https://github.com/nttcslab/eval-audio-repr) | `rsc_toolkit/models/opera_ct.py` (`TfmEncHead`, `MLPHead`), `scripts/reproduction/dump_opera_goldens.py` | NTT Software License Agreement for Evaluation ([upstream](https://github.com/nttcslab/eval-audio-repr/blob/main/LICENSE)) | Copyright Nippon Telegraph and Telephone Corporation |
| [M2D](https://github.com/nttcslab/m2d) | `rsc_toolkit/models/m2d/encoder.py` | NTT Software License Agreement for Evaluation ([upstream](https://github.com/nttcslab/m2d/blob/master/LICENSE.pdf)) | Copyright Nippon Telegraph and Telephone Corporation |
| [MAE](https://github.com/facebookresearch/mae) (sin-cos position embedding) | `rsc_toolkit/models/m2d/encoder.py` | CC BY-NC 4.0 ([text](LICENSES/CC-BY-NC-4.0.txt)) | Copyright (c) Meta Platforms, Inc. and affiliates |

Changes: the code listed above was modified to fit the toolkit's model, datamodule and callback interfaces; each file's
header comment and docstring describe the differences from the original.

## Data files

| Files | Source | License |
|---|---|---|
| `rsc_toolkit/datasets/icbhi/data/*.txt` | ICBHI 2017 Respiratory Sound Database metadata and split files, identical to the copies distributed with patch-mix_contrastive_learning and RespireNet | as distributed by those repositories (MIT) |
| `rsc_toolkit/datasets/hflungv1/data/` | [HF_Lung_V1](https://gitlab.com/techsupportHF/HF_Lung_V1) | CC BY 4.0 (`rsc_toolkit/datasets/hflungv1/data/LICENSE`) |
| `rsc_toolkit/datasets/sprsound/data/metadata.csv` | derived from [SPRSound](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound) | CC BY 4.0 |
| `rsc_toolkit/processors/data/magma_lut_bgr.npy` | matplotlib "magma" colormap (Nathaniel J. Smith, Stéfan van der Walt, Eric Firing) | CC0 1.0 |

## Pre-trained weights

Pre-trained backbones are not included in this repository; they are downloaded from their original sources and keep
their own licenses. Released fine-tuned weights (see [`weights_manifest.csv`](weights_manifest.csv)) inherit the terms of
the backbone they were fine-tuned from.
