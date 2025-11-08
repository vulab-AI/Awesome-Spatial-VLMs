# All in Tokens: Unifying Output Space of Visual Tasks via Soft Token

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/all-in-tokens-unifying-output-space-of-visual/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=all-in-tokens-unifying-output-space-of-visual)

By [Jia Ning](https://scholar.google.com/citations?user=hW0AexsAAAAJ&hl=en)\*, [Chen Li](https://github.com/LC-Edward)\*, [Zheng Zhang*](https://stupidzz.github.io/), [Zigang Geng](https://scholar.google.com/citations?user=MdFYVoAAAAAJ&hl=zh-CN), [Qi Dai](https://scholar.google.com/citations?user=NSJY12IAAAAJ), [Kun He](https://scholar.google.com/citations?user=YTQnGJsAAAAJ&hl=en), [Han Hu](https://ancientmooner.github.io/)

## Introduction
**AiT** is initially described in [arxiv](https://arxiv.org/pdf/2301.02229.pdf), which is a framework to unify the output space of visual tasks. We demonstrate a single unified model that simultaneously handles two typical visual tasks of instance segmentation and depth estimation, which have discrete/fixed-length and continuous/varied-length outputs, respectively. We propose several new techniques that take into account the particularity of visual tasks: 1) Soft tokens. We employ soft tokens to represent the task output. Unlike hard tokens in the common VQ-VAE which are assigned one-hot to discrete codebooks/vocabularies, the soft tokens are assigned softly to the codebook embeddings. Soft tokens can improve the accuracy of both the next token inference and decoding the task output; 2) Mask augmentation. Many visual tasks have corruption, undefined or invalid values in label annotations, i.e., occluded area of depth maps. We show that a mask augmentation technique can greatly benefit these tasks. With these new techniques and other designs, we show that the proposed general-purpose task solver can perform both instance segmentation and depth estimation well. Particularly, we achieve 0.275 RMSE on the specific task of NYUv2 depth estimation, setting a new record on this benchmark.

![teaser](figures/teaser.png)

## Results and Models
### Results on COCO instance segmentation
| <div style="width: 100pt"> Model | Box AP| Mask AP| VQ-VAE Model | Task-Solver Model|
|:-------------------:|:-------:|:-------:|:-------:|:-------:|
| [AiT(SwinV2-B)](ait/configs/swinv2b_640reso_inssegonly.py) | 43.3 | 34.2 | [vqvae_insseg.pt](https://msravcghub.blob.core.windows.net/ait-release/vae/vqvae_insseg.pt?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) | [model](https://msravcghub.blob.core.windows.net/ait-release/checkpoint/ait_insseg_swinv2b.pth?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D)|
| [AiT(SwinV2-B) w/o soft token](ait/configs/swinv2b_640reso_inssegonly_wosoft.py) | 43.6 | 31.1(-3.1) | [vqvae_insseg.pt](https://msravcghub.blob.core.windows.net/ait-release/vae/vqvae_insseg.pt?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) | [model](https://msravcghub.blob.core.windows.net/ait-release/checkpoint/ait_insseg_swinv2b_wosoft.pth?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) |


### Results on NYUv2 depth estimation
| <div style="width: 100pt"> Model</div> | D1 | D2 | D3 | Abs Rel | RMSE | Log10 | VQ-VAE <br> Model | Task-Solver <br> Model |
|:-------------------:|:-------:|:-------:|:--------:|:--------:|:--------:|:-------:|:-------:|:-------:|
| [AiT(SwinV2-B)](ait/configs/swinv2b_480reso_depthonly.py) | 0.934 | 0.991 | 0.998 | 0.087 | 0.305 | 0.037 | [vqvae_depth.pt](https://msravcghub.blob.core.windows.net/ait-release/vae/vqvae_depth.pt?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) |[model](https://msravcghub.blob.core.windows.net/ait-release/checkpoint/ait_depth_swinv2b_ar.pth?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) |
| [AiT-P(SwinV2-B)](ait/configs/swinv2b_480reso_parallel_depthonly.py) | 0.940 | 0.992 | 0.998 | 0.085 | 0.301 | 0.036 | [vqvae_depth.pt](https://msravcghub.blob.core.windows.net/ait-release/vae/vqvae_depth.pt?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) | [model](https://msravcghub.blob.core.windows.net/ait-release/checkpoint/ait_depth_swinv2b_parallel.pth?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) |
| [AiT(SwinV2-B) w/o soft token](ait/configs/swinv2b_480reso_depthonly_wosoft.py) | 0.932 | 0.991 | 0.998 | 0.089 | 0.318 | 0.038 | [vqvae_depth.pt](https://msravcghub.blob.core.windows.net/ait-release/vae/vqvae_depth.pt?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) | [model](https://msravcghub.blob.core.windows.net/ait-release/checkpoint/ait_depth_swinv2b_ar_wosoft.pth?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) |
| [AiT(SwinV2-L)](ait/configs/swinv2l_480reso_depthonly.py) | 0.949 | 0.993 | 0.999 | 0.079 | 0.284 | 0.034 | [vqvae_depth.pt](https://msravcghub.blob.core.windows.net/ait-release/vae/vqvae_depth.pt?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) |[model](https://msravcghub.blob.core.windows.net/ait-release/checkpoint/ait_depth_swinv2l_ar.pth?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) |
| [AiT-P(SwinV2-L)](ait/configs/swinv2l_480reso_parallel_depthonly.py) | 0.954 | 0.994 | 0.999 | 0.076 | 0.275 | 0.033 | [vqvae_depth.pt](https://msravcghub.blob.core.windows.net/ait-release/vae/vqvae_depth.pt?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) | [model](https://msravcghub.blob.core.windows.net/ait-release/checkpoint/ait_depth_swinv2l_parallel.pth?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) |

### Joint training results on COCO and NYUv2
| <div style="width: 100pt"> Model</div> | Box AP| Mask AP| RMSE | VQ-VAE Model | Task-Solver <br> Model |
|:-------------------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [AiT(SwinV2-B)](ait/configs/swinv2b_640reso_joint.py) | 42.2 | 34.1 | 0.310 | [vqvae_depth.pt](https://msravcghub.blob.core.windows.net/ait-release/vae/vqvae_depth.pt?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D)/[vqvae_insseg.pt](https://msravcghub.blob.core.windows.net/ait-release/vae/vqvae_insseg.pt?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D)  | [model](https://msravcghub.blob.core.windows.net/ait-release/checkpoint/ait_joint_swinv2b.pth?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D)|


## Usage

### Installation
We recommend using pytorch>=1.10, other packages can be found in requirements.txt. To install boundary-iou-api, please using the following command:
```bash
git clone https://github.com/bowenc0221/boundary-iou-api && cd boundary-iou-api && pip install -e .
```

### Data/Pre-training model Preparation 
1. Download the [NYU Depth V2](https://github.com/vinvino02/GLPDepth) dataset, [COCO](https://cocodataset.org/#download) datasets, our preprocess box-cropped binary instance masks, named [maskcoco](https://msravcghub.blob.core.windows.net/ait-release/data/maskcoco.tar?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D), and organize the data according to the following directory structure:

```plain
AiT
├── ait
├── vae
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── maskcoco
│   ├── nyu_depth_v2
```
2. Create the data links using following commands:

```bash
ln -s data ait/data
ln -s data vae/data
```

3. Download pre-trained backbone models [swin_v2_base_densesimmim.pth](https://msravcghub.blob.core.windows.net/ait-release/checkpoint/swin_v2_base_densesimmim.pth?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D) and [swin_v2_large_densesimmim.pth](https://msravcghub.blob.core.windows.net/ait-release/checkpoint/swin_v2_large_densesimmim.pth?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D). 


### Training
#### Training VQ-VAE on depth estimation:
```bash
cd vae
python -m torch.distributed.launch --nproc_per_node=${N_GPUS} train_depth_vqvae_dist.py  configs/depth/ait_depth_vqvae.py --cfg-options <custom-configs>
```
#### Training VQ-VAE on instance segmentation:

```bash
cd vae
python -m torch.distributed.launch --nproc_per_node=${N_GPUS} train_insseg_vqvae_dist.py  configs/insseg/ait_insseg_vqvae.py --cfg-options <custom-configs>
```

#### Training task-solver on depth estimation:
```bash
cd ait

# Train auto-regressive model
python -m torch.distributed.launch --nproc_per_node=8 code/train.py configs/swinv2b_480reso_depthonly.py --cfg-options model.backbone.init_cfg.checkpoint=swin_v2_base_densesimmim.pth model.task_heads.depth.vae_cfg.pretrained=vqvae_depth.pt # for AR training

# Train parallel model
python -m torch.distributed.launch --nproc_per_node=8 code/train.py configs/swinv2b_480reso_parallel_depthonly.py --cfg-options model.backbone.init_cfg.checkpoint=swin_v2_base_densesimmim.pth model.task_heads.depth.vae_cfg.pretrained=vqvae_depth.pt # for parallel training
```

#### Training task-solver on object detection
```bash
cd ait
python -m torch.distributed.launch --nproc_per_node=16 --nnodes=2 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} code/train.py configs/swinv2b_640reso_detonly.py --cfg-options model.backbone.init_cfg.checkpoint=swin_v2_base_densesimmim.pth
```

**Note:** We use the pre-trainined object detection model to initialize the instance segmentation models and joint-training models to save training cost, please download the pre-trained model ([ait_det_swinv2b_wodec.pth](https://msravcghub.blob.core.windows.net/ait-release/checkpoint/ait_det_swinv2b_wodec.pth?sv=2021-10-04&spr=https%2Chttp&st=2023-06-30T01%3A47%3A00Z&se=2026-01-01T01%3A47%3A00Z&sr=c&sp=rl&sig=bwb8Tpfpk2FfZxsilGa4Oc5vKEZiifZK4xs%2F6RuWF9E%3D)) before training on instance segmentation and joint training setting.

#### Training task-solver on instance segmentation
```bash
python -m torch.distributed.launch --nproc_per_node=16 code/train.py configs/swinv2b_640reso_inssegonly.py --cfg-options model.backbone.init_cfg.checkpoint=swin_v2_base_densesimmim.pth model.task_heads.insseg.vae_cfg.pretrained=vqvae_insseg.pt load_from=ait_det_swinv2b_wodec.pth
```

#### Joint training on instance segmentation and depth estimation
```bash
python -m torch.distributed.launch --nproc_per_node=16 --nnodes=4 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} code/train.py configs/swinv2b_640reso_joint.py --cfg-options model.backbone.init_cfg.checkpoint=swin_v2_base_densesimmim.pth model.task_heads.insseg.vae_cfg.pretrained=vqvae_insseg.pt model.task_heads.depth.vae_cfg.pretrained=vqvae_depth.pt load_from=ait_det_swinv2b_wodec.pth
```

### Inference
#### Evaluate  on depth estimation
```bash
cd ait

# Evaluating auto-regressive model
python -m torch.distributed.launch --nproc_per_node=8 code/train.py configs/swinv2b_480reso_depthonly.py  --cfg-options model.task_heads.depth.vae_cfg.pretrained=vqvae_depth.pt --eval <model_checkpiont>

# Evaluating parallele model
python -m torch.distributed.launch --nproc_per_node=8 code/train.py configs/swinv2b_480reso_parallel_depthonly.py  --cfg-options model.task_heads.depth.vae_cfg.pretrained=vqvae_depth.pt --eval <model_checkpiont>
```

#### Evaluate on instance segmentation
```bash
cd ait

python -m torch.distributed.launch --nproc_per_node=8 code/train.py configs/swinv2b_640reso_inssegonly.py --cfg-options model.task_heads.insseg.vae_cfg.pretrained=vqvae_insseg.pt --eval <model_checkpiont>
```

#### Evaluate on both depth estimation and instance segmentation
```bash
cd ait

python -m torch.distributed.launch --nproc_per_node=8 code/train.py configs/swinv2b_640reso_joint.py --cfg-options model.task_heads.insseg.vae_cfg.pretrained=vqvae_insseg.pt model.task_heads.depth.vae_cfg.pretrained=vqvae_depth.pt --eval <model_checkpiont>
```


## Citation
```
@article{ning2023all,
  title={All in Tokens: Unifying Output Space of Visual Tasks via Soft Token},
  author={Ning, Jia and Li, Chen and Zhang, Zheng and Geng, Zigang and Dai, Qi and He, Kun and Hu, Han},
  journal={arXiv preprint arXiv:2301.02229},
  year={2023}
}
```
