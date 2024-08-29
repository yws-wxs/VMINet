# VMINet:A Separable Self-attention Inspired by State Space Model in Computer Vision

## Abstract

Mamba is an efficient State Space Model (SSM) with linear computational complexity. Although SSMs are not suitable for handling non-causal data, Vision Mamba (ViM) methods still demonstrate good performance in tasks such as image classification and object detection.
Recent studies have shown that there is a rich theoretical connection between state space models and attention variants. We propose a novel separable self-attention method, for the first time introducing some excellent design concepts of Mamba into separable self-attention.
To ensure a fair comparison with ViMs, we introduce VMINet, a simple yet powerful prototype architecture, constructed solely by stacking our novel attention modules with the most basic down-sampling layers. Notably, VMINet differs significantly from the conventional 
Transformer architecture. Our experiments demonstrate that VMINet has achieved competitive results on image classification and high-resolution dense prediction tasks.

## ImageNet classification
### 1. Requirements
torch>=1.7.0; torchvision>=0.8.0; pyyaml; timm==0.6.13; einops; fvcore; h5py;

### 2.Train VMINet
python3 -m torch.distributed.launch --nproc_per_node=3 train_imagenet.py --data {path-to-imagenet} --model {starnet-variants} -b 256 --lr 1e-3 --weight-decay 0.025 --aa rand-m1-mstd0.5-inc1 --cutmix 0.2 --color-jitter 0. --drop-path 0.

### 3. Pretrained checkpoints
|Model|Top1|Ckpt|logs|
|:-----:|:----:|:----:|:----:|
|VMINet-XS|76.5|  |  |
|VMINet-S|79.0|  |  |
|VMINet-B|80.9|  |  |

## Acknowledgement
The development of this project referenced the source code of StarNet (https://github.com/ma-xu/Rewrite-the-Stars/tree/main/imagenet), thanks to this excellent work.

## License
The majority of Rewrite the Stars is licensed under an [Apache License 2.0](https://github.com/ma-xu/Rewrite-the-Stars/blob/main/LICENSE)
