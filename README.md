# [[Information Fusion] A channel- adaptive and plug-and- play framework for hyperspectral image analysis](https://www.sciencedirect.com/science/article/abs/pii/S1566253525008322)

This is an official PyTorch implementation of "[A channel- adaptive and plug-and- play framework for hyperspectral image analysis](https://www.sciencedirect.com/science/article/abs/pii/S1566253525008322)".

# Introduction
HyperSpectral Image (HSI) reflects rich properties of matter and facilitates distinguishing various objects, demonstrating substantial potential in a wide range of applications, including medical diagnosis and remote sensing. However, HSI exhibits variable number of channels due to the variations in acquisition equipments, which makes existing HSI analytical methods fail to utilize data from multiple equipments. To address this challenge, we first distill HSIs with varying channels into principal and residual components. We then develop a Fusion-Guided Network (FGNet) to transform the two distilled components into fused images with a fixed number of channels and perform channel-adaptive HSI analysis. To enable the fused images to maintain intensity, structure, and texture information in the original HSI, we generate pseudo labels to supervise the fusion. To facilitate the FGNet to extract more representative features, we further design a low-rank attention module (LGAM), leveraging the low-rank prior of HSI that few key information can represent a large amount of data. Moreover, the proposed framework can be applied as a plug-in to existing HSI analysis methods. We conducted extensive experiments on five HSI datasets including medical HSI segmentation task and remote sensing HSI classification task, which demonstrates the proposed method outperforms the state-of-the-art methods. We further experimentally identified that existing works can be seamlessly incorporated with our framework to achieve channel-adaptive ability and boost analytical performance.

# Installation

Please follow the instructions provided in [SpecTr](https://github.com/DeepMed-Lab-ECNU/SpecTr) to set up the environment.


# Training

## Medical Image Segmentation

### MHSI Choledoch

To train ```FGNet``` models on MHSI Choledoch, run:

``` bash
cd ./seg/
python train.py --dataset_name 'MDC' --root_path 'path/MHSI Choledoch Dataset (Preprocessed Dataset)/' \
--dataset_hyper 'MHSI' --dataset_mask 'Mask' --dataset_divide './mdc_train_val.json' \
--scale_time 4 --batch 8 --group_num 12 --MD_R 12  --output_folder 'path/MDC'
```

## Image Classification

### Whu-hi-HanChuan

To train ```FGNet``` models on Whu-hi-HanChuan, run:

``` bash
cd ./cls/
python train.py --data_name WHU_Hi_HanChuan --hsi_path path/WHU_Hi_HanChuan/WHU_Hi_HanChuan.mat \
--label_path path/WHU_Hi_HanChuan/HanChuan_split_gt.mat --batch 128 --patch_size 17 \
--output ./checkpoints/ --experiment_name WHU_Hi_HanChuan
```

# Citation
Please cite our work if you find it useful.
```bibtex
@article{CHEN2026103770,
title = {A channel- adaptive and plug-and- play framework for hyperspectral image analysis},
journal = {Information Fusion},
volume = {127},
pages = {103770},
year = {2026},
issn = {1566-2535},
author = {Taiqin Chen and Hao Sha and Yifeng Wang and Yuan Jiang and Shuai Liu and Zikun Zhou and Ke Chen and Yongbing Zhang},
}
```

## Acknowledgement 
+ SpecTr code is heavily used. [official](https://github.com/DeepMed-Lab-ECNU/SpecTr)
+ Multi-dimensional Choledoch Dataset [official](https://www.kaggle.com/datasets/hfutybx/mhsi-choledoch-dataset-preprocessed-dataset) 
+ Whu-Hi dataset [official](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm)