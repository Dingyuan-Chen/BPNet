### BPNet: Blurry dense object extraction based on buffer parsing network for high-resolution satellite remote sensing imagery

![](imgs/framework.png)

The repo is based on [Frame-Field-Learning](https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning) and [CascadePSP](https://github.com/hkchengrex/CascadePSP).

### Introduction
Despite the remarkable progress of deep learning-based object extraction in revealing the number and boundary location of geo-objects for high-resolution satellite imagery, it still faces challenges in accurately extracting blurry dense objects. Unlike general objects, blurry dense objects have limited spatial resolution, leading to inaccurate and connected boundaries. Even with the improved spatial resolution and recent boundary refinement methods for general object extraction, connected boundaries may remain undetected in blurry dense object extraction if the gap between object boundaries is less than the spatial resolution. This paper proposes a blurry dense object extraction method named the buffer parsing network (BPNet) for satellite imagery. To solve the connected boundary problem, a buffer parsing module is designed for dense boundary separation. Its essential component is a buffer parsing architecture that comprises a boundary buffer generator and an interior/boundary parsing step. This architecture is instantiated as a dual-task mutual learning head that co-learns the mutual information between the interior and boundary buffer, which estimates the dependence between the dual-task outputs. Specifically, the boundary buffer head generates a buffer region that overlaps with the interior, enabling the architecture to learn the dual-task bias and assign a reliable semantic in the overlapping region through high-confidence voting. To alleviate the inaccurate boundary location problem, BPNet incorporates a high-frequency refinement module for blurry boundary refinement. This module includes a high-frequency enhancement unit to enhance high-frequency signals at the blurry boundaries and a cascade buffer parsing refinement unit that integrates the buffer parsing architecture coarse-to-fine to recover the boundary details progressively. The proposed BPNet framework is validated on two representative blurry dense object datasets for small vehicle and agricultural greenhouse object extraction. The results indicate the superior performance of the BPNet framework, achieving 25.25% and 73.51% in contrast to the state-of-the-art PointRend method, which scored 21.92% and 63.95% in the ${AP50}_{segm}$ metric on two datasets, respectively. Furthermore, the ablation analysis of the super-resolution and building extraction methods demonstrates the significance of high-quality boundary details for subsequent practical applications, such as building vectorization.

## Installation
Please refer to INSTALL.md in [Frame-Field-Learning](https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning) and [CascadePSP](https://github.com/hkchengrex/CascadePSP) for installation and dataset preparation.

## Downloads
Download pretrained weights of the high-frequency refinement module (./models/) at [BaiduDrive](https://pan.baidu.com/s/1HYVebWlpYN6LRddnV_c7kg) (code: sj2p)

## Getting Started
1. Train a model
```
python main.py --config configs/config.gh_dataset.unet_resnet152_pretrained.json --master_port 1220 --max_epoch 300 -b 4
```

2. Test a dataset
```
python main.py --config configs/config.gh_dataset.unet_resnet152_pretrained.json --mode eval --eval_batch_size 4 --master_port 1221
python main.py --config configs/config.gh_dataset.unet_resnet152_pretrained.json --mode eval_coco --eval_batch_size 4 --master_port 1222
```

3. Inference with pretrained models
```
python main.py --run_name gh_dataset.{run_name} --in_filepath {in_filepath}/images/ --out_dirpath {out_dirpath}/results
```

## Agricultural greenhouse extraction dataset results
|Model                      |Backbone           |    ${mAP}_{segm}$     |    ${AP50}_{segm}$  |  ${AP50}_{bd}$ | Speed (fps) |
|:-------------:            |:-------------: | :-----:| :-----: | :-----:  | :-----:  |
|[Top-Down] PolygonRNN (Castrejon et al., 2017)   |R50  | 11.22 | 23.76 | 17.60 | 7.40 |
|[Top-Down] DELTA (Ma et al., 2021)   |R50  | 19.45 | 35.43 | 32.80 | 11.80 |
|[Top-Down] SOLO (Wang et al., 2020)   |R50  | 30.30 | 55.40 | 49.89 | 18.50 |
|[Top-Down] Mask R-CNN (He et al., 2017)   |R50  | 34.26 | 60.68 | 57.76 | 10.60 |
|[Bottom-Up] ResUNet (Xu et al., 2018)   |R34  | 26.89 | 47.34 | 42.36 | 26.80 |
|[Bottom-Up] ResUNet (Xu et al., 2018)   |R101  | 28.45 | 50.77 | 46.63 | 24.70 |
|[Bottom-Up] ResUNet (Xu et al., 2018)   |R152  | 30.55 | 54.20 | 50.17 | 23.10 |
|[Refinement] ECL (Liu et al., 2020)   |R152  | 30.89 | 55.41 | 50.57 | 22.80 |
|[Refinement] PointRend (Kirillov et al., 2020)   |R50  | 33.45 | 63.95 | 61.79 | 8.50 |
|[Refinement] CascadePSP (Cheng et al., 2020)   |R152  | 30.54 | 54.80 | 50.58 | 21.70 |
|[Refinement] CBR-Net (Guo et al., 2022)   |R152  | 28.79 | 46.22 | 41.47 | 22.80 |
|BPNet (proposed)   |R152  | 40.36 | 73.51 | 66.15 | 21.10 |

## Citation

```BibTeX
@article{chen2023large,
  title={Large-scale agricultural greenhouse extraction for remote sensing imagery based on layout attention network: A case study of China},
  author={Chen, Dingyuan and Ma, Ailong and Zheng, Zhuo and Zhong, Yanfei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={200},
  pages={73--88},
  year={2023},
  publisher={Elsevier}
}
@article{ma2021national,
  title={National-scale greenhouse mapping for high spatial resolution remote sensing imagery using a dense object dual-task deep learning framework: A case study of China},
  author={Ma, Ailong and Chen, Dingyuan and Zhong, Yanfei and Zheng, Zhuo and Zhang, Liangpei},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={181},
  pages={279--294},
  year={2021},
  publisher={Elsevier}
}
```
