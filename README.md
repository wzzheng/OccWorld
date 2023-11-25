# OccWorld
### [Paper](https://arxiv.org/pdf/2311.12754)

> OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving

> [Wenzhao Zheng](https://wzzheng.net/)\* $\dagger$, [Weiliang Chen](https://github.com/chen-wl20)\*, [Yuanhui Huang](https://scholar.google.com/citations?hl=zh-CN&user=LKVgsk4AAAAJ), [Borui Zhang](https://boruizhang.site/), [Yueqi Duan](https://duanyueqi.github.io/), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)

\* Equal contribution $\dagger$ Project leader

**OccWorld models the joint evolutions of 3D scenes and ego movements.**

Combined with self-supervised ([SelfOcc](https://github.com/huang-yh/SelfOcc)), LiDAR-collected ([TPVFormer](https://github.com/wzzheng/TPVFormer)), or machine-annotated ([SurroundOcc](https://github.com/weiyithu/SurroundOcc)) dense  3D occupancy, OccWorld has the potential to scale up to large-scale training, paving the way for **interpretable end-to-end large driving models**.

## Overview

![overview](./assets/overview.png)

Given past 3D occupancy observations, our self-supervised OccWorld trained can forecast future scene evolutions and ego movements jointly. This task requires a spatial understanding of the 3D scene and temporal modeling of how driving scenarios develop. We observe that OccWorld can successfully forecast the movements of surrounding agents and future map elements such as drivable areas. OccWorld even generates more reasonable drivable areas than the ground truth, demonstrating its ability to understand the scene rather than memorizing training data. Still, it fails to forecast new vehicles entering the sight, which is difficult given their absence in the inputs. 

### Framework
![framework](./assets/framework.png)

### Results

#### 4D Occupancy Forecast
![framework](./assets/4docc.png)

#### Motion Planning
![framework](./assets/planning.png)

#### Visualizations
![framework](./assets/vis.png)

## Code

Coming soon!

## Installation
1. Create conda environment with python version 3.8.0

2. Install all the packages in environment.yaml

3. Anything about the installation of mmdetection3d, please refer to [mmdetection3d](https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation)

## Preparing
1. Create soft link from data/nuscenes to your_nuscenes_path

2. Prepare the gts semantic occupancy introduced in [Occ3d](https://github.com/Tsinghua-MARS-Lab/Occ3D)

3. Download our generated train/val pickle files and put them in data/
    [nuscenes_infos_train_temporal_v3_scene.pkl]
    [nuscenes_infos_val_temporal_v3_scene.pkl]
  The dataset should be organized as follows:
  
```
OccWorld/data
    nuscenes                 -    downloaded from www.nuscenes.org
        lidarseg
        maps
        samples
        sweeps
        v1.0-trainval
        gts                  -    download from Occ3d
    nuscenes_infos_train_temporal_v3_scene.pkl
    nuscenes_infos_val_temporal_v3_scene.pkl
```

## Getting Started

1. Train the OccWorld on RTX 4090 (a VQVAE should be trained using similar command before training)
```
python train.py --py-config config/occworld.py --work-dir out/occworld
```
2. Eval the model on RTX 4090
```
python eval_metric_stp3.py --py-config config/occworld.py --work-dir out/occworld
```
## Related Projects

Our code is based on [TPVFormer](https://github.com/wzzheng/TPVFormer), [SelfOcc](https://github.com/huang-yh/SelfOcc), and [PointOcc](https://github.com/wzzheng/PointOcc). 

Also thanks to these excellent open-sourced repos:
[SurroundOcc](https://github.com/weiyithu/SurroundOcc) 
[OccFormer](https://github.com/zhangyp15/OccFormer)
[BEVFormer](https://github.com/fundamentalvision/BEVFormer)

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{zheng2023occworld,
    title={OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving},
    author={Zheng, Wenzhao and Chen, Weiliang and Huang, Yuanhui and Zhang, Borui and Duan, Yueqi and Lu, Jiwen },
    journal={arXiv preprint arXiv:},
    year={2023}
}
```
