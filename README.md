# Unsupervised Anomaly Detection

## 1. Installation
- Requirement: Python 3.10

```
git clone https://github.com/lamnguyenvu98/unsupervised-anomaly-detection.git

cd unsupervised-anomaly-detection

pip install .
```

## 2. Training
### 2.1 Dataset structure
#### 2.1.1 Structure for DFR and Padim
```
├── transistor (root)
│   ├── train
│       ├── good
│           ├── image1.png
│           └── image2.png
│           └── ...
│           └── imageN.png
│   └── test
│       ├── good
│           ├── image1.png
│           └── image2.png
│           └── ...
│           └── imageN.png
│       ├── anomaly
│           ├── image1.png
│           └── image2.png
│           └── ...
│           └── imageN.png
```

#### 2.1.2 Structure for RegAD
```
├── mvtec-ad (root)
|       ├── transistor
│       |    ├── train
│       |    |    └── good
│       |    |        ├── image1.png
│       |    |        ├── image2.png
│       |    |        ├── ...
│       |    |        └── imageN.png
│       |    └── test
│       |        ├── good
│       |        |    ├── image1.png
│       |        |    ├── image2.png
│       |        |    ├── ...
│       |        |    └── imageN.png
│       |        └── anomaly
│       |            ├── image1.png
│       |            ├── image2.png
│       |            ├── ...
│       |            └── imageN.png
|       ├── bottle
│       |    ├── train
│       |    |    └── good
│       |    |        ├── image1.png
│       |    |        ├── image2.png
│       |    |        ├── ...
│       |    |        └── imageN.png
│       |    └── test
│       |        ├── good
│       |        |    ├── image1.png
│       |        |    ├── image2.png
│       |        |    ├── ...
│       |        |    └── imageN.png
│       |        └── anomaly
│       |            ├── image1.png
│       |            ├── image2.png
│       |            ├── ...
│       |            └── imageN.png
|       ├── ...
```

### 2.2 Modify config file (configs/*.yaml)
There are some parameters need to be clarified:
- DATA_DIR: root directory of dataset

- SAVE_DIR: directory where to save model checkpoints

- CHECKPOINT_PATH: path to model checkpoint for inference or resuming training process

#### 2.2.1 Padim 
- REDUCE_FEATURES: number of features reduce to

#### 2.2.2 DFR
- NUM_LAYERS: number of backbone layers to get feature maps and aggregate them

#### 2.2.3 RegAD
- BACKBONE: name of backbone for feature extraction. Options: "resnet18", "resnet34", "resnet50", "resnet101", "resnet152".

- STN_MODE: Default is "rotation_scale". Other options: "affine", "translation", "rotation", "scale", "shear", "translation_scale", "rotation_translation", "rotation_translation_scale".

- N_SHOT: number of support set (4, 8, 16,...)

- N_TEST: number of rounds to evaluate model

- TRAIN_DATA_DIR: root directory of dataset which contain multiple classes (screw, bottle, transistor,...)

- TEST_DATA_DIR: directory of an object for model evaluation, this object shouldn't be included in TRAIN_DATA_DIR. If it's inside TRAIN_DATA_DIR, set IGNORE_CLASS to name of that object.

```
TRAIN_DATA_DIR: "/content/mvtec-ad"
TEST_DATA_DIR: "/content/mvtec-ad/transistor"
IGNORE_CLASS: "transistor"
```

`transistor` class was used for model evaluation. But it was also inside TRAIN_DATA_DIR. So that, setting IGNORE_CLASS to "transistor" helped model ignored this class during training. 

### 2.3 Train model
#### 2.3.1 RegAD
```
python train_regad.py --config configs/regad_config.yaml
```

#### 2.3.2 Padim
```
python train_padim.py --config configs/padim_config.yaml
```

#### 2.3.3 DFR
```
python train_dfr.py --config configs/dfr_config.yaml
```


## Acknowledgement

Original repo RegAD: https://github.com/MediaBrain-SJTU/RegAD
```
@inproceedings{huang2022regad,
  title={Registration based Few-Shot Anomaly Detection}
  author={Huang, Chaoqin and Guan, Haoyan and Jiang, Aofan and Zhang, Ya and Spratlin, Michael and Wang, Yanfeng},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

Original repo DFR: https://github.com/YoungGod/DFR
```
@misc{yang2020dfr,
      title={DFR: Deep Feature Reconstruction for Unsupervised Anomaly Segmentation}, 
      author={Jie Yang and Yong Shi and Zhiquan Qi},
      year={2020},
      eprint={2012.07122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Original repo Padim: https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
