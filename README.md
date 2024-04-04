# Rot-MVGaze
This is the official PyTorch implementation of the paper [Rotation-Constrained Cross-View Feature Fusion for Multi-View Appearance-based Gaze Estimation](https://arxiv.org/abs/2305.12704).
<img src='asset/teaser.jpg' width=800>

## Installation

```
git clone git@github.com:ut-vision/Rot-MVGaze.git
cd Rot-MVGaze
pip install -r requirements.txt
```

## Data
### Prepare datasets
#### ETH-XGaze
Please download the normalized XGaze_224 from the official website.

#### MPII-NV
Please refer to [Learning-by-Novel-View-Synthesis for Full-Face Appearance-Based 3D Gaze Estimation](https://arxiv.org/abs/2201.07927) or directly contact us for the data synthesis.

### Configuration
create `configs/data_path.yaml`
```
xgaze: <path to xgaze>
mpiinv: <path to mpiinv>
```

## Training

#### Exporiments names
- `xgaze2mpiinv_known` 
- `xgaze2mpiinv_novel`
- `mpiinv2xgaze_known`
- `mpiinv2xgaze_novel`
```
python main.py \
  --exp_name <exp_name> \
  --mode train \
```

## Evaluation
Download the pretrained checkpoints and run


| Experiment | Model  | Path |
| - | - | - |
| XGaze to MPII-NV (known head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/1pW_q6bZB6RWfFA7mwbI4JJH2jycgCroq/view?usp=sharing) |
| XGaze to MPII-NV (novel head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/11ze-5r3Dq1VGID856Bi7JBK4xbuXUZPg/view?usp=sharing) |
| MPII-NV to XGaze (known head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/1oY5lDGmJ0tyJQhPkuEdl5Y3NqXwNE2xv/view?usp=sharing) |
| MPII-NV to XGaze (novel head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/1x5yFR0mUZa4R67K8YnVm3-CFeF4DubUH/view?usp=sharing) |

```
python main.py \
  --exp_name <exp_name> \
  --mode test --ckpt_pretrained <path to the ckpt>
```


## Citation
```
@inproceedings{hisadome2024rotation,
  title={Rotation-Constrained Cross-View Feature Fusion for Multi-View Appearance-based Gaze Estimation},
  author={Hisadome, Yoichiro and Wu, Tianyi and Qin, Jiawei and Sugano, Yusuke},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5985--5994},
  year={2024}
}
```
<!-- ## Acknowledgements -->

## Contact
Jiawei Qin: jqin@iis.u-tokyo.ac.jp
