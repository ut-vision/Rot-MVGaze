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
create `data_path.yaml`
```
xgaze: <path to xgaze>
mpiinv: <path to mpiinv>
```

## Training

#### Exporiments
exp_names:
- `xgaze2mpiinv_known` 
- `xgaze2mpiinv_novel`
- `mpiinv2xgaze_known`
- `mpiinv2xgaze_novel`
`
```
python main.py \
  --exp_name <exp_name> \
  --mode train \
```

## Evaluation
Download the pretrained checkpoints and run


| Experiment | Model  | Path |
| - | - | - |
| XGaze to MPII-NV (known head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/1-j3jiW3oN0Hqbzz9BC58u-VXaNjL6uqf/view?usp=sharing) |
| XGaze to MPII-NV (novel head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/1R5oU6tYno92pke9F1Kj9zHxB1l89I5nu/view?usp=sharing) |
| MPII-NV to XGaze (known head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/1lESAPVbKjHp1v5V6fIQxoaWdQ3Pmi--6/view?usp=sharing) |
| MPII-NV to XGaze (novel head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/1-zqoPL53y1UuOn1qgE_9dClrfhM8HKHR/view?usp=sharing) |

```
python main.py \
  --exp_name <exp_name> \
  --mode test --ckpt_pretrained <path to the ckpt>
```


## Citation
```
@article{hisadome2023rotation,
  title={Rotation-Constrained Cross-View Feature Fusion for Multi-View Appearance-based Gaze Estimation},
  author={Hisadome, Yoichiro and Wu, Tianyi and Qin, Jiawei and Sugano, Yusuke},
  journal={arXiv preprint arXiv:2305.12704},
  year={2023}
}
```
<!-- ## Acknowledgements -->

## Contact
JIawei Qin: jqin@iis.u-tokyo.ac.jp
