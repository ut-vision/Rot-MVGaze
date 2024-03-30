# MVGaze

## Installation

```
pip install -r requirements
```

## Prepare Datasets

#### XGaze

#### MPII-NV


## Training


```
python main.py \
  -out ./logs \
  --mode train 
```

## Evaluation
Download the pretrained checkpoints and run

```
python main.py \
  -out ./logs \
  --mode test --ckpt_pretrained <path to the ckpt>
```

| Experiment | Model  | Path |
| - | - | - |
| XGaze to MPII-NV (known head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/1-j3jiW3oN0Hqbzz9BC58u-VXaNjL6uqf/view?usp=sharing) |
| XGaze to MPII-NV (novel head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/1R5oU6tYno92pke9F1Kj9zHxB1l89I5nu/view?usp=sharing) |
| MPII-NV to XGaze (known head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/1lESAPVbKjHp1v5V6fIQxoaWdQ3Pmi--6/view?usp=sharing) |
| MPII-NV to XGaze (novel head pose) | Rot-MV | [Google Drive](https://drive.google.com/file/d/1-zqoPL53y1UuOn1qgE_9dClrfhM8HKHR/view?usp=sharing) |


## Citation
```
@article{hisadome2023rotation,
  title={Rotation-Constrained Cross-View Feature Fusion for Multi-View Appearance-based Gaze Estimation},
  author={Hisadome, Yoichiro and Wu, Tianyi and Qin, Jiawei and Sugano, Yusuke},
  journal={arXiv preprint arXiv:2305.12704},
  year={2023}
}
```
## Acknowledgements
