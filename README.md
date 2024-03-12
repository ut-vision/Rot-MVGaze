# Gaze Estimation Library

## Setup
We defined some usefull commands.
```sh
source command.sh # for fish user, `source command.fish`
```
To build a Python environment, please select either conda or singularity.

- Conda
    ```
    setup-conda
    ```

- Singularity
    ```sh
    build-sif
    singularity shell --nv --bind /work path/to/gzcv.sif
    pip install -e .
    ```

## Install GazeCV
**Important**:
If you build the environment with singularity or manually setup conda, please
```sh
pip install -e .
```
at the root of this repository.
Then, `gzcv` libarary is available in editable mode.

## Train/Test
### Train and test
```sh
cd examples
python main.py --model ../configs/proposed.yaml --exp ../configs/experiments/xgaze_known_head.yaml --is_train --is_test
```
- `--level`: logging level [debug, info]
- `--model, --exp`: relative path to yaml config

### Test using pretrained weights
```sh
cd examples
python main.py --model ../configs/proposed.yaml --exp ../configs/experiments/xgaze_known_head.yaml --is_test --resume path/to/the/log/ckpt/model-best.pth
```

The trained weight and other logs will be stored at `logs` directory.
If you set the logging level `debug`, the result will be saved at `logs/debug`.

## Lint and Format
pysen formats your source code.
```sh
source command.sh
```
```
lint
```
