https://arxiv.org/abs/2205.10772

## Usage

The code is tested on Python 3.9, CUDA 11, CuDNN 8.2. To install the dependencies:
```
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

`scripts` contains the scripts to reproduce the experiments. The usage is
```
mkdir -p ~/run/liv
cd scripts
python main.py <exp_name> -ng <num_gpus> -nm <num_tasks_per_gpu> -train [exp-specific args]
```
where `exp_name` can be `dgmm`, `gp-uq` or `demand`. Each choice corresponds to one or more
experiments in paper; see the corresponding file in the `scripts/` directory for explanations.

Results will be logged into a directory of the form `~/run/liv/<date>-<time>-<expname>_/`. Each
of its subdirectory corresponds to a single experiment, within which `stdout` contains the
test metrics. The experiment-specific scripts contain instructions for the extraction of the
results.

## Acknowledgements

This repository contains files from [deepmind/ssl_hsic](https://github.com/deepmind/ssl_hsic)
and [google/flax](https://github.com/google/flax), the licenses of which are appended to the
corresponding file.
