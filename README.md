#  RESHUFFLE

This repository contains the official source code for the RESHUFFLE model, presented at **KR 2025** in our paper **"Faithful Differentiable Reasoning with Reshuffled Region-based Embeddings"**.
The repository includes the following:

1. the implementation of RESHUFFLE.
2. the code for training and testing RESHUFFLE on FB15k-237 v1-4, WN18RR v1-4, and NELL-995 v1-4 to reproduce the results presented in our paper (`run_experiments.py`).
3. an `environment.yml` to automatically set up a conda environment with all dependencies.

# Requirements

* Python 3.8
* PyTorch 2.1.0
* CUDA Toolkit 11.5.0
* PyKEEN 1.10.1
* NumPy < 2.0
* Tensorboard 2.14.0

# Installation

We have provided an `environment.yml` file that can be used to create a conda environment with all required
dependencies. Run `conda env create -f environment.yml` to create the conda environment `Env_RESHUFFLE`.
Afterward, use `conda activate Env_RESHUFFLE` to activate the environment before rerunning our experiments.

# Running RESHUFFLE

Training and evaluation of RESHUFFLE are done by executing the `run_experiments.py` script. In particular, a configuration file
must be specified for a RESHUFFLE model, containing all model, training, and evaluation parameters. The best
configuration files for FB15k-237 v1-4, WN18RR v1-4, and NELL-995 v1-4 are provided in the `Best_Configurations` directory and can be adapted to
try out different parameter configurations. To run an experiment, the following parameters need to be specified:

- `config_dir` contains the path to the model configuration directory (e.g., `config_dir=Best_Configurations/RESHUFFLE/`).
- `config_name` contains the name of the config file (e.g. `config_name=FB_v1`).
- `train` contains `true` if the model shall be trained and `false` otherwise.
- `test` contains `true` if the model shall be evaluated on the test graph and `false` otherwise.
- `gpu` contains the id of the GPU that shall be used (e.g., `gpu=0`). If this parameter is left out, the model will be trained on CPU. 
- `seeds` contains the seeds for repeated runs of the experiment (e.g., `seeds=1,2,3`).

Finally, one can run an experiment
with `python run_experiments.py config_dir=<config_dir> config_name=<config_name> train=<true|false> test=<true|false> gpu=<gpuID> seeds=<seeds>`, 
where angle brackets represent a parameter value. 

When you run an experiment with `test=true`, two sets of results will be generated:
- the complete results, saved in `Benchmarking/complete_results`. This directory includes all evaluation metrics computed during this experiment.
- the paper-relevant results, saved in `Benchmarking/short_results`. This directory includes shortened result summaries, containing only the key metrics used in our paper (for Tables 2 and 3) together with the hyperparameters used in this experiment.

# Reproducing the Results

This section explains how to reproduce the results from our inductive knowledge graph completion benchmarks (see Tables 2 and 3 in our paper presented at KR 2025)

## Results for Inductive Knowledge Graph Completion

Each experiment uses a specific configuration file that contains the best hyperparameter setting found for a model/dataset pair. For example, the configuration file specified at `Best_Configurations/RESHUFFLE/FB_v1.json` specifies the best hyperparameter setting found for RESHUFFLE on the FB15k-237 v1 dataset.
Similarly `Best_Configurations/RESHUFFLE_square/NELL_v2.json` specifies the best hyperparameter setting found for RESHUFFLE<sup>2</sup> on the NELL-995 v1 dataset and so on for other combinations.

To run a benchmark experiment on CPU, you’ll need to substitute the following variables in the upcoming commands:
* `<model>` represents the model that shall be trained:
  * `<model>` needs to be substituted by one of [`RESHUFFLE`, `RESHUFFLE_no_loop`, `RESHUFFLE_square`].
* `<config_name>` represents the config file that should be used: 
  * `<config_name>` needs to be substituted by one of [`FB_v1`, `FB_v2`, `FB_v3`, `FB_v4`, `WN_v1`, `WN_v2`, `WN_v3`, `WN_v4`, `NELL_v1`, `NELL_v2`, `NELL_v3`, `NELL_v4`].
  
Train: `python run_experiments.py train=true test=false seeds=1,2,3 config_dir=Best_Configurations/<model>/ config_name=<config_name>`

Test: `python run_experiments.py train=false test=true seeds=1,2,3 config_dir=Best_Configurations/<model>/ config_name=<config_name>`

To run a benchmark experiment on GPU, add the parameter `gpu=<gpuID>` to one of the above commands.

# Tensorboard & Convergence Time

The evolution of the model loss and validation metrics can be observed on tensorboard.
To run tensorboard, execute the provided tensorboard.sh file: `. tensorboard.sh`.

# Citation 

If you use this code or its corresponding paper, please cite our work as follows:

```
@inproceedings{
pavlovic2025reshuffle,
title={Faithful Differentiable Reasoning with Reshuffled Region-based Embeddings},
author={Aleksandar Pavlovi{\'c} and Emanuel Sallinger and Steven Schockaert},
booktitle={Proceedings of the 22nd International Conference on Principles of
           Knowledge Representation and Reasoning, {KR} 2025, Melbourne, Australia.
           November 11-17, 2025},
year={2025}
}
```

# Contact

Aleksandar Pavlović

Research Center AI, Software and Safety

University of Applied Sciences Vienna (HCW)

Vienna, Austria

<aleksandar.pavlovic@hcw.ac.at>

# Licenses

The inductive benchmark datasets FB15k-237 v1-4, WN18RR v1-4, and NELL-995 v1-4 are already included in the PyKEEN library. PyKEEN uses the MIT license. 
FB15k-237 is a subset of FB15k, which uses the CC BY 2.5 license. 
The license of FB15k-237 v1-4, WN18RR v1-4, and NELL-995 v1-4 are unknown. This project runs under the MIT license.

Copyright (c) 2025 Aleksandar Pavlović