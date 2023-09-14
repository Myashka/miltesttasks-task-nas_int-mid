# MILTestTasks-task-NAS_int-mid
==============================

Test project for Middle Researcher position at MIL Team, MIPT

## Neural Architecture Search (NAS) Experiment

This repository contains the codebase and results for the Neural Architecture Search (NAS) experiment. The goal of this experiment is to explore the SuperNet architecture and evaluate its performance on various configurations.

## Results

Result presentation presented in the `reports/results.pptx` file.

### Hypothesis
1. Training SuperNet subdivisions allows you to find a more optimal network configuration for training from scratch.

2. A deeper model can produce more consistent results between SuperNet training and training from scratch

### Configuration

**2 model configurations:**
- SuperNet 16_32
    - Init_conv – ConvBlock [1; 16]
    - Variable_block – 3x ConvBlock [16; 16]
    - Downsample_block – ConvBlock [16; 32]
    - Variable_block – 3x ConvBlock [32; 32]
    - Global_avg_pool
    - Linear – [32; 10]
- SuperNet 32_64
    - Init_conv – ConvBlock [1; 32]
    - Variable_block – 3x ConvBlock [32; 32]
    - Downsample_block – ConvBlock [32; 64]
    - Variable_block – 3x ConvBlock [64; 64]
    - Global_avg_pool
    - Linear – [64; 10]

**Training details:**
- Epochs = 200
- AdamW optimizer
    - Learning rate = 3e-0.3
    - Wight decay = 0.01
- Dataset MNIST
    - Validation data = 0.15% of train MNIST dataset
    - Train batch size = 8192
    - Validation batch size = 64
- Wandb logging

### Top-1 accuracy comparison table

|   Layers |   SuperNet_16_32 |   From scratch 16_32 |   SuperNet_32_64 |   From scratch 32_64 |
|---------:|-----------------:|---------------------:|-----------------:|---------------------:|
|      1_1 |           0.6613 |               0.9804 |           0.821  |               0.9855 |
|      1_2 |           0.8291 |               0.9915 |           0.8376 |               0.9922 |
|      1_3 |           0.8833 |               **0.9934** |           0.8667 |               0.994  |
|      2_1 |           0.9031 |               0.9857 |           0.902  |               0.9913 |
|      2_2 |           0.9479 |               0.9926 |           0.9443 |               0.994  |
|      2_3 |           **0.9578** |               0.9917 |           0.9539 |               0.9939 |
|      3_1 |           0.7435 |               0.987  |           0.9314 |               0.9925 |
|      3_2 |           0.8559 |               0.9912 |           0.9509 |               0.9937 |
|      3_3 |           0.8597 |               0.993  |           **0.956**  |               **0.9946** |


## Project Structure
```
MILTESTTASKS-TASK-NAS_INT-MID
│
├───artifacts        # Results from the training and testing processes
│   ├───test
│   │   ├───from_scratch
│   │   └───random_sampler
│   └───train
│
├───configs           # Configuration files for the experiment
│
├───data              # Directory for dataset storage (not tracked by git)
│
├───notebooks         # Jupyter notebooks for data exploration and result analysis
│
├───reports           # Results
│
├───scripts           # Script files for running the training and inference processes
│
├───src               # Source code for the experiment
│   ├───data
│   ├───eval
│   ├───logging
│   ├───losses
│   ├───metrics
│   ├───models
│   ├───train
│   └───utils
│
└───task              # Task description and related images
    └───pics
```

## Getting Started
1. **Setup Environment:**
- Install the required packages: `pip install -r requirements.txt`
- Install the local package: `pip install .`
2. **Training:**
- For training and test in the Google Colab use `notebooks/mil_colab_run.ipynb`
- Modify the `train-config.yaml` as per your experiment setup.
- Run the training script: `python scripts/train.py --config-file path/train-config.yaml`
3. **Inference:**
- Modify the `test-config.yaml` as per your test setup.
- Run the inference script: `python scripts/inference.py --config-file path/test-config.yaml`
4. **Analysis:**
- Use the provided Jupyter notebooks inside the notebooks directory to analyze the results.


## Repository Contents

- **artifacts:** Contains the results from the training and test processes.
- **configs:** YAML configuration files for setting up the experiment.
- **notebooks:** Jupyter notebooks for data exploration, debug sessions, and result analysis.
- **scripts:** Script files to run the training and inference processes.
- **src:** Main source code for the project including models, utilities, metrics, and more.
- **task:** Description of the task and related images.
