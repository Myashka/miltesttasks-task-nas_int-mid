# MILTestTasks-task-NAS_int-mid
==============================

Test project for Middle Researcher position at MIL Team, MIPT

## Neural Architecture Search (NAS) Experiment

This repository contains the codebase and results for the Neural Architecture Search (NAS) experiment. The goal of this experiment is to explore the SuperNet architecture and evaluate its performance on various configurations.

## Results

Result presentation presented in the `reports/results.pptx` file.

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
