# Overview

This repository contains the implementation of experiments conducted in the research paper titled *Generating Clinical Pathway Recommendations for Cancer Care with Offline Reinforcement Learning: A Breast Cancer Application Context* The study focuses on generating clinical pathway data for breast cancer patients using Ray to simulate patient trajectories and using Coach learns to recommend optimal clinical pathways in an offline fashion.

# Dependencies

The implementation utilizes two key libraries, Ray and Coach, both built on the TensorFlow deep learning framework. However, it's important to note that these libraries have different version requirements. To ensure a smooth execution, it is recommended to install the required packages in separate Conda environments.

## Installation

Create two separate virtual environment and install the packages.

### virtual environment

	conda create -n BreastCancerDCIS_Coach python=3.6

	conda create -n BreastCancerDCIS_Ray python=3.6

### Coach

	source activate BreastCancerDCIS_Coach
	[Installation Guide](https://github.com/IntelLabs/coach#installation)

### Ray

	source activate BreastCancerDCIS_Ray
	pip install -U "ray[rllib]==1.11.1"

## Running Experiments

To reproduce the experiments outlined in the paper, ensure that you activate the appropriate Conda environment before executing the scripts. Training agent and generating data from the agent requires Ray while Offline RL learning and policy evaluation requires Coach. Training Agent, Creating Agent generated data, Offline Learning and Evaluation all have separate command lines. 

### Experiment 1
		
		python train_agent.py -t
		python offline_learning.py -q -t -e
		python evaluate.py -q -t -e


### Experiment 2
		python create_augmented_data.py -t
		python offline_learning.py -q -t -e
		python evaluate.py -q -t -e

### Experiment 3
		python offline_learning.py -q -t -e
		python evaluate.py -q -t -e

