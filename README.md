# Spectrum Sensing Enhancement Using Reinforcement Learning

## Description
This research focuses on improving spectrum sensing performance using reinforcement learning techniques. The goal is to enhance detection probability and reduce access time in cognitive radio networks.

## Features
- Implemented reinforcement learning algorithms for spectrum optimization.
- Developed a cooperative spectrum sensing model.
- Evaluated performance metrics for signal detection efficiency.

## Repository Structure
This repository contains the following key files:

- **`functions.py`**: Contains helper functions used in the project, such as generating neighbor sets and calculating reward values.
- **`arm.py`**: Defines the Multi-Armed Bandit (MAB) problem and solves it using the Upper Confidence Bound (UCB) algorithm.
- **`constructor_function_prop.py`**: Implements the proposed complete model, relying on functions from `functions.py` and `arm.py`.
- **`constructor_function_ref.py`**: Implements the reference complete model, also dependent on `functions.py` and `arm.py`.
- **`application.py`**: Demonstrates how to use `constructor_function_prop.py` and `constructor_function_ref.py` to display results
