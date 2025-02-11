# Neural Network Classification of Top Quark Production at the Large Hadron Collider (LHC)

## Overview
This project applies machine learning techniques to classify top-quark-antiquark pair production with the Z boson (𝑡𝑡̅𝑍) at the Large Hadron Collider (LHC). The objective is to distinguish between signal events (𝑡𝑡̅𝑍) and background events (WZ production) using Decision Trees and Neural Networks.

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Discussion](#discussion)
- [Limitations and Future Work](#limitations-and-future-work)
- [References](#references)

## Introduction
The Standard Model (SM) of particle physics provides the best framework for understanding fundamental particles and forces. However, questions remain about its completeness. High Energy Physics (HEP) experiments at the LHC leverage machine learning (ML) techniques to extract signals from complex background data. This project aims to classify a rare SM process—𝑡𝑡̅𝑍 production—using ML methods.

## Objectives
- Train ML models to classify 𝑡𝑡̅𝑍 events from background WZ events.
- Optimize hyperparameters for improved classification accuracy.
- Compare Decision Trees and Neural Networks to determine the best-performing model.
- Evaluate model performance using AUC (Area Under the Curve) and other metrics.

## Dataset
The dataset consists of Monte Carlo simulations of proton-proton collisions at the ATLAS experiment, featuring:
- 741,538 samples
- 18 features related to reconstructed objects, momenta, and missing transverse momentum
- Binary labels (1 = signal, 0 = background)

## Methodology
1. **Preprocessing**:
   - Removed outliers using z-score calculations.
   - Performed feature selection using Decision Trees.
   - Split data into training (80%) and testing (20%) sets.
   - Applied cross-validation for robust evaluation.

2. **Decision Tree Classifier**:
   - Used entropy and Gini index as splitting criteria.
   - Identified top 10 most important features.
   - Optimized hyperparameters using Randomized Search.

3. **Neural Network Classifier**:
   - Implemented a feed-forward NN with:
     - 2 hidden layers (450 neurons each)
     - Activation functions: ReLU and tanh
     - Adam optimizer with a learning rate of 0.001
   - Trained using binary cross-entropy loss and backpropagation.

4. **Evaluation Metrics**:
   - AUC score (primary metric)
   - Confusion matrix analysis
   - Signal efficiency vs. background rejection

## Results
- Decision Tree achieved an AUC score of **0.87** after hyperparameter tuning.
- Neural Network achieved an AUC score of **0.88**, slightly outperforming the Decision Tree.
- Feature importance analysis identified `n_bjets` as the most significant feature.
- No significant performance difference when using only the top 10 features.

## Discussion
Despite expectations that the Neural Network would outperform the Decision Tree significantly, the results were very similar. This suggests that the relationships in the dataset are simple enough to be captured effectively by a Decision Tree. The models demonstrated strong classification performance, distinguishing 𝑡𝑡̅𝑍 from background events with high accuracy.

## Limitations and Future Work
- **Computational Constraints**: Limited access to high-performance computing resources restricted hyperparameter tuning and model complexity.
- **Real Data vs. Simulated Data**: Future work should include training on real experimental data to account for detector effects.
- **Model Improvements**:
  - Testing Recurrent Neural Networks (RNNs) for sequential dependencies.
  - Expanding hyperparameter search space for potential performance gains.
  - Exploring additional feature engineering techniques.

## References
1. [CERN - The Standard Model](https://home.cern/science/physics/standard-model)
2. Baldi, P., Sadowski, P., & Whiteson, D. (2014). "Searching for exotic particles in high-energy physics with deep learning." *Nat Communications, 5.*
3. Guest, D., Krammer, K., & Whiteson, D. (2018). "Deep learning and its applications to LHC physics." *Annual Review of Nuclear and Particle Science, 68*, 161-181.
4. Bourilkov, D. (2019). "Machine and Deep Learning Applications in Particle Physics." *International Journal of Modern Physics A, 34(35).*

---
<!-- This repository contains:
- `notebooks/`: Jupyter notebooks for data processing and model training.
- `data/`: Processed datasets used for training and testing.
- `models/`: Saved models for Decision Trees and Neural Networks.
- `results/`: Performance metrics, plots, and model evaluation summaries. -->

Contributions and improvements are welcome! 🚀

