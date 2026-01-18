<h1 align="center">Di-COT</h1>
<h2 align="center">
Divide and Contrast:<br>
Learning Robust Temporal Features without Augmentation
</h2>


This repository contains the official Pytorch implementation of the "[**Divide and Contrast (Di-COT)**]", an unsupervised framework that avoids data augmentation and multiple encoder passes by contrasting informative substructures within a window rather than individual timesteps.

![Di-COT](./visuals/DiCOT.png?raw=true "Title")

## Data


To use **Di-COT** and the baseline models, you will need access to relevant time-series datasets. The following datasets are used in this repository:

For the Large benchmark datasets, we use the preprocessed dataset from the [**Series2Vec**](https://github.com/Navidfoumani/Series2Vec) repository for the PAMAP2, WISDM2, SLEEP and SKODA datasets. 

- [**HARTH**](https://archive.ics.uci.edu/dataset/779/harth): This is a human activity recognition (HAR) dataset that contains recordings from 22 participants, each wearing two 3-axial Axivity AX3 accelerometers for approximately 2 hours in a free-living setting at a sampling rate of 50Hz.

- [**ECG**](https://physionet.org/content/afdb/1.0.0/): We use the MIT-BIH Atrial Fibrillation dataset, which includes 25 long-term electrocardiogram (ECG) recordings of human subjects with atrial fibrillation, each with a duration of 10 hours.


For the UCR and UEA benchmarks, you can download them from the [**official website**](https://www.timeseriesclassification.com/).

Make sure to place the dataset in the appropriate directory (e.g., `datasets/harth`) as specified in the configuration files.


## Usage

### Unsupervised pretraining of Di-COT

To pretrain the Di-COT model for classification, use the following command:

```bash
python pretrain.py <method> <dataset> -p <configs/<dataset>config.yml> -s < > --evaluate < >

```
- `method` specifies the self-supervised method to train.
- `dataset` specifies the dataset directory.
- `-p` specifies the configuration file.
- `-s` sets the seed for reproducibility.
- `--evaluate` [optional] define the downstream task to perform after pretraining.

### Example
For example, to pretrain Di-COT on the harth dataset with a seed of 1 and evaluate on supervised classification, run:
```bash
python pretrain.py Di-COT harth -p configs/harthconfig.yml -s 1 --evaluate supervised
```
Check the scripts/ directory for complete list of training scripts for all tasks in the paper as well as the different seeds used for reproducibility.

### Running Baseline Methods
To compare Di-COT against competitive method, you can use similar commands to pretrain the baselines. For example, to pretrain the TNC baseline on Skoda:

```bash
python pretrain.py TNC Skoda -p configs/Skodaconfig.yml -s 1 --evaluate supervised
```

## Visualizations

The figure below shows a t-SNE plot of the learned representation from all baselines on the Skoda dataset.

![t-SNE Visualization](./visuals/tSNE_DiCOT.svg?raw=true "Title")


## Acknowledgements

This repository provides reimplementations of several baselines for time-series representation learning using some parts of the codes provided by the following  works:

- [**TS2Vec**](https://github.com/zhihanyue/ts2vec): Towards Universal Representation of Time Series.

- [**Soft**](https://github.com/seunghan96/softclt?tab=readme-ov-file): Soft Contrastive Learning for Time Series.

- [**SimMTM**](https://github.com/thuml/SimMTM): A Simple Pre-Training Framework for Masked Time-Series Modeling.

- [**CoST**](https://github.com/salesforce/CoST): Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting.

- [**InfoTS**](https://github.com/chengw07/InfoTS): Time Series Contrastive Learning with Information-Aware Augmentations.

- [**TNC**](https://github.com/sanatonek/TNC_representation_learning): Unsupervised Representation Learning for TimeSeries with Temporal Neighborhood Coding.

- [**TF-C**](https://github.com/mims-harvard/TFC-pretraining): Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency.

- [**TS-TCC**](https://github.com/emadeldeen24/TS-TCC): Time-Series Representation Learning via Temporal and Contextual Contrasting.

- [**CaTT**](https://github.com/sfi-norwai/CaTT): Contrast All The Time: Learning Time Series Representation from Temporal Consistency.


Please check out the original repositories for more details.