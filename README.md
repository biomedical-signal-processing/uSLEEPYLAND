
# uSLEEPYLAND

The `uSLEEPYLAND` repository is a revised version of the [U-Time](https://github.com/perslev/U-Time), slightly redesigned for easier integration into the [SLEEPYLAND](https://github.com/biomedical-signal-processing/SLEEPYLAND) project. 
This revised version incorporates additional models to extend its utility in the context of SLEEPYLAND, providing an enhanced toolset for sleep staging analysis.


>This repository builds upon the foundational work presented in:
>
>**Mathias Perslev, Michael Hejselbak Jensen, Sune Darkner, Poul Jørgen Jennum, and Christian Igel.**  
*U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging.*  
Advances in Neural Information Processing Systems (NeurIPS 2019).
 
>While preserving the original pipeline described in U-Time, the primary focus of uSLEEPYLAND is to expose additional feature-based and deep learning models that can be seamlessly evaluated within the SLEEPYLAND framework. Note that the trained models are not included in this repository but are accessible via the SLEEPYLAND toolbox available in [Docker Hub](https://hub.docker.com/repository/docker/bspsupsi/sleepyland/general).

It is strongly recommended to follow the guidelines provided in the [SLEEPYLAND](https://github.com/biomedical-signal-processing/SLEEPYLAND) repository to fully leverage the `usleepyland` service, including access to pre-trained models and proper configuration for predictive analysis. Pre-trained models are accessible exclusively through the SLEEPYLAND toolbox on [Docker Hub](https://hub.docker.com/r/bspsupsi/sleepyland).

---

## Models for Sleep Staging

### Feature-Based Models for Sleep Staging

| **Model**        | **Description**                                                                                               | **Developed By**     | **References**                                             | **Available** |
|------------------|---------------------------------------------------------------------------------------------------------------|----------------------|-----------------------------------------------------------|---------------|
| **POPS (Luna)**   | A feature-based model for sleep stage classification using manually engineered features from PSG data.        | Luna Sleep Toolbox    | [Luna Resource](https://zzz.bwh.harvard.edu/luna/)      | ❌           |
| **YASA**          | A lightweight, open-source package for sleep staging, offering rapid sleep analysis using EEG/EOG/EMG signals. | Raphael Vallat, et al | Vallat R, Walker MP. [An open-source, high-performance tool for automated sleep staging.](https://doi.org/10.7554/eLife.70092) Elife. 2021. | ✅           |

**Note:** Feature-based models are used only for predictions and have not been retrained from scratch on datasets within the SLEEPYLAND context.

---

### Deep Learning Models for Sleep Staging

| **Model**       | **Description**                                                                                                               | **Developed By**                 | **References**                                                                                                                                                                                                              | **Available** |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| **U-Sleep**     | A robust deep learning model designed to handle various PSG setups and mixed-cohort datasets for automated sleep staging.      | Perslev, et al                   | Perslev M, Darkner S, et al. [U-Sleep: resilient high-frequency sleep staging.](https://doi.org/10.1038/s41746-021-00440-5) NPJ Digital Medicine. 2021.                                                                     | ✅          |
| **DeepResNet**  | A deep residual network model for high-frequency sleep stage classification using PSG data.                                    | Olesen, et al                    | Olesen AN, Jennum PJ, et al. [Automatic sleep stage classification with deep residual networks in a mixed-cohort setting.](https://doi.org/10.1093/sleep/zsaa161) Sleep. 2021.                                              | ✅          |
| **SleepTransformer** | A transformer-based model that provides sleep staging with interpretability and uncertainty quantification.                    | Phan, et al                      | Phan H, Mikkelsen K, et al. [Sleeptransformer: Automatic sleep staging with interpretability and uncertainty quantification.](https://doi.org/10.1109/TBME.2022.3147187) IEEE Transactions on Biomedical Engineering. 2022. | ✅          |
| **L-SeqSleepNet** | L-SeqSleepNet: Whole-cycle Long Sequence Modeling for Automatic Sleep Staging.                    | Phan, et al                      | Phan H, Lorenzen KP, et al. [L-SeqSleepNet: Whole-cycle Long Sequence Modeling for Automatic Sleep Staging.](https://doi.org/10.1109/JBHI.2023.3303197) IEEE Journal of Biomedical and Health Informatics. 2023.            | ❌          |

All models have been implemented to ensure consistency and fairness in their evaluation on the NSRR datasets. Additionally, all models are now available in a single-channel configuration (EEG or EOG) and in the multi-channel configuration (EEG+EOG), offering flexibility to match user data and requirements.

---
### Implemented Models

- **U-Sleep**. A fully convolutional neural network inspired by U-Net, designed for sleep staging from raw EEG and EOG signals. Key features include an encoder-decoder architecture with skip connections and a segment classifier that maps intermediate representations to sleep stage predictions. The model in SLEEPYLAND uses an increased initial learning rate (`η = 10^-5`) compared to the original implementation.


- **DeepResNet**. An adaptation of the ResNet architecture for 1D time-series PSG data, featuring convolutional residual blocks for feature extraction and bidirectional GRUs for temporal processing. Modifications include increased learning rate scheduler patience and adjusted initial learning rate (`η = 10^-5`).
 

- **SleepTransformer**. A fully transformer-based architecture for sequence-to-sequence sleep staging. The model includes an epoch transformer for intra-epoch processing and a sequence transformer for inter-epoch dependencies. Adjustments were made to ensure compatibility with multi-channel configurations and numerical stability.


- > **L-SeqSleepNet** ❗TODO❗


### Training Procedure and Data Management

- **Epoch Definition**: One training epoch processes $10^6$ sleep segments. Early stopping occurs after 200 epochs of no improvement, with a maximum limit of 10,000 epochs.


- **Batch Generation**: Batches of size 64 are created by sampling from the training dataset. PSG records and sleep stages are selected randomly, ensuring diverse training sequences.


- **Segment Length**: Fixed at 35 sleep epochs (17.5 minutes) for most models, except L-SeqSleepNet, which processes sequences of 100 epochs.


- **Spectrograms**: Generated on-the-fly for models requiring time-frequency images, following robust scaling per batch.


- **Data Augmentation**: Includes random noise injection and channel dropout, except for single-channel models.


### Deployment via Docker
The service is available as a Docker image for easy deployment in [Docker Hub](https://hub.docker.com/repository/docker/bspsupsi/sleepyland/general):
```bash
docker pull bspsupsi/sleepyland:usleepyland
docker run -v /path/to/data:/data bspsupsi/sleepyland:usleepyland
```

This containerized version ensures a consistent and reliable environment for running the service across various systems.

## Acknowledgments

The original [U-Time](https://github.com/perslev/U-Time) repository serves as the basis for this work. We acknowledge the foundational contributions of its authors. This revised version builds upon their efforts to expand its capabilities within the SLEEPYLAND project.

---

For any issues, feature requests, or contributions, please submit a pull request or create an issue in this repository or the [SLEEPYLAND](https://github.com/biomedical-signal-processing/SLEEPYLAND) repository.