# Spatiotemporal-Satellite-Image-Downscaling-with-Transfer-Encoders
This is a code repository for the paper: Spatiotemporal Satellite Image Downscaling with Transfer Encoders and Autoregressive Generative Models. 
This repository contains the full implementation of our **two-stage transfer learning downscaling framework**, where a **small MERRA-2 U-Net** is pretrained to learn long-range spatiotemporal structure, and its **frozen encoder** is transferred into a **large DDPM (diffusion model)** to generate **7-km G5NR-resolution dust-extinction AOD**.

The pipeline supports:

* Domain Diagnostic for Transfer Learning
* Seasonal × Regional training
* U-Net pretraining on long MERRA-2 sequences
* Trained DDPM main downscaling modeling with transferred features
* Predicted Image Stitching with halo-and-Hann
* In-data and out-of-data evaluation

---

## 📁 Repository Structure

### **large model/**

Scripts for training and evaluating the **main downscaling model: DDPM** with transferred features from the small model.

* **domain_similar_check_wd.py**
  A script to compute Wasserstein Distance between MERRA-2 and G5NR domains to verify domain similarity before transfer.

* **evaluate_large_ddpm.py**
  A model evaluation script to evaluate the reconstructed high-resolution images autoregressively predicted by the trained large DDPM models on held-out G5NR test days by computing pixel-wise RMSE, MAE, R square, NSE, and KGE.

* **large_u_ddpm_with_transfer.sh**
  A SLURM job script for running seasonal training of DDPM with transfer encoder in Compute Canada HPC clusters.

* **train_ddpm_with_transfer.py**
  A training script for training the large DDPM model with transferred model. It synthesizes all util tools scripts and other large model scripts to transfer pre-trained model, train the main model, predict and reconstruct images, and evaluate the prediction.

---

### **small model/**

Scripts for training and evaluating the **small MERRA-2 U-Net**, and generating transfer encoders.

* **evaluate_small.py**
  A model evaluation script to evaluate the autoregressive prediction of MERRA-2 day-ahead dust extinction produced by the trained small model by computing pixel-wise R square and MAE.

* **run_small_transfer_unet.sh**
  A SLURM script for training small U-Net models across seasons/areas in Compute Canada HPC clusters.

* **transfer_unet_fur_diffuser.py**
  A training script for training the small model with its util tools scripts and saving the weights of the checkpointed model.

---

### **util tools/**

Shared utilities used by small and large models.

* **get_large_seasonal_data_ready.py**
 A data preparation script shared between small and large models to generate Season × Area splits and prepare training/validation/test indices for the model to be trained.

* **fixed_transfer_data_loader.py**
 A data preparation script used by the small model for loading MERRA-2 sequences, geographic variables, elevation, etc.

* **fixed_transfer_data_loader_old.py**
 A data preparation script used by the large DDPM model. Structure is similar to the small-model data loader, but adapted for loading DDPM inputs.
  
* **torch_ddpm_downscale_halo_hann**
 An inference script used by the trained DDPM to perform in-data and out-of-data downscaling autoregressively, and reconstruct the predicted patch images with halo cropping and Hann patch-stitching to prevent edge effect.

* **torch_downscale.py**
 An inference script used by the trained small model to perform in-data and out-of-data downscaling autoregressively.

* **r_rmse_curves.py**
 A plotting script for performing exploratory data analysis and graphical diagnosis of out-of-data downscaling predictions by plotting image-wise RMSE and R square between target day and its associated lags.

* **semivariogram_ACF_qc_plot.py**
Generates empirical semivariogram and ACF/PACF diagnostics comparing G5NR, MERRA-2, and model predictions by Season × Area, for both in-data and out-of-data runs.

---

## 🚀 Project Overview

This project proposes a **transfer-learning-enhanced diffusion downscaler** for dust-extinction AOD:

1. **Small U-Net (MERRA-2)**

   * Trained autoregressively on 20+ years of MERRA-2
   * Learns long-term, stable spatiotemporal structure
   * Encoder layers' weights are frozen and transferred

2. **Large DDPM (G5NR)**

   * Receives coarse MERRA-2 field, elevation, geographic variables, and **transferred features**
   * Produces **7-km** high-resolution predictions for dust-extinction AOD at the pre-specified region and season. 
   * Uses halo-and-Hann patch stitching to reconstruct predicted patch images and reduce edge effect.

3. **Evaluation**

   * In-data Downscaling Evaluation: RMSE, MAE, R square, NSE, KGE
   * Out-of-data (OOD) Downscaling Evaluation: Semivariograms, ACF/PACF plots, RMSE plots, R square plots
   * Demonstrates strong spatial detail reconstruction & stable OOD behavior
