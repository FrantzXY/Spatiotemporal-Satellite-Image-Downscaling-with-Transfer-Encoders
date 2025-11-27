# Spatiotemporal-Satellite-Image-Downscaling-with-Transfer-Encoders
This is a code repository for the paper: Spatiotemporal Satellite Image Downscaling with Transfer Encoders and Autoregressive Generative Models. 
This repository contains the full implementation of our **two-stage transfer learning downscaling framework**, where a **small MERRA-2 U-Net** is pretrained to learn long-range spatiotemporal structure, and its **frozen encoder** is transferred into a **large DDPM (diffusion model)** to generate **7-km G5NR-resolution dust-extinction AOD**.

The pipeline supports:

* Domain Diagnostic for Transfer Learning
* Seasonal × Regional training
* U-Net pretraining on long MERRA-2 sequences
* Trained DDPM main downscaling modelling with transferred features
* Preidected Image Stitching with halo-and-Hann
* In-data and out-of-data evaluation

---

## 📁 Repository Structure

### **large model/**

End-to-end scripts for the **large downscaling DDPM** with transferred features from the small model.

* **domain_similar_check_wd.py**
  Computes Wasserstein Distance (WD) between MERRA-2 and G5NR domains to verify domain similarity before transfer.

* **evaluate_large_ddpm.py**
  Evaluates large DDPM models on held-out G5NR days (computes RMSE, MAE, R², NSE, KGE, plots, etc.).

* **get_large_seasonal_data_ready.py**
  Generates Season × Area splits and prepares training/validation/test indices for the large DDPM.

* **large_u_ddpm_with_transfer.sh**
  SLURM job script for running seasonal training of DDPM with transfer encoder (HPC pipeline).


* **train_ddpm_with_transfer.py**
  Main training script for the large DDPM model. Loads transferred features, handles patching, noise schedule, loss, checkpointing.

---

### **small model/**

Scripts for training and evaluating the **small MERRA-2 U-Net**, and generating transfer encoders.

* **evaluate_small.py**
  Evaluates the pretrained small MERRA-2 U-Net by MAE/R² on the validation/test periods.

* **run_small_transfer_unet.sh**
  SLURM script for training small U-Net models across seasons/areas.

* **transfer_unet_fur_diffuser.py**
  Loads the trained encoder weights, freezes them, and produces transfer features for the large DDPM.

---

### **util tools/**

Shared utilities used by both small and large models.

* **fixed_transfer_data_loader.py**
  Main dataloader for seasonal MERRA-2 sequences, geographic variables, elevation, etc. Used by the small model (pretraining + transfer-feature generation).

* **fixed_transfer_data_loader_old.py**
  Dataloader used by the large DDPM model. Structure is similar to the small-model loader, but adapted for loading G5NR-aligned inputs and transferred features.
  
* **torch_ddpm_downscale.py**
  Core PyTorch DDPM implementation used by the large downscaling model (scheduler, noise prediction UNet, sampling).

* **torch_downscale.py**
  Utility functions for patch extraction, halo cropping, and Hann patch-stitching are used during inference.

---

## 🚀 Project Overview

This project proposes a **transfer-learning-enhanced diffusion downscaler** for dust-extinction AOD:

1. **Small U-Net (MERRA-2)**

   * Trained autoregressively on 20+ years of MERRA-2
   * Learns long-term, stable spatiotemporal structure
   * Encoder weights are frozen and transferred

2. **Large DDPM (G5NR)**

   * Receives coarse MERRA-2 driver, elevation, geographic variables, and **transferred features**
   * Produces **7-km** high-resolution predictions
   * Uses halo-and-Hann patch stitching to remove seams

3. **Evaluation**

   * In-data: RMSE, MAE, R², NSE, KGE
   * OOD: semivariograms, ACF/PACF, lag-RMSE, lag-R²
   * Demonstrates strong spatial detail reconstruction & stable OOD behavior
