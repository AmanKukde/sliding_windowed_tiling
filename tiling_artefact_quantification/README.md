# 🧩 Tiling Artifact Metrics

This repository provides tools to **quantify and compare tiling artifacts** in model predictions.
When models process large images in smaller tiles, they often leave visible seams or statistical discontinuities at tile boundaries. Our framework measures and visualizes these differences, making it easier to compare artifact severity across different models or inference strategies.

### 🔑 Features

* **Gradient-based metrics**: Extract gradients from model outputs and compare distributions at tile **edges** vs. **centers**.
* **Histogram & divergence analysis**: Compute histograms of gradient values and quantify differences with **KL divergence** (histogram-based or Gaussian-approximated).
  
* **Flexible visualization**:

  * Heatmaps of KL divergences across tiles
  * Histograms & bar plots of edge vs. middle statistics
  * Optional Gaussian / Difference-of-Gaussians (DoG) fits for smooth approximations
* **Multi-model comparison**: Easily compare artifact patterns between different models (e.g., inner-tiling vs. sliding-window inference).

### 📊 Example Insights
* Quantify whether one model shows stronger boundary artifacts.

### 🚀 Applications

* Benchmarking **image-to-image models** (segmentation, restoration, diffusion, etc.) for seamless tiling.
* Diagnosing tiling strategies in **large-scale inference pipelines**.
* Research on **artifact-aware loss functions** or **post-processing methods**.
