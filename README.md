# OirthirSAT – Python Image Processing for Coastline Vectorisation

This repository contains the initial Python code used in the image processing pipeline. The aim is to extract a **vectorised coastline** from satellite imagery, enabling automated monitoring of coastlines for erosion. Once validated, the pipeline will be ported to C++ and deployed onboard the satellite.

## Overview

The image processing workflow is designed to handle real satellite data and convert it into a usable, high-resolution coastline vector. The pipeline must be fast, lightweight, and capable of processing images onboard with minimal error, along complex or cloud-obstructed coastlines.

## Current Functional Modules

The project is still under active development, but consists of the following major components:

### 1. Cloud Masking  
Identifies and masks out clouds from imagery to ensure downstream processing only uses valid land/water data.

### 2. Entropy-Based Image Segmentation *(Tentative)*  
This technique aims to identify and discard regions that are either entirely land or entirely water, identified by their low or high local entropy. The assumption is that **coastlines exist in areas with moderate entropy**, due to the mixed nature of land and water boundaries. Retaining only these areas is expected to reduce the total number of pixels that need full processing, lowering computation cost.

#### Potential Improvements to Entropy-Based Segmentation

- **Multi-scale entropy analysis**: Compute entropy at varying spatial resolutions to capture coastline transitions at different scales.
- **Directional entropy**: Evaluate entropy along specific axes (e.g. horizontal, vertical) to better detect structured shoreline boundaries.
- **Entropy-gradient fusion**: Combine entropy maps with image gradients (e.g. Sobel edge detection) to more accurately isolate transition zones.
- **Spectral-band entropy**: Perform entropy analysis on spectral indices (e.g. NIR, NDWI) to exploit clearer land-water separation.
- **Lightweight ML-based filtering**: Use a simple classifier trained on entropy, gradient, and spectral features to discard low-value regions efficiently.

### 3. UNET-Based Segmentation *(Tentative)*  
The goal is to train a UNET model to **directly extract coastlines**, replacing manual or spectral segmentation. However, current performance is limited. If this approach proves unreliable, the fallback use case is to use UNET for **general land-water segmentation**, supporting downstream vectorisation through more traditional means.

### 4. Alternate Coastline Extraction  
Applies spectral and clustering-based techniques to efficiently and accurately segment the image and extract the coastline.

Approaches under evaluation:  
- **NIR Exploitation**: Water absorbs NIR strongly and appears black, whilst land does not. This spectral difference is used to segment land and water, either directly or via indices like NDWI.  
- **K-Means Clustering**: Groups pixels based on colour or spectral similarity to separate land and water.  
- **Contrast Stretching**: Enhances the dynamic range of input imagery, particularly useful in hazy or low-contrast scenes.  
- **Binarisation**: Converts segmented output to a binary land/water mask suitable for vector extraction.

### 5. Marching Squares – Subpixel Vectorisation  
Once a clean binary coastline mask is available, the Marching Squares algorithm is used to extract a **sub-pixel-accurate vector** of the coastline. This improves precision in coastline tracking and reduces the appearance of jagged or aliased boundaries.

---

## Coastline Data Analysis & Post-Processing

After coastline extraction, further processing and analysis can be applied:

- **Noise Removal**: Exclude small or irrelevant features (e.g. lakes, rivers, artefacts) using filters based on blob size, location, or external map overlays.  
- **Smoothing**: Clean up jagged vector edges using curve fitting or line simplification algorithms.  
- **Temporal Differencing** *(Planned)*: Compare coastline vectors over time to track erosion, deposition, or other long-term changes.  
- **Erosion Quantification**: Calculate displacement metrics, retreat rates, and identify areas of accelerated coastal change.

---

## In Progress

- Developing and validating the alternate coastline extraction pipeline  
- Evaluating and benchmarking UNET versus entropy-based segmentation  
- Consolidating code modules into a single pipeline for integration  
- Assembling a representative dataset consistent with the onboard satellite imager for validation and stress-testing

## Known Issues and Limitations

- **UNET performance** is currently insufficient for reliable coastline vector extraction. It may be **replaced or augmented** by alternative techniques if accuracy cannot be improved.  
- **Dataset limitations** are hindering tuning and robustness checks. Access to realistic satellite data is needed to test segmentation and thresholding methods under diverse conditions.  
- **Entropy segmentation** remains unvalidated and may not generalise well across all geographies or lighting conditions.

---

## Roadmap

- [x] Cloud masking implemented  
- [x] Marching Squares implemented  
- [ ] Alternate coastline extraction in progress  
- [ ] UNET and entropy segmentation under evaluation  
- [ ] Pipeline integration and modular cleanup  
- [ ] Testing with representative satellite imagery  
- [ ] Final C++ port (to be developed in the OirthirSAT C++ repository)

## Contributors

Developed by the **OirthirSAT Software Team**.

## Licence

To be determined.
