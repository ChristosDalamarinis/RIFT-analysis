# RIFT-analysis
Collection of MNE script for RIFT-EEG analysis

Author: Christos Dalamarinis

Date: March - 2026

![CI](https://github.com/ChristosDalamarinis/RIFT-analysis/actions/workflows/blank.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![MNE](https://img.shields.io/badge/MNE-1.7-purple?logo=data:image/png;base64,)
![conda](https://img.shields.io/badge/env-miniconda-green?logo=anaconda&logoColor=white)
![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)

---

## `rift_analysis.ipynb`

Main analysis notebook implementing the full **RIFT-EEG preprocessing and analysis pipeline**.

### Purpose
Detects and quantifies EEG phase-locking to a 60 Hz visual flicker stimulus (Rapid Invisible Frequency Tagging) using coherence analysis. Processes BioSemi `.bdf` recordings across multiple subjects.

### Pipeline Structure

| Phase   |                                                     Description                                         |    Status   |
|-------  |---------------------------------------------------------------------------------------------------------|-------------|
|   1     | Data loading, metadata inspection, montage setup, PSD quality checks                                    | Implemented |
|   2     | Preprocessing: filtering, downsampling, re-referencing, ICA artifact removal, bad channel interpolation | Implemented |
|   3     | Epoching and epoch-level quality control                                                                |   Planned   |
|   4     | RIFT coherence analysis (main spectral analysis at 60 Hz)                                               |   Planned   |
|   5     | Statistical testing                                                                                     |   Planned   |
|   6     | Documentation and export                                                                                |   Planned   |

### Key Design Choices
- **Low-pass at 100 Hz** (not the standard 40 Hz) to preserve the 60 Hz tagging signal
- **NO notch at 60 Hz** — Netherlands mains frequency is 50 Hz, cleanly separated from the signal
- **Linked mastoid reference** (EXG5/EXG6)
- **ICA** fitted on a 1 Hz high-pass copy for better convergence, applied to the 0.1 Hz filtered data
- **Downsampling** from 1024 Hz → 512 Hz (Nyquist well above 60 Hz)

### Inputs
```
data/
  sub1/  sub2/  sub3/  sub4/
    *.bdf              # Raw EEG (BioSemi, 1024 Hz, 73 channels)
```

### Key Dependencies
`mne`, `mne-icalabel`, `mne-connectivity`, `numpy`, `scipy`, `pandas`, `matplotlib`, `autoreject`