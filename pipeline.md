# RIFT-EEG Data Analysis Plan in Python

Author: Christos Dalamarinis
Date: March 2026

## Overview

This document lays out a complete, step-by-step pipeline for analyzing **Rapid Invisible Frequency Tagging (RIFT)** EEG data using Python. RIFT is a frequency-tagging technique in which visual stimuli is flickered at high frequencies (here **60 Hz**) that are perceptually invisible to participants. The brain nonetheless entrains to the flicker, producing a steady-state response at the tagging frequency that can be detected in the EEG. The core analysis question is: **does the EEG signal show reliable phase-locking to the 60 Hz flicker?** This is assessed via coherence between each EEG channel and a reference signal representing the flicker.

The pipeline covers: 
1) **data loading**,
2) **quality checks**,
3) **RIFT-aware preprocessing** (with critical adjustments to filtering and sampling rate to preserve the 60 Hz signal), 
4) **epoching**, and a 
5) **Fourier-based coherence analysis**.

The primary toolchain centers on **MNE-Python**, supplemented by NumPy, SciPy, pandas, and matplotlib.

### Core Libraries

|              Library             |                              Role                                     |
|----------------------------------|-----------------------------------------------------------------------|
| `mne`                       ->   | Loading, preprocessing, epoching, spectral analysis                   |
| `numpy` / `scipy`           ->   | FFT, cross-spectral density, coherence computation, signal processing |
| `pandas`                    ->   | Behavioral data, event logs, tabular summaries                        |
| `matplotlib` / `seaborn`    ->   | Visualization (spectra, topographies, coherence maps)                 |
| `autoreject`                ->   | Automated epoch rejection and repair                                  |
| `mne-icalabel`              ->   | Automated IC classification (eye blinks, muscle, etc.)                |
| `mne-connectivity`          ->   | Spectral connectivity (coherence) between EEG and reference signal    |
| `statsmodels` / `pingouin`  ->   | Statistical testing, permutation tests                                |

---

## Phase 1 — Data Loading and Initial Inspection

### 1.1 Load the Raw Data

EEG data comes in many vendor-specific formats, however here we focus on **.bdf** format from Biosemi  `read_raw_bdf()` .

**What to do:**

```python
import mne
raw = mne.io.read_raw_bdf("sub-01.bdf", preload=True)
```

Setting `preload=True` loads the full data into memory, which is required for most preprocessing operations. 

### 1.2 Inspect the Metadata

Before touching the data, verify that the recording metadata is correct.

**Check the following:**

1) **Sampling rate (`raw.info['sfreq']`):** Confirm it matches what you expect from the recording system (e.g., 512 Hz, 1024 Hz).
2) **Channel count and names (`raw.info['ch_names']`, `raw.info['nchan']`):** Ensure the number of channels is correct and channel labels look right (no duplicates, no unnamed channels).
3) **Channel types:** Verify that channels are labeled correctly as *EEG*, *EOG*, *EMG*, *ECG*, *STIM*, or *miscellaneous*. Mislabeled channels will be processed incorrectly downstream.
4) **Recording duration (`raw.times[-1]`):** Sanity-check the total recording length against your protocol. Check for total amount of trials and blocks.
5) **Events/annotations (`raw.annotations`):** Confirm that event markers (triggers) are present and match your experimental design.

```python
print(raw.info)
print(f"Duration: {raw.times[-1]:.1f} seconds")
print(f"Channels: {raw.info['nchan']}")
print(f"Sampling rate: {raw.info['sfreq']} Hz")
print(raw.annotations)
```

### 1.3 Set Channel Types and Montage

Most formats do not embed channel type information reliably. Manually set the types for any non-EEG channels.

```python
MASTOID_CHANNELS = ['EXG5', 'EXG6']
EOG_CHANNELS     = ['EXG1', 'EXG2', 'EXG3', 'EXG4']
UNUSED_CHANNELS  = ['EXG7', 'EXG8']
```

Then attach a standard electrode montage so that MNE knows the 3D positions of each electrode. This is required for topographic plotting, interpolation, re-referencing to average, and source localization.

```python
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing="warn")
```

If your cap uses a non-standard layout, you may need to load a custom montage from a `.sfp`, `.elc`, `.tsv`, or digitization file.

### 1.4 Visual Inspection of the Raw Data

Plot the raw continuous data and scroll through it to get a feel for the recording quality.

```python
raw.plot(duration=20, n_channels=30, scalings="auto")
```

**What to look for:**

1) **Flat channels:** Channels showing zero or near-zero amplitude throughout — likely disconnected electrodes.
2) **Excessively noisy channels:** Channels with dramatically higher amplitude or obvious high-frequency noise relative to neighbors.
3) **Large slow drifts:** Very slow oscillations (< 0.1 Hz) that dominate the signal — common if the participant moved or the electrode impedance was poor.
4) **Muscle artifacts:** High-frequency bursts (often > 20 Hz) associated with jaw clenching, swallowing, or head/neck movement.
5) **Eye blinks and saccades:** Large, slow deflections most prominent at frontal electrodes (Fp1, Fp2).
6) **Electrical line noise:** A clear 50 Hz or 60 Hz oscillation (depending on country) visible as a sustained hum.
7) **Event markers:** Verify that triggers appear at the expected times and in the expected pattern.

### 1.5 Check Power Spectral Density (PSD) 

Plot the PSD of the raw data to identify frequency-domain problems before preprocessing.

```python
raw.compute_psd(fmax=100).plot()
```

This is a **general data quality check** — the goal here is to assess the overall health of the recording, not to detect the RIFT signal. At this stage the data is raw and unpreprocessed, so do not be alarmed if you cannot see a clear 60 Hz peak. That is expected and normal.

**What to look for:**
1) **Overall spectral shape** — the spectrum should follow a smooth 1/f curve (power decreasing with frequency), with a visible alpha bump around 8–13 Hz. This is your basic sanity check: *does this look like EEG?*
2) **50 Hz line noise** — a sharp spike at 50 Hz (Netherlands mains frequency) indicates electrical interference that will need to be removed in Phase 2 with a notch filter.
3) **Excessive high-frequency power** — a broad elevation above ~40 Hz across many channels usually indicates muscle artifacts.
4) **Outlier channels** — channels with a clearly different spectral profile from their neighbours (much higher or lower power across the board) are likely bad channels.
5) **Low-frequency drift** — disproportionately high power at very low frequencies (< 1 Hz) suggests slow drifts from movement or poor electrode contact.

**Note:** You may or may not see a small peak at 60 Hz here — if it is visible, that is a good sign, but its absence does not indicate a problem. Raw PSD computed over the entire recording is a blunt tool: it averages over RIFT-on and RIFT-off periods alike, discards phase information entirely, and has no noise reduction applied. Detecting the RIFT response properly requires restricting the analysis to **RIFT-on time windows**, cleaning the data, and using **phase-sensitive measures such as coherence** — all of which are handled in Phase 4.

### 1.6 Exploratory PSD on RIFT-On Epochs (Raw, Unpreprocessed)

An optional sanity check — epoch the raw data around RIFT-on triggers (trigger code **40**) and compute the PSD only over those windows, to see if restricting to flicker-active periods makes a 60 Hz elevation more visible compared to 1.5.

**Important caveats:**
- The data is still raw and unpreprocessed — artifacts, noise, and drifts are all still present
- This epoching is **temporary and exploratory only** — it operates on a copy of `raw` and does not alter it in any way
- Not seeing a 60 Hz peak here is still perfectly normal for the same reasons discussed in 1.5
- This is **not** the RIFT analysis — that happens in Phase 4 after preprocessing, using coherence

**Epoch window:** `tmin=-0.2` (200 ms before flicker onset) and `tmax=2.5` (full 2000 ms RIFT window + 500 ms after it ends).

```python
raw_copy       = raw.copy()
rift_on_events = events[events[:, 2] == 40]

epochs_explore = mne.Epochs(
    raw_copy, rift_on_events, event_id={'rift_on': 40},
    tmin=-0.2, tmax=2.5, baseline=None, picks='eeg', preload=True,
)

epochs_explore.compute_psd(fmin=40, fmax=80).plot(average=True)

del raw_copy, epochs_explore   # discard — raw is unchanged
```

---

## Phase 2 — Preprocessing

### 2.1 Exclude or Mark Bad Channels

Based on your visual inspection and PSD review, mark any channels that are clearly unusable.

```python
raw.info["bads"] = ["Cz", "T8"]  # example bad channels
```

Do not drop them yet — marking them as "bad" excludes them from key operations (like average re-referencing and ICA) while preserving them for later interpolation.

**Automated approaches:** You can also use automated methods such as `mne.preprocessing.find_bad_channels_maxwell()` (for MEG, but adaptable), RANSAC from `autoreject`, or the PREP pipeline's approach (via `pyprep`). These are especially useful when processing large datasets where visual inspection of every recording is impractical.

```python
# Example with pyprep (if installed)
from pyprep.find_noisy_channels import NoisyChannels

nd = NoisyChannels(raw, random_state=42)
nd.find_all_bads()
raw.info["bads"] = nd.get_bads()
```

### 2.2 Filtering

Filtering removes frequency content outside the band of interest. The order and parameters matter — and for RIFT-EEG, the filter settings are **critically different** from a standard (e.g ERP) pipeline.

> **⚠️ CRITICAL FOR RIFT:** Your frequency of interest is 60 Hz. A standard low-pass filter at 40 Hz (common in ERP work) would **completely destroy your RIFT signal**. The low-pass must be set well above 60 Hz, or omitted entirely.

**High-pass filter (removing slow drifts):** A high-pass filter at **0.1 Hz** is appropriate. For improved ICA quality, consider filtering a copy at 1 Hz for the ICA fit (see 2.5).

```python
raw.filter(l_freq=0.1, h_freq=None, fir_design="firwin")
```

**Low-pass filter — RIFT-adjusted:** Set the low-pass to **100 Hz or higher** to comfortably preserve the **60 Hz tagging frequency** plus some headroom for spectral estimation. If your sampling rate is high enough (≥ 500 Hz), 100 Hz is a good choice. If you also care about the second harmonic (120 Hz), set it to 150 Hz or skip the low-pass entirely.

```python
# RIFT-appropriate low-pass: preserve 60 Hz (and optionally 120 Hz harmonic)
raw.filter(l_freq=None, h_freq=100.0, fir_design="firwin")
```

Or combined:

```python
raw.filter(l_freq=0.1, h_freq=100.0, fir_design="firwin")
```

**Notch filter (removing 50 Hz line noise):** In the Netherlands, line noise is at **50 Hz**. This is close to but distinct from your **60 Hz tagging frequency**. Apply a notch filter at 50 Hz (and its harmonics) to clean up line noise **without** touching 60 Hz. Be cautious with the notch bandwidth — MNE's default is usually fine, but verify that the notch does not extend into the 55–65 Hz range.

```python
# Remove 50 Hz line noise and harmonics (NL mains frequency)
raw.notch_filter(freqs=[50, 100, 150])
```

> **Do NOT notch-filter at 60 Hz** — that is your signal of interest. If you were in a 60 Hz mains country (e.g., the US), you would need a different tagging frequency or very careful separation strategies. Fortunately, the 50 Hz / 60 Hz separation works in your favor here.

After applying the notch, re-inspect the PSD to confirm the 50 Hz peak is gone and 60 Hz content is intact:

```python
raw.compute_psd(fmax=120).plot()
```

**Important notes on filter order:**

- Always high-pass filter before ICA (ICA performance degrades with slow drifts).
- If you plan to use ICA, consider filtering a copy of the data at 1 Hz for the ICA fit, then applying the resulting ICA solution back onto the original (0.1 Hz filtered) data. This gives you the best of both worlds: clean ICA decomposition and minimal signal distortion.

### 2.3 Downsampling (Optional but Constrained)

If the data was recorded at a very high sampling rate (e.g., 2048 Hz or 5000 Hz), downsampling can reduce memory use and computation time. However, for RIFT at 60 Hz, the Nyquist theorem requires a sampling rate of at least 120 Hz, and in practice you want **at least 3–4× your frequency of interest** for clean spectral estimation.

> **For 60 Hz RIFT: do not downsample below 500 Hz.** A sampling rate of 512 Hz or 500 Hz gives you comfortable headroom. If you also want to analyze the second harmonic at 120 Hz, keep the rate at 512 Hz or above.

```python
# Only downsample if your original rate is much higher (e.g., 2048 Hz)
raw.resample(sfreq=512)
```

MNE applies an anti-aliasing low-pass filter automatically before downsampling. Do this **after** filtering but **before** ICA and epoching.

### 2.4 Re-referencing

EEG is a relative measure — you always need a reference. The choice of reference affects the spatial distribution of your data and can influence results.

**Common reference choices:**

- **Average reference:** Projects the data to the average of all (good) channels. Appropriate when you have high-density (≥ 32 channel) recordings with good spatial coverage.
- **Linked mastoids (TP9/TP10 or M1/M2):** Traditional choice for many cognitive paradigms; good for auditory and language studies.
- **Cz or other single electrode:** Sometimes used, but introduces a spatial bias.
- **REST (Reference Electrode Standardization Technique):** An infinity reference approximation; available in MNE.

```python
# Average reference (most common for high-density EEG)
raw.set_eeg_reference("average", projection=True)

# Linked mastoids
raw.set_eeg_reference(["TP9", "TP10"])   # EXG5 and EXG6 in my case
```

If you chose average reference, MNE adds it as a projection. Apply it:

```python
raw.apply_proj()
```

### 2.5 Artifact Removal with ICA

Independent Component Analysis separates the data into statistically independent sources. Some of these correspond to artifacts (eye blinks, saccades, heartbeat, muscle activity), and others to brain activity. You remove the artifact components and reconstruct the data.

**Step 1 — Fit ICA:**

```python
from mne.preprocessing import ICA

ica = ICA(
    n_components=20,      # or 0.99 for variance-explained threshold
    method="infomax",     # or "fastica", "picard"
    random_state=42,
    max_iter=800,
)
ica.fit(raw)              # uses only good channels, respects raw.info["bads"]
```

The number of components is typically set to 15–25 for 32-channel data, or up to 50–60 for 64–128 channel data. Setting it to `0.99` tells MNE to keep enough components to explain 99% of the variance.

**Step 2 — Identify artifact components:**

You can do this visually, semi-automatically, or fully automatically.

*Visual:*

```python
ica.plot_sources(raw)      # time-series of each component
ica.plot_components()      # topographic maps of each component
```

Look for components with a frontal topography and slow, large deflections (blinks), a lateral frontal topography (saccades), or a focal distribution near the temples with high-frequency content (muscle).

*Semi-automatic (correlation with EOG/ECG):*

```python
eog_indices, eog_scores = ica.find_bads_eog(raw)
ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
ica.exclude = eog_indices + ecg_indices
```

*Fully automatic (ICLabel):*

```python
from mne_icalabel import label_components

labels = label_components(raw, ica, method="iclabel")

# Exclude components classified as eye blink, muscle, heart, etc.
# labels["labels"] is a list like ["brain", "eye blink", "muscle", ...]
ica.exclude = [
    i for i, label in enumerate(labels["labels"])
    if label not in ("brain", "other")
]
```

**Step 3 — Apply ICA (remove artifacts):**

```python
raw = ica.apply(raw)
```

If you used the two-step filtering strategy (1 Hz for ICA fit, 0.1 Hz for main data), apply the ICA from the 1 Hz copy onto the 0.1 Hz data:

```python
ica.apply(raw_01hz)  # raw_01hz is the 0.1 Hz filtered original
```

### 2.6 Interpolate Bad Channels

Now that artifacts are removed, interpolate any channels you marked as bad. This uses spherical spline interpolation based on the montage.

```python
raw.interpolate_bads(reset_bads=True)
```

`reset_bads=True` clears the bads list after interpolation, so the interpolated channels are now treated as normal data channels.

---

## Phase 3 — Epoching and Epoch-Level Quality Control

### 3.1 Extract Events

Events (triggers, stimulus markers) define the time-locking points for your epochs.

```python
events, event_id = mne.events_from_annotations(raw)

# Or define your own mapping
event_id = {"target": 1, "distractor": 2, "response": 3}
```

Check that events look correct:

```python
print(f"Total events found: {len(events)}")
for key, val in event_id.items():
    count = (events[:, 2] == val).sum()
    print(f"  {key} (id={val}): {count} trials")
```

### 3.2 Create Epochs

Cut the continuous data into time-locked segments around each event.

```python
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=-0.2,           # 200 ms before stimulus
    tmax=0.8,            # 800 ms after stimulus
    baseline=None,       # apply baseline later (see 3.4)
    preload=True,
    reject=None,         # handle rejection separately (see 3.3)
    detrend=1,           # linear detrend each epoch (optional)
)
```

**Parameter choices:**

- `tmin` / `tmax`: Should encompass your time window of interest plus some padding for edge effects of time-frequency analysis.
- `detrend=1`: Applies a linear detrend within each epoch, which can help with residual slow drifts.
- Setting `reject=None` initially lets you use more sophisticated rejection methods in the next step.

### 3.3 Epoch Rejection and Repair

Bad epochs (those contaminated by residual artifacts that ICA did not fully capture) need to be identified and either rejected or repaired.

**Option A — Simple amplitude threshold:**

```python
reject_criteria = dict(eeg=150e-6)  # 150 µV peak-to-peak
epochs.drop_bad(reject=reject_criteria)
print(epochs.drop_log_stats())
```

This is fast and transparent but uses a single global threshold, which may not suit all channels equally.

**Option B — Autoreject (recommended for publication-quality work):**

`autoreject` learns per-channel rejection thresholds from the data and can repair (interpolate) mildly bad epochs rather than discarding them entirely.

```python
from autoreject import AutoReject

ar = AutoReject(random_state=42, n_jobs=-1)
epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

# Inspect what was done
reject_log.plot_epochs(epochs)
```

After rejection, verify you still have enough trials per condition for reliable analysis (a common minimum is 30–40 trials per condition for ERP work, though more is better).

```python
print(epochs_clean)
# Check per-condition counts
for condition in event_id:
    print(f"  {condition}: {len(epochs_clean[condition])} trials")
```

### 3.4 Baseline Correction

Subtract the mean amplitude during a pre-stimulus baseline period from each epoch. This removes any residual DC offset that varies from trial to trial.

```python
epochs_clean.apply_baseline(baseline=(-0.2, 0.0))
```

The baseline period is typically the interval from `tmin` to stimulus onset (0.0). Ensure this period does not overlap with your previous trial's response window or any other confound.

---

## Phase 4 — RIFT Coherence Analysis

The core analysis in RIFT-EEG is to quantify how strongly each EEG channel's signal is phase-locked to the 60 Hz flicker. This is done by computing **magnitude-squared coherence** between the EEG and a synthetic reference sinusoid at the tagging frequency. The Fourier transform is the computational engine that makes this possible, but the quantity you are estimating is coherence, not raw spectral power.

### 4.1 Why Coherence and Not Just FFT Power?

You could simply compute the FFT of each epoch and look for a peak at 60 Hz. This works (and is worth doing as a sanity check — see 4.5), but coherence is the preferred metric for two reasons. First, coherence is sensitive to both frequency content **and** phase consistency across trials, while power only captures the former. A channel could have high 60 Hz power from noise or muscle activity without any phase relationship to the flicker. Coherence filters this out. Second, the RIFT literature has consistently shown that coherence yields a higher signal-to-noise ratio than power analysis for detecting tagging responses.

### 4.2 Generate the Reference Signal

The reference signal is a pure sinusoid at your tagging frequency (60 Hz), sampled at the same rate and with the same duration as your EEG epochs. This is the "artificial 60 Hz wave" you are computing coherence against.

```python
import numpy as np

sfreq = epochs_clean.info["sfreq"]  # e.g., 512 or 2048 Hz
tagging_freq = 60.0                  # Hz

# Build a reference signal matching the epoch time axis
times = epochs_clean.times           # e.g., -0.2 to 0.8 s
reference_signal = np.sin(2 * np.pi * tagging_freq * times)
```

**Phase alignment considerations:**

If you used a fixed flicker phase across all trials (the most common design), a single reference sinusoid works. If you randomized the flicker phase across trials, you need to adjust the reference phase on a trial-by-trial basis using your stimulus log. Some setups use a photodiode to record the actual luminance modulation; if you have a photodiode channel, you can use that directly as your reference instead of a synthetic sinusoid — this accounts for any timing jitter or dropped frames in the display.

```python
# If you have a photodiode channel recorded alongside EEG:
photodiode_data = raw.copy().pick_channels(["PHOTODIODE"]).get_data().squeeze()
# Use this as the reference instead of the synthetic sinusoid
```

### 4.3 Narrowband Filtering Around the Tagging Frequency

Before computing coherence, bandpass filter the EEG epochs (and the reference signal) in a narrow band around 60 Hz. This focuses the coherence estimate on the frequency of interest and removes broadband noise.

The RIFT literature typically uses a ±1.5 to ±2.5 Hz band around the tagging frequency.

```python
from scipy.signal import butter, filtfilt

def narrowband_filter(data, sfreq, center_freq, bandwidth=2.0, order=4):
    """Apply a narrow bandpass filter around a center frequency.
    
    Parameters
    ----------
    data : array, shape (n_trials, n_channels, n_times) or (n_times,)
    sfreq : float, sampling rate in Hz
    center_freq : float, center frequency in Hz
    bandwidth : float, half-bandwidth in Hz (filter is center ± bandwidth)
    order : int, filter order
    
    Returns
    -------
    filtered : array, same shape as data
    """
    low = (center_freq - bandwidth) / (sfreq / 2)
    high = (center_freq + bandwidth) / (sfreq / 2)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1)

# Filter EEG epochs: shape (n_trials, n_channels, n_times)
eeg_data = epochs_clean.get_data()  # (n_trials, n_channels, n_times)
eeg_filtered = narrowband_filter(eeg_data, sfreq, tagging_freq, bandwidth=2.0)

# Filter reference signal (same filter for consistency)
ref_filtered = narrowband_filter(reference_signal, sfreq, tagging_freq, bandwidth=2.0)
```

### 4.4 Compute Magnitude-Squared Coherence

Coherence measures the consistency of the phase relationship between two signals at a given frequency across trials. It ranges from 0 (no consistent relationship) to 1 (perfect phase-locking).

There are two main approaches:

**Approach A — Hilbert transform method (most common in RIFT literature):**

After narrowband filtering, apply the Hilbert transform to extract the instantaneous phase. Then compute coherence as the consistency of the phase difference between EEG and reference across trials.

```python
from scipy.signal import hilbert

def compute_rift_coherence_hilbert(eeg_filtered, ref_filtered, axis=0):
    """
    Compute magnitude-squared coherence between EEG and reference
    using the Hilbert transform approach.
    
    Parameters
    ----------
    eeg_filtered : array, shape (n_trials, n_channels, n_times)
        Narrowband-filtered EEG data.
    ref_filtered : array, shape (n_times,)
        Narrowband-filtered reference signal.
    axis : int
        Trial axis (default 0).
    
    Returns
    -------
    coherence : array, shape (n_channels, n_times)
        Time-resolved coherence per channel.
    """
    n_trials = eeg_filtered.shape[0]
    
    # Hilbert transform to get analytic signal
    eeg_analytic = hilbert(eeg_filtered, axis=-1)
    ref_analytic = hilbert(ref_filtered)
    
    # Extract instantaneous phase
    eeg_phase = np.angle(eeg_analytic)
    ref_phase = np.angle(ref_analytic)
    
    # Phase difference between EEG and reference
    phase_diff = eeg_phase - ref_phase[np.newaxis, np.newaxis, :]
    
    # Coherence = magnitude of mean complex exponential of phase differences
    # across trials (this is equivalent to the PLV between EEG and reference)
    coherence = np.abs(np.mean(np.exp(1j * phase_diff), axis=axis))
    
    return coherence  # shape: (n_channels, n_times)


coherence_map = compute_rift_coherence_hilbert(eeg_filtered, ref_filtered)
print(f"Coherence shape: {coherence_map.shape}")  # (n_channels, n_times)
```

**Approach B — Cross-spectral density method (via SciPy or MNE):**

This uses the standard Welch-based coherence estimator, treating the reference as one signal and each EEG channel as the other.

```python
from scipy.signal import coherence as scipy_coherence

def compute_rift_coherence_welch(epochs_data, reference, sfreq, freq_of_interest=60.0,
                                  nperseg=512, freq_tolerance=1.0):
    """
    Compute magnitude-squared coherence at a specific frequency using Welch's method.
    
    Parameters
    ----------
    epochs_data : array, shape (n_trials, n_channels, n_times)
    reference : array, shape (n_times,)
    sfreq : float
    freq_of_interest : float
    nperseg : int, segment length for Welch
    freq_tolerance : float, Hz window around freq_of_interest to average
    
    Returns
    -------
    coh_per_channel : array, shape (n_channels,)
        Mean coherence at freq_of_interest per channel.
    freqs : array
        Frequency axis.
    coh_spectra : array, shape (n_channels, n_freqs)
        Full coherence spectrum per channel.
    """
    n_trials, n_channels, n_times = epochs_data.shape
    
    # Concatenate trials for each channel (treat as continuous)
    coh_spectra_list = []
    for ch in range(n_channels):
        # Concatenate all trials for this channel
        eeg_concat = epochs_data[:, ch, :].ravel()
        ref_concat = np.tile(reference, n_trials)
        
        freqs, coh = scipy_coherence(eeg_concat, ref_concat, fs=sfreq,
                                      nperseg=nperseg, noverlap=nperseg // 2)
        coh_spectra_list.append(coh)
    
    coh_spectra = np.array(coh_spectra_list)
    
    # Extract coherence at frequency of interest
    freq_mask = (freqs >= freq_of_interest - freq_tolerance) & \
                (freqs <= freq_of_interest + freq_tolerance)
    coh_per_channel = coh_spectra[:, freq_mask].mean(axis=1)
    
    return coh_per_channel, freqs, coh_spectra


eeg_data = epochs_clean.get_data()
coh_values, freqs, coh_spectra = compute_rift_coherence_welch(
    eeg_data, reference_signal, sfreq, freq_of_interest=60.0
)
```

**Which approach to choose?**

The Hilbert method gives you **time-resolved** coherence, which is useful for examining when during the epoch the brain tracks the flicker (e.g., does coherence increase after stimulus onset and decay when the flicker stops?). The Welch method gives you a single coherence value per channel (averaged over the epoch), plus a full coherence **spectrum** that lets you verify the peak is specifically at 60 Hz and not broadband. In practice, you should do both: Welch for spectral verification and overall coherence values, Hilbert for time-resolved dynamics.

### 4.5 Sanity Checks: FFT Power / SNR at the Tagging Frequency

Before interpreting coherence results, verify that the tagging signal is present at all by computing the FFT-based signal-to-noise ratio (SNR) at 60 Hz.

```python
def compute_tagging_snr(epochs_data, sfreq, freq_of_interest=60.0,
                         neighbor_range=(2, 5)):
    """
    Compute SNR at the tagging frequency using FFT.
    SNR = power at freq_of_interest / mean power of neighboring frequencies.
    
    Parameters
    ----------
    epochs_data : array, shape (n_trials, n_channels, n_times)
    sfreq : float
    freq_of_interest : float
    neighbor_range : tuple (min_dist, max_dist) in Hz
        Neighboring frequencies used as noise estimate, excluding
        the immediate vicinity of the target frequency.
    
    Returns
    -------
    snr_db : array, shape (n_channels,)
        SNR in decibels per channel.
    freqs : array
    power_spectrum : array, shape (n_channels, n_freqs)
    """
    n_trials, n_channels, n_times = epochs_data.shape
    
    # Compute FFT per trial, then average power across trials
    fft_data = np.fft.rfft(epochs_data, axis=-1)
    power = np.mean(np.abs(fft_data) ** 2, axis=0)  # (n_channels, n_freqs)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    
    # Find the bin closest to freq_of_interest
    foi_idx = np.argmin(np.abs(freqs - freq_of_interest))
    
    # Neighbor bins (excluding immediate neighbors)
    min_dist, max_dist = neighbor_range
    neighbor_mask = (
        ((freqs >= freq_of_interest - max_dist) & (freqs <= freq_of_interest - min_dist)) |
        ((freqs >= freq_of_interest + min_dist) & (freqs <= freq_of_interest + max_dist))
    )
    
    signal_power = power[:, foi_idx]
    noise_power = power[:, neighbor_mask].mean(axis=1)
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db, freqs, power


eeg_data = epochs_clean.get_data()
snr_db, freqs, power = compute_tagging_snr(eeg_data, sfreq, freq_of_interest=60.0)

# Print top channels by SNR
ch_names = epochs_clean.ch_names
top_channels = np.argsort(snr_db)[::-1][:10]
print("Top 10 channels by 60 Hz SNR:")
for idx in top_channels:
    print(f"  {ch_names[idx]}: {snr_db[idx]:.2f} dB")
```

You should see the highest SNR over **occipital/parietal electrodes** (Oz, O1, O2, POz, PO3, PO4, etc.), which is the expected topography for a visual frequency-tagging response.

### 4.6 Visualization

**Coherence spectrum (verify the peak is at 60 Hz):**

```python
import matplotlib.pyplot as plt

# Pick the top 6 channels by coherence
top_6 = np.argsort(coh_values)[::-1][:6]
mean_coh_spectrum = coh_spectra[top_6, :].mean(axis=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(freqs, mean_coh_spectrum, "k-", linewidth=1.5)
ax.axvline(60, color="red", linestyle="--", alpha=0.7, label="60 Hz (tagging)")
ax.axvline(50, color="gray", linestyle=":", alpha=0.5, label="50 Hz (line noise)")
ax.set_xlim(40, 80)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Coherence")
ax.set_title("Coherence Spectrum — Top 6 Channels")
ax.legend()
plt.tight_layout()
plt.show()
```

**Topographic map of coherence at 60 Hz:**

```python
import mne

fig, ax = plt.subplots(figsize=(6, 5))
mne.viz.plot_topomap(
    coh_values,
    epochs_clean.info,
    axes=ax,
    show=False,
)
ax.set_title("Coherence at 60 Hz (tagging frequency)")
plt.show()
```

**Time-resolved coherence (from the Hilbert method):**

```python
# coherence_map shape: (n_channels, n_times)
# Average across top channels
top_ch_indices = np.argsort(coherence_map.mean(axis=1))[::-1][:6]
mean_time_coherence = coherence_map[top_ch_indices, :].mean(axis=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs_clean.times, mean_time_coherence, "b-", linewidth=1.5)
ax.axvline(0, color="k", linestyle="--", alpha=0.5, label="Stimulus onset")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Coherence")
ax.set_title("Time-Resolved 60 Hz Coherence — Top 6 Channels")
ax.legend()
plt.tight_layout()
plt.show()
```

You should see coherence rising after stimulus onset (when the flicker begins) and dropping after it ends, with the highest values over posterior electrodes.

### 4.7 Compare Coherence Across Conditions

If your experimental design has different conditions (e.g., attended vs. unattended, congruent vs. incongruent), extract the coherence values per condition and compare them.

```python
# Compute coherence separately for each condition
conditions = list(event_id.keys())
coherence_by_condition = {}

for cond in conditions:
    cond_data = epochs_clean[cond].get_data()
    coh_vals, _, _ = compute_rift_coherence_welch(
        cond_data, reference_signal, sfreq, freq_of_interest=60.0
    )
    coherence_by_condition[cond] = coh_vals

# Compare at specific channels or ROI (e.g., average of occipital channels)
occipital_chs = [i for i, ch in enumerate(ch_names) if ch in ["Oz", "O1", "O2", "POz"]]

for cond in conditions:
    mean_coh = coherence_by_condition[cond][occipital_chs].mean()
    print(f"  {cond}: mean occipital coherence = {mean_coh:.4f}")
```

### 4.8 Intermodulation Frequencies (If Tagging Multiple Stimuli)

If your design tags two stimuli at different frequencies (e.g., f1 = 60 Hz and f2 = 68 Hz), their neural representations may interact, producing power at intermodulation frequencies (f2 − f1 = 8 Hz, f2 + f1 = 128 Hz). The difference frequency is usually detectable; the sum frequency is harder to find in EEG. Check for intermodulation peaks in the power spectrum as evidence of neural integration of the two tagged inputs.

```python
# If you have two tagging frequencies:
f1, f2 = 60.0, 68.0
intermod_diff = f2 - f1   # 8 Hz — look for this in the power spectrum
intermod_sum = f2 + f1     # 128 Hz — harder to detect

# Compute SNR at the intermodulation frequency
snr_intermod, _, _ = compute_tagging_snr(
    eeg_data, sfreq, freq_of_interest=intermod_diff, neighbor_range=(2, 5)
)
```

---

## Phase 5 — Statistical Testing

### 5.1 Verify the Tagging Response Exists

Before comparing conditions, confirm that the tagging response is statistically above chance. Compare coherence during the RIFT-on window to a baseline period (pre-stimulus or a period when the flicker is off) using a non-parametric cluster-based permutation test across channels.

```python
from mne.stats import permutation_cluster_1samp_test
from mne.channels import find_ch_adjacency

adjacency, _ = find_ch_adjacency(epochs_clean.info, ch_type="eeg")

# coherence_tagging: (n_channels,) coherence during RIFT window
# coherence_baseline: (n_channels,) coherence during pre-stimulus baseline
# For a group-level test, stack subjects: (n_subjects, n_channels)
# Here, example for within-subject across-trial comparison:

diff = coherence_tagging - coherence_baseline  # shape: (n_subjects, n_channels)

T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    diff,
    n_permutations=10000,
    adjacency=adjacency,
    seed=42,
)

sig_clusters = [c for c, p in zip(clusters, cluster_p_values) if p < 0.05]
print(f"Found {len(sig_clusters)} significant cluster(s) confirming tagging response")
```

### 5.2 Compare Coherence Across Conditions

To test whether coherence differs between experimental conditions (e.g., attended vs. unattended):

**ROI-based approach (if you have a priori channel selection):**

```python
from scipy.stats import ttest_rel
import pingouin as pg

# Extract mean coherence over an occipital ROI per subject, per condition
# coh_attended, coh_unattended: arrays of shape (n_subjects,)
t_stat, p_val = ttest_rel(coh_attended, coh_unattended)
cohens_d = pg.compute_effsize(coh_attended, coh_unattended, eftype="cohen")
print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")
```

**Data-driven approach (no a priori channel selection):**

Use cluster-based permutation tests to identify which channels and/or time points show a significant condition difference, correcting for multiple comparisons.

```python
from mne.stats import spatio_temporal_cluster_test

# For time-resolved coherence: (n_subjects, n_timepoints, n_channels)
X = [coh_condition_A, coh_condition_B]  # two arrays, one per condition

T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_test(
    X,
    n_permutations=10000,
    adjacency=adjacency,
    n_jobs=-1,
    seed=42,
)

sig_clusters = [c for c, p in zip(clusters, cluster_p_values) if p < 0.05]
print(f"Found {len(sig_clusters)} significant cluster(s)")
```

For factorial designs, use repeated-measures ANOVA on the coherence values extracted from your ROI:

```python
import pingouin as pg

# df with columns: subject, condition, coherence
aov = pg.rm_anova(data=df, dv="coherence", within="condition", subject="subject")
print(aov)
```

### 5.3 Reporting

For each analysis, report the following: the test used and its assumptions, the test statistic with degrees of freedom and exact p-value, the effect size (Cohen's d, partial eta-squared, etc.), the number of trials per condition after rejection, the tagging frequency and reference signal used (synthetic sinusoid or photodiode), the narrowband filter parameters (bandwidth, order), the coherence computation method (Hilbert or Welch), and for cluster tests, the cluster extent (time range, channels involved), the cluster-level p-value, and the number of permutations. Include a coherence spectrum plot showing the 60 Hz peak, a topographic map of the coherence, and time-resolved coherence traces if applicable.

---

## Phase 6 — Documentation, Reproducibility, and Export

### 6.1 Save Preprocessed Data

```python
epochs_clean.save("sub-01-epo.fif", overwrite=True)
# Or for sharing outside MNE
epochs_clean.export("sub-01-epo.set", overwrite=True)  # EEGLAB format
```

### 6.2 Save Analysis Outputs

```python
import pandas as pd
import numpy as np

# Save coherence values per channel per condition
results_df = pd.DataFrame({
    "channel": epochs_clean.ch_names,
    "coherence_60Hz": coh_values,
    "snr_60Hz_dB": snr_db,
})
results_df.to_csv("sub-01-coherence-results.csv", index=False)

# Save coherence spectra for later plotting
np.savez("sub-01-coherence-spectra.npz",
         freqs=freqs, coh_spectra=coh_spectra, ch_names=epochs_clean.ch_names)
```

### 6.3 Generate a Processing Report

```python
report = mne.Report(title="Subject 01 — RIFT")
report.add_raw(raw, title="Raw", psd=True)
report.add_ica(ica=ica, title="ICA", inst=raw)
report.add_epochs(epochs_clean, title="Epochs")
report.save("sub-01-rift-report.html", overwrite=True)
```

### 6.4 Reproducibility Checklist

To ensure your pipeline is fully reproducible: pin your environment with a `requirements.txt` or `environment.yml` specifying exact versions of MNE, NumPy, SciPy, and all dependencies. Set random seeds for ICA, autoreject, permutation tests, and any other stochastic step. Log all parameters: filter cutoffs (especially high-pass, low-pass, and notch), ICA method and component count, rejection thresholds, epoch time windows, baseline period, reference scheme, narrowband filter bandwidth and order for coherence, and whether you used a synthetic sinusoid or photodiode as the reference. Organize raw data in BIDS format using `mne-bids`. Use git for version control and keep raw data read-only, writing all derivatives to a separate directory.

---

## Quick Reference: RIFT-EEG Processing Order

1. Load raw data
2. Inspect metadata (sampling rate ≥ 512 Hz, channels, duration, events)
3. Set channel types and montage
4. Visual inspection + PSD check (look for 60 Hz peak, 50 Hz line noise)
5. Mark bad channels
6. Filter: high-pass 0.1 Hz → low-pass ≥ 100 Hz → notch at 50/100/150 Hz (NOT 60 Hz)
7. Downsample to 512 Hz if needed (never below 512 for 60 Hz RIFT)
8. Re-reference (average reference for high-density)
9. ICA: fit → identify artifact components → apply
10. Interpolate bad channels
11. Epoch around RIFT stimulus events
12. Epoch rejection / repair
13. Baseline correction
14. Generate reference sinusoid at 60 Hz (or use photodiode channel)
15. Narrowband filter around 60 Hz (±2 Hz Butterworth)
16. Compute coherence (Hilbert for time-resolved; Welch for spectral)
17. Compute FFT-based SNR at 60 Hz as sanity check
18. Visualize: coherence spectrum, topography, time course
19. Compare coherence across conditions
20. Statistical testing (cluster-based permutation or ROI t-test)
21. Save outputs, generate reports, document everything