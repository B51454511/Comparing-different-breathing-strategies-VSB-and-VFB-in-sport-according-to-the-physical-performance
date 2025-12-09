# Comparing-different-breathing-strategies-VSB-and-VFB-in-sport-according-to-the-physical-performance
Project work for the Biomedical Signal Processing course of Pázmány Péter Catholic University Faculty of Information Technology and Bionics

This project analyzes physiological audio recordings collected after a 200-meter running task to extract heart rate and breathing rate under two controlled breathing conditions:

Fast breathing (~30 breaths/min)

Slow breathing (~10 breaths/min)

Using digital signal processing and a pretrained deep learning model (YAMNet), the project aims to evaluate how controlled breathing patterns influence recovery after exercise. The pipeline is implemented in Python and designed to run smoothly in a Google Colab notebook.

# Team Members

- Roha Riaz
- Zita Wágner
- Benedek Varga

# Objectives

Load and preprocess audio collected after running 200 meters.

Extract heart rate using:

Bandpass filtering (20–60 Hz)

Hilbert-envelope analysis

Peak detection

Extract breathing rate using:

Classic low-frequency filtering (0.1–2 Hz)

A machine-learning method (TensorFlow Hub’s YAMNet)

Compare slow vs. fast breathing groups using:

Breathing rate

Heart rate

Plotted signal morphologies

Group statistical analysis

Provide reproducible code and methodology for other researchers and students.

# Installation

Clone the repository:

git clone https://github.com/your-team/project.git
cd project


Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows


Install all required Python dependencies:

pip install -r requirements.txt

# Usage
Running the Notebook

Open the main notebook (e.g. in Google Colab or Jupyter):

notebooks/analysis.ipynb


The notebook performs:

Audio loading (.m4a input)

Noise reduction (spectral gating + smoothing)

Signal filtering and envelope extraction

Heartbeat peak detection

YAMNet-based breathing detection

Plotting (signals, peaks, averages, boxplots)

Slow vs. fast group comparisons

No additional configuration is needed.

# Methods
1. Audio Preprocessing

Every audio file is processed using:

Resampling to 2000 Hz

Conversion to mono

Amplitude normalization

Silence trimming (librosa.effects.trim)

Spectral gating noise reduction (STFT-based)

Median filtering (to suppress spike noise)

These steps ensure consistent, denoised signals for subsequent analysis.

2. Heart Rate Extraction

To estimate heart rate, the pipeline uses classical DSP:

Bandpass filter 20–60 Hz (range where heart vibrations appear in microphone recordings)

Hilbert transform → amplitude envelope

Local peak detection

RR interval estimation

Conversion to beats per minute (BPM)

This method is resilient to noise and works reliably on handheld microphone recordings.

3. Breathing Rate Extraction

Two complementary approaches are used:

A. Classical Filtering Method

Bandpass: 0.1–2.0 Hz

Hilbert envelope extraction

Thresholded peak detection

Breaths per minute (BrPM)

This captures macro breathing oscillations.

B. Machine Learning Method (YAMNet)

Using TensorFlow Hub’s pretrained YAMNet audio classifier:

Audio resampled to 16 kHz

Forward pass through YAMNet → class probabilities

Identification of frames where breathing-related classes activate

Upsampling to original signal length

Peak grouping → breaths per minute

This method adds robustness when noise or speech overlaps breathing sounds.


# References

Software and tool libraries used:

Librosa (audio analysis)

SciPy Signal (filtering, Hilbert transforms)

Matplotlib (visualization)

NumPy (numerical computing)

TensorFlow (ML framework)

TensorFlow Hub / YAMNet (pretrained audio classifier)
