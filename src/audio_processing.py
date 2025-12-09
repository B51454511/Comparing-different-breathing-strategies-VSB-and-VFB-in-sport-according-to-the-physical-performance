# ==============================================================
# Utility functions
# ==============================================================

def load_and_trim_audio(path, target_sr=2000):
    """Load audio, convert to mono, normalize, remove silence."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)

    # Normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Auto-trim silence
    y_trim, _ = librosa.effects.trim(y, top_db=25)

    return y_trim, sr


def pad_signals_to_same_length(signal_list):
    max_len = max(len(s) for s in signal_list)
    mat = np.full((len(signal_list), max_len), np.nan)

    for i, sig in enumerate(signal_list):
        mat[i, :len(sig)] = sig

    return mat

# ==============================================================
# Noise reduction
# ==============================================================

def spectral_gate_noise_reduction(y, sr):
    """Simple spectral gating noise reduction using librosa."""
    S = np.abs(librosa.stft(y))
    noise_profile = np.mean(S[:, :10], axis=1, keepdims=True)
    S_denoised = np.maximum(S - noise_profile, 0)
    y_clean = librosa.istft(S_denoised * np.exp(1j * np.angle(librosa.stft(y))))
    return y_clean


def smooth_signal(y, kernel=5):
    """Use a small median filter to remove spike noise."""
    return medfilt(y, kernel)
  
# ==============================================================
# Filtering and envelope
# ==============================================================

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs * 0.5
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def bandpass_filter(y, low, high, fs, order=4):
    b, a = butter_bandpass(low, high, fs, order)
    return filtfilt(b, a, y)

def envelope_hilbert(y, fs, smoothing=0.05):
    analytic = hilbert(y)
    env = np.abs(analytic)

    win = int(fs * smoothing)
    win = max(1, win)
    smooth_env = np.convolve(env, np.ones(win)/win, mode='same')
    return smooth_env

# ==============================================================
# Group Loading
# ==============================================================

def load_group(folder, prefix):
    """Load all audio files whose name starts with prefix."""
    signals = []
    sample_rates = []

    for fname in os.listdir(folder):
        if fname.lower().startswith(prefix.lower()):
            path = os.path.join(folder, fname)
            y, sr = load_and_trim_audio(path)
            y = spectral_gate_noise_reduction(y, sr)
            y = smooth_signal(y)
            signals.append(y)
            sample_rates.append(sr)

    if not signals:
        return None, None

    mat = pad_signals_to_same_length(signals)
    return mat, sample_rates[0]


# ==============================================================
# Batch Processing - collect peaks & envelopes
# ==============================================================

def process_group(group, fs):
    """
    Process a group of signals.

    Heart rate: envelope + peak detection
    Breathing rate: YAMNet only
    Returns:
        heart_rates, breath_rates,
        heart_envs, heart_peaks_list,
        breath_envs, breath_peaks_list
    """
    heart_rates = []
    breath_rates = []
    heart_envs = []
    heart_peaks_list = []
    breath_envs = []
    breath_peaks_list = []

    for row in group:
        # Remove NaN padding
        valid = row[~np.isnan(row)]
        if len(valid) < 10:
            # Skip very short signals
            heart_rates.append(np.nan)
            breath_rates.append(np.nan)

            heart_envs.append(np.zeros(len(row)))
            breath_envs.append(np.zeros(len(row)))

            heart_peaks_list.append(np.array([]))
            breath_peaks_list.append(np.array([]))
            continue

        # -----------------------------
        # Heart rate detection
        # -----------------------------
        h_filtered = bandpass_filter(valid, 20, 60, fs)
        h_env = envelope_hilbert(h_filtered, fs)
        h_peaks, bpm = estimate_heart_rate(h_env, fs)

        # Pad heart envelope
        h_env_full = np.full(len(row), np.nan)
        h_env_full[:len(h_env)] = h_env

        heart_rates.append(bpm)
        heart_envs.append(h_env_full)
        heart_peaks_list.append(h_peaks)

        # -----------------------------
        # Breathing detection — YAMNet
        # -----------------------------
        try:
            b_peaks, brpm, b_env = detect_breaths_yamnet(valid, fs)
        except Exception:
            b_peaks = np.array([], dtype=int)
            brpm = np.nan
            b_env = np.zeros(len(valid))

        # Pad breathing envelope
        b_env_full = np.full(len(row), np.nan)
        b_env_full[:len(b_env)] = b_env

        breath_rates.append(brpm)
        breath_envs.append(b_env_full)
        breath_peaks_list.append(b_peaks)

    return (
        np.array(heart_rates),
        np.array(breath_rates),
        heart_envs, heart_peaks_list,
        breath_envs, breath_peaks_list
    )
