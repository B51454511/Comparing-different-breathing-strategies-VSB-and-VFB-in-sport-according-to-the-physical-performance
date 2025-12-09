# ==============================================================
# Heartbeat Detection
# ==============================================================

def estimate_heart_rate(env, fs, min_hr=40, max_hr=180):
    min_dist = int(fs * 60 / max_hr)

    peaks, _ = find_peaks(env,
                          distance=min_dist,
                          prominence=np.mean(env) * 0.5)

    if len(peaks) < 2:
        return peaks, np.nan

    intervals = np.diff(peaks) / fs
    valid = intervals[(intervals > 0.3) & (intervals < 2.0)]

    if len(valid) == 0:
        return peaks, np.nan

    bpm = 60.0 / np.mean(valid)
    return peaks, bpm

# ==============================================================
# Breathing Detection (0.1–2 Hz, adaptive)
# ==============================================================

def detect_breaths(y, fs):
    """Breathing detection : low frequency extraction + envelope."""
    low = 0.1
    high = 2.0
    filtered = bandpass_filter(y, low, high, fs)

    env = envelope_hilbert(filtered, fs, smoothing=1.0)

    threshold = np.mean(env) + 0.4 * np.std(env)

    peaks, _ = find_peaks(env, distance=fs*2.0)  # <= 30 breaths/min
    peaks = [p for p in peaks if env[p] > threshold]

    brpm = len(peaks) * 60 / (len(y)/fs)
    return np.array(peaks), brpm, env

# ==============================================================
# YAMNet integration (needs resample to 16k)
# ==============================================================
_YAMNET_MODEL = None
_YAMNET_CLASSNAMES = None

def _load_yamnet():
    global _YAMNET_MODEL, _YAMNET_CLASSNAMES
    if _YAMNET_MODEL is None:
        # Load model
        _YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1")
        # load class map bundled with model
        class_map_path = _YAMNET_MODEL.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path) as csvfile:
            reader = csv.DictReader(csvfile)
            _YAMNET_CLASSNAMES = [row['display_name'] for row in reader]
    return _YAMNET_MODEL, _YAMNET_CLASSNAMES
