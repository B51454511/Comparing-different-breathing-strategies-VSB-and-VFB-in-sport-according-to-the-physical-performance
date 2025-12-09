# Load YAMNet
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Get class names from YAMNet model
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
with tf.io.gfile.GFile(class_map_path) as csvfile:
    reader = csv.DictReader(csvfile)
    yamnet_class_names = [row['display_name'] for row in reader]
try:
    YAMNET_CLASS_ID = yamnet_class_names.index('Wind')
    #print(f"Found 'Breathing' class with index {YAMNET_CLASS_ID}.")
except ValueError:
    print("Warning: 'Breathing' class not found in YAMNet output. Defaulting to index 0.")
    YAMNET_CLASS_ID = 0

def detect_breaths_yamnet(y_clean, sr):
    """
    Improved breathing detector using YAMNet.
    - Automatically selects likely breathing-related classes from YAMNet's class names.
    - Uses a robust threshold (percentile-based) to avoid empty detections.
    - Computes frame duration from model outputs for correct mapping to original sample indices.
    Returns:
        peaks (np.array of sample indices at original sr),
        brpm (breaths per minute, or np.nan),
        env (float array same length as y_clean, 0..1 indicator-like envelope)
    """
    if len(y_clean) < 100:
        return np.array([], dtype=int), np.nan, np.zeros_like(y_clean)

    # ensure global model & class names are loaded
    global _YAMNET_MODEL, _YAMNET_CLASSNAMES
    if _YAMNET_MODEL is None or _YAMNET_CLASSNAMES is None:
        _load_yamnet()

    # Resample to 16k (YAMNet expects 16k)
    y_16k = librosa.resample(y_clean, orig_sr=sr, target_sr=16000).astype(np.float32)

    # Run YAMNet
    scores, embeddings, spectrogram = _YAMNET_MODEL(y_16k)
    scores = scores.numpy()  # shape: (num_frames, num_classes)
    if scores.ndim == 1:
        # ensure 2D
        scores = scores[np.newaxis, :]

    n_frames = scores.shape[0]
    total_duration = len(y_16k) / 16000.0
    if n_frames <= 0 or total_duration <= 0:
        return np.array([], dtype=int), np.nan, np.zeros_like(y_clean)

    # compute frame duration dynamically (seconds per frame)
    frame_dt = total_duration / n_frames  # robust instead of hard-coded 0.48s

    # choose breathing-related classes automatically
    candidate_indices = []
    lower_names = [cn.lower() for cn in _YAMNET_CLASSNAMES]
    keywords = ("breath", "sigh", "inhale", "exhal", "resp", "wind")
    for i, nm in enumerate(lower_names):
        if any(k in nm for k in keywords):
            candidate_indices.append(i)

    # fallback if nothing found: try some reasonable candidates
    if not candidate_indices:
        for fallback in ("breathing", "sigh", "speech", "wind"):
            try:
                candidate_indices.append(lower_names.index(fallback))
            except ValueError:
                pass

    # final fallback: just use column 0 (should rarely happen)
    if not candidate_indices:
        candidate_indices = [0]

    # sum probabilities across selected class columns to get a single "breath score"
    breath_prob = np.sum(scores[:, candidate_indices], axis=1)

    # make threshold robust:
    # use a percentile threshold + small margin, and ensure at least a minimal absolute threshold
    perc = max(60, 60)  # you can tweak percentile (60..80)
    thresh_percentile = np.percentile(breath_prob, perc)
    thresh = max(thresh_percentile, np.mean(breath_prob) + 0.25 * np.std(breath_prob), 0.01)

    active_frames = np.where(breath_prob > thresh)[0]
    if len(active_frames) == 0:
        # no breaths detected — return zeros-shaped envelope
        return np.array([], dtype=int), np.nan, np.zeros(len(y_clean))

    # Merge consecutive/nearby active frames into events (frame units)
    merged_frames = []
    if len(active_frames) > 0:
        current = active_frames[0]
        for f in active_frames[1:]:
            # if the gap is more than 1 frame, consider it a new event
            if f - current > 1:
                merged_frames.append(current)
            current = f
        merged_frames.append(current)

    # Convert merged frame indices to sample indices in ORIGINAL sr
    peaks = []
    for fi in merged_frames:
        t = fi * frame_dt              # seconds from start of resampled signal
        # map time back to original sample rate
        sample = int(round(t * sr))
        if 0 <= sample < len(y_clean):
            peaks.append(sample)
    peaks = np.array(peaks, dtype=int)

    # compute breaths-per-minute
    total_time_sec = len(y_clean) / float(sr)
    if total_time_sec > 0:
        # further merge peaks closer than 0.8s (avoid double-counting)
        if len(peaks) > 1:
            merged = []
            current = peaks[0]
            for p in peaks[1:]:
                if (p - current) / sr > 0.8:
                    merged.append(current)
                current = p
            merged.append(current)
            br_count = len(merged)
        else:
            br_count = len(peaks)
        brpm = br_count * 60.0 / total_time_sec
    else:
        brpm = np.nan

    # Build an upsampled "env" indicator for plotting
    # frame_env length = n_frames, 1 if breath_prob > thresh else 0
    frame_env = (breath_prob > thresh).astype(float)

    # Upsample frame_env to original sample rate by repeating each frame for frame_dt*sr samples.
    reps = max(1, int(np.round(frame_dt * sr)))
    env = np.repeat(frame_env, reps)

    # trim/pad env to len(y_clean)
    if len(env) < len(y_clean):
        env = np.pad(env, (0, len(y_clean) - len(env)), 'constant', constant_values=0)
    else:
        env = env[:len(y_clean)]

    # smooth env a bit to make plotting nicer (simple moving average)
    win = max(1, int(sr * 0.2))  # 200 ms smoothing
    env = np.convolve(env, np.ones(win)/win, mode='same')

    return peaks, brpm, env
