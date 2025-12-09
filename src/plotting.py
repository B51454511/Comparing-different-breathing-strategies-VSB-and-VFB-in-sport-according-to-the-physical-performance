# ==============================================================
# Plotting each processed signal + peaks
# ==============================================================

def plot_group_signals_with_peaks(group, env_list, peaks_list, fs, title):
    t = np.arange(group.shape[1]) / fs

    plt.figure(figsize=(14, 8))
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    for i, env in enumerate(env_list):
        offset = i * 1.2  # vertical spacing
        plt.plot(t, env + offset, alpha=0.8, zorder=1)

        peaks = peaks_list[i]
        if len(peaks) > 0:
            plt.plot(t[peaks], env[peaks] + offset, "ro", markersize=4, zorder=10)

        plt.text(0, offset, f"#{i+1}", fontsize=10)

    plt.grid(True)
    plt.show()


# ==============================================================
# Plot averages
# ==============================================================

def plot_avg_signal(group, fs, title):
    #avg = group.mean(axis=0)
    avg = np.nanmean(group, axis=0)
    t = np.arange(len(avg)) / fs

    plt.figure(figsize=(12,4))
    plt.plot(t, avg)
    plt.title(f"Average Signal — {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def plot_group_stats(slow_vals, fast_vals, metric_name):
    plt.figure(figsize=(8,5))
    plt.boxplot([slow_vals, fast_vals], labels=["Slow", "Fast"])
    plt.title(f"{metric_name} Comparison")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.show()
