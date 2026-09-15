import numpy as np
g = np.load("/home/moon/Desktop/ISSC_wCue/mvpa_results/cv_group_scores.npy")   # (n_subj, n_times)
t = np.load("/home/moon/Desktop/ISSC_wCue/mvpa_results/times.npy")
m = g.mean(0)
peak_t = t[np.argmax(m)]
print(f"peak AUC {m.max():.3f} at {peak_t*1000:.0f} ms")