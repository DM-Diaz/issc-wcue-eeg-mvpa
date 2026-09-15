import numpy as np
from mvpa_io import load_subject

ep, _ = load_subject(1)
sf = ep.info["sfreq"]
n_samples = len(ep.times)
n_fft = min(256, n_samples)          # cannot exceed epoch length

psd = ep.compute_psd(method="welch", fmax=sf / 2, n_fft=n_fft, verbose=False)
f = psd.freqs
p = psd.get_data().mean(axis=(0, 1))     # average over epochs and channels
p = p / p.max()

print(f"sfreq = {sf}, info lowpass = {ep.info['lowpass']}, "
      f"epoch samples = {n_samples}, n_fft = {n_fft}")
for hz in (10, 20, 30, 40, 50, 62.5, 80, 100, 120):
    if hz > f.max():
        continue
    i = np.argmin(np.abs(f - hz))
    print(f"  {f[i]:>6.1f} Hz : {p[i]:.2e}  ({10*np.log10(p[i]):+6.1f} dB)")

frac = p[f > 62.5].sum() / p.sum()
print(f"\npower above 62.5 Hz as fraction of total: {frac:.5f}")