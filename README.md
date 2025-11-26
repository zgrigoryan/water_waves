# water_waves

Simple 1D standing-wave simulation for a shallow channel, followed by an FFT of the midpoint signal and a quick-and-dirty sonification to WAV.

## What it does
- Sets up a 1D domain with fixed ends, initializes a sine-shaped surface displacement, and advances the linear wave equation with an explicit second-order scheme.
- Records the midpoint elevation over time, computes a one-sided FFT, and plots the frequency spectrum.
- Speeds up the sub-Hz signal, resamples it to ~3 seconds at 44.1 kHz, and writes `lake_wave_sonification.wav`.

## Run locally (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy
python water_waves.py
```
You should see a Matplotlib spectrum plot and a generated `lake_wave_sonification.wav` in the project folder.

## Next steps
- Add damping or windowing before the FFT to reduce boundary reflections and get cleaner spectra.
- Track energy over time to sanity-check stability.
- Explore a dispersive or nonlinear shallow-water model for more realistic behavior.
- Port to C++ (e.g., Eigen + FFTW) if you want faster or larger grids; reuse the same scheme and boundary conditions.
