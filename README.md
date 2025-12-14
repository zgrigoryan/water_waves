# water_waves

Numerical simulation of **2D shallow-basin surface waves** with a **variable depth field**, optional **damping** and **shoreline forcing**, plus **FFT-based frequency analysis** and **audification (sonification) to WAV**.

This project is a lightweight, educational pipeline intended for building intuition and producing reproducible outputs (plots + audio) in an idealized setting.

---

## What it does

- Builds a **2D rectangular basin** with **Dirichlet boundaries** (`η = 0` on all sides).
- Prescribes a **synthetic bathymetry** `h(x,y)` (deeper in the center), giving spatially varying wave speed  
  `c(x,y) = sqrt(g h(x,y))`.
- Runs **two cases automatically** in one execution:
  - **unimodal**: single standing-wave mode (clean spectrum)
  - **multimodal**: additional mode + localized bump + shoreline forcing (richer spectrum)
- Records the elevation at a **probe point** (off-center by default to avoid sitting at modal nodes).
- Produces:
  - midpoint/probe **time series plot**
  - **FFT frequency spectrum** plot
  - **audification WAV** file per case
  - **depth field plot**
  - several **wave field snapshots**
  - optional animation (may require extra tooling)

Outputs are saved under `./results_waves/`.

---

## Files

- `water_waves.py` — main simulation + plotting + audio export
- `results_waves/` — generated plots, snapshots, and WAV files

---

## Requirements

- Python 3.9+ recommended
- Packages: `numpy`, `matplotlib`, `scipy`

> **Optional:** For MP4 animations, install **ffmpeg**. Otherwise the script can save GIFs (slower) or skip animation.

---

## Run locally (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy
python water_waves.py
````

After it finishes, check the `results_waves/` folder for:

* `depth_field_hxy.png`
* `unimodal_fft_spectrum.png`, `multimodal_fft_spectrum.png`
* `unimodal_audification.wav`, `multimodal_audification.wav`
* snapshot images and comparison plots

---

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy matplotlib scipy
py water_waves.py
```

If PowerShell blocks activation scripts, run this once (Windows setting):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then re-run the activation line.

---

## Notes on probe location (important!)

If you measure exactly at the geometric midpoint, some modes can be **invisible** because the midpoint can lie on a **node** of the mode shape.
To avoid this, the code uses an off-center probe (example):

```python
ix = int(0.37 * (Nx - 1))
iy = int(0.61 * (Ny - 1))
```

This makes unimodal vs multimodal spectra more distinguishable.

---

## Optional: animations

* If **ffmpeg** is installed, MP4 saving is usually fast.
* Without ffmpeg, GIF export can be slow for many frames.

If you don’t need animation for the report, set:

```python
save_animation = False
```

Snapshots + plots are usually enough for a Results section.

---

## Next steps

* Add a **log-scale FFT plot** (or plot in dB) so smaller peaks aren’t visually crushed by the main peak.
* Record **multiple probes** (midpoint + off-center) and compare spectra.
* Replace synthetic depth with **real bathymetry** and drive forcing with **real wind data**.
* Compare boundary conditions (Dirichlet vs Neumann vs absorbing) and discuss spectral changes.
* Extend to nonlinear shallow-water equations for amplitude-dependent effects.
