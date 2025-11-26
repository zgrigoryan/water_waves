"""
Simple 1D standing-wave simulation for a shallow channel, plus FFT and
sonification of the midpoint signal. Uses an explicit second-order scheme with
fixed-end boundaries (zero displacement at both ends).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# ----------------------------
# 1. Parameters
# ----------------------------
L = 100.0          # domain length (m, arbitrary)
N = 201            # number of spatial points
dx = L / (N - 1)

g = 9.81           # gravity (m/s^2)
h = 5.0            # effective depth (m, choose something reasonable)
c = np.sqrt(g * h) # wave speed

T = 60.0           # total simulation time (s)
CFL = 0.9          # Courant number < 1 for stability
dt = CFL * dx / c
Nt = int(T / dt)

A = 0.1            # initial wave amplitude (m)

# ----------------------------
# 2. Grids and initial condition
# ----------------------------
x = np.linspace(0, L, N)
eta_prev = A * np.sin(np.pi * x / L)   # eta at t=0
eta = eta_prev.copy()                  # will become eta at t=dt

sigma = (c * dt / dx)**2  # stability factor used in the stencil

# First time step (zero initial velocity assumed)
eta[1:-1] = eta_prev[1:-1] + 0.5 * sigma * (
    eta_prev[2:] - 2 * eta_prev[1:-1] + eta_prev[:-2]
)
eta[0] = 0.0              # fixed boundary (wall) on the left
eta[-1] = 0.0             # fixed boundary (wall) on the right

# ----------------------------
# 3. Time stepping
# ----------------------------
mid_index = N // 2
eta_mid = [eta_prev[mid_index]]  # record midpoint time series
eta_mid.append(eta[mid_index])

for n in range(1, Nt):
    eta_next = np.zeros_like(eta)
    eta_next[1:-1] = (    # interior update: 2nd-order explicit wave equation
        2 * eta[1:-1]
        - eta_prev[1:-1]
        + sigma * (eta[2:] - 2 * eta[1:-1] + eta[:-2])
    )
    # Boundary conditions
    eta_next[0] = 0.0
    eta_next[-1] = 0.0

    # shift time levels
    eta_prev, eta = eta, eta_next

    eta_mid.append(eta[mid_index])

eta_mid = np.array(eta_mid)
time = np.arange(len(eta_mid)) * dt

# ----------------------------
# 4. Fourier transform
# ----------------------------
fft_vals = np.fft.rfft(eta_mid)          # one-sided spectrum (real input)
freqs = np.fft.rfftfreq(len(eta_mid), dt)

plt.figure()
plt.plot(freqs, np.abs(fft_vals))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency spectrum of surface elevation at midpoint")
plt.xlim(0, 2.0)  # zoom in on low frequencies if needed
plt.grid(True)
plt.show()

# ----------------------------
# 5. Sonification
# ----------------------------
# Normalize
signal = eta_mid / np.max(np.abs(eta_mid))

# Speed up the signal to make sub-Hz content audible
speed_factor = 50
new_length = len(signal) // speed_factor
signal_fast = np.interp(
    np.linspace(0, len(signal)-1, new_length),
    np.arange(len(signal)),
    signal
)

# Choose sample rate for audio
fs = 44100  # standard audio sample rate

# Simple repetition or stretching to a nicer length if needed
duration_seconds = 3
target_len = duration_seconds * fs
signal_resampled = np.interp(
    np.linspace(0, len(signal_fast)-1, target_len),
    np.arange(len(signal_fast)),
    signal_fast
)

# Ensure float32 in [-1,1]
audio = signal_resampled.astype(np.float32)

write("lake_wave_sonification.wav", fs, audio)
print("Saved audio file: lake_wave_sonification.wav")

# TODOs for richer analysis/realism:
# - Add light damping to reduce boundary reflections; compare spectra with/without.
# - Apply a window before the FFT or extract individual modes via peak picking.
# - Track energy over time to sanity-check numerical stability.
# - Extend to 2D or use a dispersive model (e.g., Boussinesq) for shallow water.
