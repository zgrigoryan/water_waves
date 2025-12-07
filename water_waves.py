"""
Simple 2D standing-wave simulation for a shallow rectangular basin, plus FFT and
sonification of the midpoint signal. Uses an explicit second-order scheme with
fixed-edge boundaries (zero displacement along all sides).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# ----------------------------
# 1. Parameters
# ----------------------------
Lx = 10.0         # domain length in x (m, arbitrary)
Ly = 10.0         # domain length in y (m, arbitrary)
Nx = 201           # number of points in x
Ny = 201           # number of points in y

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

g = 9.81           # gravity (m/s^2)
h = 5.0            # effective depth (m)
c = np.sqrt(g * h) # wave speed c = sqrt(g h)

T = 60.0           # total simulation time (s)

# CFL condition for 2D: c*dt/dx <= 1/sqrt(2) (assuming dx = dy)
# Pick a Courant number safely below the limit.
CFL = 0.5
dt = CFL * dx / c
Nt = int(T / dt)

A = 0.1            # initial wave amplitude (m)

print(f"dx = {dx:.4f}, dt = {dt:.4f}, Nt = {Nt}, CFL = {c*dt/dx:.3f}")

# ----------------------------
# 2. Grids and initial condition
# ----------------------------
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing="ij")  # X[i,j] = x_i, Y[i,j] = y_j

# Fundamental standing mode in 2D: sin(pi x / Lx) * sin(pi y / Ly)
A1 = 0.1
A2 = 0.05

eta_prev = (
    A1 * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly) +
    A2 * np.sin(2 * np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
)
eta = eta_prev.copy()  # will become eta at t = dt

# sigma = (c * dt / dx)^2 for the 2D Laplacian with dx = dy
sigma = (c * dt / dx) ** 2

# First time step (zero initial velocity assumed)
# eta^1 = eta^0 + 0.5 * sigma * Laplacian(eta^0)
eta[1:-1, 1:-1] = eta_prev[1:-1, 1:-1] + 0.5 * sigma * (
    eta_prev[2:, 1:-1] + eta_prev[:-2, 1:-1] +
    eta_prev[1:-1, 2:] + eta_prev[1:-1, :-2] -
    4.0 * eta_prev[1:-1, 1:-1]
)

# Fixed-edge (Dirichlet) boundaries: eta = 0 on all sides
eta[0, :] = 0.0
eta[-1, :] = 0.0
eta[:, 0] = 0.0
eta[:, -1] = 0.0

# ----------------------------
# 3. Time stepping
# ----------------------------
ix_mid = Nx // 2
iy_mid = Ny // 2

eta_mid = [eta_prev[ix_mid, iy_mid]]  # record midpoint time series at t = 0
eta_mid.append(eta[ix_mid, iy_mid])   # at t = dt

for n in range(1, Nt):
    eta_next = np.zeros_like(eta)

    # Interior update: 2D wave equation with 5-point Laplacian
    eta_next[1:-1, 1:-1] = (
        2.0 * eta[1:-1, 1:-1]
        - eta_prev[1:-1, 1:-1]
        + sigma * (
            eta[2:, 1:-1] + eta[:-2, 1:-1] +
            eta[1:-1, 2:] + eta[1:-1, :-2] -
            4.0 * eta[1:-1, 1:-1]
        )
    )

    # Fixed boundaries
    eta_next[0, :] = 0.0
    eta_next[-1, :] = 0.0
    eta_next[:, 0] = 0.0
    eta_next[:, -1] = 0.0

    # Shift time levels
    eta_prev, eta = eta, eta_next

    # Record midpoint signal
    eta_mid.append(eta[ix_mid, iy_mid])

eta_mid = np.array(eta_mid)
time = np.arange(len(eta_mid)) * dt

# ----------------------------
# 4. Fourier transform (midpoint signal)
# ----------------------------
fft_vals = np.fft.rfft(eta_mid)
freqs = np.fft.rfftfreq(len(eta_mid), dt)

plt.figure()
plt.plot(freqs, np.abs(fft_vals))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency spectrum of surface elevation at basin midpoint")
plt.xlim(0, 2.0)  # zoom on low frequencies
plt.grid(True)
plt.show()

# ----------------------------
# 5. Sonification 
# ----------------------------

# Normalize
signal = eta_mid / np.max(np.abs(eta_mid))

# Speed up the signal to push low frequencies into the audible range
speed_factor = 50
new_length = len(signal) // speed_factor
signal_fast = np.interp(
    np.linspace(0, len(signal) - 1, new_length),
    np.arange(len(signal)),
    signal
)

# Choose audio sampling rate
fs = 44100  # standard audio sample rate

# Stretch or compress to a fixed duration for nicer playback
duration_seconds = 3
target_len = duration_seconds * fs
signal_resampled = np.interp(
    np.linspace(0, len(signal_fast) - 1, target_len),
    np.arange(len(signal_fast)),
    signal_fast
)

# Ensure float32 in [-1, 1]
audio = signal_resampled.astype(np.float32)

write("lake_wave_sonification_2d.wav", fs, audio)
print("Saved audio file: lake_wave_sonification_2d.wav")