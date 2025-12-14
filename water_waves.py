"""
2D shallow-basin wave simulation with:
- variable depth h(x,y) -> variable wave speed c(x,y)
- small damping
- unimodal and multimodal initial conditions 
- optional time-periodic "wind forcing" near one shore (used in multimodal case)
- FFT of midpoint signal
- audification (sonification) of midpoint signal
- saved plots + snapshots (+ optional animation)

Dirichlet (fixed) boundaries: eta = 0 on all sides.

Outputs go into: ./results_waves/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import write


# ----------------------------
# Utility
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_savefig(path: str, dpi: int = 300) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def try_save_animation(ani, out_path_mp4: str, out_path_gif: str, fps: int = 20) -> None:
    """
    Tries to save MP4 (ffmpeg) first, then GIF (pillow). If neither works, skips.
    """
    try:
        ani.save(out_path_mp4, fps=fps, dpi=200)
        print(f"[OK] Saved animation: {out_path_mp4}")
        return
    except Exception as e_mp4:
        print(f"[WARN] MP4 save failed ({e_mp4}). Trying GIF...")

    try:
        ani.save(out_path_gif, writer="pillow", fps=fps)
        print(f"[OK] Saved animation: {out_path_gif}")
    except Exception as e_gif:
        print(f"[WARN] GIF save failed ({e_gif}). Animation not saved.")


def audify_signal(signal: np.ndarray, fs: int, speed_factor: int, duration_seconds: int) -> np.ndarray:
    """
    Time-compress + resample signal into fixed audio duration.
    """
    # Normalize safely
    maxv = np.max(np.abs(signal))
    if maxv < 1e-12:
        return np.zeros(fs * duration_seconds, dtype=np.float32)

    s = signal / maxv

    # Speed up (compress time axis)
    new_length = max(2, len(s) // max(1, speed_factor))
    s_fast = np.interp(
        np.linspace(0, len(s) - 1, new_length),
        np.arange(len(s)),
        s
    )

    # Resample to fixed duration
    target_len = int(duration_seconds * fs)
    s_resamp = np.interp(
        np.linspace(0, len(s_fast) - 1, target_len),
        np.arange(len(s_fast)),
        s_fast
    )

    return s_resamp.astype(np.float32)


# ----------------------------
# Core simulation
# ----------------------------
def build_depth_field(X, Y, Lx, Ly, g, h0, h1, sigma_h):
    # Depth: deeper in center (Gaussian hump)
    h = h0 + h1 * np.exp(-((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2) / (2 * sigma_h ** 2))
    c_grid = np.sqrt(g * h)
    return h, c_grid


def initial_condition(X, Y, Lx, Ly, A1, A2, use_bump, bump_amp, bump_sigma):
    # Fundamental mode + optional second mode
    eta0 = (
        A1 * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
        + A2 * np.sin(3 * np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
    )

    if use_bump:
        bump = bump_amp * np.exp(-((X - Lx / 3) ** 2 + (Y - 2 * Ly / 3) ** 2) / (2 * bump_sigma ** 2))
        eta0 += bump

    return eta0


def run_case(case_name: str, cfg: dict, out_dir: str, h: np.ndarray, c_grid: np.ndarray):
    """
    Runs one simulation case and saves all per-case outputs.
    Returns dict with arrays used for combined plots.
    """
    # Unpack
    Lx, Ly = cfg["Lx"], cfg["Ly"]
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    g = cfg["g"]
    T = cfg["T"]
    gamma = cfg["gamma"]
    CFL = cfg["CFL"]
    save_every = cfg["save_every"]

    use_forcing = cfg["use_forcing"]
    F0 = cfg["F0"]
    f_forcing = cfg["f_forcing"]

    A1, A2 = cfg["A1"], cfg["A2"]
    use_bump = cfg["use_bump"]
    bump_amp = cfg["bump_amp"]
    bump_sigma = cfg["bump_sigma"]

    fs = cfg["fs"]
    speed_factor = cfg["speed_factor"]
    audio_duration_seconds = cfg["audio_duration_seconds"]

    # Grid spacing
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    if abs(dx - dy) > 1e-12:
        print("[WARN] dx != dy; the scheme assumes dx ~ dy. Using dx in sigma.")

    # CFL timestep based on max wave speed
    c_max = float(np.max(c_grid))
    dt = CFL * dx / (c_max * np.sqrt(2))
    Nt = int(T / dt)

    sigma_grid = (c_grid * dt / dx) ** 2

    print(f"\n[{case_name}] dx={dx:.4f}, dt={dt:.5f}, Nt={Nt}, CFL_max={c_max*dt/dx:.3f}, "
          f"h in [{h.min():.2f},{h.max():.2f}], forcing={use_forcing}, bump={use_bump}, A2={A2}")

    # Coordinate arrays (matching shape)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Initial condition: eta at t=0
    eta_prev = initial_condition(X, Y, Lx, Ly, A1, A2, use_bump, bump_amp, bump_sigma)

    # eta at t=dt (first step with zero initial velocity)
    eta = eta_prev.copy()
    lap_prev = (
        eta_prev[2:, 1:-1] + eta_prev[:-2, 1:-1] +
        eta_prev[1:-1, 2:] + eta_prev[1:-1, :-2] -
        4.0 * eta_prev[1:-1, 1:-1]
    )
    eta[1:-1, 1:-1] = eta_prev[1:-1, 1:-1] + 0.5 * sigma_grid[1:-1, 1:-1] * lap_prev

    # Dirichlet boundaries
    eta[0, :] = 0.0
    eta[-1, :] = 0.0
    eta[:, 0] = 0.0
    eta[:, -1] = 0.0
    eta_prev[0, :] = 0.0
    eta_prev[-1, :] = 0.0
    eta_prev[:, 0] = 0.0
    eta_prev[:, -1] = 0.0

    # Forcing mask
    forcing_mask = np.zeros_like(X)
    if use_forcing:
        forcing_mask[X < 0.2 * Lx] = 1.0

    # Probe signal
    ix_mid = int(0.37 * (Nx - 1))
    iy_mid = int(0.61 * (Ny - 1))

    eta_mid = [eta_prev[ix_mid, iy_mid], eta[ix_mid, iy_mid]]


    frames = [eta_prev.copy()]
    frame_times = [0.0]

    # Time stepping
    for n in range(1, Nt):
        t_n = n * dt
        eta_next = np.zeros_like(eta)

        lap_eta = (
            eta[2:, 1:-1] + eta[:-2, 1:-1] +
            eta[1:-1, 2:] + eta[1:-1, :-2] -
            4.0 * eta[1:-1, 1:-1]
        )

        if use_forcing:
            F = F0 * np.sin(2 * np.pi * f_forcing * t_n) * forcing_mask
            F_interior = F[1:-1, 1:-1]
        else:
            F_interior = 0.0

        eta_next[1:-1, 1:-1] = (
            (2.0 - gamma * dt) * eta[1:-1, 1:-1]
            - (1.0 - gamma * dt) * eta_prev[1:-1, 1:-1]
            + sigma_grid[1:-1, 1:-1] * lap_eta
            + (dt ** 2) * F_interior
        )

        # Dirichlet boundaries
        eta_next[0, :] = 0.0
        eta_next[-1, :] = 0.0
        eta_next[:, 0] = 0.0
        eta_next[:, -1] = 0.0

        eta_prev, eta = eta, eta_next

        eta_mid.append(eta[ix_mid, iy_mid])

        if n % save_every == 0:
            frames.append(eta.copy())
            frame_times.append(t_n)

    eta_mid = np.array(eta_mid)
    t_arr = np.arange(len(eta_mid)) * dt

    # ----------------------------
    # Save plots: time series + FFT
    # ----------------------------
    # Time series
    plt.figure()
    plt.plot(t_arr, eta_mid)
    plt.xlabel("Time (s)")
    plt.ylabel("Midpoint elevation (m)")
    plt.title(f"Midpoint surface elevation: {case_name}")
    plt.grid(True)
    safe_savefig(os.path.join(out_dir, f"{case_name}_midpoint_timeseries.png"))

    # FFT
    fft_vals = np.fft.rfft(eta_mid)
    freqs = np.fft.rfftfreq(len(eta_mid), dt)

    plt.figure()
    plt.plot(freqs, np.abs(fft_vals))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(f"Frequency spectrum at basin midpoint: {case_name}")
    plt.xlim(0, cfg["fft_xlim_hz"])
    plt.grid(True)
    safe_savefig(os.path.join(out_dir, f"{case_name}_fft_spectrum.png"))

    # ----------------------------
    # Sonification (WAV)
    # ----------------------------
    audio = audify_signal(eta_mid, fs=fs, speed_factor=speed_factor, duration_seconds=audio_duration_seconds)
    wav_path = os.path.join(out_dir, f"{case_name}_audification.wav")
    write(wav_path, fs, audio)
    print(f"[OK] Saved audio: {wav_path}")

    # ----------------------------
    # Wave field snapshots
    # ----------------------------
    frames_arr = np.array(frames)
    frame_times_arr = np.array(frame_times)
    vmin = float(frames_arr.min())
    vmax = float(frames_arr.max())

    snap_indices = np.linspace(0, len(frames_arr) - 1, cfg["n_snapshots"], dtype=int)
    for k in snap_indices:
        plt.figure()
        im = plt.imshow(
            frames_arr[k],
            extent=[0, Lx, 0, Ly],
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(im, label="Surface elevation (m)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"{case_name}: Surface elevation, t = {frame_times_arr[k]:.2f} s")
        safe_savefig(os.path.join(out_dir, f"{case_name}_snapshot_t{frame_times_arr[k]:.2f}.png"))

    # ----------------------------
    # Optional animation
    # ----------------------------
    if cfg["save_animation"]:
        fig, ax = plt.subplots()
        im = ax.imshow(
            frames_arr[0],
            extent=[0, Lx, 0, Ly],
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(im, ax=ax, label="Surface elevation (m)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{case_name}: Surface elevation, t = {frame_times_arr[0]:.2f} s")

        def update(i):
            im.set_data(frames_arr[i])
            ax.set_title(f"{case_name}: Surface elevation, t = {frame_times_arr[i]:.2f} s")
            return [im]

        ani = FuncAnimation(fig, update, frames=len(frames_arr), interval=50, blit=True)

        mp4_path = os.path.join(out_dir, f"{case_name}_animation.mp4")
        gif_path = os.path.join(out_dir, f"{case_name}_animation.gif")
        try_save_animation(ani, mp4_path, gif_path, fps=cfg["anim_fps"])
        plt.close(fig)

    return {
        "case_name": case_name,
        "t": t_arr,
        "eta_mid": eta_mid,
        "freqs": freqs,
        "fft_mag": np.abs(fft_vals),
        "dt": dt
    }


# ----------------------------
# Main
# ----------------------------
def main():
    out_dir = "results_waves"
    ensure_dir(out_dir)

    # ----------------------------
    # Global parameters
    # ----------------------------
    Lx = 10.0
    Ly = 10.0
    Nx = 201
    Ny = 201

    g = 9.81
    T = 60.0
    gamma = 0.05
    CFL = 0.5

    # Depth field parameters
    h0 = 4.0
    h1 = 3.0
    sigma_h = Lx / 4.0

    # Sonification settings
    fs = 44100
    speed_factor = 80
    audio_duration_seconds = 3

    # Output settings
    save_every = 10
    fft_xlim_hz = 2.0
    n_snapshots = 4

    # Optional animation saving (needs ffmpeg or pillow available)
    save_animation = True
    anim_fps = 20

    # Build grid and depth once (shared across both cases)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    h, c_grid = build_depth_field(X, Y, Lx, Ly, g, h0, h1, sigma_h)

    # Save depth field plot (shared)
    plt.figure()
    depth_plot = plt.contourf(X, Y, h, levels=40)
    plt.colorbar(depth_plot, label="Depth h(x, y) (m)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Variable Depth Field h(x, y)")
    plt.gca().set_aspect("equal", adjustable="box")
    safe_savefig(os.path.join(out_dir, "depth_field_hxy.png"))
    print(f"[OK] Saved depth field: {os.path.join(out_dir, 'depth_field_hxy.png')}")

    # ----------------------------
    # Case definitions
    # ----------------------------
    base_cfg = dict(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, g=g,
        T=T, gamma=gamma, CFL=CFL,
        save_every=save_every,
        fs=fs, speed_factor=speed_factor, audio_duration_seconds=audio_duration_seconds,
        fft_xlim_hz=fft_xlim_hz,
        n_snapshots=n_snapshots,
        save_animation=save_animation,
        anim_fps=anim_fps,
    )

    # Unimodal
    cfg_uni = dict(base_cfg)
    cfg_uni.update(
        A1=0.10,
        A2=0.00,          # unimodal means A2=0
        use_bump=False,
        bump_amp=0.00,
        bump_sigma=Lx / 10.0,
        use_forcing=False,
        F0=0.0,
        f_forcing=0.0
    )

    # Multimodal
    cfg_multi = dict(base_cfg)
    cfg_multi.update(
        A1=0.10,
        A2=0.06,          # FIXED: multimodal must have nonzero A2
        use_bump=True,
        bump_amp=0.05,
        bump_sigma=Lx / 10.0,
        use_forcing=True,
        F0=0.02,
        f_forcing=0.2
    )

    # ----------------------------
    # Run both cases
    # ----------------------------
    res_uni = run_case("unimodal", cfg_uni, out_dir, h, c_grid)
    res_multi = run_case("multimodal", cfg_multi, out_dir, h, c_grid)

    # ----------------------------
    # Combined comparison plots
    # ----------------------------
    # Time series comparison 
    nmin = min(len(res_uni["t"]), len(res_multi["t"]))
    plt.figure()
    plt.plot(res_uni["t"][:nmin], res_uni["eta_mid"][:nmin], label="unimodal")
    plt.plot(res_multi["t"][:nmin], res_multi["eta_mid"][:nmin], label="multimodal")
    plt.xlabel("Time (s)")
    plt.ylabel("Midpoint elevation (m)")
    plt.title("Midpoint time series comparison")
    plt.grid(True)
    plt.legend()
    safe_savefig(os.path.join(out_dir, "comparison_midpoint_timeseries.png"))

    # FFT comparison (normalize for readability)
    fmax = fft_xlim_hz
    plt.figure()
    mask_u = res_uni["freqs"] <= fmax
    mask_m = res_multi["freqs"] <= fmax

    u_mag = res_uni["fft_mag"][mask_u]
    m_mag = res_multi["fft_mag"][mask_m]

    u_mag = u_mag / (np.max(u_mag) + 1e-12)
    m_mag = m_mag / (np.max(m_mag) + 1e-12)

    plt.plot(res_uni["freqs"][mask_u], u_mag, label="unimodal (normalized)")
    plt.plot(res_multi["freqs"][mask_m], m_mag, label="multimodal (normalized)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized magnitude")
    plt.title("FFT spectrum comparison (normalized)")
    plt.grid(True)
    plt.legend()
    safe_savefig(os.path.join(out_dir, "comparison_fft_spectra.png"))

    print("\n=== DONE ===")
    print(f"All results saved in: {os.path.abspath(out_dir)}")
    print("Key outputs:")
    print("  - depth_field_hxy.png")
    print("  - unimodal_* (timeseries, fft, snapshots, audification, optional animation)")
    print("  - multimodal_* (timeseries, fft, snapshots, audification, optional animation)")
    print("  - comparison_midpoint_timeseries.png")
    print("  - comparison_fft_spectra.png")


if __name__ == "__main__":
    main()
