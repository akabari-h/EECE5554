#!/usr/bin/env python3
"""
Yaw Estimation for Lab 4.

Four methods compared:
1. Raw magnetometer yaw
2. Calibrated magnetometer yaw
3. Gyro-integrated yaw
4. Complementary filter yaw
5. IMU Euler yaw (for final comparison only)

All analysis uses data_driving.bag with calibration from circles.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from read_bags import read_imu_data

OUTPUT_DIR = Path.home() / 'EECE5554' / 'LAB4' / 'analysis' / 'plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR   = Path.home() / 'EECE5554' / 'LAB4' / 'data'


def mag_to_yaw(mag_x, mag_y):
    """
    Convert calibrated magnetometer X and Y to yaw angle in degrees.

    Why atan2?
    atan2(y, x) gives the angle of the vector (x, y) from the
    positive X axis in the range [-180, 180] degrees.
    For a magnetometer lying flat, this gives the heading relative
    to magnetic north.

    Args:
        mag_x, mag_y: Calibrated magnetometer arrays (Gauss)

    Returns:
        yaw in degrees, range [-180, 180]
    """
    yaw_rad = np.arctan2(mag_y, mag_x)
    yaw_deg = np.degrees(yaw_rad)
    return yaw_deg


def integrate_gyro_yaw(gyro_z, time_sec):
    """
    Integrate gyroscope Z-axis rate to get yaw angle.

    Why cumulative trapezoidal integration?
    The gyro measures angular rate (rad/s). Integrating it over
    time gives the accumulated angle change (rad). The trapezoidal
    rule is more accurate than simple rectangular integration because
    it accounts for the linear change between samples.

    The initial yaw is set to 0 — this gives relative yaw change,
    not absolute heading. We will align it with the magnetometer later.

    Args:
        gyro_z: Angular rate around Z axis (rad/s)
        time_sec: Time array (seconds)

    Returns:
        Integrated yaw in degrees
    """
    dt = np.diff(time_sec)

    # Trapezoidal integration: cumulative sum of trapezoids
    # Each trapezoid area = (y[i] + y[i+1]) / 2 * dt[i]
    yaw_rad = np.zeros(len(gyro_z))
    for i in range(1, len(gyro_z)):
        yaw_rad[i] = yaw_rad[i-1] + \
                     (gyro_z[i-1] + gyro_z[i]) / 2.0 * dt[i-1]

    yaw_deg = np.degrees(yaw_rad)
    return yaw_deg


def butter_lowpass(data, cutoff_hz, fs, order=4):
    """
    Apply a Butterworth low-pass filter.

    Why low-pass for magnetometer?
    The magnetometer gives absolute heading but is noisy at high
    frequencies due to vibration and electromagnetic interference.
    A low-pass filter keeps the slow-varying heading information
    and removes the high-frequency noise.

    Args:
        data: Input signal
        cutoff_hz: Cutoff frequency in Hz
        fs: Sample rate in Hz
        order: Filter order (higher = steeper rolloff)

    Returns:
        Filtered signal
    """
    nyq = fs / 2.0
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def butter_highpass(data, cutoff_hz, fs, order=4):
    """
    Apply a Butterworth high-pass filter.

    Why high-pass for integrated gyro?
    The integrated gyro gives accurate short-term heading changes
    but drifts over time due to bias accumulation. A high-pass
    filter keeps the accurate fast-varying component and removes
    the slow drift (which appears as a low-frequency signal).

    Args:
        data: Input signal
        cutoff_hz: Cutoff frequency in Hz
        fs: Sample rate in Hz
        order: Filter order

    Returns:
        Filtered signal
    """
    nyq = fs / 2.0
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)


def complementary_filter(mag_yaw_lp, gyro_yaw_hp):
    """
    Combine low-pass magnetometer and high-pass gyro yaw.

    The complementary filter equation is:
        yaw_cf = mag_yaw_lp + gyro_yaw_hp

    Why does this work?
    - Low-pass magnetometer: accurate heading at low frequencies
      (long-term absolute reference), free from vibration noise
    - High-pass integrated gyro: accurate heading changes at high
      frequencies (short-term dynamics), free from bias drift
    - Together they complement each other — hence the name

    The two filters are complementary in the frequency domain:
    H_lp(f) + H_hp(f) ≈ 1 for all frequencies when they share
    the same cutoff frequency.

    Args:
        mag_yaw_lp: Low-pass filtered magnetometer yaw
        gyro_yaw_hp: High-pass filtered integrated gyro yaw

    Returns:
        Combined complementary filter yaw
    """
    return mag_yaw_lp + gyro_yaw_hp


def unwrap_yaw(yaw_deg):
    """
    Unwrap yaw to remove discontinuities at ±180 degrees.

    When yaw crosses ±180 degrees it wraps around causing large
    jumps in the plot. np.unwrap works in radians so we convert,
    unwrap, then convert back.
    """
    return np.degrees(np.unwrap(np.radians(yaw_deg)))


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':

    print("="*60)
    print("YAW ESTIMATION")
    print("="*60)

    # Load calibration parameters from circles data
    cal_params = np.load(OUTPUT_DIR / 'cal_params.npy')
    x_offset, y_offset, x_scale, y_scale = cal_params
    print(f"\nUsing calibration: offset=({x_offset:.4f}, {y_offset:.4f})")
    print(f"                   scale =({x_scale:.4f}, {y_scale:.4f})")

    # Load driving data
    print("\nLoading driving data...")
    imu = read_imu_data(DATA_DIR / 'data_driving')
    fs  = imu.attrs['fs']
    t   = imu['time_sec'].values

    # ===== STEP 1: MAGNETOMETER YAW =====
    # Raw yaw (before calibration)
    yaw_mag_raw = mag_to_yaw(
        imu['mag_x'].values,
        imu['mag_y'].values
    )

    # Apply calibration to driving data
    mag_x_cal = (imu['mag_x'].values - x_offset) * x_scale
    mag_y_cal = (imu['mag_y'].values - y_offset) * y_scale

    # Calibrated yaw
    yaw_mag_cal = mag_to_yaw(mag_x_cal, mag_y_cal)

    # Unwrap to remove ±180 jumps
    yaw_mag_raw_uw = unwrap_yaw(yaw_mag_raw)
    yaw_mag_cal_uw = unwrap_yaw(yaw_mag_cal)
    yaw_mag_cal_uw = medfilt(yaw_mag_cal_uw, kernel_size=51)

    print(f"\nMagnetometer yaw range (calibrated): "
          f"{yaw_mag_cal_uw.min():.1f} to {yaw_mag_cal_uw.max():.1f} deg")

    # ===== STEP 2: GYRO INTEGRATED YAW =====
    yaw_gyro = integrate_gyro_yaw(imu['gyro_z'].values, t)

    # Align gyro yaw starting point with magnetometer
    # This ensures both start at the same absolute heading
    yaw_gyro = yaw_gyro - yaw_gyro[0] + yaw_mag_cal_uw[0]

    print(f"Gyro yaw drift over recording: "
          f"{yaw_gyro[-1] - yaw_mag_cal_uw[-1]:.2f} deg")

    # ===== STEP 3: CHOOSE CUTOFF FREQUENCY =====
    # Look at the frequency content of both signals to choose cutoff
    # Rule of thumb for complementary filter:
    # - Magnetometer: noisy at >0.1 Hz, trust at <0.1 Hz
    # - Gyro: drifts at <0.1 Hz, trust at >0.1 Hz
    # We use 0.1 Hz as the crossover frequency
    CUTOFF_HZ = 0.05

    print(f"\nComplementary filter cutoff: {CUTOFF_HZ} Hz")

    # ===== STEP 4: COMPLEMENTARY FILTER =====
    # Low-pass filter the calibrated magnetometer yaw
    yaw_mag_lp = butter_lowpass(yaw_mag_cal_uw, CUTOFF_HZ, fs)

    # High-pass filter the integrated gyro yaw
    yaw_gyro_hp = butter_highpass(yaw_gyro, CUTOFF_HZ, fs)
    
    ALPHA = 0.02
    # Combine
    yaw_cf = ALPHA * yaw_mag_lp + (1.0 - ALPHA) * yaw_gyro
    print(f"Complementary filter alpha: {ALPHA} "
          f"(magnetometer weight: {ALPHA*100:.0f}%, "
          f"gyro weight: {(1-ALPHA)*100:.0f}%)")

    # ===== STEP 5: IMU EULER YAW (for comparison only) =====
    yaw_imu = unwrap_yaw(imu['yaw'].values)
    # ===== ALIGN ALL METHODS TO SAME STARTING POINT =====
    # Each method has a different absolute reference.
    # We align all methods so they start at the same value
    # (the IMU Euler yaw at t=0) for fair comparison.
    #This is standard practice when comparing heading estimates.
    reference_start = yaw_imu[0]

    yaw_mag_cal_uw = yaw_mag_cal_uw - yaw_mag_cal_uw[0] + reference_start
    yaw_gyro       = yaw_gyro       - yaw_gyro[0]       + reference_start
    yaw_mag_lp     = yaw_mag_lp     - yaw_mag_lp[0]     + reference_start
    yaw_gyro_hp    = yaw_gyro_hp    - yaw_gyro_hp[0]     # HP filter is relative
    yaw_cf         = yaw_mag_lp + yaw_gyro_hp
    yaw_mag_raw_uw = yaw_mag_raw_uw - yaw_mag_raw_uw[0] + reference_start

    print(f"\nAll methods aligned to start at {reference_start:.2f} degrees")

    # ===== PLOT 1: Raw vs Calibrated Magnetometer Yaw =====
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, yaw_mag_raw_uw, color='red',
            linewidth=0.8, alpha=0.7, label='Raw Magnetometer Yaw')
    ax.plot(t, yaw_mag_cal_uw, color='blue',
            linewidth=0.8, label='Calibrated Magnetometer Yaw')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Yaw (degrees)')
    ax.set_title('Magnetometer Yaw: Raw vs Calibrated')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'yaw_mag_raw_vs_cal.png', dpi=150)
    plt.close()
    print("\nSaved: yaw_mag_raw_vs_cal.png")

    # ===== PLOT 2: LP Filter, HP Filter, Complementary Filter =====
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, yaw_mag_lp,  color='blue',
            linewidth=1.0, label=f'LP Magnetometer (cutoff={CUTOFF_HZ} Hz)')
    ax.plot(t, yaw_gyro_hp, color='orange',
            linewidth=1.0, label=f'HP Integrated Gyro (cutoff={CUTOFF_HZ} Hz)')
    ax.plot(t, yaw_cf,      color='green',
            linewidth=1.2, label='Complementary Filter')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Yaw (degrees)')
    ax.set_title('LP Magnetometer, HP Gyro, and Complementary Filter Yaw')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'yaw_complementary_filter.png', dpi=150)
    plt.close()
    print("Saved: yaw_complementary_filter.png")

    # ===== PLOT 3: All Four Yaw Methods =====
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, yaw_mag_cal_uw, color='blue',
            linewidth=0.8, alpha=0.8, label='Calibrated Magnetometer')
    ax.plot(t, yaw_gyro,       color='orange',
            linewidth=0.8, alpha=0.8, label='Integrated Gyro')
    ax.plot(t, yaw_cf,         color='green',
            linewidth=1.0, label='Complementary Filter')
    ax.plot(t, yaw_imu,        color='red',
            linewidth=0.8, alpha=0.8, label='IMU Euler Yaw')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Yaw (degrees)')
    ax.set_title('Yaw Comparison: Four Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'yaw_four_methods.png', dpi=150)
    plt.close()
    print("Saved: yaw_four_methods.png")

    # Save yaw data for use in dead reckoning
    np.save(OUTPUT_DIR / 'yaw_cf.npy', yaw_cf)
    np.save(OUTPUT_DIR / 'yaw_mag_cal.npy', yaw_mag_cal_uw)
    np.save(OUTPUT_DIR / 'time_imu.npy', t)
    print("\nYaw arrays saved for dead reckoning analysis")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*60)
