#!/usr/bin/env python3
"""
Forward Velocity Estimation for Lab 4.

Two methods:
1. Integrate forward acceleration (accel_x) from IMU
2. Compute velocity from GPS position changes

The IMU-based velocity requires adjustment because:
- Gravity component may be present if IMU is tilted
- Bias in accelerometer causes velocity to drift
- We need to subtract the mean bias during stationary periods
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from read_bags import read_imu_data, read_gps_data

OUTPUT_DIR = Path.home() / 'EECE5554' / 'LAB4' / 'analysis' / 'plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR   = Path.home() / 'EECE5554' / 'LAB4' / 'data'


def integrate_acceleration(accel, time_sec):
    """
    Integrate acceleration to get velocity using trapezoidal rule.

    Why trapezoidal?
    More accurate than rectangular integration because it accounts
    for the linear variation between samples. For a 40 Hz signal
    this gives significantly better results over long recordings.

    Args:
        accel: Forward acceleration array (m/s²)
        time_sec: Time array (seconds)

    Returns:
        velocity array (m/s)
    """
    dt = np.diff(time_sec)
    velocity = np.zeros(len(accel))
    for i in range(1, len(accel)):
        velocity[i] = velocity[i-1] + \
                      (accel[i-1] + accel[i]) / 2.0 * dt[i-1]
    return velocity


def compute_gps_velocity(gps_df):
    """
    Compute velocity from GPS position changes.

    Why use UTM coordinates?
    UTM gives positions in meters, so we can directly compute
    velocity as distance/time without any coordinate conversion.

    velocity = sqrt(dE² + dN²) / dt

    where dE and dN are changes in UTM Easting and Northing.
    This gives the magnitude of the horizontal velocity vector.

    Args:
        gps_df: GPS DataFrame with utm_easting, utm_northing, time_sec

    Returns:
        gps_time: Time array for GPS velocity (midpoints between readings)
        gps_vel:  Velocity magnitude array (m/s)
    """
    dt       = np.diff(gps_df['time_sec'].values)
    d_east   = np.diff(gps_df['utm_easting'].values)
    d_north  = np.diff(gps_df['utm_northing'].values)

    # Euclidean distance between consecutive GPS points
    distance = np.sqrt(d_east**2 + d_north**2)

    # Velocity = distance / time
    gps_vel  = distance / dt

    # Time at midpoints between GPS readings
    gps_time = (gps_df['time_sec'].values[:-1] +
                gps_df['time_sec'].values[1:]) / 2.0

    return gps_time, gps_vel


def remove_accel_bias(accel, time_sec, stationary_duration=15.0):
    """
    Remove accelerometer bias using the initial stationary period.

    Why remove bias?
    The accelerometer has a constant offset (bias) that causes
    the integrated velocity to drift linearly over time. During
    the initial 15 seconds when the car was stationary, the
    true forward acceleration should be zero. Any non-zero mean
    during this period is the bias.

    We also subtract the gravity component. If the IMU is tilted
    by angle θ from horizontal, accel_x measures g*sin(θ) even
    when stationary. This appears as a constant offset.

    Args:
        accel: Raw acceleration array
        time_sec: Time array
        stationary_duration: Seconds at start when car was stationary

    Returns:
        bias-corrected acceleration array
    """
    # Find samples during stationary period
    stationary_mask = time_sec < stationary_duration
    bias = accel[stationary_mask].mean()

    print(f"  Accelerometer bias (from first {stationary_duration}s): {bias:.6f} m/s²")

    return accel - bias

def apply_zupt(vel_imu, t_imu, accel_corrected,
               vel_gps, t_gps, stop_threshold=0.5):
    """
    Segment-based ZUPT — re-integrate from zero at each stop.

    Why segment-based?
    Linear interpolation only removes some drift. The real fix
    is to treat each driving segment independently — start from
    zero velocity at each stop and integrate only within that
    segment. This prevents drift from one segment contaminating
    the next.

    Algorithm:
    1. Find all GPS stop times
    2. Find corresponding IMU indices
    3. For each segment between stops, integrate from zero
    4. Concatenate all segments

    Args:
        vel_imu: Initial integrated velocity (unused — we re-integrate)
        t_imu: IMU time array
        accel_corrected: Bias-corrected acceleration
        vel_gps: GPS velocity array
        t_gps: GPS time array
        stop_threshold: GPS velocity below this = stopped (m/s)

    Returns:
        Corrected IMU velocity array
    """
    # Find GPS indices where car is stopped
    stopped_gps_idx = np.where(vel_gps < stop_threshold)[0]

    if len(stopped_gps_idx) == 0:
        print("  No stop events found")
        return vel_imu

    # Get stop times from GPS
    stop_times = t_gps[stopped_gps_idx]

    # Find corresponding IMU indices for each stop time
    stop_imu_idx = []
    for st in stop_times:
        idx = np.argmin(np.abs(t_imu - st))
        stop_imu_idx.append(idx)

    # Remove duplicates and sort
    stop_imu_idx = sorted(set(stop_imu_idx))
    print(f"  Found {len(stop_imu_idx)} stop indices in IMU data")

    # Build segment boundaries
    # Each segment goes from one stop to the next
    boundaries = [0] + stop_imu_idx + [len(t_imu) - 1]
    boundaries = sorted(set(boundaries))

    # Re-integrate each segment starting from zero
    vel_out = np.zeros(len(t_imu))

    for seg in range(len(boundaries) - 1):
        i_start = boundaries[seg]
        i_end   = boundaries[seg + 1]

        if i_end <= i_start:
            continue

        # Get segment data
        t_seg     = t_imu[i_start:i_end + 1]
        accel_seg = accel_corrected[i_start:i_end + 1]
        dt_seg    = np.diff(t_seg)

        # Integrate from zero within this segment
        v_seg    = np.zeros(len(t_seg))
        for i in range(1, len(t_seg)):
            v_seg[i] = v_seg[i-1] + \
                       (accel_seg[i-1] + accel_seg[i]) / 2.0 * dt_seg[i-1]

        vel_out[i_start:i_end + 1] = v_seg

    print(f"  Processed {len(boundaries)-1} segments")
    print(f"  Corrected velocity range: "
          f"{vel_out.min():.2f} to {vel_out.max():.2f} m/s")

    return vel_out


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':

    print("="*60)
    print("FORWARD VELOCITY ESTIMATION")
    print("="*60)

    # Load data
    print("\nLoading driving data...")
    imu = read_imu_data(DATA_DIR / 'data_driving')
    gps = read_gps_data(DATA_DIR / 'data_driving')

    t_imu = imu['time_sec'].values
    fs    = imu.attrs['fs']

    # ===== STEP 1: RAW VELOCITY FROM ACCELERATION =====
    print("\nComputing raw velocity from acceleration...")
    accel_raw = imu['accel_x'].values

    # Raw velocity (before bias removal)
    vel_raw = integrate_acceleration(accel_raw, t_imu)

    print(f"  Raw velocity range: {vel_raw.min():.2f} to "
          f"{vel_raw.max():.2f} m/s")

    # ===== STEP 2: BIAS-CORRECTED VELOCITY =====
    print("\nRemoving accelerometer bias...")
    accel_corrected = remove_accel_bias(accel_raw, t_imu,
                                        stationary_duration=15.0)
    
    # Initial integration (will be corrected by ZUPT)
    vel_corrected = integrate_acceleration(accel_corrected, t_imu)
    print(f"  After bias removal: {vel_corrected.min():.2f} to "
          f"{vel_corrected.max():.2f} m/s")

    # ===== STEP 2B: APPLY ZUPT =====
    print("\nApplying Zero Velocity Updates (ZUPT)...")

    # Need GPS velocity first for ZUPT
    t_gps_temp, vel_gps_temp = compute_gps_velocity(gps)
    vel_corrected = apply_zupt(vel_corrected, t_imu, accel_corrected,
                               vel_gps_temp, t_gps_temp,
                               stop_threshold=0.5)

    print(f"  Final max speed: {vel_corrected.max():.2f} m/s "
          f"({vel_corrected.max()*3.6:.1f} km/h)")

    # ===== STEP 3: GPS VELOCITY =====
    print("\nComputing GPS velocity...")
    t_gps, vel_gps = compute_gps_velocity(gps)

    print(f"  GPS velocity range: {vel_gps.min():.2f} to "
          f"{vel_gps.max():.2f} m/s")
    print(f"  GPS max speed: {vel_gps.max():.2f} m/s "
          f"({vel_gps.max()*3.6:.1f} km/h)")

    # ===== PLOT 1: Raw vs Corrected Velocity from Acceleration =====
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_imu, vel_raw,       color='red',
            linewidth=0.8, alpha=0.7, label='Raw (before bias removal)')
    ax.plot(t_imu, vel_corrected, color='blue',
            linewidth=0.8, label='Corrected (after bias removal)')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Forward Velocity from Accelerometer: Before and After Adjustment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'velocity_accel_raw_vs_corrected.png', dpi=150)
    plt.close()
    print("\nSaved: velocity_accel_raw_vs_corrected.png")

    # ===== PLOT 2: Corrected IMU Velocity vs GPS Velocity =====
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_imu, vel_corrected, color='blue',
            linewidth=0.8, label='IMU Velocity (corrected)')
    ax.plot(t_gps, vel_gps,       color='green',
            linewidth=1.2, label='GPS Velocity')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Forward Velocity: IMU vs GPS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'velocity_imu_vs_gps.png', dpi=150)
    plt.close()
    print("Saved: velocity_imu_vs_gps.png")

    # Save velocity data for dead reckoning
    np.save(OUTPUT_DIR / 'velocity_corrected.npy', vel_corrected)
    np.save(OUTPUT_DIR / 'time_gps.npy', t_gps)
    np.save(OUTPUT_DIR / 'velocity_gps.npy', vel_gps)
    print("\nVelocity arrays saved for dead reckoning analysis")

    # Print statistics for report
    print(f"\n{'='*55}")
    print("VELOCITY STATISTICS")
    print(f"{'='*55}")
    print(f"IMU max speed  : {vel_corrected.max():.2f} m/s "
          f"({vel_corrected.max()*3.6:.1f} km/h)")
    print(f"GPS max speed  : {vel_gps.max():.2f} m/s "
          f"({vel_gps.max()*3.6:.1f} km/h)")
    print(f"IMU mean speed : {np.abs(vel_corrected).mean():.2f} m/s")
    print(f"GPS mean speed : {vel_gps.mean():.2f} m/s")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*60)
