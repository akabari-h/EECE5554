#!/usr/bin/env python3

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BAG_PATH = Path.home() / 'EECE5554' / 'LAB3' / 'data' / '5_hour_imu_data'
MSG_TYPE   = 'imu_interfaces/msg/IMUmsg'
OUTPUT_DIR = Path.home() / 'EECE5554' / 'LAB3' / 'analysis' / 'plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_imu_data(bag_path, msg_type, max_messages=None):

    db_files = list(Path(bag_path).glob('*.db3'))
    if not db_files:
        raise FileNotFoundError(f"No .db3 file in {bag_path}")

    print(f"Reading: {db_files[0]}")
    conn = sqlite3.connect(str(db_files[0]))
    cursor = conn.cursor()

    # Find /imu topic
    cursor.execute("SELECT id FROM topics WHERE name='/imu'")
    result = cursor.fetchone()
    if result is None:
        raise ValueError("No /imu topic found in bag")
    topic_id = result[0]

    # Query messages — optionally limit count for speed
    if max_messages:
        cursor.execute(
            "SELECT timestamp, data FROM messages "
            "WHERE topic_id=? LIMIT ?",
            (topic_id, max_messages)
        )
    else:
        cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=?",
            (topic_id,)
        )

    rows = cursor.fetchall()
    conn.close()
    print(f"Loaded {len(rows)} messages")

    # Deserialize all messages
    msg_class = get_message(msg_type)

    timestamps = []
    gyro_x, gyro_y, gyro_z     = [], [], []
    accel_x, accel_y, accel_z  = [], [], []

    for i, (ts, data) in enumerate(rows):
        if i % 50000 == 0:
            print(f"  Deserializing message {i}/{len(rows)}...")
        msg = deserialize_message(data, msg_class)
        timestamps.append(ts)
        gyro_x.append(msg.imu.angular_velocity.x)
        gyro_y.append(msg.imu.angular_velocity.y)
        gyro_z.append(msg.imu.angular_velocity.z)
        accel_x.append(msg.imu.linear_acceleration.x)
        accel_y.append(msg.imu.linear_acceleration.y)
        accel_z.append(msg.imu.linear_acceleration.z)

    # Convert to numpy arrays for fast computation
    timestamps = np.array(timestamps, dtype=np.float64) / 1e9
    dt_array   = np.diff(timestamps)

    # Calculate actual sample rate from median time step
    # Using median instead of mean to avoid outliers from
    # occasional dropped messages or recording pauses
    dt = np.median(dt_array)
    fs = 1.0 / dt
    print(f"Estimated sample rate: {fs:.2f} Hz")
    print(f"Total duration: {(timestamps[-1]-timestamps[0])/3600:.2f} hours")

    return {
        'timestamps': timestamps,
        'dt': dt,
        'fs': fs,
        'gyro_x':  np.array(gyro_x),
        'gyro_y':  np.array(gyro_y),
        'gyro_z':  np.array(gyro_z),
        'accel_x': np.array(accel_x),
        'accel_y': np.array(accel_y),
        'accel_z': np.array(accel_z),
    }


def compute_allan_deviation(data, dt):

    N = len(data)

    # Generate cluster sizes m as powers of 2 up to N/2
    # Powers of 2 give logarithmically spaced points on the plot
    # We go up to N//2 to ensure at least 2 non-overlapping clusters
    max_m = N // 2
    m_values = np.unique(
        np.round(np.logspace(0, np.log10(max_m), 200)).astype(int)
    )
    m_values = m_values[m_values >= 1]
    m_values = m_values[m_values <= max_m]

    taus  = []
    adevs = []

    for m in m_values:
        tau = m * dt

        # Compute cluster averages using cumulative sum trick
        # This is much faster than looping over each cluster
        # cumsum[i] = sum of data[0:i]
        cumsum = np.cumsum(np.insert(data, 0, 0))

        # Cluster sum for each starting position
        # cluster_sums[i] = sum of data[i:i+m]
        cluster_sums = (cumsum[m:] - cumsum[:-m])

        # Number of complete clusters starting at each position
        n_clusters = len(cluster_sums)

        if n_clusters < 2:
            continue
        phase = np.cumsum(data) * dt  # integrated signal (angle for gyro)

        if len(phase) < 2 * m + 1:
            continue

        # Sum of squared differences of adjacent cluster averages
        # This implements the standard AVAR formula efficiently
        diffs = cluster_sums[m:] - cluster_sums[:-m]
        n_diffs = len(diffs)

        if n_diffs < 1:
            continue

        avar = np.sum(diffs ** 2) / (2.0 * m**2 * dt**2 * n_diffs)
        adev = np.sqrt(avar)

        taus.append(tau)
        adevs.append(adev)

    return np.array(taus), np.array(adevs)


def extract_noise_parameters(taus, adevs):

    taus  = np.array(taus)
    adevs = np.array(adevs)

    # Find bias stability (minimum point)
    min_idx = np.argmin(adevs)
    B     = adevs[min_idx]
    tau_B = taus[min_idx]

    # Angle Random Walk (N): value of ADEV at tau = 1 second
    # interpolated from the -1/2 slope region (left of minimum)
    # Find the region with slope closest to -1/2
    left_mask = taus < tau_B
    if np.sum(left_mask) > 2:
        log_tau  = np.log10(taus[left_mask])
        log_adev = np.log10(adevs[left_mask])
        slopes   = np.diff(log_adev) / np.diff(log_tau)
        # Find index where slope is closest to -0.5
        arw_idx  = np.argmin(np.abs(slopes - (-0.5)))
        # N at tau=1 from that slope line
        tau_ref  = 1.0
        N = 10 ** (log_adev[arw_idx] +
                   (-0.5) * (np.log10(tau_ref) - log_tau[arw_idx]))
    else:
        # Fallback: interpolate ADEV at tau=1
        N = float(np.interp(1.0, taus, adevs))

    # Rate Random Walk (K): value at tau = 3 seconds on +1/2 slope
    right_mask = taus > tau_B
    if np.sum(right_mask) > 2:
        log_tau  = np.log10(taus[right_mask])
        log_adev = np.log10(adevs[right_mask])
        slopes   = np.diff(log_adev) / np.diff(log_tau)
        # Find index where slope is closest to +0.5
        rrw_idx  = np.argmin(np.abs(slopes - 0.5))
        tau_ref  = 3.0
        K = 10 ** (log_adev[rrw_idx] +
                   0.5 * (np.log10(tau_ref) - log_tau[rrw_idx]))
        K = K / np.sqrt(3)
    else:
        K = float(np.interp(3.0, taus, adevs)) / np.sqrt(3)

    return {'N': N, 'B': B, 'K': K, 'tau_B': tau_B}


def plot_allan_deviation(taus, adevs, params, title, filename, unit):

    fig, ax = plt.subplots(figsize=(10, 7))

    # Main ADEV curve
    ax.loglog(taus, adevs, 'b-', linewidth=2, label='Allan Deviation')

    # Mark bias stability minimum
    ax.loglog(params['tau_B'], params['B'], 'ro', markersize=10,
              label=f"Bias Stability B = {params['B']:.2e} {unit}")

    # Draw N reference line (slope = -1/2)
    tau_n_line = np.array([taus[0], params['tau_B']])
    adev_n_line = params['N'] * tau_n_line ** (-0.5)
    ax.loglog(tau_n_line, adev_n_line, 'g--', linewidth=1.5,
              label=f"ARW N = {params['N']:.2e} {unit}·√s")

    # Draw K reference line (slope = +1/2)
    if params['tau_B'] < taus[-1]:
        tau_k_line = np.array([params['tau_B'], taus[-1]])
        adev_k_line = params['K'] * np.sqrt(3) * tau_k_line ** (0.5)
        ax.loglog(tau_k_line, adev_k_line, 'm--', linewidth=1.5,
                  label=f"RRW K = {params['K']:.2e} {unit}/√s")

    # Reference slope lines for visual guidance
    tau_ref = np.array([taus[0], taus[-1]])
    # -1/2 slope guide
    ax.loglog(tau_ref,
              adevs[0] * (tau_ref / taus[0]) ** (-0.5),
              'k:', linewidth=0.8, alpha=0.4, label='slope = -1/2')
    # +1/2 slope guide
    ax.loglog(tau_ref,
              adevs[-1] * (tau_ref / taus[-1]) ** (0.5),
              'k-.', linewidth=0.8, alpha=0.4, label='slope = +1/2')

    ax.set_xlabel('Averaging Time τ (seconds)', fontsize=12)
    ax.set_ylabel(f'Allan Deviation ({unit})', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def print_parameters_table(results, sensor_name, unit):
    print(f"\n{'='*60}")
    print(f"{sensor_name} Noise Parameters")
    print(f"{'='*60}")
    print(f"{'Axis':<6} {'ARW/VRW N':<20} {'Bias Stab B':<20} {'RRW/AccRW K':<20}")
    print(f"{'':6} {f'({unit}·√s)':<20} {f'({unit})':<20} {f'({unit}/√s)':<20}")
    print(f"{'-'*66}")
    for axis, params in results.items():
        print(f"{axis:<6} {params['N']:<20.4e} "
              f"{params['B']:<20.4e} "
              f"{params['K']:<20.4e}")

if __name__ == '__main__':

    print("="*60)
    print("ALLAN VARIANCE ANALYSIS — Lab 3")
    print("="*60)

    print("\nReading bag file (this may take a minute)...")
    data = read_imu_data(BAG_PATH, MSG_TYPE, max_messages=200000)

    dt = data['dt']

    print("\nComputing gyroscope Allan Deviation...")
    gyro_results = {}

    for axis, key in [('X', 'gyro_x'), ('Y', 'gyro_y'), ('Z', 'gyro_z')]:
        print(f"  Processing gyro {axis}...")
        taus, adevs = compute_allan_deviation(data[key], dt)
        params = extract_noise_parameters(taus, adevs)
        gyro_results[axis] = params

        plot_allan_deviation(
            taus, adevs, params,
            title=f'Allan Deviation — Gyroscope {axis}-axis',
            filename=f'allan_gyro_{axis.lower()}.png',
            unit='rad/s'
        )

    print_parameters_table(gyro_results, 'GYROSCOPE', 'rad/s')

    print("\nComputing accelerometer Allan Deviation...")
    accel_results = {}

    for axis, key in [('X', 'accel_x'), ('Y', 'accel_y'), ('Z', 'accel_z')]:
        print(f"  Processing accel {axis}...")
        taus, adevs = compute_allan_deviation(data[key], dt)
        params = extract_noise_parameters(taus, adevs)
        accel_results[axis] = params

        plot_allan_deviation(
            taus, adevs, params,
            title=f'Allan Deviation — Accelerometer {axis}-axis',
            filename=f'allan_accel_{axis.lower()}.png',
            unit='m/s²'
        )

    print_parameters_table(accel_results, 'ACCELEROMETER', 'm/s²')

    print(f"\nAll Allan Variance plots saved to: {OUTPUT_DIR}")
    print("="*60)
