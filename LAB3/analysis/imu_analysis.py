#!/usr/bin/env python3

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_imu_bag(bag_path):

    bag_folder = Path(bag_path)
    db_files = list(bag_folder.glob('*.db3'))

    if not db_files:
        raise FileNotFoundError(f"No .db3 file found in {bag_path}")

    print(f"Reading: {db_files[0]}")
    conn = sqlite3.connect(str(db_files[0]))
    cursor = conn.cursor()

    # Find the /imu topic ID and message type
    cursor.execute("SELECT id, name, type FROM topics")
    topics = cursor.fetchall()

    imu_topic_id = None
    imu_msg_type = None
    for topic_id, name, msg_type in topics:
        if name == '/imu':
            imu_topic_id = topic_id
            imu_msg_type = msg_type
            break

    if imu_topic_id is None:
        raise ValueError(f"No /imu topic found. Available: {[t[1] for t in topics]}")

    print(f"Found topic /imu with type: {imu_msg_type}")

    # Get the Python class for deserialization
    msg_class = get_message(imu_msg_type)

    # Query all messages for the /imu topic
    cursor.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ?",
        (imu_topic_id,)
    )
    rows = cursor.fetchall()
    print(f"Found {len(rows)} IMU messages")
    conn.close()

    # Deserialize each message and extract fields
    data_list = []
    for timestamp, data in rows:
        msg = deserialize_message(data, msg_class)

        # Convert quaternion back to Euler angles for plotting
        # This reverses the conversion we did in the driver
        q = msg.imu.orientation
        roll, pitch, yaw = quaternion_to_euler(q.w, q.x, q.y, q.z)

        data_list.append({
            'timestamp_ns': timestamp,

            # Euler angles in degrees (converted from quaternion)
            'roll':  np.degrees(roll),
            'pitch': np.degrees(pitch),
            'yaw':   np.degrees(yaw),

            # Angular velocity in rad/s (from gyroscope)
            'gyro_x': msg.imu.angular_velocity.x,
            'gyro_y': msg.imu.angular_velocity.y,
            'gyro_z': msg.imu.angular_velocity.z,

            # Linear acceleration in m/s²
            'accel_x': msg.imu.linear_acceleration.x,
            'accel_y': msg.imu.linear_acceleration.y,
            'accel_z': msg.imu.linear_acceleration.z,

            # Magnetic field in Tesla (already converted in driver)
            # Multiply by 1e4 to convert back to Gauss for plotting
            # Gauss is more intuitive for Earth's magnetic field (~0.5 Gauss)
            'mag_x': msg.mag_field.magnetic_field.x * 1e4,
            'mag_y': msg.mag_field.magnetic_field.y * 1e4,
            'mag_z': msg.mag_field.magnetic_field.z * 1e4,
        })

    df = pd.DataFrame(data_list)

    # Convert timestamp to seconds starting from zero
    # This makes the x-axis of time series plots read "0 to N seconds"
    # rather than a large Unix timestamp number
    df['time_sec'] = (df['timestamp_ns'] - df['timestamp_ns'].iloc[0]) / 1e9

    return df


def quaternion_to_euler(w, x, y, z):

    # Roll (rotation around X axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotation around Y axis)
    # Clamp to [-1, 1] to handle floating point errors at ±90°
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (rotation around Z axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def plot_time_series(df, output_dir):

    # --- Gyroscope ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['time_sec'], df['gyro_x'], label='X', linewidth=0.8)
    ax.plot(df['time_sec'], df['gyro_y'], label='Y', linewidth=0.8)
    ax.plot(df['time_sec'], df['gyro_z'], label='Z', linewidth=0.8)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Angular Rate (rad/s)')
    ax.set_title('Gyroscope - Stationary Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'gyro_timeseries.png', dpi=150)
    plt.close()
    print("Saved: gyro_timeseries.png")

    # --- Accelerometer ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['time_sec'], df['accel_x'], label='X', linewidth=0.8)
    ax.plot(df['time_sec'], df['accel_y'], label='Y', linewidth=0.8)
    ax.plot(df['time_sec'], df['accel_z'], label='Z', linewidth=0.8)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Linear Acceleration (m/s²)')
    ax.set_title('Accelerometer - Stationary Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'accel_timeseries.png', dpi=150)
    plt.close()
    print("Saved: accel_timeseries.png")

    # --- Magnetometer ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['time_sec'], df['mag_x'], label='X', linewidth=0.8)
    ax.plot(df['time_sec'], df['mag_y'], label='Y', linewidth=0.8)
    ax.plot(df['time_sec'], df['mag_z'], label='Z', linewidth=0.8)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Magnetic Field (Gauss)')
    ax.set_title('Magnetometer - Stationary Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mag_timeseries.png', dpi=150)
    plt.close()
    print("Saved: mag_timeseries.png")

    # --- Orientation (Euler Angles) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['time_sec'], df['roll'],  label='Roll',  linewidth=0.8)
    ax.plot(df['time_sec'], df['pitch'], label='Pitch', linewidth=0.8)
    ax.plot(df['time_sec'], df['yaw'],   label='Yaw',   linewidth=0.8)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Orientation (Euler Angles) - Stationary Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'orientation_timeseries.png', dpi=150)
    plt.close()
    print("Saved: orientation_timeseries.png")


def plot_orientation_histograms(df, output_dir):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col, label in zip(
        axes,
        ['roll', 'pitch', 'yaw'],
        ['Roll', 'Pitch', 'Yaw']
    ):
        # Center around the median
        median_val = df[col].median()
        centered = df[col] - median_val

        mean_val = df[col].mean()
        std_val  = df[col].std()

        ax.hist(centered, bins=50, color='steelblue',
                edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--',
                   linewidth=1.5, label=f'Median: {median_val:.4f}°')
        ax.axvline(mean_val - median_val, color='green',
                   linestyle='--', linewidth=1.5,
                   label=f'Mean offset: {mean_val - median_val:.4f}°')

        ax.set_xlabel(f'{label} deviation from median (degrees)')
        ax.set_ylabel('Frequency (count)')
        ax.set_title(f'{label} Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        print(f"\n{label}:")
        print(f"  Mean   : {mean_val:.6f} deg")
        print(f"  Median : {median_val:.6f} deg")
        print(f"  Std Dev: {std_val:.6f} deg")

    plt.suptitle('Orientation Noise Distribution Around Median',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'orientation_histograms.png', dpi=150)
    plt.close()
    print("\nSaved: orientation_histograms.png")


def print_summary_statistics(df):

    print("\n" + "="*55)
    print("SUMMARY STATISTICS — STATIONARY DATA")
    print("="*55)
    print(f"Total messages : {len(df)}")
    print(f"Duration       : {df['time_sec'].iloc[-1]:.1f} seconds")

    for sensor, cols, unit in [
        ('Gyroscope',     ['gyro_x',  'gyro_y',  'gyro_z'],  'rad/s'),
        ('Accelerometer', ['accel_x', 'accel_y', 'accel_z'], 'm/s²'),
        ('Magnetometer',  ['mag_x',   'mag_y',   'mag_z'],   'Gauss'),
        ('Orientation',   ['roll',    'pitch',   'yaw'],     'deg'),
    ]:
        print(f"\n{sensor} ({unit}):")
        for col in cols:
            axis = col.split('_')[-1].upper() if '_' in col else col
            print(f"  {axis:6s} — Mean: {df[col].mean():10.6f}  "
                  f"Std: {df[col].std():10.6f}  "
                  f"Min: {df[col].min():10.6f}  "
                  f"Max: {df[col].max():10.6f}")


if __name__ == '__main__':

    BAG_PATH   = Path.home() / 'EECE5554' / 'LAB3' / 'data' / 'stationary_imu'
    OUTPUT_DIR = Path.home() / 'EECE5554' / 'LAB3' / 'analysis' / 'plots'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*55)
    print("IMU STATIONARY DATA ANALYSIS — Lab 3")
    print("="*55)

    # Read the bag file
    df = read_imu_bag(BAG_PATH)

    # Generate all plots
    print("\nGenerating time series plots...")
    plot_time_series(df, OUTPUT_DIR)

    print("\nGenerating orientation histograms...")
    plot_orientation_histograms(df, OUTPUT_DIR)

    # Print statistics
    print_summary_statistics(df)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*55)
