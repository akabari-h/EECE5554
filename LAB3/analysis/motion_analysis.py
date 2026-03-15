#!/usr/bin/env python3

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import sys
sys.path.insert(0, str(Path(__file__).parent))
from imu_analysis import read_imu_bag, quaternion_to_euler

BAG_PATH = Path.home() / 'EECE5554' / 'LAB3' / 'data' / 'dynamic_motion'

IMU_VIDEO_OFFSET = 17.407

CLIP_TIMES = [
    (21.9, 'Clip 1 - Pitch motion about Y axis (gyro_y peak = -2.33 rad/s)'),
    (61.0, 'Clip 2 - Roll motion about X axis (gyro_x peak = -18.84 rad/s)'),
    (78.3, 'Clip 3 - Yaw motion about Z axis (gyro_z peak = -4.52 rad/s)'),
]

OUTPUT_DIR = Path.home() / 'EECE5554' / 'LAB3' / 'analysis' / 'plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_DURATION = 5.0  # seconds per clip


def plot_full_timeseries(df, output_dir):

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # Gyroscope
    axes[0].plot(df['time_sec'], df['gyro_x'], label='X', linewidth=0.8)
    axes[0].plot(df['time_sec'], df['gyro_y'], label='Y', linewidth=0.8)
    axes[0].plot(df['time_sec'], df['gyro_z'], label='Z', linewidth=0.8)
    axes[0].set_ylabel('Angular Rate (rad/s)')
    axes[0].set_title('Gyroscope')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accelerometer
    axes[1].plot(df['time_sec'], df['accel_x'], label='X', linewidth=0.8)
    axes[1].plot(df['time_sec'], df['accel_y'], label='Y', linewidth=0.8)
    axes[1].plot(df['time_sec'], df['accel_z'], label='Z', linewidth=0.8)
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].set_title('Accelerometer')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Magnetometer
    axes[2].plot(df['time_sec'], df['mag_x'], label='X', linewidth=0.8)
    axes[2].plot(df['time_sec'], df['mag_y'], label='Y', linewidth=0.8)
    axes[2].plot(df['time_sec'], df['mag_z'], label='Z', linewidth=0.8)
    axes[2].set_ylabel('Magnetic Field (Gauss)')
    axes[2].set_title('Magnetometer')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Orientation
    axes[3].plot(df['time_sec'], df['roll'],  label='Roll',  linewidth=0.8)
    axes[3].plot(df['time_sec'], df['pitch'], label='Pitch', linewidth=0.8)
    axes[3].plot(df['time_sec'], df['yaw'],   label='Yaw',   linewidth=0.8)
    axes[3].set_ylabel('Angle (degrees)')
    axes[3].set_title('Orientation (Euler Angles)')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.suptitle('Dynamic Motion — Full Time Series', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'motion_full_timeseries.png', dpi=150)
    plt.close()
    print("Saved: motion_full_timeseries.png")
    print("Use this plot to identify interesting 5-second windows.")
    print("Update CLIP_TIMES in the script with your chosen start times.")


def plot_clip(df, start_time, duration, label, clip_number, output_dir):

    mask = (df['time_sec'] >= start_time) & \
           (df['time_sec'] < start_time + duration)
    clip = df[mask].copy()

    if len(clip) == 0:
        print(f"WARNING: No data found for clip starting at {start_time}s")
        print(f"  Available time range: 0 to {df['time_sec'].max():.1f}s")
        return

    clip['time_sec'] = clip['time_sec'] - start_time

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Gyroscope clip
    axes[0].plot(clip['time_sec'], clip['gyro_x'], label='X', linewidth=1.2)
    axes[0].plot(clip['time_sec'], clip['gyro_y'], label='Y', linewidth=1.2)
    axes[0].plot(clip['time_sec'], clip['gyro_z'], label='Z', linewidth=1.2)
    axes[0].set_ylabel('Angular Rate (rad/s)')
    axes[0].set_title('Gyroscope')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accelerometer clip
    axes[1].plot(clip['time_sec'], clip['accel_x'], label='X', linewidth=1.2)
    axes[1].plot(clip['time_sec'], clip['accel_y'], label='Y', linewidth=1.2)
    axes[1].plot(clip['time_sec'], clip['accel_z'], label='Z', linewidth=1.2)
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].set_title('Accelerometer')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Orientation clip
    axes[2].plot(clip['time_sec'], clip['roll'],  label='Roll',  linewidth=1.2)
    axes[2].plot(clip['time_sec'], clip['pitch'], label='Pitch', linewidth=1.2)
    axes[2].plot(clip['time_sec'], clip['yaw'],   label='Yaw',   linewidth=1.2)
    axes[2].set_ylabel('Angle (degrees)')
    axes[2].set_title('Orientation (Euler Angles)')
    axes[2].set_xlabel('Time within clip (seconds)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f'Clip {clip_number}: {label}\n'
        f'(t = {start_time:.1f}s to {start_time + duration:.1f}s)',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    filename = f'motion_clip_{clip_number}.png'
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


if __name__ == '__main__':

    print("="*55)
    print("IMU DYNAMIC MOTION ANALYSIS — Lab 3")
    print("="*55)

    # Check if bag file exists
    if not BAG_PATH.exists():
        print(f"\nWARNING: Bag file not found at {BAG_PATH}")
        print("Update BAG_PATH in the script when you have the motion bag.")
        print("\nPrinting video alignment methodology instead:")
        exit(0)

    # Read the bag file
    df = read_imu_bag(BAG_PATH)

    # Step 1: Plot full time series to identify interesting clips
    print("\nGenerating full time series plot...")
    plot_full_timeseries(df, OUTPUT_DIR)

    # Step 2: Plot the three 5-second clips
    print("\nGenerating 5-second clips...")
    for i, (start_time, label) in enumerate(CLIP_TIMES, start=1):
        plot_clip(df, start_time, CLIP_DURATION, label, i, OUTPUT_DIR)


    print(f"\nAll motion plots saved to: {OUTPUT_DIR}")
    print("="*55)
