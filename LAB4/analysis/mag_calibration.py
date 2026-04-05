#!/usr/bin/env python3
"""
Magnetometer Calibration for Lab 4.

Uses circles data to compute hard-iron and soft-iron corrections.
Hard-iron: offset of ellipse center from origin
Soft-iron: scaling to make axes equal length (circle instead of ellipse)

Steps:
1. Plot raw mag X-Y to see the distorted ellipse
2. Fit an ellipse to find center offsets (hard-iron)
3. Scale axes to equalize radii (soft-iron)
4. Verify corrected data forms a circle centered at origin
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from read_bags import read_imu_data

OUTPUT_DIR = Path.home() / 'EECE5554' / 'LAB4' / 'analysis' / 'plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.home() / 'EECE5554' / 'LAB4' / 'data'


def compute_hard_iron_offset(mag_x, mag_y):
    """
    Compute hard-iron offset as the midpoint of the min/max range.

    Why midpoint method?
    When driving in circles, the magnetometer traces an ellipse.
    The center of that ellipse is the hard-iron offset — a constant
    magnetic bias added by nearby permanent magnets in the car.
    The simplest estimate of the center is the average of the
    min and max values on each axis.

    Args:
        mag_x, mag_y: Raw magnetometer X and Y arrays

    Returns:
        x_offset, y_offset: Hard-iron offsets to subtract
    """
    x_offset = (mag_x.max() + mag_x.min()) / 2.0
    y_offset = (mag_y.max() + mag_y.min()) / 2.0
    return x_offset, y_offset


def compute_soft_iron_scale(mag_x, mag_y, x_offset, y_offset):
    """
    Compute soft-iron scaling factors to equalize axis radii.

    Why scaling?
    After removing the hard-iron offset, the ellipse is centered
    at the origin but still elongated in one direction due to
    soft-iron effects. We scale each axis by the ratio of its
    radius to the average radius so all axes have the same range.

    Args:
        mag_x, mag_y: Raw magnetometer arrays
        x_offset, y_offset: Hard-iron offsets

    Returns:
        x_scale, y_scale: Multiplicative scale factors
    """
    # Radius in each axis after removing offset
    x_radius = (mag_x.max() - mag_x.min()) / 2.0
    y_radius = (mag_y.max() - mag_y.min()) / 2.0

    # Average radius
    avg_radius = (x_radius + y_radius) / 2.0

    # Scale factors: multiply each axis to reach average radius
    x_scale = avg_radius / x_radius
    y_scale = avg_radius / y_radius

    return x_scale, y_scale


def apply_calibration(mag_x, mag_y, x_offset, y_offset, x_scale, y_scale):
    """
    Apply hard-iron and soft-iron corrections.

    Correction formula:
        mag_x_cal = (mag_x - x_offset) * x_scale
        mag_y_cal = (mag_y - y_offset) * y_scale

    Args:
        mag_x, mag_y: Raw values
        x_offset, y_offset: Hard-iron offsets
        x_scale, y_scale: Soft-iron scales

    Returns:
        Corrected mag_x and mag_y arrays
    """
    mag_x_cal = (mag_x - x_offset) * x_scale
    mag_y_cal = (mag_y - y_offset) * y_scale
    return mag_x_cal, mag_y_cal


def plot_mag_xy(mag_x_raw, mag_y_raw, mag_x_cal, mag_y_cal,
                x_offset, y_offset, output_dir):
    """
    Plot magnetometer X-Y before and after calibration side by side.

    The raw plot should show an offset ellipse.
    The calibrated plot should show a circle centered at origin.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw data
    axes[0].scatter(mag_x_raw, mag_y_raw, s=5, alpha=0.5,
                    color='red', label='Raw data')
    axes[0].scatter(x_offset, y_offset, s=100, color='blue',
                    marker='+', zorder=5, label=f'Center ({x_offset:.4f}, {y_offset:.4f})')
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].axvline(0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Mag X (Gauss)')
    axes[0].set_ylabel('Mag Y (Gauss)')
    axes[0].set_title('Magnetometer X-Y Before Calibration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')

    # Calibrated data
    axes[1].scatter(mag_x_cal, mag_y_cal, s=5, alpha=0.5,
                    color='green', label='Calibrated data')
    axes[1].scatter(0, 0, s=100, color='blue', marker='+',
                    zorder=5, label='Center (0, 0)')
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].axvline(0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Mag X (Gauss)')
    axes[1].set_ylabel('Mag Y (Gauss)')
    axes[1].set_title('Magnetometer X-Y After Calibration')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    plt.suptitle('Hard-Iron and Soft-Iron Magnetometer Calibration',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'mag_calibration_xy.png', dpi=150)
    plt.close()
    print("Saved: mag_calibration_xy.png")


def plot_mag_timeseries(imu_df, mag_x_cal, mag_y_cal, output_dir):
    """
    Plot time series of magnetometer X and Y before and after calibration.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t = imu_df['time_sec']

    # X axis
    axes[0].plot(t, imu_df['mag_x'], color='red',
                 linewidth=0.8, label='Raw Mag X', alpha=0.7)
    axes[0].plot(t, mag_x_cal, color='green',
                 linewidth=0.8, label='Calibrated Mag X')
    axes[0].set_ylabel('Mag X (Gauss)')
    axes[0].set_title('Magnetometer X — Before and After Calibration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Y axis
    axes[1].plot(t, imu_df['mag_y'], color='red',
                 linewidth=0.8, label='Raw Mag Y', alpha=0.7)
    axes[1].plot(t, mag_y_cal, color='green',
                 linewidth=0.8, label='Calibrated Mag Y')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Mag Y (Gauss)')
    axes[1].set_title('Magnetometer Y — Before and After Calibration')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Magnetometer Time Series Before and After Calibration',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'mag_calibration_timeseries.png', dpi=150)
    plt.close()
    print("Saved: mag_calibration_timeseries.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':

    print("="*60)
    print("MAGNETOMETER CALIBRATION")
    print("="*60)

    # Load circles data for calibration
    print("\nLoading circles data...")
    imu_circles = read_imu_data(DATA_DIR / 'data_going_in_circles')

    mag_x = imu_circles['mag_x'].values
    mag_y = imu_circles['mag_y'].values

    # Compute calibration parameters
    x_offset, y_offset = compute_hard_iron_offset(mag_x, mag_y)
    x_scale, y_scale   = compute_soft_iron_scale(
        mag_x, mag_y, x_offset, y_offset
    )

    print(f"\nCalibration Parameters:")
    print(f"  Hard-iron offsets: x = {x_offset:.6f} Gauss")
    print(f"                     y = {y_offset:.6f} Gauss")
    print(f"  Soft-iron scales:  x = {x_scale:.6f}")
    print(f"                     y = {y_scale:.6f}")

    # Apply calibration to circles data
    mag_x_cal, mag_y_cal = apply_calibration(
        mag_x, mag_y, x_offset, y_offset, x_scale, y_scale
    )

    # Generate plots
    print("\nGenerating calibration plots...")
    plot_mag_xy(mag_x, mag_y, mag_x_cal, mag_y_cal,
                x_offset, y_offset, OUTPUT_DIR)
    plot_mag_timeseries(imu_circles, mag_x_cal, mag_y_cal, OUTPUT_DIR)

    # Save calibration parameters for use in subsequent analysis
    # These will be imported by the yaw estimation script
    np.save(OUTPUT_DIR / 'cal_params.npy', np.array([
        x_offset, y_offset, x_scale, y_scale
    ]))
    print("\nCalibration parameters saved to: cal_params.npy")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*60)
