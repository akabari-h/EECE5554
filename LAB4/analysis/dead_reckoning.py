#!/usr/bin/env python3
"""
Dead Reckoning Navigation for Lab 4.

Uses IMU velocity and heading to estimate vehicle trajectory
and compares with GPS ground truth.

Steps:
1. Compute omega*Xdot and compare to y_ddot_obs
2. Rotate forward velocity using heading to get (ve, vn)
3. Integrate (ve, vn) to get trajectory (xe, xn)
4. Compare with GPS trajectory
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from read_bags import read_imu_data, read_gps_data

OUTPUT_DIR = Path.home() / 'EECE5554' / 'LAB4' / 'analysis' / 'plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR   = Path.home() / 'EECE5554' / 'LAB4' / 'data'


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':

    print("="*60)
    print("DEAD RECKONING ANALYSIS")
    print("="*60)

    # ===== LOAD DATA =====
    print("\nLoading data...")
    imu = read_imu_data(DATA_DIR / 'data_driving')
    gps = read_gps_data(DATA_DIR / 'data_driving')

    t_imu = imu['time_sec'].values
    dt    = np.median(np.diff(t_imu))
    fs    = 1.0 / dt

    # Load saved arrays from previous analyses
    vel_corrected = np.load(OUTPUT_DIR / 'velocity_corrected.npy')
    yaw_cf        = np.load(OUTPUT_DIR / 'yaw_cf.npy')

    print(f"  IMU samples : {len(t_imu)}")
    print(f"  GPS samples : {len(gps)}")

    # ===== STEP 1: COMPARE omega*Xdot vs y_ddot_obs =====
    print("\nStep 1: Computing omega*Xdot vs y_ddot_obs...")

    # omega = gyro_z (yaw rate in rad/s)
    omega = imu['gyro_z'].values

    # Xdot = forward velocity
    x_dot = vel_corrected

    # omega * Xdot = centripetal acceleration term
    # This should equal the observed lateral acceleration y_ddot_obs
    omega_x_dot = omega * x_dot

    # Observed lateral acceleration (y axis of IMU)
    y_ddot_obs = imu['accel_y'].values

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t_imu, omega_x_dot, color='blue',
                 linewidth=0.8, label='ω·Ẋ (computed)')
    axes[0].plot(t_imu, y_ddot_obs,  color='red',
                 linewidth=0.8, alpha=0.7, label='ÿ_obs (measured)')
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].set_title('Comparison: ω·Ẋ vs Observed Lateral Acceleration ÿ_obs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Difference
    diff = omega_x_dot - y_ddot_obs
    axes[1].plot(t_imu, diff, color='green',
                 linewidth=0.8, label='Difference (ω·Ẋ - ÿ_obs)')
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Difference (m/s²)')
    axes[1].set_title('Residual: ω·Ẋ - ÿ_obs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Dead Reckoning: Centripetal Acceleration Analysis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'omega_xdot_vs_yddot.png', dpi=150)
    plt.close()
    print("  Saved: omega_xdot_vs_yddot.png")

    # Statistics for report
    corr = np.corrcoef(omega_x_dot, y_ddot_obs)[0, 1]
    rmse = np.sqrt(np.mean(diff**2))
    print(f"  Correlation between ω·Ẋ and ÿ_obs: {corr:.4f}")
    print(f"  RMSE of difference: {rmse:.4f} m/s²")

    # ===== STEP 2: ROTATE VELOCITY TO EASTING/NORTHING =====
    print("\nStep 2: Rotating velocity to Easting/Northing...")

    # Convert yaw from degrees to radians
    yaw_rad = np.radians(yaw_cf)

    # Rotate forward velocity vector using heading
    # ve = Vforward * sin(yaw)  — Easting component
    # vn = Vforward * cos(yaw)  — Northing component
    # Why sin/cos?
    # In NED frame, yaw=0 means pointing North.
    # North component = V*cos(yaw), East component = V*sin(yaw)
    ve = vel_corrected * np.sin(yaw_rad)
    vn = vel_corrected * np.cos(yaw_rad)

    print(f"  ve range: {ve.min():.2f} to {ve.max():.2f} m/s")
    print(f"  vn range: {vn.min():.2f} to {vn.max():.2f} m/s")

    # ===== STEP 3: INTEGRATE TO GET TRAJECTORY =====
    print("\nStep 3: Integrating velocity to get trajectory...")

    # Cumulative trapezoidal integration
    xe = np.zeros(len(t_imu))
    xn = np.zeros(len(t_imu))

    dt_arr = np.diff(t_imu)
    for i in range(1, len(t_imu)):
        xe[i] = xe[i-1] + (ve[i-1] + ve[i]) / 2.0 * dt_arr[i-1]
        xn[i] = xn[i-1] + (vn[i-1] + vn[i]) / 2.0 * dt_arr[i-1]

    print(f"  IMU trajectory range:")
    print(f"    Easting : {xe.min():.1f} to {xe.max():.1f} m")
    print(f"    Northing: {xn.min():.1f} to {xn.max():.1f} m")

    # ===== STEP 4: GPS TRAJECTORY =====
    print("\nStep 4: Computing GPS trajectory...")

    # Center GPS trajectory at origin
    gps_e = gps['utm_easting'].values  - gps['utm_easting'].values[0]
    gps_n = gps['utm_northing'].values - gps['utm_northing'].values[0]
    t_gps = gps['time_sec'].values

    print(f"  GPS trajectory range:")
    print(f"    Easting : {gps_e.min():.1f} to {gps_e.max():.1f} m")
    print(f"    Northing: {gps_n.min():.1f} to {gps_n.max():.1f} m")

    # ===== STEP 5: ALIGN STARTING HEADING =====
    print("\nStep 5: Aligning IMU and GPS trajectories...")

    # Both tracks start at (0,0) by construction.
    # Now align the initial heading direction so the first
    # straight segment of both tracks points in the same direction.

    # Find heading of first straight segment from GPS
    # Use first 10 GPS points where car is moving
    gps_vel_mag = np.sqrt(np.diff(gps_e)**2 + np.diff(gps_n)**2)
    moving_idx  = np.where(gps_vel_mag > 0.5)[0]

    if len(moving_idx) > 5:
        seg_end   = min(moving_idx[0] + 10, len(gps_e) - 1)
        gps_heading_start = np.arctan2(
            gps_e[seg_end] - gps_e[moving_idx[0]],
            gps_n[seg_end] - gps_n[moving_idx[0]]
        )
    else:
        gps_heading_start = 0.0

    # Find heading of first straight segment from IMU
    imu_moving = np.where(np.abs(vel_corrected) > 0.5)[0]
    if len(imu_moving) > 200:
        seg_end_imu = min(imu_moving[0] + 200, len(xe) - 1)
        imu_heading_start = np.arctan2(
            xe[seg_end_imu] - xe[imu_moving[0]],
            xn[seg_end_imu] - xn[imu_moving[0]]
        )
    else:
        imu_heading_start = 0.0

    # Rotation angle to align
    rotation_angle = gps_heading_start - imu_heading_start
    print(f"  GPS initial heading  : {np.degrees(gps_heading_start):.2f} deg")
    print(f"  IMU initial heading  : {np.degrees(imu_heading_start):.2f} deg")
    print(f"  Rotation applied     : {np.degrees(rotation_angle):.2f} deg")

    # Apply rotation to IMU trajectory
    cos_r = np.cos(rotation_angle)
    sin_r = np.sin(rotation_angle)
    xe_rot = xe * cos_r - xn * sin_r
    xn_rot = xe * sin_r + xn * cos_r

    # ===== STEP 6: SCALE FACTOR =====
    # Compute scale factor to match total path lengths
    gps_total_dist = np.sum(np.sqrt(np.diff(gps_e)**2 +
                                    np.diff(gps_n)**2))
    imu_total_dist = np.sum(np.sqrt(np.diff(xe_rot)**2 +
                                    np.diff(xn_rot)**2))

    if imu_total_dist > 0:
        scale_factor = gps_total_dist / imu_total_dist
    else:
        scale_factor = 1.0

    print(f"  GPS total distance: {gps_total_dist:.1f} m")
    print(f"  IMU total distance: {imu_total_dist:.1f} m")
    print(f"  Scale factor      : {scale_factor:.4f}")

    xe_scaled = xe_rot * scale_factor
    xn_scaled = xn_rot * scale_factor

    # ===== PLOT: TRAJECTORY COMPARISON =====
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Without scaling
    axes[0].plot(gps_e,    gps_n,    color='green',
                 linewidth=1.5, label='GPS Track')
    axes[0].plot(xe_rot,   xn_rot,   color='blue',
                 linewidth=1.0, alpha=0.8, label='IMU Dead Reckoning')
    axes[0].scatter(0, 0, color='red', s=100,
                    zorder=5, label='Start')
    axes[0].set_xlabel('Easting (m)')
    axes[0].set_ylabel('Northing (m)')
    axes[0].set_title('Trajectory: GPS vs IMU (No Scaling)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')

    # With scaling
    axes[1].plot(gps_e,    gps_n,    color='green',
                 linewidth=1.5, label='GPS Track')
    axes[1].plot(xe_scaled, xn_scaled, color='blue',
                 linewidth=1.0, alpha=0.8,
                 label=f'IMU Dead Reckoning (scale={scale_factor:.3f})')
    axes[1].scatter(0, 0, color='red', s=100,
                    zorder=5, label='Start')
    axes[1].set_xlabel('Easting (m)')
    axes[1].set_ylabel('Northing (m)')
    axes[1].set_title('Trajectory: GPS vs IMU (With Scaling)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')

    plt.suptitle('Dead Reckoning: IMU vs GPS Trajectory',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'trajectory_comparison.png', dpi=150)
    plt.close()
    print("\nSaved: trajectory_comparison.png")

    # ===== COMPUTE POSITION ERROR OVER TIME =====
    # Interpolate GPS to IMU timestamps for error computation
    interp_e = interp1d(t_gps, gps_e, bounds_error=False,
                        fill_value='extrapolate')
    interp_n = interp1d(t_gps, gps_n, bounds_error=False,
                        fill_value='extrapolate')

    gps_e_imu_time = interp_e(t_imu)
    gps_n_imu_time = interp_n(t_imu)

    pos_error = np.sqrt((xe_scaled - gps_e_imu_time)**2 +
                        (xn_scaled - gps_n_imu_time)**2)

    # Find how long IMU stays within 2m of GPS
    within_2m = np.where(pos_error < 2.0)[0]
    if len(within_2m) > 0:
        last_within_2m = t_imu[within_2m[-1]]
        print(f"\n  IMU stays within 2m of GPS for: "
              f"{last_within_2m:.1f} seconds")
    else:
        print("\n  IMU never within 2m of GPS")

    # Plot position error over time
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_imu, pos_error, color='red', linewidth=0.8)
    ax.axhline(2.0, color='blue', linestyle='--',
               linewidth=1.5, label='2m threshold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Dead Reckoning Position Error vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'position_error.png', dpi=150)
    plt.close()
    print("  Saved: position_error.png")

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*60}")
    print("DEAD RECKONING SUMMARY")
    print(f"{'='*60}")
    print(f"  Correlation ω·Ẋ vs ÿ_obs : {corr:.4f}")
    print(f"  RMSE ω·Ẋ vs ÿ_obs        : {rmse:.4f} m/s²")
    print(f"  GPS total distance        : {gps_total_dist:.1f} m")
    print(f"  IMU total distance        : {imu_total_dist:.1f} m")
    print(f"  Scale factor applied      : {scale_factor:.4f}")
    print(f"  Mean position error       : {pos_error.mean():.2f} m")
    print(f"  Max position error        : {pos_error.max():.2f} m")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*60)