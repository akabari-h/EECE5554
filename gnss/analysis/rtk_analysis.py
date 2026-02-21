#!/usr/bin/env python3

# ===== IMPORTS =====
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ROS2 deserialization tools
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_bag_to_dataframe(bag_path, topic_name):

    bag_folder = Path(bag_path)

    # Find the .db3 file inside the bag folder
    # glob('*.db3') searches for any file ending in .db3
    db_files = list(bag_folder.glob('*.db3'))

    if not db_files:
        raise FileNotFoundError(f"No .db3 file found in {bag_path}")

    db_path = db_files[0]
    print(f"Reading: {db_path}")

    # Connect to the SQLite database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Query the topics table to find the topic ID and message type
    # Each topic has a unique ID used to link messages to topics
    cursor.execute("SELECT id, name, type FROM topics")
    topics = cursor.fetchall()

    target_topic_id = None
    target_msg_type = None

    for topic_id, name, msg_type in topics:
        if name == topic_name:
            target_topic_id = topic_id
            target_msg_type = msg_type
            break

    if target_topic_id is None:
        raise ValueError(f"Topic '{topic_name}' not found in bag. "
                         f"Available topics: {[t[1] for t in topics]}")

    print(f"Found topic '{topic_name}' with type: {target_msg_type}")

    # get_message() converts a string like 'gps_msg/msg/Customrtk'
    # into the actual Python class — needed for deserialization
    msg_class = get_message(target_msg_type)

    # Query all messages for this topic
    cursor.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ?",
        (target_topic_id,)
    )
    rows = cursor.fetchall()
    print(f"Found {len(rows)} messages")

    # Deserialize each message and extract fields into a list of dicts
    data_list = []
    for timestamp, data in rows:
        # deserialize_message() reconstructs the ROS2 message object
        # from raw bytes using the message class definition
        msg = deserialize_message(data, msg_class)

        entry = {
            'timestamp': timestamp,       # Bag timestamp in nanoseconds
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude,
            'utm_easting': msg.utm_easting,
            'utm_northing': msg.utm_northing,
            'utc': msg.utc,
            'zone': msg.zone,
            'letter': msg.letter,
        }

        # Add RTK-specific fields only if they exist in the message
        # hasattr() checks if an attribute exists on an object
        # This makes the function work for BOTH /gps and /rtk topics
        if hasattr(msg, 'fix_quality'):
            entry['fix_quality'] = msg.fix_quality
        if hasattr(msg, 'hdop'):
            entry['hdop'] = msg.hdop

        data_list.append(entry)

    conn.close()

    # Convert list of dicts to a DataFrame
    df = pd.DataFrame(data_list)

    # Create a time column in seconds, starting from 0
    # This makes time-based plots easier to read
    df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9

    return df

def plot_scatter(df, title, output_file):

    # Center data by subtracting first reading
    easting = df['utm_easting'] - df['utm_easting'].iloc[0]
    northing = df['utm_northing'] - df['utm_northing'].iloc[0]

    plt.figure(figsize=(8, 7))

    # s=10 sets marker size, alpha=0.6 adds transparency to see density
    plt.scatter(easting, northing, s=10, alpha=0.6, color='blue', label='GPS Readings')

    # Mark the reference/first point
    plt.scatter(0, 0, s=100, color='red', marker='x', zorder=5, label='Reference Point')

    plt.xlabel('Easting (meters from start)', fontsize=12)
    plt.ylabel('Northing (meters from start)', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Equal axes so 1 meter looks the same in both directions
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_error_histogram(df, known_easting, known_northing, title, output_file):

    # Calculate Euclidean distance from each reading to known position
    errors = np.sqrt(
        (df['utm_easting'] - known_easting) ** 2 +
        (df['utm_northing'] - known_northing) ** 2
    )

    plt.figure(figsize=(9, 5))

    # bins=30 divides the error range into 30 bars
    plt.hist(errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

    # Add vertical line for mean error
    mean_err = errors.mean()
    plt.axvline(mean_err, color='red', linestyle='--', linewidth=2,
                label=f'Mean Error: {mean_err:.3f} m')

    plt.xlabel('Position Error (meters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")

    return {
        'mean': mean_err,
        'std': errors.std(),
        'max': errors.max(),
        'min': errors.min()
    }


def plot_altitude_vs_time(df, title, output_file):

    plt.figure(figsize=(11, 5))

    plt.plot(df['time_sec'], df['altitude'], color='blue', linewidth=1)

    # Add mean line for reference
    mean_alt = df['altitude'].mean()
    plt.axhline(mean_alt, color='red', linestyle='--', linewidth=1.5,
                label=f'Mean: {mean_alt:.2f} m')

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Altitude (meters)', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_fix_quality(df, title, output_file):

    if 'fix_quality' not in df.columns:
        print("Skipping fix quality plot - no fix_quality column")
        return

    plt.figure(figsize=(11, 4))

    # step-wise plot suits discrete values like fix quality
    plt.step(df['time_sec'], df['fix_quality'], color='green', linewidth=1.2, where='post')

    # Add horizontal reference lines for each fix type
    plt.axhline(4, color='blue', linestyle=':', alpha=0.6, label='RTK Fixed (4)')
    plt.axhline(5, color='orange', linestyle=':', alpha=0.6, label='RTK Float (5)')
    plt.axhline(1, color='red', linestyle=':', alpha=0.6, label='Standalone GPS (1)')

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Fix Quality', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")

def print_statistics(df, name):
  
    print(f"\n{'='*55}")
    print(f"Statistics: {name}")
    print(f"{'='*55}")
    print(f"  Total messages  : {len(df)}")
    print(f"  Duration        : {df['time_sec'].iloc[-1]:.1f} seconds")
    print(f"  Latitude  Mean  : {df['latitude'].mean():.7f} deg")
    print(f"  Longitude Mean  : {df['longitude'].mean():.7f} deg")
    print(f"  Altitude  Mean  : {df['altitude'].mean():.3f} m")
    print(f"  Altitude  Std   : {df['altitude'].std():.3f} m")
    print(f"  UTM Easting Std : {df['utm_easting'].std():.3f} m")
    print(f"  UTM Northing Std: {df['utm_northing'].std():.3f} m")

    if 'hdop' in df.columns:
        print(f"  HDOP Mean       : {df['hdop'].mean():.3f}")
    if 'fix_quality' in df.columns:
        counts = df['fix_quality'].value_counts().to_dict()
        print(f"  Fix Quality     : {counts}")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == '__main__':

    # ===== PATHS =====
    data_dir   = Path.home() / 'EECE5554' / 'gnss' / 'data'
    output_dir = Path.home() / 'EECE5554' / 'gnss' / 'analysis' / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== KNOWN REFERENCE POSITIONS FROM LAB2 README =====
    # These are the true positions (from README.md) converted to UTM
    # We use utm library to convert lat/lon to easting/northing
    import utm

    # Open RTK reference: 42.3382544, -71.0865255
    open_ref = utm.from_latlon(42.3382544, -71.0865255)
    OPEN_KNOWN_EASTING  = open_ref[0]
    OPEN_KNOWN_NORTHING = open_ref[1]

    # Occluded RTK reference: 42.3372702, -71.0869305
    occ_ref = utm.from_latlon(42.3372702, -71.0869305)
    OCC_KNOWN_EASTING  = occ_ref[0]
    OCC_KNOWN_NORTHING = occ_ref[1]

    print(f"Open Reference    -> E: {OPEN_KNOWN_EASTING:.2f}, N: {OPEN_KNOWN_NORTHING:.2f}")
    print(f"Occluded Reference-> E: {OCC_KNOWN_EASTING:.2f}, N: {OCC_KNOWN_NORTHING:.2f}")

    # ============================================================
    # ANALYZE OPEN RTK DATA
    # ============================================================
    print("\n" + "="*55)
    print("ANALYZING: Open RTK Data")
    print("="*55)

    df_open_rtk = read_bag_to_dataframe(data_dir / 'openRTK', '/rtk')
    print_statistics(df_open_rtk, "Open RTK")

    plot_scatter(
        df_open_rtk,
        'RTK GPS - Open Area: Northing vs Easting',
        output_dir / 'rtk_open_scatter.png'
    )

    open_rtk_errors = plot_error_histogram(
        df_open_rtk, OPEN_KNOWN_EASTING, OPEN_KNOWN_NORTHING,
        'RTK GPS - Open Area: Position Error Histogram',
        output_dir / 'rtk_open_histogram.png'
    )
    print(f"\n  Error Stats (Open RTK):")
    print(f"    Mean : {open_rtk_errors['mean']:.3f} m")
    print(f"    Std  : {open_rtk_errors['std']:.3f} m")
    print(f"    Max  : {open_rtk_errors['max']:.3f} m")

    plot_altitude_vs_time(
        df_open_rtk,
        'RTK GPS - Open Area: Altitude vs Time',
        output_dir / 'rtk_open_altitude.png'
    )

    plot_fix_quality(
        df_open_rtk,
        'RTK GPS - Open Area: Fix Quality vs Time',
        output_dir / 'rtk_open_fixquality.png'
    )

    # ============================================================
    # ANALYZE OCCLUDED RTK DATA
    # ============================================================
    print("\n" + "="*55)
    print("ANALYZING: Occluded RTK Data")
    print("="*55)

    df_occ_rtk = read_bag_to_dataframe(data_dir / 'occludedRTK', '/rtk')
    print_statistics(df_occ_rtk, "Occluded RTK")

    plot_scatter(
        df_occ_rtk,
        'RTK GPS - Occluded Area: Northing vs Easting',
        output_dir / 'rtk_occluded_scatter.png'
    )

    occ_rtk_errors = plot_error_histogram(
        df_occ_rtk, OCC_KNOWN_EASTING, OCC_KNOWN_NORTHING,
        'RTK GPS - Occluded Area: Position Error Histogram',
        output_dir / 'rtk_occluded_histogram.png'
    )
    print(f"\n  Error Stats (Occluded RTK):")
    print(f"    Mean : {occ_rtk_errors['mean']:.3f} m")
    print(f"    Std  : {occ_rtk_errors['std']:.3f} m")
    print(f"    Max  : {occ_rtk_errors['max']:.3f} m")

    plot_altitude_vs_time(
        df_occ_rtk,
        'RTK GPS - Occluded Area: Altitude vs Time',
        output_dir / 'rtk_occluded_altitude.png'
    )

    plot_fix_quality(
        df_occ_rtk,
        'RTK GPS - Occluded Area: Fix Quality vs Time',
        output_dir / 'rtk_occluded_fixquality.png'
    )

    # ============================================================
    # ANALYZE WALKING RTK DATA
    # ============================================================
    print("\n" + "="*55)
    print("ANALYZING: Walking RTK Data")
    print("="*55)

    df_walk_rtk = read_bag_to_dataframe(data_dir / 'walkingRTK', '/rtk')
    print_statistics(df_walk_rtk, "Walking RTK")

    plot_scatter(
        df_walk_rtk,
        'RTK GPS - Walking: Northing vs Easting',
        output_dir / 'rtk_walking_scatter.png'
    )

    plot_altitude_vs_time(
        df_walk_rtk,
        'RTK GPS - Walking: Altitude vs Time',
        output_dir / 'rtk_walking_altitude.png'
    )

    plot_fix_quality(
        df_walk_rtk,
        'RTK GPS - Walking: Fix Quality vs Time',
        output_dir / 'rtk_walking_fixquality.png'
    )

    print("\n" + "="*55)
    print("ANALYSIS COMPLETE!")
    print(f"Plots saved to: {output_dir}")
    print("="*55)