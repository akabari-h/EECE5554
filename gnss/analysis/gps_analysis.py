#!/usr/bin/env python3
"""
GPS Data Analysis Script for EECE 5554 Lab 1

This script reads GPS data from ROS2 bag files and generates:
1. Scatterplots of Northing vs Easting
2. Histograms of position error
3. Altitude vs Time plots
4. Statistical analysis of GPS accuracy

Author: [Your Name]
Date: [Date]
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import ROS2 message deserialization utilities
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_bag_to_dataframe(bag_path):
    """
    Read a ROS2 bag file and extract GPS messages into a pandas DataFrame.
    
    This function directly reads the SQLite database (.db3 file) that ROS2
    uses to store bag data. We query the messages table and deserialize
    each message using ROS2's built-in tools.
    
    Args:
        bag_path (str): Path to the bag folder (e.g., 'stationary_open')
        
    Returns:
        pandas.DataFrame: DataFrame containing all GPS message fields
    """
    
    # Construct path to the .db3 file inside the bag folder
    bag_folder = Path(bag_path)
    db_files = list(bag_folder.glob('*.db3'))
    
    if not db_files:
        raise FileNotFoundError(f"No .db3 file found in {bag_path}")
    
    db_path = db_files[0]
    print(f"Reading bag file: {db_path}")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get topic information to find the GPS topic ID
    cursor.execute("SELECT id, name, type FROM topics")
    topics = cursor.fetchall()
    
    # Find the /gps topic
    gps_topic_id = None
    gps_msg_type = None
    for topic_id, topic_name, msg_type in topics:
        if topic_name == '/gps':
            gps_topic_id = topic_id
            gps_msg_type = msg_type
            break
    
    if gps_topic_id is None:
        raise ValueError("No /gps topic found in bag file")
    
    print(f"Found topic /gps with type: {gps_msg_type}")
    
    # Get the message class for deserialization
    # get_message() converts string like 'gps_msg/msg/GpsMsg' to the actual Python class
    msg_class = get_message(gps_msg_type)
    
    # Query all messages from the /gps topic
    cursor.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ?",
        (gps_topic_id,)
    )
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} GPS messages")
    
    # Extract data from each message into a list of dictionaries
    data_list = []
    for timestamp, data in rows:
        # Deserialize the binary data into a ROS2 message object
        msg = deserialize_message(data, msg_class)
        
        # Extract fields into a dictionary
        data_list.append({
            'timestamp': timestamp,  # Bag timestamp in nanoseconds
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude,
            'hdop': msg.hdop,
            'utm_easting': msg.utm_easting,
            'utm_northing': msg.utm_northing,
            'utc': msg.utc,
            'zone': msg.zone,
            'letter': msg.letter
        })
    
    conn.close()
    
    # Convert list of dictionaries to pandas DataFrame
    df = pd.DataFrame(data_list)
    
    # Convert timestamp from nanoseconds to seconds for easier plotting
    df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9
    
    return df


def plot_stationary_scatter(df, title, output_file):
    """
    Create a scatterplot of Northing vs Easting for stationary data.
    
    The lab requires subtracting the first value to scale the plot,
    which centers the data around (0, 0) and shows the spread in meters.
    
    Args:
        df (DataFrame): GPS data
        title (str): Plot title
        output_file (str): Filename to save the plot
    """
    
    # Subtract first value to center the data (as specified in lab)
    easting_centered = df['utm_easting'] - df['utm_easting'].iloc[0]
    northing_centered = df['utm_northing'] - df['utm_northing'].iloc[0]
    
    # Create the figure
    plt.figure(figsize=(10, 8))
    
    # Scatterplot with small markers to see density
    plt.scatter(easting_centered, northing_centered, s=10, alpha=0.6, c='blue')
    
    # Mark the first point (reference position)
    plt.scatter(0, 0, s=100, c='red', marker='x', label='Reference (First Point)')
    
    # Labels and title
    plt.xlabel('Easting (meters from start)', fontsize=12)
    plt.ylabel('Northing (meters from start)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Equal aspect ratio so distances look correct
    plt.axis('equal')
    
    # Save the figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved scatterplot: {output_file}")


def plot_error_histogram(df, known_easting, known_northing, title, output_file):
    """
    Create a histogram of position errors from a known reference position.
    
    Position error is calculated as the Euclidean distance from each
    measurement to the known true position.
    
    Args:
        df (DataFrame): GPS data
        known_easting (float): True easting coordinate from Google Earth
        known_northing (float): True northing coordinate from Google Earth
        title (str): Plot title
        output_file (str): Filename to save the plot
    """
    
    # Calculate Euclidean distance from each point to the known position
    # Error = sqrt((measured_easting - true_easting)^2 + (measured_northing - true_northing)^2)
    errors = np.sqrt(
        (df['utm_easting'] - known_easting)**2 + 
        (df['utm_northing'] - known_northing)**2
    )
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    
    # Plot histogram with automatic binning
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Add vertical line for mean error
    mean_error = errors.mean()
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                label=f'Mean Error: {mean_error:.2f} m')
    
    # Labels and title
    plt.xlabel('Position Error (meters)', fontsize=12)
    plt.ylabel('Frequency (count)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved histogram: {output_file}")
    
    # Return statistics for the report
    return {
        'mean_error': mean_error,
        'std_error': errors.std(),
        'max_error': errors.max(),
        'min_error': errors.min()
    }


def plot_altitude_vs_time(df, title, output_file):
    """
    Create a plot of altitude versus time.
    
    This shows how altitude measurements vary over the recording period,
    which can reveal GPS drift and atmospheric effects.
    
    Args:
        df (DataFrame): GPS data
        title (str): Plot title
        output_file (str): Filename to save the plot
    """
    
    plt.figure(figsize=(12, 5))
    
    # Plot altitude over time
    plt.plot(df['time_sec'], df['altitude'], 'b-', linewidth=1)
    
    # Add horizontal line for mean altitude
    mean_alt = df['altitude'].mean()
    plt.axhline(mean_alt, color='red', linestyle='--', linewidth=1,
                label=f'Mean: {mean_alt:.2f} m')
    
    # Labels and title
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Altitude (meters)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved altitude plot: {output_file}")


def plot_walking_with_line_fit(df, title, output_file):
    """
    Create a scatterplot for walking data with a line of best fit.
    
    The lab assumes you walked in a straight line, so we fit a linear
    regression and calculate the error from this ideal line.
    
    Args:
        df (DataFrame): GPS data
        title (str): Plot title
        output_file (str): Filename to save the plot
        
    Returns:
        dict: Statistics about the line fit error
    """
    
    # Center the data
    easting = df['utm_easting'] - df['utm_easting'].iloc[0]
    northing = df['utm_northing'] - df['utm_northing'].iloc[0]
    
    # Fit a line using numpy polyfit (degree 1 = linear)
    # This finds coefficients [slope, intercept] that minimize squared error
    coefficients = np.polyfit(easting, northing, 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    
    # Create the fitted line
    easting_line = np.linspace(easting.min(), easting.max(), 100)
    northing_line = slope * easting_line + intercept
    
    # Calculate perpendicular distance from each point to the line
    # Formula: |ax + by + c| / sqrt(a^2 + b^2) where line is ax + by + c = 0
    # Our line: y = mx + b  =>  mx - y + b = 0  =>  a=slope, b=-1, c=intercept
    distances = np.abs(slope * easting - northing + intercept) / np.sqrt(slope**2 + 1)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot the GPS points
    plt.scatter(easting, northing, s=10, alpha=0.6, c='blue', label='GPS Data')
    
    # Plot the fitted line
    plt.plot(easting_line, northing_line, 'r-', linewidth=2, 
             label=f'Best Fit Line (slope={slope:.4f})')
    
    # Mark start and end points
    plt.scatter(easting.iloc[0], northing.iloc[0], s=100, c='green', 
                marker='o', label='Start')
    plt.scatter(easting.iloc[-1], northing.iloc[-1], s=100, c='red', 
                marker='s', label='End')
    
    # Labels and title
    plt.xlabel('Easting (meters from start)', fontsize=12)
    plt.ylabel('Northing (meters from start)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save the figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved walking plot: {output_file}")
    
    return {
        'mean_deviation': distances.mean(),
        'std_deviation': distances.std(),
        'max_deviation': distances.max()
    }


def print_statistics(df, name):
    """
    Print summary statistics for a GPS dataset.
    
    Args:
        df (DataFrame): GPS data
        name (str): Dataset name for display
    """
    
    print(f"\n{'='*50}")
    print(f"Statistics for: {name}")
    print(f"{'='*50}")
    print(f"Number of points: {len(df)}")
    print(f"Duration: {df['time_sec'].iloc[-1]:.1f} seconds")
    print(f"\nLatitude:  Mean={df['latitude'].mean():.7f}°, Std={df['latitude'].std():.7f}°")
    print(f"Longitude: Mean={df['longitude'].mean():.7f}°, Std={df['longitude'].std():.7f}°")
    print(f"Altitude:  Mean={df['altitude'].mean():.2f}m, Std={df['altitude'].std():.2f}m")
    print(f"HDOP:      Mean={df['hdop'].mean():.2f}, Max={df['hdop'].max():.2f}")
    print(f"\nUTM Easting:  Mean={df['utm_easting'].mean():.2f}m, Std={df['utm_easting'].std():.2f}m")
    print(f"UTM Northing: Mean={df['utm_northing'].mean():.2f}m, Std={df['utm_northing'].std():.2f}m")


# ===== MAIN EXECUTION =====
if __name__ == '__main__':
    
    # Set paths to your bag files
    data_dir = Path.home() / 'EECE5554' / 'LAB1' / 'Data'
    output_dir = Path.home() / 'EECE5554' / 'LAB1' / 'Analysis' / 'plots'
    
    # Create output directory for plots
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # IMPORTANT: Set your known positions from Google Earth here!
    # These are the "true" positions where you collected stationary data
    # You need to convert lat/lon from Google Earth to UTM coordinates
    # ============================================================
    
    # Example placeholder values - REPLACE WITH YOUR ACTUAL COORDINATES!
    # You can use an online converter like: https://www.latlong.net/lat-long-utm.html
    
    OPEN_KNOWN_EASTING = 328313.84    # Replace with your actual value
    OPEN_KNOWN_NORTHING = 4690070.22  # Replace with your actual value
    
    OCCLUDED_KNOWN_EASTING = 328301.45    # Replace with your actual value
    OCCLUDED_KNOWN_NORTHING = 4690149.75  # Replace with your actual value
    
    # ============================================================
    # ANALYZE STATIONARY OPEN DATA
    # ============================================================
    print("\n" + "="*60)
    print("ANALYZING: Stationary Open Data")
    print("="*60)
    
    try:
        df_open = read_bag_to_dataframe(data_dir / 'stationary_open1_0')
        print_statistics(df_open, "Stationary Open")
        
        plot_stationary_scatter(
            df_open,
            'Stationary GPS Data - Open Area (Northing vs Easting)',
            output_dir / 'stationary_open_scatter.png'
        )
        
        error_stats_open = plot_error_histogram(
            df_open, OPEN_KNOWN_EASTING, OPEN_KNOWN_NORTHING,
            'Position Error Histogram - Open Area',
            output_dir / 'stationary_open_histogram.png'
        )
        print(f"\nError Statistics (Open):")
        print(f"  Mean Error: {error_stats_open['mean_error']:.2f} m")
        print(f"  Std Error:  {error_stats_open['std_error']:.2f} m")
        
        plot_altitude_vs_time(
            df_open,
            'Altitude vs Time - Open Area',
            output_dir / 'stationary_open_altitude.png'
        )
        
    except Exception as e:
        print(f"Error processing stationary_open: {e}")
    
    # ============================================================
    # ANALYZE STATIONARY OCCLUDED DATA
    # ============================================================
    print("\n" + "="*60)
    print("ANALYZING: Stationary Occluded Data")
    print("="*60)
    
    try:
        df_occluded = read_bag_to_dataframe(data_dir / 'stationary_occluded_0')
        print_statistics(df_occluded, "Stationary Occluded")
        
        plot_stationary_scatter(
            df_occluded,
            'Stationary GPS Data - Occluded Area (Northing vs Easting)',
            output_dir / 'stationary_occluded_scatter.png'
        )
        
        error_stats_occluded = plot_error_histogram(
            df_occluded, OCCLUDED_KNOWN_EASTING, OCCLUDED_KNOWN_NORTHING,
            'Position Error Histogram - Occluded Area',
            output_dir / 'stationary_occluded_histogram.png'
        )
        print(f"\nError Statistics (Occluded):")
        print(f"  Mean Error: {error_stats_occluded['mean_error']:.2f} m")
        print(f"  Std Error:  {error_stats_occluded['std_error']:.2f} m")
        
        plot_altitude_vs_time(
            df_occluded,
            'Altitude vs Time - Occluded Area',
            output_dir / 'stationary_occluded_altitude.png'
        )
        
    except Exception as e:
        print(f"Error processing stationary_occluded: {e}")
    
    # ============================================================
    # ANALYZE WALKING DATA
    # ============================================================
    print("\n" + "="*60)
    print("ANALYZING: Walking Data")
    print("="*60)
    
    try:
        df_walking = read_bag_to_dataframe(data_dir / 'walking_0')
        print_statistics(df_walking, "Walking")
        
        walking_stats = plot_walking_with_line_fit(
            df_walking,
            'Walking GPS Data - Northing vs Easting with Line Fit',
            output_dir / 'walking_scatter.png'
        )
        print(f"\nLine Fit Statistics (Walking):")
        print(f"  Mean Deviation from Line: {walking_stats['mean_deviation']:.2f} m")
        print(f"  Max Deviation from Line:  {walking_stats['max_deviation']:.2f} m")
        
        plot_altitude_vs_time(
            df_walking,
            'Altitude vs Time - Walking',
            output_dir / 'walking_altitude.png'
        )
        
    except Exception as e:
        print(f"Error processing walking: {e}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"Plots saved to: {output_dir}")
    print("="*60)
