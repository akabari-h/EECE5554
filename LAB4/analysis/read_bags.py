#!/usr/bin/env python3
"""
MCAP Bag Reader for Lab 4.
Reads lab4_pkg IMU and GPS messages from MCAP files.
Uses rosbags with custom type registration from embedded schema.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


# ============================================================
# DEFINE MESSAGE TYPES FROM SCHEMA
# ============================================================

# GPS message definition exactly as extracted from MCAP
GPS_MSG_DEF = """
std_msgs/Header header
float64 latitude
float64 longitude
float64 altitude
float64 hdop
float64 utm_easting
float64 utm_northing
float64 utc
string zone
string letter
"""

# IMU message definition exactly as extracted from MCAP
IMU_MSG_DEF = """
std_msgs/Header header
sensor_msgs/Imu imu
sensor_msgs/MagneticField mag_field
string raw
"""


def get_custom_typestore():
    """
    Create a typestore with custom lab4_pkg message types registered.

    Why do we need this?
    The rosbags library needs to know the structure of custom message
    types to deserialize them. The standard typestore only knows about
    standard ROS2 types. We register our custom types on top of that.
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Register custom message types
    add_types = {}
    add_types.update(
        get_types_from_msg(GPS_MSG_DEF, 'lab4_pkg/msg/GpsMsg')
    )
    add_types.update(
        get_types_from_msg(IMU_MSG_DEF, 'lab4_pkg/msg/IMUmsg')
    )
    typestore.register(add_types)

    return typestore


def read_imu_data(bag_path):
    """
    Read IMU messages from MCAP bag file.

    Returns DataFrame with all IMU fields.
    Magnetic field converted from Tesla to Gauss (× 1e4).
    Quaternion converted back to Euler angles for analysis.
    """
    typestore = get_custom_typestore()
    bag_path  = Path(bag_path)

    timestamps = []
    gyro_x, gyro_y, gyro_z     = [], [], []
    accel_x, accel_y, accel_z  = [], [], []
    mag_x, mag_y, mag_z        = [], [], []
    qx, qy, qz, qw             = [], [], [], []

    print(f"Reading IMU from: {bag_path.name}")

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == '/imu']
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)

            timestamps.append(timestamp)

            # Angular velocity (rad/s)
            gyro_x.append(msg.imu.angular_velocity.x)
            gyro_y.append(msg.imu.angular_velocity.y)
            gyro_z.append(msg.imu.angular_velocity.z)

            # Linear acceleration (m/s²)
            accel_x.append(msg.imu.linear_acceleration.x)
            accel_y.append(msg.imu.linear_acceleration.y)
            accel_z.append(msg.imu.linear_acceleration.z)

            # Magnetic field: Tesla → Gauss (× 1e4)
            mag_x.append(msg.mag_field.magnetic_field.x * 1e4)
            mag_y.append(msg.mag_field.magnetic_field.y * 1e4)
            mag_z.append(msg.mag_field.magnetic_field.z * 1e4)

            # Quaternion orientation
            qx.append(msg.imu.orientation.x)
            qy.append(msg.imu.orientation.y)
            qz.append(msg.imu.orientation.z)
            qw.append(msg.imu.orientation.w)

    print(f"  Loaded {len(timestamps)} IMU messages")

    df = pd.DataFrame({
        'timestamp_ns': timestamps,
        'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
        'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
        'mag_x': mag_x, 'mag_y': mag_y, 'mag_z': mag_z,
        'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
    })

    # Time in seconds starting from 0
    df['time_sec'] = (df['timestamp_ns'] - df['timestamp_ns'].iloc[0]) / 1e9

    # Convert quaternion to Euler angles
    df['roll'], df['pitch'], df['yaw'] = zip(
        *df.apply(lambda r: quaternion_to_euler(r.qw, r.qx, r.qy, r.qz), axis=1)
    )
    # Convert to degrees
    df['roll']  = np.degrees(df['roll'])
    df['pitch'] = np.degrees(df['pitch'])
    df['yaw']   = np.degrees(df['yaw'])

    # IMU sample rate
    dt_median = np.median(np.diff(df['time_sec'].values))
    df.attrs['dt'] = dt_median
    df.attrs['fs'] = 1.0 / dt_median

    print(f"  Duration  : {df['time_sec'].iloc[-1]:.1f} seconds")
    print(f"  Sample rate: {df.attrs['fs']:.2f} Hz")

    return df


def read_gps_data(bag_path):
    """
    Read GPS messages from MCAP bag file.

    Returns DataFrame with all GPS fields.
    """
    typestore = get_custom_typestore()
    bag_path  = Path(bag_path)

    timestamps = []
    latitudes, longitudes = [], []
    altitudes, hdops      = [], []
    eastings, northings   = [], []
    utcs                  = []

    print(f"Reading GPS from: {bag_path.name}")

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == '/gps']
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)

            timestamps.append(timestamp)
            latitudes.append(msg.latitude)
            longitudes.append(msg.longitude)
            altitudes.append(msg.altitude)
            hdops.append(msg.hdop)
            eastings.append(msg.utm_easting)
            northings.append(msg.utm_northing)
            utcs.append(msg.utc)

    print(f"  Loaded {len(timestamps)} GPS messages")

    df = pd.DataFrame({
        'timestamp_ns': timestamps,
        'latitude':  latitudes,
        'longitude': longitudes,
        'altitude':  altitudes,
        'hdop':      hdops,
        'utm_easting':  eastings,
        'utm_northing': northings,
        'utc':       utcs,
    })

    # Time in seconds starting from 0
    df['time_sec'] = (df['timestamp_ns'] - df['timestamp_ns'].iloc[0]) / 1e9

    print(f"  Duration  : {df['time_sec'].iloc[-1]:.1f} seconds")

    return df


def quaternion_to_euler(w, x, y, z):
    """Convert quaternion to roll, pitch, yaw in radians."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll      = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw       = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ============================================================
# TEST
# ============================================================
if __name__ == '__main__':
    data_dir = Path.home() / 'EECE5554' / 'LAB4' / 'data'

    print("="*60)
    print("Testing bag reader")
    print("="*60)

    # Test circles bag
    imu_circles = read_imu_data(data_dir / 'data_going_in_circles')
    gps_circles  = read_gps_data(data_dir / 'data_going_in_circles')

    print(f"\nCircles IMU — mag_x range: {imu_circles['mag_x'].min():.4f} to {imu_circles['mag_x'].max():.4f} Gauss")
    print(f"Circles GPS — lat range  : {gps_circles['latitude'].min():.6f} to {gps_circles['latitude'].max():.6f}")

    # Test driving bag
    imu_driving = read_imu_data(data_dir / 'data_driving')
    gps_driving  = read_gps_data(data_dir / 'data_driving')

    print(f"\nDriving IMU — gyro_x mean: {imu_driving['gyro_x'].mean():.6f} rad/s")
    print(f"Driving GPS — UTM easting range: {gps_driving['utm_easting'].min():.2f} to {gps_driving['utm_easting'].max():.2f}")

    print("\nAll bags read successfully!")
