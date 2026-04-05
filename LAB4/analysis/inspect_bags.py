#!/usr/bin/env python3
"""
Inspect MCAP bag files to understand message structure
before writing analysis code.
"""

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path

def inspect_bag(bag_path):
    print(f"\n{'='*60}")
    print(f"Inspecting: {bag_path}")
    print(f"{'='*60}")

    typestore = get_typestore(Stores.LATEST)

    with Reader(bag_path) as reader:
        # Print all connections (topics)
        print("\nTopics found:")
        for conn in reader.connections:
            print(f"  Topic: {conn.topic}")
            print(f"  Type : {conn.msgtype}")
            print(f"  Count: {conn.msgcount}")

        # Read first IMU message and print all fields
        print("\nFirst /imu message fields:")
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == '/imu':
                try:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    print(f"  Available attributes: {[a for a in dir(msg) if not a.startswith('_')]}")
                    # Try to print header
                    if hasattr(msg, 'header'):
                        print(f"  header.stamp.sec    : {msg.header.stamp.sec}")
                        print(f"  header.stamp.nanosec: {msg.header.stamp.nanosec}")
                        print(f"  header.frame_id     : {msg.header.frame_id}")
                    # Try common IMU field names
                    for attr in ['imu', 'linear_acceleration', 'angular_velocity',
                                 'accel', 'gyro', 'mag_field', 'magnetometer']:
                        if hasattr(msg, attr):
                            print(f"  {attr}: {getattr(msg, attr)}")
                except Exception as e:
                    print(f"  Error deserializing: {e}")
                break

        # Read first GPS message and print all fields
        print("\nFirst /gps message fields:")
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == '/gps':
                try:
                    msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                    print(f"  Available attributes: {[a for a in dir(msg) if not a.startswith('_')]}")
                    for attr in dir(msg):
                        if not attr.startswith('_'):
                            try:
                                print(f"  {attr}: {getattr(msg, attr)}")
                            except:
                                pass
                except Exception as e:
                    print(f"  Error deserializing: {e}")
                break

if __name__ == '__main__':
    data_dir = Path.home() / 'EECE5554' / 'LAB4' / 'data'

    inspect_bag(data_dir / 'data_going_in_circles')
    inspect_bag(data_dir / 'data_driving')
