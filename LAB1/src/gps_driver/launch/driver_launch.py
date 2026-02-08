#!/usr/bin/env python3
"""
Launch file for GPS Driver Node - EECE 5554 Lab 1

Usage:
    ros2 launch gps_driver driver_launch.py port:=/dev/ttyUSB0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Generate and return the launch description.
    
    This function is required by ROS2 - it looks for this exact name.
    """
    
    # Declare a launch argument called 'port'
    # Users can override this with port:=/dev/pts/X on command line
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyUSB0',
        description='Serial port path for GPS device'
    )
    
    # Get the value of the 'port' argument for use in the node
    port_config = LaunchConfiguration('port')
    
    # Define the GPS driver node
    gps_driver_node = Node(
        package='gps_driver',
        executable='driver',
        name='gps_driver_node',
        output='screen',
        arguments=['--port', port_config]
    )
    
    # Return all actions in the launch description
    return LaunchDescription([
        port_arg,
        gps_driver_node
    ])