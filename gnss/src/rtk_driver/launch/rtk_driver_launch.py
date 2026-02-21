#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Generate and return the launch description for the RTK driver.
    """
    
    # Declare the port argument (can be overridden at launch time)
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyUSB0',
        description='Serial port path for RTK GPS device'
    )
    
    # Get the port configuration value
    port_config = LaunchConfiguration('port')
    
    # Define the RTK driver node
    rtk_driver_node = Node(
        package='rtk_driver',
        executable='rtk_driver',
        name='rtk_driver_node',
        output='screen',
        arguments=['--port', port_config]
    )
    
    # Return the launch description
    return LaunchDescription([
        port_arg,
        rtk_driver_node
    ])
