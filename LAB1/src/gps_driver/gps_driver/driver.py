#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPS Driver for EECE 5554 Lab 1
Reads GPGGA sentences from serial port, parses them, and publishes to ROS2.

Author: [Your Name]
Date: [Date]
"""

# ===== IMPORTS =====
import rclpy                          # ROS2 Python client library
from rclpy.node import Node           # Base class for ROS2 nodes
import serial                         # PySerial for reading from GPS device
import utm                            # UTM coordinate conversion library
import argparse                       # For parsing command-line arguments

# Import our custom message type
from gps_msg.msg import GpsMsg

# Import Header message for timestamps
from std_msgs.msg import Header


class GPSDriverNode(Node):
    """
    ROS2 Node that reads GPS data from a serial port and publishes parsed data.
    
    This class inherits from rclpy.node.Node, which provides all the ROS2
    functionality like creating publishers, timers, and logging.
    
    Why use a class?
    - Encapsulation: All GPS-related data and functions are grouped together
    - State management: The serial port connection persists across callbacks
    - ROS2 pattern: Node classes are the standard way to write ROS2 Python nodes
    """
    
    def __init__(self, port_path):
        """
        Initialize the GPS driver node.
        
        Args:
            port_path (str): Path to serial device (e.g., '/dev/ttyUSB0' or '/dev/pts/6')
        
        The __init__ method is Python's constructor - it runs once when we create
        a GPSDriverNode object. We use it to set up everything the node needs.
        """
        # Call parent class constructor with our node name
        # 'gps_driver_node' will appear in ros2 node list
        super().__init__('gps_driver_node')
        
        # Store the port path as an instance variable (self.xxx)
        # Instance variables persist for the lifetime of the object
        self.port_path = port_path
        
        # ===== CREATE THE PUBLISHER =====
        # create_publisher(msg_type, topic_name, queue_size)
        # - GpsMsg: The message type we'll publish
        # - '/gps': Topic name (must match what lab specifies)
        # - 10: Queue size (how many messages to buffer if subscribers are slow)
        self.publisher = self.create_publisher(GpsMsg, '/gps', 10)
        
        # Log that we're starting (appears in terminal)
        # get_logger() returns this node's logger for printing messages
        self.get_logger().info(f'GPS Driver starting on port: {port_path}')
        
        # ===== OPEN SERIAL CONNECTION =====
        # try/except handles errors gracefully if the port doesn't exist
        try:
            # serial.Serial() opens a connection to the serial device
            # - port: Device path
            # - baudrate: 4800 is standard for GPS NMEA output
            # - timeout: Seconds to wait for data before returning (prevents infinite blocking)
            self.serial_port = serial.Serial(port_path, baudrate=4800, timeout=1.0)
            self.get_logger().info('Serial port opened successfully')
        except serial.SerialException as e:
            # If we can't open the port, log error and raise exception to stop the node
            self.get_logger().error(f'Failed to open serial port: {e}')
            raise e
        
        # ===== CREATE A TIMER FOR READING DATA =====
        # create_timer(period_seconds, callback_function)
        # This calls self.timer_callback every 0.1 seconds (10 Hz)
        # Why use a timer instead of a while loop?
        # - Allows ROS2 to handle other callbacks (like shutdown signals)
        # - More responsive and doesn't block the ROS2 executor
        self.timer = self.create_timer(0.1, self.timer_callback)
    
    def timer_callback(self):
        """
        Called periodically by the timer to read and process GPS data.
        
        This is a callback function - we don't call it directly, ROS2's
        executor calls it automatically based on the timer we created.
        """
        # Read one line from serial port
        # readline() returns bytes, decode() converts to string
        # errors='ignore' skips any corrupted characters
        try:
            line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
        except Exception as e:
            self.get_logger().warn(f'Serial read error: {e}')
            return
        
        # Only process GPGGA sentences (or GNGGA for multi-constellation receivers)
        if not (line.startswith('$GPGGA') or line.startswith('$GNGGA')):
            return  # Ignore other sentence types
        
        # Parse the GPGGA sentence
        parsed_data = self.parse_gpgga(line)
        
        # If parsing failed (invalid data), skip this reading
        if parsed_data is None:
            self.get_logger().debug(f'Failed to parse: {line}')
            return
        
        # Create and publish the ROS message
        self.publish_gps_message(parsed_data)
    
    def parse_gpgga(self, sentence):
        """
        Parse a GPGGA NMEA sentence and extract all relevant fields.
        
        Args:
            sentence (str): Raw GPGGA sentence string
            
        Returns:
            dict: Parsed data dictionary, or None if parsing fails
            
        This is a method (function inside a class). We use 'self' as the first
        parameter to access other methods and variables of this object.
        """
        # Split sentence by commas into a list of fields
        # Example: "$GPGGA,181908.00,..." becomes ["$GPGGA", "181908.00", ...]
        fields = sentence.split(',')
        
        # Validate: GPGGA should have at least 15 fields
        if len(fields) < 15:
            return None
        
        # Check fix quality (field index 6)
        # '0' or empty means no GPS fix - can't get valid position
        fix_quality = fields[6]
        if fix_quality == '' or fix_quality == '0':
            return None
        
        # ===== PARSE UTC TIME (Field 1) =====
        utc_raw = fields[1]
        if utc_raw == '':
            return None
        
        # Extract hours, minutes, seconds from HHMMSS.ss format
        # String slicing: utc_raw[0:2] gets characters at index 0 and 1
        hours = int(utc_raw[0:2])
        minutes = int(utc_raw[2:4])
        seconds = float(utc_raw[4:])  # [4:] means "from index 4 to the end"
        utc_time = hours * 3600 + minutes * 60 + seconds
        
        # ===== PARSE LATITUDE (Fields 2-3) =====
        lat_raw = fields[2]
        lat_dir = fields[3]
        
        if lat_raw == '' or lat_dir == '':
            return None
        
        # Convert DDMM.MMMMM to decimal degrees
        # [:2] gets first 2 characters (degrees), [2:] gets the rest (minutes)
        lat_degrees = float(lat_raw[:2])
        lat_minutes = float(lat_raw[2:])
        latitude = lat_degrees + (lat_minutes / 60.0)
        
        # Apply hemisphere sign (South is negative)
        if lat_dir == 'S':
            latitude = -latitude
        
        # ===== PARSE LONGITUDE (Fields 4-5) =====
        lon_raw = fields[4]
        lon_dir = fields[5]
        
        if lon_raw == '' or lon_dir == '':
            return None
        
        # Longitude uses 3 digits for degrees (range 0-180)
        lon_degrees = float(lon_raw[:3])
        lon_minutes = float(lon_raw[3:])
        longitude = lon_degrees + (lon_minutes / 60.0)
        
        # West is negative
        if lon_dir == 'W':
            longitude = -longitude
        
        # ===== PARSE REMAINING FIELDS =====
        # The 'if fields[x] else default' pattern handles empty fields
        num_satellites = int(fields[7]) if fields[7] else 0
        hdop = float(fields[8]) if fields[8] else 0.0
        altitude = float(fields[9]) if fields[9] else 0.0
        
        # ===== CONVERT TO UTM =====
        # utm.from_latlon returns a tuple: (easting, northing, zone_number, zone_letter)
        utm_result = utm.from_latlon(latitude, longitude)
        utm_easting = utm_result[0]
        utm_northing = utm_result[1]
        zone_number = utm_result[2]
        zone_letter = utm_result[3]
        
        # Return all data as a dictionary
        # Dictionaries store key-value pairs, accessed like data['latitude']
        return {
            'utc': utc_time,
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude,
            'hdop': hdop,
            'utm_easting': utm_easting,
            'utm_northing': utm_northing,
            'zone': zone_number,
            'letter': zone_letter
        }
    
    def publish_gps_message(self, data):
        """
        Create a GpsMsg and publish it to the /gps topic.
        
        Args:
            data (dict): Parsed GPS data from parse_gpgga()
        """
        # Create an instance of our custom message
        msg = GpsMsg()
        
        # ===== FILL IN THE HEADER =====
        # The header contains timestamp and coordinate frame information
        
        # Create timestamp from GPS UTC time (NOT system time!)
        # Lab requirement: use GPGGA time to avoid sync issues
        # ROS2 timestamps have two parts: seconds and nanoseconds
        utc_seconds = int(data['utc'])                    # Whole seconds
        utc_nanoseconds = int((data['utc'] % 1) * 1e9)   # Fractional part as nanoseconds
        
        # Set the timestamp
        # builtin_interfaces.msg.Time has sec and nanosec fields
        msg.header.stamp.sec = utc_seconds
        msg.header.stamp.nanosec = utc_nanoseconds
        
        # Set frame_id to identify which sensor this data came from
        # Lab specifies: "GPS1_Frame"
        msg.header.frame_id = "GPS1_Frame"
        
        # ===== FILL IN GPS DATA =====
        msg.latitude = data['latitude']
        msg.longitude = data['longitude']
        msg.altitude = data['altitude']
        msg.hdop = data['hdop']
        msg.utm_easting = data['utm_easting']
        msg.utm_northing = data['utm_northing']
        msg.utc = data['utc']
        msg.zone = data['zone']
        msg.letter = data['letter']
        
        # Publish the message
        # All subscribers to '/gps' will receive this message
        self.publisher.publish(msg)
        
        # Log at debug level (won't show unless you set log level to debug)
        self.get_logger().debug(
            f'Published: Lat={data["latitude"]:.6f}, Lon={data["longitude"]:.6f}'
        )
    
    def destroy_node(self):
        """
        Clean up when the node shuts down.
        
        This method overrides the parent class's destroy_node() to add
        our own cleanup code (closing the serial port).
        """
        # Close serial port if it's open
        if hasattr(self, 'serial_port') and self.serial_port.is_open:
            self.serial_port.close()
            self.get_logger().info('Serial port closed')
        
        # Call parent class cleanup
        super().destroy_node()


def main(args=None):
    """
    Main entry point for the GPS driver node.
    
    This function is called when you run the node (either directly or via ros2 run).
    It sets up argument parsing, initializes ROS2, creates the node, and spins.
    
    Args:
        args: Command-line arguments (passed automatically by ROS2)
    """
    # ===== PARSE COMMAND-LINE ARGUMENTS =====
    # argparse is Python's standard library for handling command-line arguments
    # This allows running: python3 driver.py --port /dev/ttyUSB0
    parser = argparse.ArgumentParser(description='GPS Driver Node for EECE 5554')
    
    # add_argument defines what arguments the program accepts
    # '--port': The flag name (use --port /dev/ttyUSB0)
    # type=str: Expect a string value
    # default: Value if not specified
    # help: Description shown with --help
    parser.add_argument(
        '--port',
        type=str,
        default='/dev/ttyUSB0',
        help='Serial port path (e.g., /dev/ttyUSB0 or /dev/pts/6 for emulator)'
    )
    
    # parse_known_args() parses our arguments and ignores ROS2's internal arguments
    # Returns a tuple: (our_args, remaining_args)
    parsed_args, _ = parser.parse_known_args()
    
    # ===== INITIALIZE ROS2 =====
    # This must be called before creating any nodes
    rclpy.init(args=args)
    
    # ===== CREATE AND RUN THE NODE =====
    try:
        # Create our GPS driver node
        node = GPSDriverNode(port_path=parsed_args.port)
        
        # spin() keeps the node running and processing callbacks
        # It blocks here until the node is shut down (Ctrl+C)
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        # ===== CLEANUP =====
        # Always clean up, even if there was an error
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


# This block only runs if you execute this file directly (not when imported)
if __name__ == '__main__':
    main()
