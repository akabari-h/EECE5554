#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===== IMPORTS =====
import rclpy
from rclpy.node import Node
import serial
import utm
import argparse

# Import our custom RTK message type
# Customrtk includes fix_quality and hdop in addition to standard GPS fields
from gps_msg.msg import Customrtk


class RTKDriverNode(Node):
    """
    ROS2 Node that reads RTK GPS data from a serial port and publishes parsed data.
    
    This driver is designed for RTK GNSS receivers that output GNGGA sentences.
    GNGGA uses the "GN" prefix indicating multi-constellation GNSS data
    (GPS + GLONASS + Galileo + BeiDou), as opposed to GPGGA which is GPS-only.
    
    The key addition for RTK is the fix_quality field:
        0 = Invalid
        1 = Standalone GPS (3-5 meter accuracy)
        2 = DGPS (1-3 meter accuracy)
        4 = RTK Fixed (1-5 centimeter accuracy)
        5 = RTK Float (20-100 centimeter accuracy)
    """
    
    def __init__(self, port_path):
        """
        Initialize the RTK driver node.
        
        Args:
            port_path (str): Path to serial device (e.g., '/dev/ttyUSB0' or '/dev/pts/X')
        """
        # Initialize the node with a descriptive name
        super().__init__('rtk_driver_node')
        
        # Store the port path for reference
        self.port_path = port_path
        
        # Create publisher for the RTK message on the /rtk topic
        # Queue size of 10 provides buffer for the 1 Hz GPS data rate
        self.publisher = self.create_publisher(Customrtk, '/rtk', 10)
        
        # Log startup information
        self.get_logger().info(f'RTK Driver starting on port: {port_path}')
        
        # Open the serial connection to the GPS receiver
        # RTK receivers typically use 4800 or 115200 baud depending on model
        # The sensor emulator uses 4800 baud by default
        try:
            self.serial_port = serial.Serial(port_path, baudrate=4800, timeout=1.0)
            self.get_logger().info('Serial port opened successfully')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            raise e
        
        # Create a timer to periodically read from the serial port
        # 10 Hz (0.1 second period) ensures we don't miss any 1 Hz GPS messages
        self.timer = self.create_timer(0.1, self.timer_callback)
    
    def timer_callback(self):
        """
        Timer callback that reads and processes GPS data.
        
        Called at 10 Hz by the ROS2 timer. Reads a line from the serial port,
        checks if it's a GNGGA sentence, parses it, and publishes the result.
        """
        # Read one line from the serial port
        try:
            line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
        except Exception as e:
            self.get_logger().warn(f'Serial read error: {e}')
            return
        
        # Only process GNGGA or GPGGA sentences
        # GNGGA = multi-constellation GNSS, GPGGA = GPS only
        # We accept both for maximum compatibility
        if not (line.startswith('$GNGGA') or line.startswith('$GPGGA')):
            return
        
        # Parse the NMEA sentence
        parsed_data = self.parse_gngga(line)
        
        # Skip if parsing failed (invalid data)
        if parsed_data is None:
            self.get_logger().debug(f'Failed to parse: {line}')
            return
        
        # Create and publish the ROS message
        self.publish_rtk_message(parsed_data)
    
    def parse_gngga(self, sentence):
        """
        Parse a GNGGA/GPGGA NMEA sentence and extract all relevant fields.
        
        The GNGGA sentence format is identical to GPGGA:
        $GNGGA,HHMMSS.ss,DDMM.MMMM,N,DDDMM.MMMM,W,Q,Sats,HDOP,Alt,M,Geoid,M,Age,ID*CS
        
        Args:
            sentence (str): Raw NMEA sentence string
            
        Returns:
            dict: Dictionary containing all parsed fields, or None if parsing fails
        """
        # Split the sentence into comma-separated fields
        fields = sentence.split(',')
        
        # Validate minimum field count (GNGGA should have at least 15 fields)
        if len(fields) < 15:
            return None
        
        # Extract and validate fix quality (field 6)
        # This is crucial for RTK - it tells us the positioning mode
        fix_quality_str = fields[6]
        if fix_quality_str == '' or fix_quality_str == '0':
            # No valid fix - cannot extract reliable position
            return None
        fix_quality = int(fix_quality_str)
        
        # ===== PARSE UTC TIME (Field 1) =====
        # Format: HHMMSS.ss (e.g., "181908.00" = 18:19:08.00 UTC)
        utc_raw = fields[1]
        if utc_raw == '':
            return None
        
        hours = int(utc_raw[0:2])
        minutes = int(utc_raw[2:4])
        seconds = float(utc_raw[4:])
        utc_time = hours * 3600 + minutes * 60 + seconds
        
        # ===== PARSE LATITUDE (Fields 2-3) =====
        # Format: DDMM.MMMM where DD = degrees, MM.MMMM = minutes
        lat_raw = fields[2]
        lat_dir = fields[3]
        
        if lat_raw == '' or lat_dir == '':
            return None
        
        # Latitude: first 2 digits are degrees, rest is minutes
        lat_degrees = float(lat_raw[:2])
        lat_minutes = float(lat_raw[2:])
        latitude = lat_degrees + (lat_minutes / 60.0)
        
        # Apply hemisphere sign (South is negative)
        if lat_dir == 'S':
            latitude = -latitude
        
        # ===== PARSE LONGITUDE (Fields 4-5) =====
        # Format: DDDMM.MMMM where DDD = degrees, MM.MMMM = minutes
        lon_raw = fields[4]
        lon_dir = fields[5]
        
        if lon_raw == '' or lon_dir == '':
            return None
        
        # Longitude: first 3 digits are degrees, rest is minutes
        lon_degrees = float(lon_raw[:3])
        lon_minutes = float(lon_raw[3:])
        longitude = lon_degrees + (lon_minutes / 60.0)
        
        # Apply hemisphere sign (West is negative)
        if lon_dir == 'W':
            longitude = -longitude
        
        # ===== PARSE HDOP (Field 8) =====
        # HDOP indicates horizontal position quality
        # Lower is better: < 1 = ideal, 1-2 = excellent, 2-5 = good, > 5 = poor
        hdop = float(fields[8]) if fields[8] else 0.0
        
        # ===== PARSE ALTITUDE (Field 9) =====
        altitude = float(fields[9]) if fields[9] else 0.0
        
        # ===== CONVERT TO UTM COORDINATES =====
        # UTM provides Easting/Northing in meters for easy distance calculations
        utm_result = utm.from_latlon(latitude, longitude)
        utm_easting = utm_result[0]
        utm_northing = utm_result[1]
        zone_number = utm_result[2]
        zone_letter = utm_result[3]
        
        # Return all parsed data as a dictionary
        return {
            'utc': utc_time,
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude,
            'hdop': hdop,
            'fix_quality': fix_quality,
            'utm_easting': utm_easting,
            'utm_northing': utm_northing,
            'zone': zone_number,
            'letter': zone_letter
        }
    
    def publish_rtk_message(self, data):
        """
        Create and publish a Customrtk message.
        
        Args:
            data (dict): Parsed GPS data from parse_gngga()
        """
        # Create message instance
        msg = Customrtk()
        
        # ===== FILL HEADER =====
        # Use GPS UTC time for timestamp (not system time) to ensure consistency
        utc_seconds = int(data['utc'])
        utc_nanoseconds = int((data['utc'] % 1) * 1e9)
        
        msg.header.stamp.sec = utc_seconds
        msg.header.stamp.nanosec = utc_nanoseconds
        msg.header.frame_id = "RTK_Frame"
        
        # ===== FILL POSITION DATA =====
        msg.latitude = data['latitude']
        msg.longitude = data['longitude']
        msg.altitude = data['altitude']
        msg.utm_easting = data['utm_easting']
        msg.utm_northing = data['utm_northing']
        msg.utc = data['utc']
        msg.zone = data['zone']
        msg.letter = data['letter']
        
        # ===== FILL RTK-SPECIFIC FIELDS =====
        msg.fix_quality = data['fix_quality']
        msg.hdop = data['hdop']
        
        # Publish the message
        self.publisher.publish(msg)
        
        # Log at debug level with fix quality info
        fix_types = {1: 'GPS', 2: 'DGPS', 4: 'RTK Fixed', 5: 'RTK Float'}
        fix_name = fix_types.get(data['fix_quality'], 'Unknown')
        self.get_logger().debug(
            f'Published: Lat={data["latitude"]:.7f}, Lon={data["longitude"]:.7f}, '
            f'Fix={fix_name}, HDOP={data["hdop"]}'
        )
    
    def destroy_node(self):
        """
        Clean up when the node shuts down.
        """
        if hasattr(self, 'serial_port') and self.serial_port.is_open:
            self.serial_port.close()
            self.get_logger().info('Serial port closed')
        super().destroy_node()


def main(args=None):
    """
    Main entry point for the RTK driver node.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='RTK GPS Driver Node for EECE 5554')
    parser.add_argument(
        '--port',
        type=str,
        default='/dev/ttyUSB0',
        help='Serial port path (e.g., /dev/ttyUSB0 or /dev/pts/X for emulator)'
    )
    
    parsed_args, _ = parser.parse_known_args()
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    try:
        # Create and run the node
        node = RTKDriverNode(port_path=parsed_args.port)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()