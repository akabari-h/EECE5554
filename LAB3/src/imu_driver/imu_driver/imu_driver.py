#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import serial
import argparse
import time

from sensor_msgs.msg import Imu, MagneticField

from imu_msg.msg import IMUmsg


class IMUDriverNode(Node):
    
    def __init__(self, port_path):
        super().__init__('imu_driver_node')

        self.port_path = port_path

        
        self.publisher = self.create_publisher(IMUmsg, '/imu', 10)

        self.get_logger().info(f'IMU Driver starting on port: {port_path}')


        try:
            self.serial_port = serial.Serial(
                port_path, baudrate=115200, timeout=1.0
            )
            self.get_logger().info('Serial port opened successfully')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            raise e

        self.configure_imu_rate(40)

        self.timer = self.create_timer(0.01, self.timer_callback)

    def configure_imu_rate(self, rate_hz):

        cmd_body = f'VNWRG,07,{rate_hz}'

        checksum = 0
        for char in cmd_body:
          
            checksum ^= ord(char)

     
        cmd = f'${cmd_body}*{checksum:02X}\r\n'

        self.get_logger().info(f'Configuring IMU rate to {rate_hz} Hz: {cmd.strip()}')

    
        self.serial_port.write(cmd.encode())

  
        time.sleep(0.1)

    def timer_callback(self):

        try:
        
            line = self.serial_port.readline().decode(
                'utf-8', errors='ignore'
            ).strip()
        except Exception as e:
            self.get_logger().warn(f'Serial read error: {e}')
            return

    
        if not line.startswith('$VNYMR'):
            return

     
        parsed = self.parse_vnymr(line)
        if parsed is None:
            self.get_logger().debug(f'Failed to parse: {line}')
            return

    
        self.publish_imu_message(parsed, line)

    def parse_vnymr(self, sentence):

        try:
            
            fields = sentence.split(',')

  
            if len(fields) != 13:
                return None
            
            yaw   = float(fields[1])
            pitch = float(fields[2])
            roll  = float(fields[3])

            mag_x = float(fields[4])
            mag_y = float(fields[5])
            mag_z = float(fields[6])

            accel_x = float(fields[7])
            accel_y = float(fields[8])
            accel_z = float(fields[9])

            gyro_x = float(fields[10])
            gyro_y = float(fields[11])
            gyro_z = float(fields[12].split('*')[0])

            return {
                'yaw':    yaw,   'pitch': pitch, 'roll':  roll,
                'mag_x':  mag_x, 'mag_y': mag_y, 'mag_z': mag_z,
                'accel_x':accel_x,'accel_y':accel_y,'accel_z':accel_z,
                'gyro_x': gyro_x,'gyro_y': gyro_y,'gyro_z': gyro_z,
            }

        except (ValueError, IndexError) as e:

            self.get_logger().debug(f'Parse error: {e}')
            return None

    def publish_imu_message(self, data, raw_line):

        msg = IMUmsg()

        now = self.get_clock().now()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'IMU1_Frame'

        import math
        roll_r  = math.radians(data['roll'])
        pitch_r = math.radians(data['pitch'])
        yaw_r   = math.radians(data['yaw'])

        cy = math.cos(yaw_r   * 0.5)
        sy = math.sin(yaw_r   * 0.5)
        cp = math.cos(pitch_r * 0.5)
        sp = math.sin(pitch_r * 0.5)
        cr = math.cos(roll_r  * 0.5)
        sr = math.sin(roll_r  * 0.5)

        msg.imu.orientation.w = cr*cp*cy + sr*sp*sy
        msg.imu.orientation.x = sr*cp*cy - cr*sp*sy
        msg.imu.orientation.y = cr*sp*cy + sr*cp*sy
        msg.imu.orientation.z = cr*cp*sy - sr*sp*cy

        msg.imu.orientation_covariance[0]         = -1.0
        msg.imu.angular_velocity_covariance[0]    = -1.0
        msg.imu.linear_acceleration_covariance[0] = -1.0

        msg.imu.angular_velocity.x = data['gyro_x']
        msg.imu.angular_velocity.y = data['gyro_y']
        msg.imu.angular_velocity.z = data['gyro_z']

        msg.imu.linear_acceleration.x = data['accel_x']
        msg.imu.linear_acceleration.y = data['accel_y']
        msg.imu.linear_acceleration.z = data['accel_z']

        msg.mag_field.magnetic_field.x = data['mag_x'] * 1e-4
        msg.mag_field.magnetic_field.y = data['mag_y'] * 1e-4
        msg.mag_field.magnetic_field.z = data['mag_z'] * 1e-4

        msg.mag_field.magnetic_field_covariance[0] = -1.0

        msg.raw_str = raw_line

        self.publisher.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'serial_port') and self.serial_port.is_open:
            self.serial_port.close()
            self.get_logger().info('Serial port closed')
        super().destroy_node()


def main(args=None):

    parser = argparse.ArgumentParser(description='IMU Driver for VN-100')
    parser.add_argument(
        '--port', type=str, default='/dev/ttyUSB0',
        help='Serial port path (e.g., /dev/ttyUSB0 or /dev/pts/X)'
    )
    parsed_args, _ = parser.parse_known_args()

    rclpy.init(args=args)

    try:
        node = IMUDriverNode(port_path=parsed_args.port)
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