from setuptools import setup
import os
from glob import glob

package_name = 'imu_driver'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files so ROS2 can find them
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@northeastern.edu',
    description='IMU driver node for EECE 5554 Lab 3',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Format: executable_name = package.module:function
            # This creates a command called imu_driver that calls main()
            # in the imu_driver.py file inside the imu_driver package
            'imu_driver = imu_driver.imu_driver:main',
        ],
    },
)
