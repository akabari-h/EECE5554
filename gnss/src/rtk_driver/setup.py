from setuptools import setup
import os
from glob import glob

package_name = 'rtk_driver'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@northeastern.edu',
    description='RTK GPS driver node for EECE 5554 Lab 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rtk_driver = rtk_driver.rtk_driver:main',
        ],
    },
)
