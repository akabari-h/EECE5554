# LAB4: Dead Reckoning Navigation

## Team Member Uploading Driver
GitHub username: tajane19

## How to Run

### Prerequisites
- ROS2 Humble
- GPS puck on /dev/ttyUSB0
- VectorNav VN-100 IMU on /dev/ttyUSB2

### Build
```bash
cd LAB4
colcon build
source install/setup.bash
```

### Launch Both Drivers
```bash
ros2 launch gps_driver combined_launch.py \
    gps_port:=/dev/ttyUSB0 \
    imu_port:=/dev/ttyUSB2
```

### Record Data
```bash
ros2 bag record -o data_driving /gps /imu
```

### Run Analysis
```bash
cd analysis
python3 mag_calibration.py
python3 yaw_estimation.py
python3 velocity_estimation.py
python3 dead_reckoning.py
```
