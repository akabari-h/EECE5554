cmake_minimum_required(VERSION 3.8)
project(gps_msg)

# Default to C99 and C++14 standards
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Enable compiler warnings for better code quality
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ===== FIND DEPENDENCIES =====
# These are the packages needed to build and generate messages
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ===== GENERATE MESSAGE INTERFACES =====
# This is the critical part - it tells CMake to generate Python/C++ code
# from our .msg file definition
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Customgps.msg"
  "msg/Customrtk.msg"
  "msg/GpsMsg.msg"
  DEPENDENCIES std_msgs
)

# ===== TESTING (optional but good practice) =====
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ===== EXPORT PACKAGE =====
ament_package()
