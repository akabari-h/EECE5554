# Header contains timestamp and frame_id
std_msgs/Header header

# Geographic coordinates in decimal degrees (signed floats)
float64 latitude
float64 longitude

# Altitude above mean sea level in meters
float64 altitude

# UTM coordinates in meters
float64 utm_easting
float64 utm_northing

# UTC time in seconds since midnight
float64 utc

# UTM zone information
int32 zone
string letter

# RTK-specific fields
int32 fix_quality
float64 hdop