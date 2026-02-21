# GPSmsg.msg - Custom GPS message for EECE 5554 Lab 1
# Contains parsed GPGGA data with UTM conversion

# Header contains timestamp and frame_id
# The timestamp should use GPS UTC time, NOT system time
std_msgs/Header header

# Geographic coordinates in decimal degrees (signed floats)
# Latitude: positive = North, negative = South
# Longitude: positive = East, negative = West
float64 latitude
float64 longitude

# Altitude above mean sea level in meters
float64 altitude

# Horizontal Dilution of Precision - indicates position accuracy
# Lower HDOP = better accuracy (1.0 is ideal, >20 is poor)
float64 hdop

# UTM coordinates in meters
# Easting: distance east within the UTM zone
# Northing: distance north from the equator
float64 utm_easting
float64 utm_northing

# UTC time in seconds since midnight
float64 utc

# UTM zone information
# Zone: integer 1-60 representing the UTM zone number
# Letter: character representing the latitude band (C-X, excluding I and O)
int32 zone
string letter