"""

Import the generate_pngs function which generates 1 png
8 bit 3 channel rgb


"""
from osgeo import gdal
import numpy

# Use numpngw because pillow can't do 16 bit PNG
# we would get 2^8 times more detail
from numpngw import write_png

# Use GDAL to read each band
# Band -> numpy array
def get_bands(filepath):
  im = gdal.Open(filepath)
  r = numpy.array(im.GetRasterBand(1).ReadAsArray())
  g = numpy.array(im.GetRasterBand(2).ReadAsArray())
  b = numpy.array(im.GetRasterBand(3).ReadAsArray())
  return (r, g, b)

# Normalize w Yaw's function
# Function to normalize the grid values
def normalize(array):
  """Normalizes numpy arrays into scale 0.0 - 1.0"""
  array_min, array_max = array.min(), array.max()
  return ((array - array_min)/(array_max - array_min))

# Name without .png ending!
# Multiple files will be generated
def generate_pngs(filepath, name):
  r, g, b = get_bands(filepath)
  rn = normalize(r)
  gn = normalize(g)
  bn = normalize(b)
  
  # 16 bit
  rgb16 = numpy.dstack((rn*65535,gn*65535,bn*65535)).astype('uint16')
  write_png(name + "_16.png", rgb16)
  
  # 8 bit
  rgb = numpy.dstack((rn*255, gn*255, bn*255)).astype('uint8')
  write_png(name + "_8.png", rgb)
