import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os
from pycrs.parse import from_epsg_code

inPath = r"Individual_VBM_Images/s13_28.2018"
outPath = r"VBM_tifs/s13_28.2018.tif"

query = os.path.join(inPath, "*.tif")

files = glob.glob(query)

tifs = []

for tif in files:
  print("Opening " + tif)
  tifs.append(rasterio.open(tif))

src = rasterio.open(files[0])
epsg_identifier = src.crs.data['lat_1']
print(epsg_identifier)

mosaic, out_trans = merge(tifs)

out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans,
                 "crs": src.crs.data,
                 })


with rasterio.open(outPath, 'w', **out_meta) as dest:
  dest.write(mosaic)


#show(rasterio.open(outPath))






"""

import shutil
import os


source = r"Individual_VBM_Images/N16_89_West.2017_part1"
dest1 = r"Individual_VBM_Images/N16_89_West.2017_part2"


files = os.listdir(source)
i=0
for f in files:
    i+=1
    if i<300:
        shutil.move(source+"/"+f, dest1)
        
"""
