import numpy as np
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
from pycrs.parse import from_epsg_code
import json
import utm
from pyproj import Proj, transform
from matplotlib import pyplot as plt 

class Cropper:
    def __init__(self, filePath):
        self.filepath = filePath
    
    def crop(self, lat, long, outPath):
        inProj = Proj(init='epsg:4326')

        with rio.open(self.filepath, "r") as raster:

            epsg_identifier = raster.crs.data['lat_1']
            if epsg_identifier == 36.7666666666667 or epsg_identifier == 36.76666666666667:
              epsg_code = 2925
            if epsg_identifier == 38.0333333333333:
              epsg_code = 2924
            
            outProj = Proj(init='epsg:'+str(epsg_code))
            x,y = transform(inProj,outProj,long,lat)   
            
            square = box(x-50, y-50, x+49.5, y+49.5)
            gdf = gpd.GeoDataFrame({'geometry': square}, index = [0], crs=raster.crs.data)
            coords = self.__getFeatures(gdf)
            
            out_img, out_transform = mask(raster, coords, crop = True)
            
            prop_zeros = np.count_nonzero(out_img == 0)/(out_img.shape[0]*out_img.shape[1]*out_img.shape[2])
            if prop_zeros > 0.5:
              raise ValueError("Too many zero pixels")
              
            out_meta = raster.meta.copy()
            out_meta.update({"driver": "GTiff", "height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform, "crs":raster.crs.data})
           
            with rio.open(outPath, "w", **out_meta) as dest:
              dest.write(out_img)
            
    def __getFeatures(self, gdf):
        return [json.loads(gdf.to_json())['features'][0]['geometry']]
  
  

### Small test case ###
if __name__ == "__main__":
      crop = Cropper(r"VBM_tifs/S13_87.2017.tif")
      crop.crop(37.51753757, -77.199464637, "testcrop.tif")
      
#im = plt.imread('testcrop.tif')      
#im.shape