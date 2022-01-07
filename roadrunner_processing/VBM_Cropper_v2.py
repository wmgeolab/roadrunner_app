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

class Cropper:
    def __init__(self, filePath):
        self.filepath = filePath
    
    def crop(self, lat_start, long_start, lat_mid, long_mid, lat_end, long_end, outPath):
        inProj = Proj(init='epsg:4326')

        with rio.open(self.filepath, "r") as raster:

            epsg_identifier = raster.crs.data['lat_1']
            if epsg_identifier == 36.7666666666667 or epsg_identifier == 36.76666666666667:
              epsg_code = 2925
            if epsg_identifier == 38.0333333333333:
              epsg_code = 2924
            
            outProj = Proj(init='epsg:'+str(epsg_code))
            x_start,y_start = transform(inProj,outProj,long_start,lat_start)   
            x_mid,y_mid = transform(inProj,outProj,long_mid,lat_mid) 
            x_end,y_end = transform(inProj,outProj,long_end,lat_end) 
            
            minx, miny = min(x_start,x_mid,x_end), min(y_start,y_mid,y_end)
            maxx, maxy = max(x_start,x_mid,x_end), max(y_start,y_mid,y_end)
            
            square = box(minx-5, miny-5, maxx+5, maxy+5)
            gdf = gpd.GeoDataFrame({'geometry': square}, index = [0], crs=raster.crs.data)
            coords = self.__getFeatures(gdf)
            
            out_img, out_transform = mask(raster, coords, crop = True)
            
            prop_zeros = np.count_nonzero(out_img == 0)/(out_img.shape[0]*out_img.shape[1]*out_img.shape[2])
            if prop_zeros > 0.2:
              raise ValueError("Too many zero pixels")
              
            out_meta = raster.meta.copy()
            out_meta.update({"driver": "GTiff", "height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform, "crs":raster.crs.data})
           
            with rio.open(outPath, "w", **out_meta) as dest:
              dest.write(out_img)
            
    def __getFeatures(self, gdf):
        return [json.loads(gdf.to_json())['features'][0]['geometry']]
  
  

### Small test case ###
if __name__ == "__main__":
      crop = Cropper(r"VBM_tifs/s03_95.tif")
      crop.crop(37.126228, -80.438430, r"testcrop.tif")
      
      
      
