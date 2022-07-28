from VBM_Cropper_v3 import Cropper
import mysql.connector
import csv
import pandas as pd
from pandas import DataFrame
from VBM_tif_convert import generate_pngs
from datetime import date
import json
import os
import glob
import sys
from shutil import copy2
import warnings
warnings.filterwarnings("ignore")

croppers = []

inPath = r"VBM_tifs"
query = os.path.join(inPath, "*.tif")
files = glob.glob(query)
for file in files:
  croppers.append(Cropper(file))
  
def execute_crop(lat, long, out):
  for cropper in croppers:
    try:
      cropper.crop(lat, long, out)
      break
    except ValueError as e:
      continue
    raise ValueError


log_fieldnames = ['id', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filename', 'time', 'phoneID']
df = pd.read_csv("final_log.csv", usecols=log_fieldnames)


for index, row in df.iterrows():
  line = df["line"][index]
  line = line.replace("\'", "\"")
  ln = json.loads(line)
  coord = ln['coordinates'][1]
  filename = df["filename"][index]
  try:
    execute_crop(coord[1], coord[0], "Ethan_tifs/"+filename+"_raw.tif")
    generate_pngs("Ethan_tifs/"+filename+"_raw.tif", "/var/lib/cdsw/share/Ethan_pngs/"+filename) 
  except AttributeError:
    print('Error with filename: ' + filename)

