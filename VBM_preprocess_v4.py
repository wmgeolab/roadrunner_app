from VBM_Cropper_v2 import Cropper
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
  
def execute_crop(lat_start, long_start, lat_mid, long_mid, lat_end, long_end, out):
  for cropper in croppers:
    try:
      cropper.crop(lat_start, long_start, lat_mid, long_mid, lat_end, long_end, out)
      break
    except ValueError as e:
      continue
    raise ValueError
    
    
#Get the latest ID in database copy
db_copy = open(r"database_copy.csv", "r")
db_rows = db_copy.readlines()[1:]
for rows in db_rows:
  pass
if not db_rows:
  last_db_ID = 0
else:
  last = rows.split(",")
  last_db_ID = int(last[0].rstrip())
db_copy.close()
  

#connect to database
conn = mysql.connector.connect(
  host="mysql.geodesc.org",
  user=os.environ["DB_USER"],
  passwd=os.environ["DB_PASS"],
  database="roadrunner"
  )

#get cursor
cursor = conn.cursor()    

# update copy of database
query = ("SELECT id, uuid, line, distance, iriXp, iriYp, iriZp, iriX, iriY, iriZ, filepath, time, phoneID FROM main "
         "WHERE id > " + str(last_db_ID))  
cursor.execute(query)

with open(r"database_copy.csv", "a+") as db_log:
  db_fieldnames = ['id', 'uuid', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filepath', 'time', 'phoneID']
  writer = csv.DictWriter(db_log, fieldnames=db_fieldnames)
  for(idNum, uuID, line, distance, iriXp, iriYp, iriZp, iriX, iriY, iriZ, filepath, time, phoneID) in cursor:  
    writer.writerow(dict({"id":idNum, "uuid": uuID, "line":line, "distance":distance, "iriXp": iriXp, "iriYp":iriYp, "iriZp":iriZp, "iriX":iriX, "iriY":iriY, "iriZ":iriZ, "filepath":filepath, "time":time, "phoneID":phoneID}))
  
cursor.close()
conn.close()   



# record fails amount and last ID at start
old_fails = open("fails.txt", "r")
old_fails_rows = old_fails.readlines()
old_fails.close()

num_old_fails = len(old_fails_rows)

if num_old_fails == 0:
  max_fails_ID = 0
else:
  all_fails_ids = []
  for f in old_fails_rows:
    current_row = f.split(",")
    fail_id = int(current_row[0])
    all_fails_ids.append(fail_id)
  max_fails_ID = max(all_fails_ids)




#
# unchecked
#


#Get the highest ID in log.csv
log = open("/var/lib/cdsw/share/log.csv", "r")
log_rows = log.readlines()[1:]
log.close()

if not log_rows:
  max_log_ID = 0
else:  
  all_ids = []
  for l in log_rows:
    current_row = l.split(",")
    current_row_id = int(current_row[0].rstrip())
    all_ids.append(current_row_id)
  max_log_ID = max(all_ids)


df = pd.read_csv(r"database_copy.csv", usecols=db_fieldnames)
df = df.drop_duplicates(subset={'line','uuid','distance','iriXp','iriYp','iriZp','iriX','iriY','iriZ','time','phoneID'}, keep='first')
extrm_indeces = df[(df['iriXp'] < 0.0001) | (df['iriYp'] < 0.0001) | (df['iriZp'] < 0.0001) | (df['iriXp'] > 20) | (df['iriYp'] > 20) | (df['iriZp'] > 20)].index
df = df.drop(extrm_indeces)
df.to_csv('database_copy_duplicates_&_extremes_removed.csv',index=False)



log_fieldnames = ['id', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filename', 'time', 'phoneID']

for index, row in df.iterrows():
  current_row_id = df["id"][index]
  phone_id = df["phoneID"][index]
  if current_row_id > max_log_ID and current_row_id > max_fails_ID:
    line = df["line"][index]
    line = line.replace("\'", "\"")
    ln = json.loads(line)
    # line is given in long lat
    start = ln['coordinates'][0]
    mid = ln['coordinates'][1]
    end = ln['coordinates'][len(ln['coordinates'])-1]
    # generate filename
    today = date.today()
    filename = str(today.year)+today.strftime('%m')+today.strftime('%d')+'_'+ str(current_row_id)
      
    try:
      execute_crop(start[1], start[0], mid[1], mid[0], end[1], end[0], "/var/lib/cdsw/share/processing/VBM/raw/"+filename+"_raw.tif")
      # generate pngs from tif
      generate_pngs("/var/lib/cdsw/share/processing/VBM/raw/"+filename+"_raw.tif", "/var/lib/cdsw/share/processing/VBM/png/"+filename)
      with open("/var/lib/cdsw/share/log.csv", "a+") as log:
        writer = csv.DictWriter(log, fieldnames=log_fieldnames)
        writer.writerow(dict({"id":current_row_id, "line":df["line"][index], "distance":df["distance"][index], "iriXp":df["iriXp"][index], 
                              "iriYp":df["iriYp"][index], "iriZp":df["iriZp"][index], "iriX":df["iriX"][index], "iriY":df["iriY"][index], 
                              "iriZ":df["iriZ"][index], "filename":filename, "time":df["time"][index], "phoneID":df["phoneID"][index]})) 
    except AttributeError:
      print("ERROR! FAILED FOR ID " + str(current_row_id))
      with open("fails.txt", "a") as fails:
        fails.write(str(current_row_id)+"," + str(phone_id) + "\n")
      print("Lat: " + str(mid[1]))
      print("Long: " + str(mid[0]))
    
  else:
    continue
        
        



#
# fails
#


# Check if fails file at start is empty and, if not, attempt to process fails
if not old_fails_rows:
  pass
else:
  fails = open("fails.txt", "r")
  fails_rows = fails.readlines()
  fails.close()

  i=0
  for fail in fails_rows[0:num_old_fails]:
    fail_id = int(fail.split(",")[0].rstrip())
    try:
      phone_id = str(fail.split(",")[1].rstrip())
    except IndexError as e:
      phone_id = ""
    index = df[df['id'] == fail_id].index[0]
    line = df["line"][index]
    line = line.replace("\'", "\"")
    ln = json.loads(line)
    # line is given in long lat
    start = ln['coordinates'][0]
    mid = ln['coordinates'][1]
    end = ln['coordinates'][len(ln['coordinates'])-1]
    # generate filename
    today = date.today()
    filename = str(today.year)+today.strftime('%m')+today.strftime('%d')+'_'+ str(df["id"][index])

    try:
      execute_crop(start[1], start[0], mid[1], mid[0], end[1], end[0], "/var/lib/cdsw/share/processing/VBM/raw/"+filename+"_raw.tif")
      # generate pngs from tif
      generate_pngs("/var/lib/cdsw/share/processing/VBM/raw/"+filename+"_raw.tif", "/var/lib/cdsw/share/processing/VBM/png/"+filename)
      with open("/var/lib/cdsw/share/log.csv", "a+") as log:
        writer = csv.DictWriter(log, fieldnames=log_fieldnames)
        writer.writerow(dict({"id":df["id"][index], "line":df["line"][index], "distance":df["distance"][index], "iriXp":df["iriXp"][index], 
                              "iriYp":df["iriYp"][index], "iriZp":df["iriZp"][index], "iriX":df["iriX"][index], "iriY":df["iriY"][index], 
                              "iriZ":df["iriZ"][index], "filename":filename, "time":df["time"][index], "phoneID":df["phoneID"][index]}))
      del fails_rows[i]
      with open("fails.txt", "w") as new_fails:
        for row in fails_rows:
          new_fails.write(row)
    except AttributeError:
      print("ERROR! FAILED FOR ID " + str(df["id"][index]))
      print("Lat: " + str(mid[1]))
      print("Long: " + str(mid[0]))
      i += 1

        
        
        
         
copy2("/var/lib/cdsw/share/log.csv", "/home/cdsw/log_copy.csv") 
       


  

# check for missing ids in log
log_fieldnames = ['id', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filename', 'time', 'phoneID']
log_df = pd.read_csv(r"/var/lib/cdsw/share/log.csv", usecols=log_fieldnames)  

fails_df = pd.read_csv(r"fails.txt",header=None)
fails_df.columns = ["id", "phoneID"]

IDs = []
for index, row in df.iterrows():
  ID = df["id"][index]
  try:
    log_index = log_df[log_df['id'] == ID].index[0]
  except IndexError:  
    try:
      fails_index = fails_df[fails_df['id'] == ID].index[0]
    except IndexError:
      IDs.append(ID)

if not IDs:
  pass
else:           
  raise ValueError("There are missing ids in log.csv")  


        












"""
# create database_copy headers
db_fieldnames = ['id', 'uuid', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filepath', 'time', 'phoneID']
with open("database_copy.csv", "w") as f:
  writer = csv.writer(f, delimiter=',')
  writer.writerow(db_fieldnames)
"""


"""
# create log headers
log_fieldnames = ['id', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filename', 'time', 'phoneID']
with open("log.csv", "w") as f:
  writer = csv.writer(f, delimiter=',')
  writer.writerow(log_fieldnames)
"""



"""
# refill fails.txt
db_fieldnames = ['id', 'uuid', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filepath', 'time', 'phoneID']
df = pd.read_csv(r"database_copy.csv", usecols=db_fieldnames)

log = open(r"log_copy.csv", "r")
log_rows = log.readlines()[1:]
all_log_ids = []
for l in log_rows:
  current_row = l.split(",")
  current_row_id = int(current_row[0].rstrip())
  if current_row_id > 129890:
    all_log_ids.append(current_row_id)

    
all_database_ids = []
fail_ids = []
for i in df['id']:
  if i > 129890:
    all_database_ids.append(i)
    if i not in all_log_ids:
      fail_ids.append(i)

with open("fails.txt", "a") as fails:     
  for num in fail_ids:
    row = str(num)+'\n'
    fails.write(row)


"""



"""
# fill missing phoneIDs in fails.txt
fails = open("fails copy 1.txt", "r")
fails_rows = fails.readlines()
fails.close()

db_fieldnames = ['id', 'uuid', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filepath', 'time', 'phoneID']
df = pd.read_csv(r"database_copy.csv", usecols=db_fieldnames)

for fail in fails_rows:
  fail_id = int(fail.split(",")[0].rstrip())
  index = df[df['id'] == fail_id].index[0]
  phoneid = df["phoneID"][index]
  with open("fails copy 3.txt", "a") as fails:
    fails.write(str(fail_id)+"," + str(phoneid) + "\n")

"""

  
"""
# remove duplicates & extremes in fails.txt  
rm fails_copy_2.txt
fails = open("fails copy 1.txt", "r")
fails_rows = fails.readlines()
fails.close()
  
db_fieldnames = ['id', 'uuid', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filepath', 'time', 'phoneID']
df = pd.read_csv(r"database_copy_duplicates_&_extremes_removed.csv", usecols=db_fieldnames)  
  
for fail in fails_rows:
  fail_id = int(fail.split(",")[0].rstrip())
  try:
    index = df[df['id'] == fail_id].index[0]
    phoneid = df["phoneID"][index]
    with open("fails_copy_2.txt", "a+") as fails:
      fails.write(str(fail_id)+"," + str(phoneid) + "\n")  
  except IndexError:  
    continue
"""  



"""
# check for and correct missing ids in log
db_fieldnames = ['id', 'uuid', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filepath', 'time', 'phoneID']
df = pd.read_csv(r"database_copy_duplicates_&_extremes_removed.csv", usecols=db_fieldnames)  

log_fieldnames = ['id', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filename', 'time', 'phoneID']
log_df = pd.read_csv(r"/var/lib/cdsw/share/log.csv", usecols=log_fieldnames)  

fails_df = pd.read_csv(r"fails.txt",header=None)
fails_df.columns = ["id", "phoneID"]

IDs = []
for index, row in df.iterrows():
  ID = df["id"][index]
  try:
    log_index = log_df[log_df['id'] == ID].index[0]
  except IndexError:  
    try:
      fails_index = fails_df[fails_df['id'] == ID].index[0]
    except IndexError:
      IDs.append(ID)

if not IDs:
  print("empty")
else:           
  miss_df = DataFrame(IDs,columns=['id'])   
        
  with open("/var/lib/cdsw/share/log.csv", "a+") as log:
    fieldnames = ['id', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filename', 'time', 'phoneID']
    writer = csv.DictWriter(log, fieldnames=fieldnames)
    for miss_index, row in miss_df.iterrows():
      current_row_id = miss_df["id"][miss_index]
      index = df[df['id'] == current_row_id].index[0]
      phone_id = df["phoneID"][index]
      line = df["line"][index]
      line = line.replace("\'", "\"")
      ln = json.loads(line)
      # line is given in long lat
      start = ln['coordinates'][0]
      end = ln['coordinates'][len(ln['coordinates'])-1]

      # generate filename
      today = date.today()
      filename = str(today.year)+today.strftime('%m')+today.strftime('%d')+'_'+ str(current_row_id)
      # Crop on midpoint
      midpoint = ((start[1]+start[1])/2, (start[0]+start[0])/2)

      try:
        execute_crop(midpoint[0], midpoint[1], "/var/lib/cdsw/share/processing/VBM/raw/"+filename+"_raw.tif")
        # generate pngs from tif
        generate_pngs("/var/lib/cdsw/share/processing/VBM/raw/"+filename+"_raw.tif", "/var/lib/cdsw/share/processing/VBM/png/"+filename)
        writer.writerow(dict({"id":current_row_id, "line":df["line"][index], "distance":df["distance"][index], "iriXp":df["iriXp"][index], 
                                "iriYp":df["iriYp"][index], "iriZp":df["iriZp"][index], "iriX":df["iriX"][index], "iriY":df["iriY"][index], 
                                "iriZ":df["iriZ"][index], "filename":filename, "time":df["time"][index], "phoneID":df["phoneID"][index]})) 
      except AttributeError:
        print("ERROR! FAILED FOR ID " + str(current_row_id))
        with open("fails.txt", "a") as fails:
          fails.write(str(current_row_id)+"," + str(phone_id) + "\n")
        print("Lat: " + str(midpoint[0]))
        print("Long: " + str(midpoint[1]))

      
"""



"""
#remove directory with many files
import shutil
source = "/var/lib/cdsw/share/processing/VBM/png"
shutil.rmtree(source)
"""


"""       
# fix non-iterating log ids  
fieldnames = ['id', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filename', 'time', 'phoneID']
log_df = pd.read_csv("log.csv", usecols=fieldnames)

for index, row in log_df.iterrows():   
  if log_df["id"][index] == 305602:
    new_id = log_df["filename"][index][9:15]  
    log_df["id"][index] = new_id

log_df.to_csv('new_log.csv',index=False)  
"""




