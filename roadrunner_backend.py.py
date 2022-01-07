import mysql.connector
import json
import os


# Opens a connection to the database
def connect():
  try:
    conn = mysql.connector.connect(
      host = "mysql.geodesc.org",
      user = os.environ["DB_USER"],
      passwd = os.environ["DB_PASS"],
      database = "roadrunner"
    )
  except Exception as e:
    print("Error connecting")
    print(e)
    return None
  else:
    return conn
  return None

# Inserts into table
def insert_row(conn, data):
  sql = '''INSERT INTO main(uuid, line, distance, iriXp, iriYp, iriZp, iriX, iriY, iriZ, time, phoneID)
           VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'''
  cur = conn.cursor()
  try:
    cur.execute(sql, data)
    conn.commit()
    return ("Successfully inserted with ID of " + str(cur.lastrowid))
  except Exception as e:
    print(e)
    print(data)
    return ("Failed to Insert!  Duplicate ID or UUID?")
  cur.close()

# Initial setup and testing
def setUp():
  conn = connect()
  
  create_table_sql = """CREATE TABLE IF NOT EXISTS `main` (
                      `id` INT NOT NULL AUTO_INCREMENT,
                      `uuid` TEXT,
                      `line` TEXT,
                      `distance` REAL,
                      `iriXp` REAL,
                      `iriYp` REAL,
                      `iriZp` REAL,
                      `iriX` REAL,
                      `iriY` REAL,
                      `iriZ` REAL,
                      `filepath` TEXT,
                      `time` REAL,
                      `phoneID` TEXT,
                      PRIMARY KEY (`id`), UNIQUE KEY `uuid` (`uuid`(255))
                      );"""
  
  if conn is not None:
    cursor = conn.cursor()
    cursor.execute(create_table_sql)
  else:
    print("Error establishing database connection")   
  conn.close()


# Primary function we will receive data into
# Function will parse the data and insert it into the table
def receive_data(args):
  #print(args)
  #setUp()
  geojsonstring = json.loads(args['geojson_string'])
  line = str(geojsonstring['features'][0]['geometry'])
  properties = geojsonstring['features'][0]['properties']
  uuid = properties['ID']
  distance = properties['DISTANCE']
  iriXp = properties['IRIphoneX']
  iriYp = properties['IRIphoneY']
  iriZp = properties['IRIphoneZ']
  iriX = properties['IRIearthX']
  iriY = properties['IRIearthY']
  iriZ = properties['IRIearthZ']
  time = properties['timestamp']
  phoneID = properties['PHONEID']
  conn = connect()
  data = (uuid, line, distance, iriXp, iriYp, iriZp, iriX, iriY, iriZ, time, phoneID)
  print(insert_row(conn, data))
  conn.commit()
  conn.close()
  print("SUCCESS")
  return "Successfully inserted new rows!"