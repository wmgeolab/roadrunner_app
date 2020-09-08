------------------------------------------------------------------------------
           ____  ____  ___ ____     ____  _   _    _   _ _   _ ______ ____
          / __ \/ __ \/   |  _ \   / _  \| | | |  / | / / | / /  ___// _  \
         / /_/ / / / / /\ | | | | / /_/ /| | | | /  |/ /  |/ /  __| / /_/ /
       	/ _  _/ /_/ / __  | |_| |/ _  _/ | |_| |/ /|  / /|  /  /___/ _  _/
       /_|_|_|\____/_/  \_|____ /_|_|_|  \_____/_/ |_/_/ |_/______/_|_|_|
        
******************************************************************************
RoadRunner: Road Roughness Collection App

By Eric Nubbe
Email: enubbe@email.wm.edu
Last Updated : May 2020
******************************************************************************

-------------
INDEX
-------------
 1-Introduction

 2-File and Function Descriptions
	a) Files For Both Versions
		i) location
	       ii) SettingsFragment
	      iii) AlertDialogGPS
	       iv) SegmentHandler
		v) SurfaceDoctorInterface
	       vi) SurfaceDoctorEvent
	      vii) SurfaceDoctorPoint
	     viii) Vector Algebra

	b) v1 Specific Files
		i) SensorActivity
	c) v2 Specific Files
		i) SensorActivity2
	       ii) SensorService
              iii) Notification Bar

 3-How the App Works
	a) General Communication Between Files
	b) Data Collection - Activity 		  (RoadRunnerv1)
	c) Data Collection - Activity and Service (RoadRunnerv2)
	d) Communication with Server/Sending Data
	e) Other

 4-Modifying the App - Where to Look

==============================================================================
1-Introduction
==============================================================================

Hello! If you're reading this file, you're probably either trying to modify the
app, are studying it for educational purposes, or you were randomly clicking 
through the GeoDev Cloudera folder and found this by accident. Whatever the 
reason, this file should help you understand how RoadRunner works.

RoadRunner is an app developed for GeoDev, a subteam of William & Mary's 
geoLab. geoLab is the world's largest provider of open data on administrative 
boundaries and works on many different GIS, machine learning, and satellite
imagery projects. The goal of this particular project is to train a neural 
network to be able to predict the roughness of a road through satellite 
imagery alone. In order to build this network, we needed data to build the 
model. This is where RoadRunner comes in to play. When a user turns on the app
and drives, the app collects roughness data using the phone's acceleramators.
It then sends this data to a backend server.

Much of the code for this project that calculates the roughness came from the 
open-source AndroidIRI by mebuie. Using their code allowed me to focus on 
collecting data and connecting to the server to train the network rather than
the math that goes into calculating the road roughness. Their original project 
can be found here: https://github.com/mebuie/AndroidIRI

==============================================================================
2-File and Function Descriptions
==============================================================================

RoadRunner has a lot of files and it is not always clear what each one is 
supposed to be doing. Here are descirptions of what each file does along with
descriptions of certain functions that are important or confusing.

a) Files For Both Versions

	i) location

	   This file defines the locations class, which holds the current
	   IRI values for the three axes (x,y,z). It also defines functions
	   that set and retrieve the differnt IRI values.

       ii) SettingsFragment

	   This file extends the PreferenceFragmentCompat class and presents
	   a Preference object to the user. It allows the user to change 
           preferences listed in the preference.xml file. These preferences
	   include segment distance and minimum logging speed.

      iii) AlertDialogGPS
	   
	   This file builds an alert when the user does not have their GPS
	   enabled. It offers the user two buttons: one the enables the 
	   GPS and one that leaves GPS turned off.

       iv) SegmentHandler

	   A very important file. This file builds the strings that contain
	   all the IRI information and are eventually sent to the server. It
	   creates a new string every time the selected logging distance is 
           driven and saves the old one in the storage.

	   The class recieves input in the form of Sensor events - gravity,
	   acceleration, etc - and adds the data as SurfaceDoctorPoint.
	   When the segment is finalized (by completing a segment or the speed
	   becoming too fast/slow), it calls the finalizeSegment method. This
	   method builds both a table with the information and calls the
	   saveResults method. This is also where the IRI is calculated!

	   The saveResult method creates a GeoJSON with all the relevant
	   information (but not all the data that was in the corresponding
	   table!) and saves it in the private storage directory. 
	   
	   NOTE: If you are using an emulator, the private storage directory 
	   may be labelled with a name that does not correspond to the actual 
	   folder name in the file system. To find out where the files are 
	   saved, print the absolute path of the file and navigate to it on 
	   the Device File Explorer.

	   Each string is given a unique ID through the UUIS.randomUUID method
	   (search for "uniqueID" to find it).

	v) SurfaceDoctorInterface

	   The interface for the SurfaceDoctorEvent (helpful, I know). This was
	   from the AndroidIRI project and I kept it in even though it seems 
	   useless.

       vi) SurfaceDoctorEvent

	   Defines the SurfaceDoctorEvent object. Each instance of the class has
	   a few doubles - IRI values for each axes and a distance. 
	   
      vii) SurfaceDoctorPoint

	   Defines a class for individual points. A point is created and updated 
	   when the set distance is travelled. Each point has gravity, acceleration,
	   and time created/finalized. Each segment is made up of many points (see
           SegmentHandler for list of SurfaceDoctorPoints that make up a segment).

     viii) Vector Algebra

	   Switches the phone's accerlation data from the phone's personal xyz axes
           system to the Earth's axes. Basically, it allows the phone to be in any
	   orientation when data is collected and the output IRI values (x,y,z) to 
           always refer to the same directions.

b) v1 Specific Files

	i) SensorActivity

	   This is the heart, brain, and small intestine of the app. It extends the
           AppCompatActivity class, which means it has a menu on the top (for the
	   user preferences) and is an Activity (it updates the GUI).	

	   When the app starts, it opens the XML layout associated with the activity
	   and starts it. All the buttons and textfields are initialized as well as
	   the sensors. When the user wants to collect data, they hit the button.
 	   The button calls methods from the other files which continually collect
	   data (more details in part 3). When the user wants to send the data to the
	   server, they simply click the "Push Data" button. 

	   NOTE: This button pushes ALL collected geoJSONS to the server. The app does
	   not explicitly delete files after pushing them in case there is a failure
	   somewhere in the pipeline. Therefore, it tries to send all the files it has
	   and lets the server decide which strings are repeats and which are new data
 	   by using the string's ID.

c) v2 Specific Files
	
	i) SensorActivity2

	   This file is essentially juts the GUI components from the SensorActivity in
	   v1. It still has all the buttons and text fields, but it no longer collects
	   the data. Instead, it is bound to the SensorService which collects the data,
	   manages the segments, and sends some information back to the activity to be
	   displayed on the GUI.

       ii) SensorService

	   This file is the other half of the v1 SensorActivity. It is responsible for
	   calling all the data collection and segment handler methods from the old file.
	   It recieves some information from SensorActivity2 (like updates to the GPS
	   permissions) and sends some information back to update the screen.

	   The purpose of this file is to allow the app to collect data when the screen
	   is off or the app has been pushed to the background. Activities can only run
	   when the screen is on and the app is running on the main screen. Putting all
	   the data collection parts of the app in the service means that the phone can 
	   have other GUI-required tasks open without stopping the data collection 
	   (remember, never text and drive/collect geospatial data).

      iii) Notification Bar
  
           This is a cookie-cutter file that simply creates a notication bar when the 
           service is running. The bar is created in the onCreate method of the service
   	   and just lets the user know that the app is collecting data.

==============================================================================
3-How the App Works
==============================================================================

a) General Communication Between Files

   Let's start with the files in the SurfaceDoctor folder. These files are fairly
   self-contained between eachother and all work together to build the geoJSONs 
   that the server collects.

   When a SegmentHandler is created, many fields are instantiated. Many of these
   fields are used to hold data collected by the phone's sensors. One important 
   field is a private list of SurfaceDoctorPoints. These points are what makes up 
   a single segment.

   When the setSurfaceDoctorAccelerometer(SensorEvent sensorEvent) is called in 
   SegmentHandler, the start/stop time, gravity, and accleration values are all 
   recorded. Then, these values are saved as a SurfaceDoctorPoint in the list of 
   points created earlier. This series of events happnes every time there is a new
   accerlation reading.

   There are similar methods for listening for Gravity, Magnometer, and Location. 
   These methods save the information in fields but do not create new 
   surfaceDoctorPoints.

   The executeSurfaceDoctor method is the main logic control for the SegmentHandler.
   It checks that the phone is moving within the max/min speed preferences and 
   that the segment has not exceded the maxDistance (length of each segment). When 
   either of those things happends, the Segement Handler calls finalizeSegment 
   and resetSegment.

   The finalizeSegment method builds a table with all the information from that
   segment and then saves the segment into memory with the saveResults method.
   This is also where the acutally IRI values are calculated! 
   It also passes a surfaceDoctorEvent to a listener. This listener passes the 
   information in the surfaceDoctorEvent (IRI values)to the activity which then
   updates the GUI with those measurements.

   Those are the main points of the SurfaceDoctor files. But who calls the methods 
   in the files? And how? The answer depends on which version of the app you are 
   looking at.

b) Data Collection - Activity (RoadRunnerv1)

   The first half of the SensorActivity is mostly import statements, creating
   buttons and text fields, and initialzing all the sensor listeners in the
   onCreate/onStart methods. The true communication and data collection begins
   in the section marked "Begin Sensor Callbacks".
   
   NOTE: The segementHandler is created in the enableGPS method that appears
   in the "Permissions and Settings" sectoin prior to the sensor callbacks.
   This segmentHandler object is what the activiy communicates with in the
   sensor callbacks.

   The onSensorChanged method continually recieves sensor events from the 
   different sensors it created in the first part of the file. It's response
   depends on which sensor is calling it. If it is the accerlation sensor,
   the method records the acceleration and calls the 
   setSurfaceDoctorAccelerometer method of the segmentHandler (see part a of
   this section of the guide). The method also displays the orientation and
   accerlation.

   In the section marked "Surface Doctor", the activity updates the IRI text
   fields with the most recent IRI reading. It accomplishes this through the
   surfaceDoctorEvent method. Remember, the segmentHandler calls this method
   and passes a surfaceDoctorEvent object in the finalizeSegment method.

c) Data Collection - Activity and Service (RoadRunnerv2)

   This version of the function has one major difference: the one activity
   has been spilt into an activity and a service. An Android Activity updates
   the GUI, while a Service is used to complete long-running tasks in the
   background. The purpose of splitting the main functions of the app into
   these two parts is so the app can collect data while the phone is asleep
   or another app is in the foreground. 

   All the GUI components from the original activity are still in the activity.
   All the methods that collect data and interact with the segmentHandler have
   been moved to the service. 

   The service is created when the Activity sends an intent with the startService
   method. The service is not bound to the activity - note that the onBind method
   in the service does nothing. Instead, the service is started in the foreground
   where it collects data somewhat independently of the activity. The service
   passes information - such as the current IRI values - to the activity through
   Broadcasts, which the activity listens for in its BroadcastReceivers. The 
   activity also passes information such as updates to the preferences or permissions.
    
   Since the service saves data in the memory of the phone, the activty does not 
   have to communicate with the service to push data to the server. Instead, when 
   the Send Data button is pushed, the activity simply looks in the shared folder.

d) Communication with Server/Sending Data
   
   In either version of the app, the methods for sending the collected data
   to the server can be found in the activities.

   First, in the first part of the activity an onClickListener is set on the
   "Push Data" button. This listener executes the sendHTMLTask command. It is
   also where the URL for the server is defined/passed to the method. If you
   want to change where the data is passed, just change this URL.

   Second, in the onCreate method there is a part that begins with a comment
   "Android Volley". I have to admit ... the next 20 lines are copy-and-pasted
   from StackOverflow. Without this code, the sendHTMLTask method raises
   some sort of security exception. Including these 20 lines here fixes that
   problem. If you ever get some sort of security error, check here first.

   The central part of server communication is the sendHTMLTask class. It is
   located at the bottom of the activity file. It extends the AsyncTask class.
   This means that when the method is executed, it runs on a background thread
   to avoid slowing the other methods (like data collection) in case it takes 
   too long. When the method is passed a URL, it:
   	- loops through every file in the phone's storage ending with "geojson" 
        - for every file it finds, it reads the file as a string
        - it then puts that string geoJSON into a JSON 
        - finally, it posts (sends) that string to the URL

   Notice that the line "file.delete" is commented out. The app never explicity
   deletes these files in case something happens to the data in the server and
   we want to push again. Of course, if that app's data is cleared then the
   files might be deleted in that manner.

e) Other

   The VectorAlegbra file is used by SegmentHandler to get the orientation of
   the phone and cast it into a regular Earth coordinate system. 

   The surfaceDoctorInterface just lists a function prototype used by the 
   surfaceDoctorEvent. In the finalizeSegement method in SegmentHandler,
   the listener calls this event, which passes the surfaceDoctorEvent to the
   main activity so the IRI information can be displayed on screen.


==============================================================================
4-Modifying the App - Where to Look
==============================================================================

If your goal is to change the app in some way, this section should help you.
(If it doesn't help, let me just apologize in advance.) I've organized it in
a Q&A style.

Q: We changed the server and now we need the app to send data to a different 
   place. What do I need to change?

A: All you need to change is the URL passed to the sendHTMLTask class. Look
   at the pushFilesListener in the activity and simply change the URL. There
   is also an access key in the method that pushes the JSONS (located at the
   bottom of the Activity) that might also need to be changed.

Q: I'm having trouble POST-ing the data to the server, I keep getting error 
   codes like 404 or 500. What should I do?

A: This app uses Android Volley (Android's recommened HTTP library) to make
   HTTP requests. I would recommend looking up their documentation if the
   app is returning classic HTTP errors like 404 instead of actually posting
   data to the server. 

Q: I want to change the format of the JSON passed to the server (but not the
   geoJSON that actually has the roughness data). Where should I look?

A: The JSON is built in the sendHTMLTask class in the activity. The method
   doInBackground first converts the geoJSONs in the storage into a string
   and then puts the strings into JSON files. To modify the JSON sent to the
   server, look at the code under the comment "HTTP Post geojson" and either
   add, remove, or modify the fields of the JSON. If the server is having 
   trouble reading the new JSON, put one in a JSON parser to make sure that
   you formatted it correctly.

Q: I want to change the geoJSON, the one that has all the data. What should I
   modify?

A: You'll most likely want to look within the SegmentHandler. The geoJSON is
   built in the finalizeSegment and saveResults methods. The finalizeSegment
   method builds a table and fills it with information from each of the 
   points in the list of surfaceDoctorPoints. It then passes this table (as
   a string) into the saveResults method.

   The saveResults method builds the geoJSON and saves it. If you want to 
   change the format of the geoJSON, change where they are saved on the phone,
   or save other information, this is the method you should modify.
   
Q: I want the screen to display more/less/different information. Where do I
   begin?

A: First, you'll need to modify the text fields in the XML layout file that
   is connected to the activity. Look in the res folder and then in the 
   layout subfolder.

   Next, you'll need to modify the fields in the acitivity. Search for the
   fields that are of type "TextView". The are connected to the XML file 
   in the onCreate method. If you want to removed some of these fields or
   add any, this is the place to do it.
 
   The IRI values that appear on screen are created and updated in the 
   onSurfaceDoctorEvent method. This method is called by the segmentHandler,
   so it is possible that you will need to look there if you want to 
   modify these TextViews.

Q: Why are there three IRI measurments? Which one is important?

A: The app calculates an X, Y, and Z IRI. The X is east/west, the Y is 
   north/south, and the Z is up/down. Since we want to know how bumpy the
   roads are, the Z IRI value is the one we care about.

Q: How can I test the app if I change it?

A: You can either use the Android Emulator or plug in an Android phone. I
   would recommend using an Android phone if you have one. At one point, the 
   app worked on the emulator but had a small problem when running on actual
   harware.

   If you want to test data collection but don't want to drive around with
   a phone plugged into a laptop or are using an emulator, you can simulate 
   movenment using a GPX file (a type of XML used by GPS). When using the
   emulator, click the three dots to open up the advance option. Then look
   for a button that says "GPX". You will be able to select your GPX fiel 
   and make the emulator think it is travelling along that path.
