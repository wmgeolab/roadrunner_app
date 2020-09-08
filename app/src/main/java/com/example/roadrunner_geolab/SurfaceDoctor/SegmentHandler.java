package com.example.roadrunner_geolab.SurfaceDoctor;

import android.content.Context;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;
import android.location.Location;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public class SegmentHandler {

    // Default user input parameters.
    private boolean units = true;
    private int maxDistance = 10;
    private int maxSpeed = 8000;
    private int minSpeed = 20;

    private Context context;
    private SensorManager sensorManager;

    private String uniqueID = "\"" + UUID.randomUUID().toString() + "\"";

    private long accelerometerStartTime = 0;
    private long accelerometerStopTime = 0;
    private List<com.example.roadrunner_geolab.SurfaceDoctor.SurfaceDoctorPoint> surfaceDoctorPoints = new ArrayList<com.example.roadrunner_geolab.SurfaceDoctor.SurfaceDoctorPoint>();

    private float[] gravity = new float[3];
    private float[] magnetometer = new float[3];

    private float alpha = 0.8f;

    private boolean hasLocationPairs = false;
    private Location endPoint;
    private ArrayList<double[]> segmentCoordinates = new ArrayList<>();


    public static double[] IRI;


    private float totalAccumulatedDistance = 0.0f;

    // Required to create an Event.
    private com.example.roadrunner_geolab.SurfaceDoctor.SurfaceDoctorInterface listener;
    public void setSomeEventListener (com.example.roadrunner_geolab.SurfaceDoctor.SurfaceDoctorInterface listener) {
        this.listener = listener;
    }

    /**
     * Constructors
     *
     * @param context
     */
    public SegmentHandler(Context context, SensorManager sm) {

        this.context = context;
        sensorManager = sm;
    }


    /**
     * Method for setting SurfaceDoctor settings that were defined by the user.
     *
     * @param inputUnits
     * @param inputSegmentDistance
     * @param inputMaxSpeed
     * @param inputMinSpeed
     */
    public void setSurfaceDoctorPreferences(boolean inputUnits, int inputSegmentDistance,
                                             int inputMaxSpeed, int inputMinSpeed) {
        // TODO: This needs implementing.
        units = inputUnits;
        maxDistance = inputSegmentDistance;
        maxSpeed = inputMaxSpeed;
        minSpeed = inputMinSpeed;
    }


    /**
     * Method for receiving accelerometer data from sensors.
     *
     * @param sensorEvent
     */
    public void setSurfaceDoctorAccelerometer(SensorEvent sensorEvent) {

        // We need time between accelerometer events to calculate the IRI.
        // TODO: Verify what time this is.
        accelerometerStartTime = accelerometerStopTime;
        accelerometerStopTime = sensorEvent.timestamp;

        // Check if we have location pairs and start & stop times. If so, we start collecting SurfaceDoctorPoint objects.
        // hasLocationPairs is set to true by the setSurfaceDoctorLocation method when to location event's have been
        // recorded.
        if ( hasLocationPairs && accelerometerStartTime > 0 ) {

            float[] inputAccelerometer = sensorEvent.values.clone();

            // Low-pass filter for isolating gravity.
            float[] adjustedGravity = new float[3];

            adjustedGravity[0] = alpha * gravity[0] + (1 - alpha) * inputAccelerometer[0];
            adjustedGravity[1] = alpha * gravity[1] + (1 - alpha) * inputAccelerometer[1];
            adjustedGravity[2] = alpha * gravity[2] + (1 - alpha) * inputAccelerometer[2];

            // High-pass filter for removing gravity.
            float[] linearAccelerationPhone = new float[3];

            linearAccelerationPhone[0] = inputAccelerometer[0] - adjustedGravity[0];
            linearAccelerationPhone[1] = inputAccelerometer[1] - adjustedGravity[1];
            linearAccelerationPhone[2] = inputAccelerometer[2] - adjustedGravity[2];

            // Convert Accelerometer from phone coordinate system to earth coordinate system.
            float[] linearAccelerationEarth = com.example.roadrunner_geolab.SurfaceDoctor.VectorAlgebra.earthAccelerometer(
                    linearAccelerationPhone, magnetometer,
                    gravity, sensorManager);

            // For each measurement of the accelerometer, create a SurfaceDoctorPoint object.
            com.example.roadrunner_geolab.SurfaceDoctor.SurfaceDoctorPoint surfaceDoctorPoint = new com.example.roadrunner_geolab.SurfaceDoctor.SurfaceDoctorPoint(
                    uniqueID,
                    linearAccelerationPhone,
                    linearAccelerationEarth,
                    gravity,
                    magnetometer,
                    accelerometerStartTime,
                    accelerometerStopTime);
            surfaceDoctorPoints.add(surfaceDoctorPoint);
        }
    }


    /**
     * Receives Gravity from Android Sensor Callback
     *
     * @param sensorEvent
     */
    public void setSurfaceDoctorGravity(SensorEvent sensorEvent ) {
        gravity = sensorEvent.values.clone();
    }

    public void setSurfaceDoctorMagnetometer(SensorEvent sensorEvent) { magnetometer = sensorEvent.values.clone(); }


    /** Method for receiving Location data from the GPS sensor.
     *
     * @param inputLocation
     */
    public void setSurfaceDoctorLocation(Location inputLocation) {

        // If we have a pair of location objects, let's run through our logic.
        if (endPoint != null) {

            // Let's first tell our accelerometer sensors to start summing data.
            hasLocationPairs = true;

            // Now let's make the current point the old point, and update the current point with the new point. We'll
            // use these location pairs to extract data later.
            Location startPoint = endPoint;
            endPoint = inputLocation;

            // We're logging, let's process the data.
            executeSurfaceDoctor(startPoint, endPoint );

            // TODO: We'll need to create an event that lets MainActivity know if we're within speed, logging, etc. This could also be handled on MainActivity side.
            double lineBearing = inputLocation.getBearing();

        } else {
            // This is our first point, our logic depends on a comparison of two location objects, so let's do nothing
            // until we get that second location.
            endPoint = inputLocation;
        }
    }

    /**
     *  This is the main logic handler of the SegmentHandler.
     *
     *  This is fired at every GPS location callback from Android.
     */
    private void executeSurfaceDoctor(Location locationStart, Location locationEnd) {

        // The approximate distance in meters.
        float lineDistance = locationStart.distanceTo(locationEnd);

        // The speed in m/s
        float lineSpeed = locationEnd.getSpeed();

        // Long / Lat of the Location pairs.
        double[] coordinatesStart = new double[]{ locationStart.getLongitude(), locationStart.getLatitude() };
        double[] coordinatesLast = new double[]{ locationEnd.getLongitude(), locationEnd.getLatitude() };


        Log.d("segment123", String.valueOf(maxDistance));
        Log.d("segment333", String.valueOf(totalAccumulatedDistance));

        // We're within speed and haven't reached the end of a segment, let's add the distance between the coordinate
        // pairs to the total distance of the segment.
        if ( isWithinSpeed( lineSpeed ) && !isSegmentEnd() ) {
            // Append the distance between the coordinate points to the total distance.
            totalAccumulatedDistance += lineDistance;
            // Append the new coordinates to the ArrayList so we can create a polyline later.
            segmentCoordinates.add(coordinatesStart);

            Log.i("IRI", "Distance: " + totalAccumulatedDistance);
        }
        // We're within speed and reached the end of a segment, let's finalize the segment.

        //isWithinSpeed( lineSpeed ) &&
        else if (isSegmentEnd() ) {
            // Append the distance between the coordinate points to the total distance.
            totalAccumulatedDistance += lineDistance;
            // Append the new coordinates to the ArrayList so we can create a polyline later.
            segmentCoordinates.add(coordinatesStart);
            // The segement is done, add the last coordinate.
            segmentCoordinates.add(coordinatesLast);

            // This is the end of a segment, let's send it off to be finalized.
            finalizeSegment(uniqueID, totalAccumulatedDistance, segmentCoordinates, surfaceDoctorPoints);

            // Let's reset for the next segment.
            // TODO: Is this safe here?
            resetSegment(false);
        }
        // We've exceeded our speed threshold, we need to do a hard reset.
        // TODO: What if we lose coordinate pairs or accelerometer data.
        else if ( !isWithinSpeed( lineSpeed )) {
            // The hard rest will clear out our location pairs, as well.
            resetSegment(true );
        }
        else {
            Log.i("SEG", "A condition was met that we didn't think about. ");
        }
    }

    /**
     *  Finalizes a road segment
     *
     *  Executes when the segment distance threshold has been met.      *
     */
    private void finalizeSegment(String id ,double distance, ArrayList<double[]> polyline, List<com.example.roadrunner_geolab.SurfaceDoctor.SurfaceDoctorPoint> measurements) {

        // Create the header for the output table.
        StringBuilder tableString = new StringBuilder("id,AccPhoneX,AccPhoneY,AccPhoneZ,AccEarthX,AccEarthY,AccEarthZ,");
        tableString.append("GravityX,GravityY,GravityZ,");
        tableString.append("MagnetX,MagnetY,MagnetZ,");
        tableString.append("Created,sStart,sStop,Distance");
        tableString.append(System.getProperty("line.separator"));

        double[] totalVerticalDisplacementPhone = new double[3];
        double[] totalVerticalDisplacementEarth = new double[3];

        // First get the total vertical displacement of the segment in both Earth and Phone coordinate system.
        for (int i = 0; i < measurements.size(); i++ ) {
            int previousIndex = i - 1;

            // Append the SurfaceDoctorPoint as a row in tableString so we can output a table later.
            tableString.append(measurements.get(i).getRowString());
            tableString.append(", " + distance);
            tableString.append(System.getProperty("line.separator"));

            // Vertical Displacement equals the absolute value of the current longitudinal offset minus the previous
            // longitudinal offset.
            if ( previousIndex >= 0 ) {

                totalVerticalDisplacementPhone[0] += Math.abs( measurements.get(i).getVertDissX(false) - measurements.get(previousIndex).getVertDissX(false) );
                totalVerticalDisplacementPhone[1] += Math.abs( measurements.get(i).getVertDissY(false) - measurements.get(previousIndex).getVertDissY(false) );
                totalVerticalDisplacementPhone[2] += Math.abs( measurements.get(i).getVertDissZ(false) - measurements.get(previousIndex).getVertDissZ(false) );

                totalVerticalDisplacementEarth[0] += Math.abs( measurements.get(i).getVertDissX(true) - measurements.get(previousIndex).getVertDissX(true) );
                totalVerticalDisplacementEarth[1] += Math.abs( measurements.get(i).getVertDissY(true) - measurements.get(previousIndex).getVertDissY(true) );
                totalVerticalDisplacementEarth[2] += Math.abs( measurements.get(i).getVertDissZ(true) - measurements.get(previousIndex).getVertDissZ(true) );
            }
        }

        // IRI(mm/m) = (total vertical displacement * 1000) / segment distance.
        // TODO: Need to allow the user to output IRI in mm/m or m/km.
        double[] totalIRIPhone = new double[3];
        double[] totalIRIEarth = new double[3];

        totalIRIPhone[0] = (totalVerticalDisplacementPhone[0] * 1000) / distance;
        totalIRIPhone[1] = (totalVerticalDisplacementPhone[1] * 1000) / distance;
        totalIRIPhone[2] = (totalVerticalDisplacementPhone[2] * 1000) / distance;

        totalIRIEarth[0] = (totalVerticalDisplacementEarth[0] * 1000) / distance;
        totalIRIEarth[1] = (totalVerticalDisplacementEarth[1] * 1000) / distance;
        totalIRIEarth[2] = (totalVerticalDisplacementEarth[2] * 1000) / distance;

        Log.i("IRI", "Xphone " + totalIRIPhone[0] + " Yphone " + totalIRIPhone[1] + " Zphone " + totalIRIPhone[2] +
                " XEarth " + totalIRIEarth[0] + " YEarth " + totalIRIEarth[1] + " ZEarth " + totalIRIEarth[2]);

        // Let's pass the segment data to the SurfaceDoctorEvent so it can be used in the main activity.
        // The SurfaceDoctorEvent will be triggered in the MainActivity.
        if (listener != null) {
            com.example.roadrunner_geolab.SurfaceDoctor.SurfaceDoctorEvent e = new com.example.roadrunner_geolab.SurfaceDoctor.SurfaceDoctorEvent();
            e.type = "TYPE_SEGMENT_IRI";
            e.x = totalIRIPhone[0];
            e.y = totalIRIPhone[1];
            e.z = totalIRIPhone[2];
            e.distance = distance;
            // TODO: Assign output to SurfaceDoctorEvent class.
            listener.onSurfaceDoctorEvent(e);
        }
        saveResults(id, distance, totalIRIPhone, totalIRIEarth, polyline, tableString.toString());
    }

    private void saveResults(String id, double distance, double[] phoneIRI, double[] earthIRI, ArrayList<double[]> polyline, String table) {
        // TODO: Instead of one file per segment, append multiple segments to one file.
        Log.d("location333", "saveeve");
        Map<String, Double> hm = new HashMap<String, Double>();
        hm.put("XearthIRI", earthIRI[0]);
        hm.put("YearthIRI", earthIRI[1]);
        hm.put("ZearthIRI", earthIRI[2]);
        //userRef.setValue(hm);
        Log.d("location3333",  " XEarth " + earthIRI[0] + " YEarth " + earthIRI[1] + " ZEarth " + earthIRI[2]);

        // Check if we have access to external storage?
        if ( isExternalStorageWritable() ) {
            // TODO: Save file as geoJSON or EsriJSON.

            // Add results to GeoJSON.

            // Create geoJSON string.
            // Tried using JASONobject and JASONArray, but couldn't git rid of quotes around coordinates. Didn't have
            // access to GSON library due to network permissions.
            StringBuilder str = new StringBuilder("{\"type\": \"FeatureCollection\", \"features\": [");
            str.append("{\"type\": \"Feature\", \"geometry\":{ \"type\": \"LineString\",");
            str.append("\"coordinates\":" + Arrays.deepToString(polyline.toArray()) + "},");
            str.append("\"properties\": {");
            str.append("\"ID\":" + id + ",");
            str.append("\"DISTANCE\":" + distance + ",");
            str.append("\"timestamp\":" + System.currentTimeMillis() + ",");
            str.append("\"IRIphoneX\":" + phoneIRI[0] + ",");
            str.append("\"IRIphoneY\":" + phoneIRI[1] + ",");
            str.append("\"IRIphoneZ\":" + phoneIRI[2] + ",");
            str.append("\"IRIearthX\":" + earthIRI[0] + ",");
            str.append("\"IRIearthY\":" + earthIRI[1] + ",");
            str.append("\"IRIearthZ\":" + earthIRI[2] + "}}");

            str.append("]}");

            // Let's save the geoJSON string.
            File file = getPrivateStorageDirectory(context, String.valueOf(accelerometerStartTime) + ".geojson");
            try {
                file.createNewFile();
                //Log.e("FILE", file.getAbsolutePath());
                OutputStreamWriter fstream = new OutputStreamWriter( new FileOutputStream(file), StandardCharsets.UTF_16);
                fstream.write(str.toString());
                fstream.flush();
                fstream.close();
                //Log.e("SAVED", "Saved geoJson:\n");
            } catch (IOException e) {
                //Log.e("ERROR", "Failed to create file with error:\n");
            }

            // Let's save the table as a txt file
            byte[] tableBytes = table.getBytes();
            File tableFile = getPrivateStorageDirectory(context, String.valueOf(accelerometerStartTime) + ".csv");
            try {
                FileOutputStream fos = new FileOutputStream(tableFile);
                try {
                    fos.write(tableBytes);
                    fos.close();
                } catch (IOException e) {
                    Log.e("ERROR", "IO exception");
                }
            } catch (FileNotFoundException e) {
                Log.e("ERROR", "File not found");
            }



        } else {
            // TODO: If no access external storage, let's save to internal until we do get access.
        }
    }


    private boolean isWithinSpeed( float inputSpeed ) {
        // TODO: Need to handle which units the user selected.
        float mph = inputSpeed * 2.23694f;
        return minSpeed <= mph && mph <= maxSpeed;
    }


    private boolean isSegmentEnd() {
        // TODO: Need to handle user's input units.
        return totalAccumulatedDistance >= maxDistance;
    }

    /**
     * Resets all the parameters.
     *
     * Used to restart the logging of a road segment.
     *
     * @param hardReset boolean
     */
    private void resetSegment(boolean hardReset) {

        // Get new segment ID
        uniqueID =  "\"" + UUID.randomUUID().toString() + "\"";

        // Reset the segment distance to zero.
        totalAccumulatedDistance = 0;

        // Clear all accelerometer measurements from the ArrayList.
        surfaceDoctorPoints.clear();

        // Clear the list of coordinates that make the polyline. 
        segmentCoordinates.clear();

        // Hard resets are used when a sensor loses connectivity or passes a threshold.
        if ( hardReset ) {
            hasLocationPairs = false;
            endPoint = null;
            accelerometerStopTime = 0;
        }
    }


    /* Checks if external storage is available for read and write */
    public boolean isExternalStorageWritable() {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state)) {
            return true;
        }
        return false;
    }

    /** Creates an empty file that can be written to.
     *
     * @param context
     * @param fileName
     * @return
     */
    private File getPrivateStorageDirectory(Context context, String fileName) {
        File file = new File(context.getExternalFilesDir("geoJson"), fileName);
        return file;
    }

}
