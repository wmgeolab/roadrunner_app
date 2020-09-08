package com.example.roadrunner_geolab.SurfaceDoctor;

import android.hardware.SensorManager;

public class VectorAlgebra {

    /**
     * Converts accelerometer readings from the Phone's xyz coordinates to the Earth's coodrinate system.
     *
     * X axis = East / West
     * Y axis = North / South magnetic pole
     * Z axis = Up / Down
     *
     * @param accelerometer float[] An array containing the phone's accelerometer data [0:X, 1:Y, 2:Z]
     * @param magnetometer float[]
     * @param gravity float[]
     * @param sensorManager SensorManager Android SensorManager
     * @return
     */
    public static float[] earthAccelerometer(float[] accelerometer, float[] magnetometer, float[] gravity, SensorManager sensorManager) {

        float[] phoneAcceleration = new float[4];
        phoneAcceleration[0] = accelerometer[0];
        phoneAcceleration[1] = accelerometer[1];
        phoneAcceleration[2] = accelerometer[2];
        phoneAcceleration[3] = 0;

        // Change the device relative acceleration values to earth relative values
        // X axis -> East
        // Y axis -> North Pole
        // Z axis -> Sky

        float[] R = new float[16], I = new float[16], earthAcceleration = new float[16], earthAccelerationFinal = new float[3];

        sensorManager.getRotationMatrix(R, I, gravity, magnetometer);

        float[] inv = new float[16];

        android.opengl.Matrix.invertM(inv, 0, R, 0);
        android.opengl.Matrix.multiplyMV(earthAcceleration, 0, inv, 0, phoneAcceleration, 0);

        earthAccelerationFinal[0] = earthAcceleration[0];
        earthAccelerationFinal[1] = earthAcceleration[1];
        earthAccelerationFinal[2] = earthAcceleration[2];

        return earthAccelerationFinal;

    }


    /**
     * Returns the phone's orientation values in radians.
     *
     * @param accelerometer
     * @param magnetometer
     * @param sensorManager
     * @return
     */
    public static float[] phoneOrientation(float[] accelerometer, float[] magnetometer, SensorManager sensorManager) {

        // Empty Float array to hold the rotation matrix.
        float[] rotationMatrix = new float[9];
        // Empty Float array to hold the azimuth, pitch, and roll.
        float orientationValues[] = new float[3];

        // Not sure exactly how this works, but populates the matrix with the input data. rotationOK returns true if the
        // .getRotationMatrix method is successful.
        // "You can transform any vector from the phone's coordinate system to the Earth's coordinate system by
        // multiplying it with the rotation matrix."
        boolean rotationOK = sensorManager.getRotationMatrix(rotationMatrix,
          null, accelerometer, magnetometer);

        // If the getRotationMatrix method is successful run the following code,
        // TODO Do I need this at all?.
        if (rotationOK) {

            sensorManager.getOrientation(rotationMatrix, orientationValues);

        }

        return orientationValues;
    }


    /**
     * Converts phone orientation values from radians to degrees.
     *
     * @param inputRadians float[] An array of orientation values.
     * @return
     */
    public static double[] radiansToDegrees(float[] inputRadians) {

        double[] outputDegrees = new double[inputRadians.length];

        for (int i=0; i < inputRadians.length; i++) {
            outputDegrees[i] = inputRadians[i] * (180/Math.PI);
        }

        return outputDegrees;
    }
}
