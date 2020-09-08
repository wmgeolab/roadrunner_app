package com.example.roadrunner_geolab.SurfaceDoctor;

public class SurfaceDoctorPoint {
    public String id;

    private float[] linearAccelerometerPhone;
    private float[] linearAccelerometerEarth;

    public float[] gravity;
    public float[] magnetometer;

    public long timeCreated;

    private long tStart;
    private long tStop;


    public SurfaceDoctorPoint(String inId, float[] accelPhone, float[] accelEarth, float[] inGravity, float[] inMagnetometer, long startTime, long stopTime) {
        id = inId;
        linearAccelerometerPhone = accelPhone;
        linearAccelerometerEarth = accelEarth;
        gravity = inGravity;
        magnetometer = inMagnetometer;
        tStart = startTime;
        tStop = stopTime;

        timeCreated = System.currentTimeMillis();
    }

    public double getVertDissX(boolean returnEarthCoordinateSystem) {
        double timeDiff = (tStop - tStart) / 1000000000.0;

        if (returnEarthCoordinateSystem) {
            return linearAccelerometerEarth[0] * timeDiff * timeDiff;
        } else {
            return linearAccelerometerPhone[0] * timeDiff * timeDiff;
        }
    }

    public double getVertDissY(boolean returnEarthCoordinateSystem) {
        double timeDiff = (tStop - tStart) / 1000000000.0;

        if (returnEarthCoordinateSystem) {
            return linearAccelerometerEarth[1] * timeDiff * timeDiff;
        } else {
            return linearAccelerometerPhone[1] * timeDiff * timeDiff;
        }
    }


    public double getVertDissZ(boolean returnEarthCoordinateSystem) {
        double timeDiff = (tStop - tStart) / 1000000000.0;

        if (returnEarthCoordinateSystem) {
            return linearAccelerometerEarth[2] * timeDiff * timeDiff;
        } else {
            return linearAccelerometerPhone[2] * timeDiff * timeDiff;
        }
    }

    public String getRowString() {
        StringBuilder str = new StringBuilder(id + ", ");
        str.append(linearAccelerometerPhone[0] + ", ");
        str.append(linearAccelerometerPhone[1] + ", ");
        str.append(linearAccelerometerPhone[2] + ", ");
        str.append(linearAccelerometerEarth[0] + ", ");
        str.append(linearAccelerometerEarth[1] + ", ");
        str.append(linearAccelerometerEarth[2] + ", ");
        str.append(gravity[0] + ", ");
        str.append(gravity[1] + ", ");
        str.append(gravity[2] + ", ");
        str.append(magnetometer[0] + ", ");
        str.append(magnetometer[1] + ", ");
        str.append(magnetometer[2] + ", ");
        str.append(timeCreated + ", ");
        str.append(tStart + ", ");
        str.append(tStop);

        return  str.toString();
    }
}
