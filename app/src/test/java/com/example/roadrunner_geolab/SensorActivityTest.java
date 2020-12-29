package com.example.roadrunner_geolab;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

public class SensorActivityTest extends SensorActivity {
    SensorActivity sensorActivity;
    @Before
    public void setUp() throws Exception {
        //create mock sensorActivity
        sensorActivity = mock(SensorActivity.class);
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void testPostData() {
        //call successCallback when postData is called
        doAnswer( invocation -> {
            sensorActivity.successCallback();
            return null;
        }).when(sensorActivity).postData();

        sensorActivity.postData();

        //check that successCallback called one time
        verify(sensorActivity, times(1)).successCallback();
    }
}